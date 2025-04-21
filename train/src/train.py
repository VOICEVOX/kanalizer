# we train a s2s model to predict the katakana phonemes from
# English phonemes
import argparse
from datetime import datetime
from functools import partial
import json
import os
from pathlib import Path
import random
import shutil
import subprocess

from g2p_en import G2p
from schedulefree import RAdamScheduleFree
import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import yaml

from config import Config
from constants import EOS_IDX, SOS_IDX, ascii_entries, kanas
from evaluator import Evaluator


class Model(nn.Module):
    def __init__(self, config: Config):
        super(Model, self).__init__()
        self.e_emb = nn.Embedding(len(ascii_entries), config.dim)
        self.k_emb = nn.Embedding(len(kanas), config.dim)
        self.encoder = nn.GRU(
            config.dim, config.dim, batch_first=True, bidirectional=True
        )
        self.encoder_fc = nn.Sequential(
            nn.Linear(2 * config.dim, config.dim),
            nn.Tanh(),
        )
        self.pre_decoder = nn.GRU(config.dim, config.dim, batch_first=True)
        self.post_decoder = nn.GRU(2 * config.dim, config.dim, batch_first=True)
        self.attn = nn.MultiheadAttention(config.dim, 4, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(config.dim, len(kanas))

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        src: [B, Ts]
        tgt: [B, Tt]
        src_mask: [B, Ts]
        tgt_mask: [B, Tt]
        """
        e_emb = self.e_emb(src)
        k_emb = self.k_emb(tgt)
        k_emb = k_emb[:, :-1]
        enc_out, _ = self.encoder(e_emb)
        enc_out = self.encoder_fc(enc_out)
        dec_out, _ = self.pre_decoder(k_emb)
        attn_out, _ = self.attn.forward(
            dec_out, enc_out, enc_out, key_padding_mask=~src_mask
        )
        x = torch.cat([dec_out, attn_out], dim=-1)
        x, _ = self.post_decoder(x)
        x = self.fc(x)
        return x

    def inference(self, src):
        # Assume both src and tgt are unbatched
        sos_idx = SOS_IDX
        eos_idx = EOS_IDX
        src = src.unsqueeze(0)
        src_emb = self.e_emb(src)
        enc_out, _ = self.encoder(src_emb)
        enc_out = self.encoder_fc(enc_out)
        res = [sos_idx]
        h1 = None
        h2 = None
        count = 0
        while res[-1] != eos_idx and count < 16:
            dec = torch.tensor([res[-1]]).unsqueeze(0).to(src.device)
            dec_emb = self.k_emb(dec)
            dec_out, h1 = self.pre_decoder(dec_emb, h1)
            attn_out, _ = self.attn(dec_out, enc_out, enc_out)
            x = torch.cat([dec_out, attn_out], dim=-1)
            x, h2 = self.post_decoder(x, h2)
            x = self.fc(x)
            idx = torch.argmax(x, dim=-1)
            res.append(idx.cpu().item())
            count += 1
        return res


class MyDataset(Dataset):
    def __init__(self, path, device, max_words: int | None):
        """
        reads a json line file
        """
        super().__init__()
        self.g2p = G2p()
        with open(path, "r") as file:
            lines = file.readlines()
        self.data = [json.loads(line) for line in lines]
        if max_words is not None:
            self.data = random.sample(self.data, min(max_words, len(self.data)))
        self.device = device
        self.c_dict = {c: i for i, c in enumerate(ascii_entries)}
        self.kata_dict = {c: i for i, c in enumerate(kanas)}
        self.sos_idx = SOS_IDX
        self.eos_idx = EOS_IDX
        self.cache_en = {}
        self.cache_kata = {}
        self.return_full = False

    def __len__(self):
        return len(self.data)

    def c2k(self, eng):
        eng = [self.c_dict[c] for c in eng]
        return eng

    def set_return_full(self, flag: bool):
        """
        Returns the full dataset, it's for bleu score calculation
        """
        self.return_full = flag

    def __getitem__(self, idx):
        if idx in self.cache_en:
            return self.cache_en[idx], self.cache_kata[idx]
        item = self.data[idx]
        eng = item["word"]
        katas = item["kata"]
        eng = self.c2k(eng)
        eng = [self.sos_idx] + eng + [self.eos_idx]
        # katas is a list of katakana words
        # if not return_full, we randomly select one of them
        # else we return all of them
        if not self.return_full:
            kata = katas[random.randint(0, len(katas) - 1)]
            kata = [self.kata_dict[c] for c in kata]
            kata = [self.sos_idx] + kata + [self.eos_idx]
            en = torch.tensor(eng).to(self.device)
            kana = torch.tensor(kata).to(self.device)
            self.cache_en[idx] = en
            self.cache_kata[idx] = kana
            return en, kana
        else:
            kata = []
            for k in katas:
                k = [self.kata_dict[c] for c in k]
                k = [self.sos_idx] + k + [self.eos_idx]
                kata.append(torch.tensor(k).to(self.device))
            en = torch.tensor(eng).to(self.device)
            self.cache_en[idx] = en
            self.cache_kata[idx] = kata
            return en, kata


def lens2mask(lens, max_length):
    mask = torch.zeros(len(lens), max_length).bool()
    for i, le in enumerate(lens):
        mask[i, :le] = True
    return mask


def collate_fn(batch, device):
    engs = [x[0] for x in batch]
    katas = [x[1] for x in batch]
    eng_lens = [len(x) for x in engs]
    kata_lens = [len(x) for x in katas]
    eng_mask = lens2mask(eng_lens, max(eng_lens))
    kata_mask = lens2mask(kata_lens, max(kata_lens))
    engs = pad_sequence(engs, batch_first=True, padding_value=0)
    katas = pad_sequence(katas, batch_first=True, padding_value=0)
    engs, katas, eng_mask, kata_mask = [
        x.to(device) for x in [engs, katas, eng_mask, kata_mask]
    ]
    return engs, katas, eng_mask, kata_mask


def infer(src, model):
    model = model.eval()
    res = model.inference(src)
    # return to words
    res = [kanas[i] for i in res]
    # also for english phonemes
    src = [ascii_entries[i] for i in src]
    return src, res


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    args = parser.parse_args()

    config = Config.from_dict(yaml.safe_load(args.config.read_text()))
    print(f"Using config: {config}")

    torch.manual_seed(config.seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    model = Model(config).to(device)
    train_dataset = MyDataset(config.train_data, device, max_words=None)
    eval_dataset = MyDataset(config.eval_data, device, max_words=config.eval_max_words)
    batch_size = 256 if use_cuda else 64
    print(f"Batch size: {batch_size}")

    output_dir = args.output or Path(
        os.path.join(
            "outputs", datetime.now().strftime(f"%Y_%m_%d_%H_%M_%S_{args.config.stem}")
        )
    )

    print(f"Output dir: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    best_scores = []

    shutil.copyfile(
        args.config,
        output_dir / "config.yml",
    )
    git_sha = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
    )
    with open(output_dir / "train_info.yml", "w") as file:
        yaml.dump(
            {
                "git_sha": git_sha,
                "use_cuda": use_cuda,
            },
            file,
        )

    train_dl = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, device=device),
        drop_last=True,
    )
    eval_dl = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, device=device),
    )

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = RAdamScheduleFree(model.parameters(), lr=config.optimizer_lr)
    writer = SummaryWriter(log_dir=output_dir)
    evaluator = Evaluator(eval_dataset)
    epochs = config.max_epochs
    steps = 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.train()
        for eng, kata, e_mask, k_mask in tqdm(train_dl, desc=f"Epoch {epoch} train"):
            optimizer.zero_grad()
            out = model(eng, kata, e_mask, k_mask)
            loss = criterion(out.transpose(1, 2), kata[:, 1:])
            writer.add_scalar("Loss/train", loss.item(), steps)
            loss.backward()
            optimizer.step()
            steps += 1
        model.eval()
        optimizer.eval()

        total_loss = 0
        total = 0
        with torch.no_grad():
            for eng, kata, e_mask, k_mask in tqdm(eval_dl, desc=f"Epoch {epoch} eval"):
                out = model(eng, kata, e_mask, k_mask)
                loss = criterion(out.transpose(1, 2), kata[:, 1:])
                total_loss += loss.item() * len(out)
                total += len(out)

        writer.add_scalar("Loss/eval", total_loss / total, epoch)
        print(f"Epoch {epoch} Loss: {total_loss / total}")

        # take a sample and inference it
        sample = eval_dataset[random.randint(0, len(eval_dataset) - 1)]
        src, tgt = sample
        src, pred = infer(src, model)
        print(f"Epoch {epoch} Sample: {src} -> {pred}")

        bleu = evaluator.evaluate(model)
        writer.add_scalar("BLEU", bleu, epoch)
        print(f"Epoch {epoch} BLEU: {bleu}")

        save_best_models(epoch, model, output_dir, config, best_scores, bleu)
        save_last_models(epoch, model, output_dir, config)


def save_best_models(
    current_epoch: int,
    model: Model,
    output_dir: Path,
    config: Config,
    best_scores: list[tuple[int, Tensor]],
    bleu: Tensor,
):
    best_scores.append((current_epoch, bleu))
    best_scores.sort(key=lambda x: x[1], reverse=True)

    removed_epoch = None
    if len(best_scores) > config.num_best_models_to_keep:
        removed_epoch, _ = best_scores.pop()

    if removed_epoch != current_epoch:
        torch.save(
            model.state_dict(),
            output_dir / f"model-best-e{current_epoch}.pth",
        )
        if removed_epoch is not None:
            path = output_dir / f"model-best-e{removed_epoch}.pth"
            print(f"Removing {path}")
            os.remove(path)


def save_last_models(
    current_epoch: int, model: Model, output_dir: Path, config: Config
):
    torch.save(
        model.state_dict(),
        output_dir / f"model-e{current_epoch}.pth",
    )
    if current_epoch - config.num_last_models_to_keep > 0:
        old = current_epoch - config.num_last_models_to_keep
        old_path = output_dir / f"model-e{old}.pth"
        if old_path.exists():
            print(f"Removing {old_path}")
            os.remove(old_path)


if __name__ == "__main__":
    train()
