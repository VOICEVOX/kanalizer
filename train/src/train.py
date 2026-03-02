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
from typing import Literal

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


def word_to_tensor(word: str, device: torch.device) -> torch.Tensor:
    c_dict = {c: i for i, c in enumerate(ascii_entries)}
    indices = [SOS_IDX] + [c_dict[c] for c in word] + [EOS_IDX]
    return torch.tensor(indices, device=device)


class CheckpointManager:
    def __init__(
        self,
        label: str,
        model: nn.Module,
        output_dir: Path,
        num_models_to_keep: int,
        compare_mode: Literal["max", "min"],
    ) -> None:
        self.label = label
        self.model = model
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._limit = num_models_to_keep
        self._compare_mode = compare_mode
        self._models: dict[int, float] = {}

    def update(self, new_epoch: int, score: int | float) -> None:
        worst_epoch = self._worst({**self._models, new_epoch: score})

        if len(self._models) < self._limit or worst_epoch != new_epoch:
            self._models[new_epoch] = score
            torch.save(self.model.state_dict(), self._path_for(new_epoch))

        if len(self._models) > self._limit:
            self._models.pop(worst_epoch)
            self._path_for(worst_epoch).unlink()

    def _worst(self, models: dict[int, float]) -> int:
        _factor = 1 if self._compare_mode == "max" else -1
        return min(models.items(), key=lambda x: x[1] * _factor)[0]

    def _path_for(self, epoch: int) -> Path:
        return self.output_dir / f"model-{self.label}-e{epoch}.pth"


class Model(nn.Module):
    def __init__(self, config: Config):
        super(Model, self).__init__()
        self.e_emb = nn.Embedding(len(ascii_entries), config.dim)
        self.k_emb = nn.Embedding(len(kanas), config.dim)
        self.encoder = nn.GRU(
            config.dim, config.dim, batch_first=True, bidirectional=True
        )
        self.encoder_norm = nn.LayerNorm(2 * config.dim)
        self.encoder_fc = nn.Sequential(
            nn.Linear(2 * config.dim, config.dim),
            nn.Tanh(),
        )
        self.pre_decoder = nn.GRU(config.dim, config.dim, batch_first=True)
        self.pre_dec_norm = nn.LayerNorm(config.dim)
        self.attn = nn.MultiheadAttention(config.dim, 4, batch_first=True, dropout=0.1)
        self.attn_norm = nn.LayerNorm(config.dim)
        self.post_decoder = nn.GRU(2 * config.dim, config.dim, batch_first=True)
        self.post_dec_norm = nn.LayerNorm(config.dim)
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
        enc_out = self.encoder_norm(enc_out)
        enc_out = self.encoder_fc(enc_out)
        dec_out, _ = self.pre_decoder(k_emb)
        dec_out = self.pre_dec_norm(dec_out)
        attn_out, _ = self.attn.forward(
            dec_out, enc_out, enc_out, key_padding_mask=~src_mask
        )
        attn_out = self.attn_norm(attn_out)
        x = torch.cat([dec_out, attn_out], dim=-1)
        x, _ = self.post_decoder(x)
        x = self.post_dec_norm(x)
        x = self.fc(x)
        return x

    def inference(self, src):
        # Assume both src and tgt are unbatched
        sos_idx = SOS_IDX
        eos_idx = EOS_IDX
        src = src.unsqueeze(0)
        src_emb = self.e_emb(src)
        enc_out, _ = self.encoder(src_emb)
        enc_out = self.encoder_norm(enc_out)
        enc_out = self.encoder_fc(enc_out)
        res = [sos_idx]
        h1 = None
        h2 = None
        count = 0
        while res[-1] != eos_idx and count < 16:
            dec = torch.tensor([res[-1]]).unsqueeze(0).to(src.device)
            dec_emb = self.k_emb(dec)
            dec_out, h1 = self.pre_decoder(dec_emb, h1)
            dec_out = self.pre_dec_norm(dec_out)
            attn_out, _ = self.attn(dec_out, enc_out, enc_out)
            attn_out = self.attn_norm(attn_out)
            x = torch.cat([dec_out, attn_out], dim=-1)
            x, h2 = self.post_decoder(x, h2)
            x = self.post_dec_norm(x)
            x = self.fc(x)
            # 1ステップ目はeosを出力させない
            if count == 0:
                x_ = x.clone()
                x_[0, 0, eos_idx] = float("-inf")
                idx = torch.argmax(x_, dim=-1)
            else:
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
        with open(path, "r", encoding="utf-8") as file:
            lines = file.readlines()
        self.data = [json.loads(line) for line in lines]
        if max_words is not None:
            self.data = random.sample(self.data, min(max_words, len(self.data)))
        self.device = device
        self.kata_dict = {c: i for i, c in enumerate(kanas)}
        self.sos_idx = SOS_IDX
        self.eos_idx = EOS_IDX
        self.cache_en = {}
        self.cache_kata = {}
        self.return_full = False

    def __len__(self):
        return len(self.data)

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
        en = word_to_tensor(eng, self.device)
        # katas is a list of katakana words
        # if not return_full, we randomly select one of them
        # else we return all of them
        if not self.return_full:
            kata = katas[random.randint(0, len(katas) - 1)]
            kata = [self.kata_dict[c] for c in kata]
            kata = [self.sos_idx] + kata + [self.eos_idx]
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

    config = Config.from_dict(yaml.safe_load(args.config.read_text(encoding="utf-8")))
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
    train_dataset, test_dataset, eval_dataset = prepare_datasets(config, device)
    batch_size = 256
    print(f"Batch size: {batch_size}")

    output_dir = args.output or Path(
        os.path.join(
            "outputs", datetime.now().strftime(f"%Y_%m_%d_%H_%M_%S_{args.config.stem}")
        )
    )

    print(f"Output dir: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 最新epochを保存する
    latest_keeper = CheckpointManager(
        "latest", model, output_dir, config.num_last_models_to_keep, "max"
    )
    # BLEUスコアが高いものを保存する
    bleu_keeper = CheckpointManager(
        "best-bleu", model, output_dir, config.num_best_models_to_keep, "max"
    )
    # Loss/evalが低いものを保存する
    loss_keeper = CheckpointManager(
        "best-loss", model, output_dir, config.num_best_models_to_keep, "min"
    )

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
    test_dl = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, device=device),
    )
    eval_dl = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, device=device),
    )

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = RAdamScheduleFree(
        model.parameters(), lr=config.optimizer_lr, weight_decay=config.weight_decay
    )
    writer = SummaryWriter(log_dir=output_dir)
    test_evaluator = Evaluator(test_dataset)
    eval_evaluator = Evaluator(eval_dataset)
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

        # calculate loss
        test_loss = calculate_loss("test", model, criterion, test_dl, epoch)
        eval_loss = calculate_loss("eval", model, criterion, eval_dl, epoch)
        test_bleu = calculate_bleu(model, test_evaluator)
        eval_bleu = calculate_bleu(model, eval_evaluator)

        report_scalar(
            "loss",
            "test",
            test_loss,
            epoch,
            writer,
        )
        report_scalar(
            "loss",
            "eval",
            eval_loss,
            epoch,
            writer,
        )
        report_scalar(
            "bleu",
            "test",
            test_bleu,
            epoch,
            writer,
        )
        report_scalar(
            "bleu",
            "eval",
            eval_bleu,
            epoch,
            writer,
        )

        # take a sample and inference it
        sample = test_dataset[random.randint(0, len(test_dataset) - 1)]
        src, tgt = sample
        src, pred = infer(src, model)
        print(f"Epoch {epoch} Sample: {src} -> {pred}")

        # save the model
        latest_keeper.update(epoch, epoch)
        bleu_keeper.update(epoch, float(test_bleu))
        loss_keeper.update(epoch, test_loss)


def prepare_datasets(config: Config, device: torch.device):
    train_and_test_dataset = MyDataset(config.train_data, device, max_words=None)
    train_dataset, test_dataset = torch.utils.data.random_split(
        train_and_test_dataset,
        [
            1.0 - config.test_ratio,
            config.test_ratio,
        ],
    )

    eval_dataset = MyDataset(config.eval_data, device, max_words=config.eval_max_words)

    return (
        train_dataset,
        test_dataset,
        eval_dataset,
    )


def calculate_loss(
    label: str,
    model: Model,
    criterion: nn.Module,
    dl: DataLoader,
    epoch: int,
):
    total_loss = 0
    total = 0
    with torch.no_grad():
        for eng, kata, e_mask, k_mask in tqdm(dl, desc=f"Epoch {epoch} {label}"):
            out = model(eng, kata, e_mask, k_mask)
            loss = criterion(out.transpose(1, 2), kata[:, 1:])
            total_loss += loss.item() * len(out)
            total += len(out)

    return total_loss / total


def calculate_bleu(
    model: Model,
    evaluator: Evaluator,
) -> Tensor:
    return evaluator.evaluate(model)


def report_scalar(
    kind: str,
    label: str,
    value: Tensor,
    epoch: int,
    writer: SummaryWriter,
):
    writer.add_scalar(f"{kind}/{label}", value, epoch)
    print(f"Epoch {epoch} {label} {kind}: {value}")


if __name__ == "__main__":
    train()
