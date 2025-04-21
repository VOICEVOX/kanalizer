"""
英単語列からカタカナ列を推論し、trainデータと同じjsonl形式で出力するスクリプト。
各単語ごとに {"word": ..., "kata": [...]} 形式のjsonを1行ずつ標準出力する。
例:
    uv run src/infer.py kanalizer ./outputs/2025_03_14_23_43_01_example
"""

import argparse
import json
import os
from pathlib import Path

import torch
import yaml
from config import Config
from constants import EOS_IDX, SOS_IDX, ascii_entries, kanas

from train import Model, MyDataset


def main():
    args = parse_args()
    device = get_device()
    config = load_config(args.output_dir)
    model = load_model(args.output_dir, config, device)
    for word in args.words:
        src_tensor = word_to_tensor(word, device)
        katakana = infer_katakana(model, src_tensor)
        obj = {"word": word, "kata": ["".join(katakana)]}
        print(json.dumps(obj, ensure_ascii=False))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("words", nargs="+", help="Input word(s) to infer")
    parser.add_argument("output_dir", help="Output directory with trained model")
    return parser.parse_args()


def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    return device


def load_config(output_dir: str) -> Config:
    config_path = Path(output_dir) / "config.yml"
    config = Config.from_dict(yaml.safe_load(config_path.read_text()))
    return config


def load_model(output_dir: str, config: Config, device: torch.device) -> Model:
    models = [f for f in os.listdir(output_dir) if f.startswith("model-e")]
    if not models:
        raise RuntimeError("No model found in output_dir")
    models.sort(key=lambda x: int(x.split("-")[1][1:-4]))
    model_path = Path(output_dir) / models[-1]
    model = Model(config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def word_to_tensor(word: str, device: torch.device) -> torch.Tensor:
    c_dict = {c: i for i, c in enumerate(ascii_entries)}
    indices = [c_dict.get(c, None) for c in word.lower()]
    if None in indices:
        raise ValueError("Input word contains invalid character")
    indices = [SOS_IDX] + indices + [EOS_IDX]
    return torch.tensor(indices, dtype=torch.long, device=device)


def infer_katakana(model: Model, src_tensor: torch.Tensor) -> list[str]:
    res = model.inference(src_tensor)
    return [kanas[i] for i in res if i != SOS_IDX and i != EOS_IDX]


if __name__ == "__main__":
    main()
