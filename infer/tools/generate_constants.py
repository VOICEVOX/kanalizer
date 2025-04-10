import subprocess
import sys
from common import train_root, infer_root

sys.path.append(str(train_root / "src"))

import constants  # type: ignore -- sys.path.append で追加したパスが pyright に認識されないため


destination = infer_root / "crates" / "kanalizer-rs" / "src" / "constants.rs"
content = f"""
// `generate_constants.py` により生成。
// このファイルは直接編集しないでください。

pub const SOS_IDX: usize = {constants.SOS_IDX};
pub const EOS_IDX: usize = {constants.EOS_IDX};
pub const KANAS: &[&str] = &[
{", ".join([f'"{kana}"' for kana in constants.kanas])}
];
pub const ASCII_ENTRIES: &[&str] = &[
{", ".join([f'"{entry}"' for entry in constants.ascii_entries])}
];
"""

formatted_content = subprocess.run(
    ["rustfmt", "--edition", "2024"],
    input=content.encode(),
    stdout=subprocess.PIPE,
).stdout.decode()

with open(destination, "w") as f:
    f.write(formatted_content)
