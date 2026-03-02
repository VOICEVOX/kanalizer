"""
UniDicの辞書をダウンロードして、英単語と読みの対応を抽出するスクリプト。
単語は以下の条件を満たすもののみを抽出する：
- 大文字始まり、小文字2文字以上
- 英単語と単純なカタカナ読みが異なる（例えばSQL -> エスキューエルは除外される）
"""

import requests
import zipfile_deflate64 as zipfile
from tqdm import tqdm
import csv
import json
import re
import os


def main():
    if not os.path.exists("./vendor/unidic-cwj-202302_full.zip"):
        download_unidic("./vendor/unidic-cwj-202302_full.zip")
    if not os.path.exists("./vendor/unidic-lex.csv"):
        print("Extracting lex.csv")
        with zipfile.ZipFile("./vendor/unidic-cwj-202302_full.zip", "r") as zip_ref:
            zip_ref.extract("lex.csv", "./vendor")
        os.rename("./vendor/lex.csv", "./vendor/unidic-lex.csv")

    filter_unidic()


def download_unidic(path: str):
    url = "https://clrd.ninjal.ac.jp/unidic_archive/2302/unidic-cwj-202302_full.zip"
    print(f"Downloading {url} to {path}.tmp")
    r = requests.get(url, allow_redirects=True, stream=True)
    total_size = int(r.headers.get("content-length", 0))
    pb = tqdm(total=total_size, unit="B", unit_scale=True)
    with open(path + ".tmp", "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                pb.update(len(chunk))
    pb.close()
    print("Renaming to", path)

    if os.path.exists(path):
        os.remove(path)
    os.rename(path + ".tmp", path)


def filter_unidic():
    filter = re.compile(r"^[Ａ-Ｚ][ａ-ｚ]{2,}$")
    filtered = {}
    with open("./vendor/unidic-lex.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in tqdm(reader, desc="Filtering"):
            if filter.match(row[0]) and simple_alphabet_to_katakana(row[0]) != row[24]:
                filtered[zenkaku_to_hankaku(row[0]).lower()] = row[24]

    with open("./vendor/unidic_words.jsonl", "w", encoding="utf-8") as f:
        for k, v in tqdm(filtered.items(), desc="Writing"):
            f.write(json.dumps({"word": k, "kata": [v]}, ensure_ascii=False) + "\n")


def zenkaku_to_hankaku(eng: str):
    kata = ""
    for c in eng:
        if "Ａ" <= c <= "Ｚ":
            kata += chr(ord(c) - ord("Ａ") + ord("A"))
        elif "ａ" <= c <= "ｚ":
            kata += chr(ord(c) - ord("ａ") + ord("a"))
        else:
            kata += c

    return kata


def simple_alphabet_to_katakana(eng: str):
    kata = ""
    for char in eng:
        c = zenkaku_to_hankaku(char).lower()
        if c in ALPHABET_MAP:
            kata += ALPHABET_MAP[c]
    return kata


ALPHABET_MAP = {
    "a": "エー",
    "b": "ビー",
    "c": "シー",
    "d": "ディー",
    "e": "イー",
    "f": "エフ",
    "g": "ジー",
    "h": "エイチ",
    "i": "アイ",
    "j": "ジェー",
    "k": "ケー",
    "l": "エル",
    "m": "エム",
    "n": "エヌ",
    "o": "オー",
    "p": "ピー",
    "q": "キュー",
    "r": "アール",
    "s": "エス",
    "t": "ティー",
    "u": "ユー",
    "v": "ブイ",
    "w": "ダブリュー",
    "x": "エックス",
    "y": "ワイ",
    "z": "ゼット",
}

if __name__ == "__main__":
    main()
