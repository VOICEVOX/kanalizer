# kanalizer

英単語から読みを推測するライブラリ。\
関連Issue：[VOICEVOX/voicevox_project#65](https://github.com/VOICEVOX/voicevox_project/issues/65)

## リポジトリ構造

このリポジトリは以下の構造になっています。

- `infer/`：読みを推測するためのコード。
- `train/`：モデルを学習するためのコード。
- `dataset/`：データセットを生成するためのコード。

## Hugging Face

Hugging Face にて、データセットと学習済みモデルを公開しています。\
- データセット：<https://huggingface.co/datasets/VOICEVOX/kanalizer-dataset>
- モデル：<https://huggingface.co/VOICEVOX/kanalizer-model>

## GitHub Actions のバージョン固定

[pinact](https://github.com/suzuki-shunsuke/pinact) を使って GitHub Actions のバージョンを full-length commit SHA に固定しています。
プルリクエストを送ると自動でテストされます。

```bash
# バージョンを固定する
pinact run

# バージョンを更新して固定する
pinact run --update --min-age 7
```

## ライセンス

このリポジトリのコードはMITライセンスのもとで公開されています。
