# kanalizer/dataset

LLMを用いて、英単語から読みを推測するためのデータセットを生成します。

## データセットの生成方法

1. `config.example.yml`を`config.yml`にコピーします。
2. `config.yml`を編集します。\
   設定値についてはコメントを参照してください。\
   また、[YAML Language Server](https://github.com/redhat-developer/yaml-language-server)を使用すると、設定値の補完が可能です。
3. `pnpm run start`を実行します。

## データセットの結合

```
pnpm run tools:mergeDatasets [結合元のデータセット1] [結合元のデータセット2] ... [結合先のデータセット]
```
