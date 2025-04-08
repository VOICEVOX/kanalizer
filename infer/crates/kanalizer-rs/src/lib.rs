//! # kanalizer-rs
//!
//! [Patchethium氏のe2k](https://github.com/Patchethium/e2k)をRustに移植したものです。
//!
//! ## 使い方
//!
//! ```rust
//! // 文字列をカタカナに変換する例
//! let src = "kanalizer";
//! let dst = kanalizer::convert(src).perform().unwrap();
//!
//! assert_eq!(dst, "カナライザー");
//! ```
//!
//! ## Features
//! ### `compress_model`
//! brotliを使用してモデルを圧縮します。
//! このfeatureはデフォルトで有効です。
//!
//! ### `getrandom_on_wasm32_unknown`
//! wasm32-unknown-unknownでのTopK/TopPサンプリングに`getrandom`を使用します。
//! このfeatureを有効にしてコンパイルするには[getrandomのドキュメント](https://docs.rs/getrandom/latest/getrandom/#webassembly-support)を参照してください。
//! オフの場合、Hashと適当な値を使用してサンプリングします。
//!

mod constants;
mod error;
mod inference;
mod layers;

use std::{collections::HashSet, num::NonZero, sync::LazyLock};

pub use constants::{ASCII_ENTRIES, KANAS};
pub use error::*;
pub use inference::*;

static KANALIZER: LazyLock<Kanalizer> = LazyLock::new(Kanalizer::new);

/// 推論を行うためのオプションを指定する構造体。
pub struct ConvertBuilder {
    word: String,
    options: ConvertOptions,
}

impl ConvertBuilder {
    fn new(word: &str) -> Self {
        Self {
            word: word.to_string(),
            options: ConvertOptions::default(),
        }
    }

    /// デコードの最大長を指定する。
    pub fn with_max_length(mut self, max_length: NonZero<usize>) -> Self {
        self.options.max_length = max_length;
        self
    }

    /// デコードに使うアルゴリズムを指定する。
    pub fn with_strategy(mut self, strategy: &Strategy) -> Self {
        self.options.strategy = strategy.clone();
        self
    }

    /// 入力を検証するかどうかを指定する。
    /// falseの場合、無効な文字は無視されます。
    pub fn with_strict(mut self, strict: bool) -> Self {
        self.options.strict = strict;
        self
    }

    /// 推論を行う。
    pub fn perform(self) -> Result<String> {
        KANALIZER.convert(&self.word, &self.options)
    }
}

/// 推論を行う。
pub fn convert(word: &str) -> ConvertBuilder {
    ConvertBuilder::new(word)
}

/// Kanalizerの入力に使える文字の一覧。
pub static INPUT_CHARS: LazyLock<HashSet<char>> = LazyLock::new(|| {
    [
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
        's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    ]
    .iter()
    .copied()
    .collect()
});

/// Kanalizerから出力されうる文字の一覧。
pub static OUTPUT_CHARS: LazyLock<HashSet<char>> = LazyLock::new(|| {
    [
        'ァ', 'ア', 'ィ', 'イ', 'ゥ', 'ウ', 'ェ', 'エ', 'ォ', 'オ', 'カ', 'ガ', 'キ', 'ギ', 'ク',
        'グ', 'ケ', 'ゲ', 'コ', 'ゴ', 'サ', 'ザ', 'シ', 'ジ', 'ス', 'ズ', 'セ', 'ゼ', 'ソ', 'ゾ',
        'タ', 'ダ', 'チ', 'ヂ', 'ッ', 'ツ', 'ヅ', 'テ', 'デ', 'ト', 'ド', 'ナ', 'ニ', 'ヌ', 'ネ',
        'ノ', 'ハ', 'バ', 'パ', 'ヒ', 'ビ', 'ピ', 'フ', 'ブ', 'プ', 'ヘ', 'ベ', 'ペ', 'ホ', 'ボ',
        'ポ', 'マ', 'ミ', 'ム', 'メ', 'モ', 'ャ', 'ヤ', 'ュ', 'ユ', 'ョ', 'ヨ', 'ラ', 'リ', 'ル',
        'レ', 'ロ', 'ヮ', 'ワ', 'ヰ', 'ヱ', 'ヲ', 'ン', 'ヴ', 'ー',
    ]
    .iter()
    .copied()
    .collect()
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_input_chars() {
        // INPUT_CHARSがASCII_ENTRIESのサブセットであることを確認する。
        let ascii_entries: HashSet<_> = constants::ASCII_ENTRIES
            .iter()
            .map(|&c| c.to_string())
            .collect();
        assert!(
            INPUT_CHARS
                .iter()
                .map(|&c| c.to_string())
                .collect::<HashSet<_>>()
                .is_subset(&ascii_entries)
        );
    }

    #[test]
    fn test_output_chars() {
        // OUTPUT_CHARSがKANASのサブセットであることを確認する。
        let kana_entries: HashSet<_> = constants::KANAS.iter().map(|&c| c.to_string()).collect();
        assert!(
            OUTPUT_CHARS
                .iter()
                .map(|&c| c.to_string())
                .collect::<HashSet<_>>()
                .is_subset(&kana_entries)
        );
    }
}
