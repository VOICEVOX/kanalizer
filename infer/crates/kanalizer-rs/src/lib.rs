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

use std::{collections::HashSet, sync::LazyLock};

pub use constants::{ASCII_ENTRIES, KANAS};
pub use error::*;
pub use inference::*;

static KANALIZER: LazyLock<Kanalizer> = LazyLock::new(Kanalizer::new);

/// 変換を行うためのオプションを指定する構造体。
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
    /// Noneの場合、入力長+2になります。
    pub fn with_max_length(mut self, max_length: MaxLength) -> Self {
        self.options.max_length = max_length;
        self
    }

    /// デコードに使うアルゴリズムを指定する。
    pub fn with_strategy(mut self, strategy: &Strategy) -> Self {
        self.options.strategy = strategy.clone();
        self
    }

    /// 入力に無効な文字が含まれている場合にエラーを返すかどうかを指定する。
    /// falseの場合、無効な文字は無視されます。
    pub fn with_error_on_invalid_input(mut self, error_on_invalid_input: bool) -> Self {
        self.options.error_on_invalid_input = error_on_invalid_input;
        self
    }

    /// 変換が終了しなかった場合にエラーを返すかどうかを指定する。
    pub fn with_error_on_incomplete(mut self, error_on_incomplete: bool) -> Self {
        self.options.error_on_incomplete = error_on_incomplete;
        self
    }

    /// 変換を行う。
    pub fn perform(self) -> Result<String> {
        KANALIZER.convert(&self.word, &self.options)
    }
}

/// 変換を行う。
pub fn convert(word: &str) -> ConvertBuilder {
    ConvertBuilder::new(word)
}

/// Kanalizerの入力に使える文字の一覧。
pub static INPUT_CHARS: LazyLock<HashSet<char>> = LazyLock::new(|| {
    [
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
        's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    ]
    .into()
});

/// Kanalizerから出力されうる文字の一覧。
pub static OUTPUT_CHARS: LazyLock<HashSet<char>> = LazyLock::new(|| {
    [
        'ァ', 'ア', 'ィ', 'イ', 'ゥ', 'ウ', 'ェ', 'エ', 'ォ', 'オ', 'カ', 'ガ', 'キ', 'ギ', 'ク',
        'グ', 'ケ', 'ゲ', 'コ', 'ゴ', 'サ', 'ザ', 'シ', 'ジ', 'ス', 'ズ', 'セ', 'ゼ', 'ソ', 'ゾ',
        'タ', 'ダ', 'チ', 'ヂ', 'ッ', 'ツ', 'ヅ', 'テ', 'デ', 'ト', 'ド', 'ナ', 'ニ', 'ヌ', 'ネ',
        'ノ', 'ハ', 'バ', 'パ', 'ヒ', 'ビ', 'ピ', 'フ', 'ブ', 'プ', 'ヘ', 'ベ', 'ペ', 'ホ', 'ボ',
        'ポ', 'マ', 'ミ', 'ム', 'メ', 'モ', 'ャ', 'ヤ', 'ュ', 'ユ', 'ョ', 'ヨ', 'ラ', 'リ', 'ル',
        'レ', 'ロ', 'ヮ', 'ワ', 'ヲ', 'ン', 'ヴ', 'ー',
    ]
    .into()
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
        #[rustfmt::skip] // TODO: Rust 2024に移行したらこの抑制をやめる
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
        #[rustfmt::skip] // TODO: Rust 2024に移行したらこの抑制をやめる
        assert!(
            OUTPUT_CHARS
                .iter()
                .map(|&c| c.to_string())
                .collect::<HashSet<_>>()
                .is_subset(&kana_entries)
        );
    }
}
