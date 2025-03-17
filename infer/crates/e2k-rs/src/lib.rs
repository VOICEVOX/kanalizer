//! # e2k-rs
//!
//! [Patchethium氏のe2k](https://github.com/Patchethium/e2k)をRustに移植したものです。
//!
//! ## 使い方
//!
//! ```rust
//! // 文字列をカタカナに変換する例
//! let src = "constants";
//! let c2k = e2k::C2k::new(32);
//! let dst = c2k.infer(src);
//!
//! dbg!(dst); // "コンスタンツ"
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
mod inference;
mod layers;

pub use constants::{ASCII_ENTRIES, KANAS};
pub use inference::*;
