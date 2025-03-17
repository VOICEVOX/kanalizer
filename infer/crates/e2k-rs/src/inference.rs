use crate::{constants, layers};
use educe::Educe;
use itertools::Itertools;
use std::{collections::HashMap, hash::Hash};

/// デコードに使うアルゴリズム。
///
/// [StrategyTopK] 、 [StrategyTopP] も参照。
#[derive(Debug)]
pub enum Strategy {
    Greedy,
    TopK(StrategyTopK),
    TopP(StrategyTopP),
}

/// Top-Kアルゴリズムのパラメータ。
#[derive(Debug, Educe)]
#[educe(Default)]
pub struct StrategyTopK {
    #[educe(Default(expression = 3))]
    pub k: usize,
}

/// Top-Pアルゴリズムのパラメータ。
#[derive(Debug, Educe)]
#[educe(Default)]
pub struct StrategyTopP {
    #[educe(Default(expression = 0.9))]
    pub top_p: f32,
    #[educe(Default(expression = 1.0))]
    pub temperature: f32,
}

#[cfg(any(
    not(all(target_arch = "wasm32", target_os = "unknown")),
    feature = "getrandom_on_wasm32_unknown"
))]
fn generate_random<T: num_traits::ToBytes>(_data: &[T]) -> usize {
    use rand::Rng;
    let mut rng = rand::rng();
    rng.random::<f64>()
        .to_bits()
        .wrapping_rem(usize::MAX as u64) as usize
}

#[cfg(all(
    target_arch = "wasm32",
    target_os = "unknown",
    not(feature = "getrandom_on_wasm32_unknown")
))]
fn generate_random<T: num_traits::ToBytes>(data: &[T]) -> usize {
    use std::hash::Hasher;

    static HASH_SEED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    let random_pointer = Box::into_raw(Box::new(())) as usize;
    let val = HASH_SEED.fetch_add(random_pointer, std::sync::atomic::Ordering::Relaxed);
    val.hash(&mut hasher);
    for d in data {
        d.to_le_bytes().hash(&mut hasher);
    }

    let data = hasher.finish() as usize;
    HASH_SEED.fetch_add(data, std::sync::atomic::Ordering::Relaxed);
    data
}

struct S2s {
    e_emb: layers::Embedding,
    k_emb: layers::Embedding,
    encoder: layers::Gru,
    encoder_reverse: layers::Gru,
    encoder_fc: layers::Linear,
    pre_decoder: layers::Gru,
    post_decoder: layers::Gru,
    attn: layers::Mha,
    fc: layers::Linear,
    max_length: usize,

    strategy: Strategy,
}

fn get_array_f16<E, D>(
    weights: &safetensors::SafeTensors,
    key: &str,
) -> ndarray::ArrayBase<ndarray::OwnedRepr<E>, D>
where
    E: ndarray_safetensors::Float16ConversionSupportedElement,
    D: ndarray::Dimension,
{
    ndarray_safetensors::parse_fp16_tensor_view_data(
        &weights
            .tensor(key)
            .unwrap_or_else(|e| panic!("model corrupted: {key} not found, {e:?}")),
    )
    .unwrap_or_else(|e| panic!("model corrupted: failed to parse {key}, {e:?}"))
    .into_dimensionality()
    .unwrap_or_else(|e| panic!("model corrupted: dimension mismatch in {key}, {e:?}"))
}

impl S2s {
    fn new(weights: safetensors::SafeTensors, max_length: usize) -> Self {
        let e_emb = layers::Embedding::new(get_array_f16(&weights, "e_emb.weight"));
        let k_emb = layers::Embedding::new(get_array_f16(&weights, "k_emb.weight"));
        let encoder = layers::Gru::new(
            layers::GruCell::new(
                get_array_f16(&weights, "encoder.weight_ih_l0"),
                get_array_f16(&weights, "encoder.weight_hh_l0"),
                get_array_f16(&weights, "encoder.bias_ih_l0"),
                get_array_f16(&weights, "encoder.bias_hh_l0"),
            ),
            false,
        );
        let encoder_reverse = layers::Gru::new(
            layers::GruCell::new(
                get_array_f16(&weights, "encoder.weight_ih_l0_reverse"),
                get_array_f16(&weights, "encoder.weight_hh_l0_reverse"),
                get_array_f16(&weights, "encoder.bias_ih_l0_reverse"),
                get_array_f16(&weights, "encoder.bias_hh_l0_reverse"),
            ),
            true,
        );
        let encoder_fc = layers::Linear::new(
            get_array_f16(&weights, "encoder_fc.0.weight"),
            get_array_f16(&weights, "encoder_fc.0.bias"),
        );
        let pre_decoder = layers::Gru::new(
            layers::GruCell::new(
                get_array_f16(&weights, "pre_decoder.weight_ih_l0"),
                get_array_f16(&weights, "pre_decoder.weight_hh_l0"),
                get_array_f16(&weights, "pre_decoder.bias_ih_l0"),
                get_array_f16(&weights, "pre_decoder.bias_hh_l0"),
            ),
            false,
        );
        let post_decoder = layers::Gru::new(
            layers::GruCell::new(
                get_array_f16(&weights, "post_decoder.weight_ih_l0"),
                get_array_f16(&weights, "post_decoder.weight_hh_l0"),
                get_array_f16(&weights, "post_decoder.bias_ih_l0"),
                get_array_f16(&weights, "post_decoder.bias_hh_l0"),
            ),
            false,
        );
        let attn = layers::Mha::new(
            get_array_f16(&weights, "attn.in_proj_weight"),
            get_array_f16(&weights, "attn.in_proj_bias"),
            get_array_f16(&weights, "attn.out_proj.weight"),
            get_array_f16(&weights, "attn.out_proj.bias"),
            4,
        );
        let fc = layers::Linear::new(
            get_array_f16(&weights, "fc.weight"),
            get_array_f16(&weights, "fc.bias"),
        );
        let strategy = Strategy::Greedy;
        Self {
            e_emb,
            k_emb,
            encoder,
            encoder_reverse,
            encoder_fc,
            pre_decoder,
            post_decoder,
            attn,
            fc,
            max_length,
            strategy,
        }
    }

    fn greedy(&self, step_dec: &ndarray::ArrayView1<f32>) -> usize {
        let max = *step_dec
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let argmax = step_dec.iter().position(|&x| x == max).unwrap();
        argmax
    }

    fn top_k(&self, step_dec: &ndarray::ArrayView1<f32>, k: usize) -> usize {
        let step_dec = step_dec.to_vec();
        let random = generate_random(&step_dec);
        let mut indices = (0..step_dec.len()).collect::<Vec<_>>();
        indices.sort_unstable_by(|&i, &j| step_dec[j].partial_cmp(&step_dec[i]).unwrap());
        indices.truncate(k);

        indices[random % indices.len()]
    }

    fn top_p(&self, step_dec: &ndarray::ArrayView1<f32>, top_p: f32, temperature: f32) -> usize {
        let random = generate_random(&step_dec.to_vec());
        let step_dec = step_dec.exp() / temperature;
        let sum = step_dec.sum();
        let step_dec = step_dec / sum;
        let mut sorted = step_dec.iter().copied().enumerate().collect::<Vec<_>>();
        sorted.sort_unstable_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
        let mut i = 0;
        let mut cumsum = 0.0;
        while cumsum < top_p {
            cumsum += sorted[i].1;
            i += 1;
        }
        let candidates = sorted[..i].iter().map(|(i, _)| *i).collect::<Vec<_>>();

        candidates[random % candidates.len()]
    }

    fn decode(&self, x: &ndarray::ArrayView1<f32>) -> usize {
        match &self.strategy {
            Strategy::Greedy => self.greedy(x),
            Strategy::TopK(StrategyTopK { k }) => self.top_k(x, *k),
            Strategy::TopP(StrategyTopP { top_p, temperature }) => {
                self.top_p(x, *top_p, *temperature)
            }
        }
    }

    fn forward(&self, source: &ndarray::Array1<usize>) -> ndarray::Array1<usize> {
        let e_emb = self.e_emb.forward(source);
        let (enc_out, _) = self.encoder.forward(&e_emb.view(), None);
        let (enc_out_rev, _) = self.encoder_reverse.forward(&e_emb.view(), None);
        let enc_out = ndarray::concatenate(
            ndarray::Axis(enc_out.ndim() - 1),
            &[enc_out.view(), enc_out_rev.view()],
        )
        .unwrap();
        let enc_out = self.encoder_fc.forward_2d(&enc_out.view());
        let enc_out = enc_out.mapv(|x| x.tanh());
        let mut result = vec![constants::SOS_IDX];
        let mut h1: Option<ndarray::Array1<f32>> = None;
        let mut h2: Option<ndarray::Array1<f32>> = None;
        for _ in 0..self.max_length {
            let dec_emb = self
                .k_emb
                .forward(&ndarray::Array1::from_elem(1, *result.last().unwrap()));
            let (dec_out, h1_) = match h1 {
                Some(h1) => self.pre_decoder.forward(&dec_emb.view(), Some(h1.view())),
                None => self.pre_decoder.forward(&dec_emb.view(), None),
            };
            h1 = Some(h1_);
            let attn_out = self
                .attn
                .forward(&dec_out.view(), &enc_out.view(), &enc_out.view());
            let x = ndarray::concatenate(
                ndarray::Axis(dec_out.ndim() - 1),
                &[dec_out.view(), attn_out.view()],
            )
            .unwrap();
            let (x, h2_) = match h2 {
                Some(h2) => self.post_decoder.forward(&x.view(), Some(h2.view())),
                None => self.post_decoder.forward(&x.view(), None),
            };
            h2 = Some(h2_);
            let x = self.fc.forward_2d(&x.view());
            let x = x.index_axis(ndarray::Axis(0), 0);
            result.push(self.decode(&x));
            if result.last().unwrap() == &constants::EOS_IDX {
                break;
            }
        }

        ndarray::Array1::from(result)
    }
}

/// [C2k] の基底となる構造体。
/// 基本的には[C2k]を使ってください。
pub struct BaseE2k<I: Hash + Eq, O: Clone> {
    s2s: S2s,
    in_table: HashMap<I, usize>,
    out_table: HashMap<usize, O>,
}

impl<I: Hash + Eq, O: Clone> BaseE2k<I, O> {
    /// 新しいインスタンスを生成する。
    ///
    /// # Arguments
    ///
    /// - `tensors`: モデルの重み。必要な値についてはS2sの実装を参照してください。
    /// - `in_table`: 入力のテーブル。キーが入力、値がモデルの入力に変換されるインデックス。
    /// - `out_table`: 出力のテーブル。キーがモデルの出力に変換されるインデックス、値が出力。
    /// - `max_length`: 読みの最大長。
    pub fn new(
        tensors: safetensors::SafeTensors,
        in_table: HashMap<I, usize>,
        out_table: HashMap<usize, O>,
        max_length: usize,
    ) -> Self {
        Self {
            s2s: S2s::new(tensors, max_length),
            in_table,
            out_table,
        }
    }
    fn infer(&self, input: &[I]) -> Vec<O> {
        let source = input
            .iter()
            .filter_map(|c| self.in_table.get(c).copied())
            .collect_vec();
        if source.is_empty() {
            return Vec::new();
        }
        let source = [constants::SOS_IDX]
            .into_iter()
            .chain(source)
            .chain([constants::EOS_IDX]);
        let source = ndarray::Array1::from_iter(source);
        let target = self.s2s.forward(&source);
        target
            .iter()
            .skip(1)
            .take_while(|&&x| x != constants::EOS_IDX)
            .map(|&x| self.out_table[&x].clone())
            .collect()
    }

    fn set_decode_strategy(&mut self, strategy: Strategy) {
        self.s2s.strategy = strategy;
    }
}

/// 英単語 -> カタカナの変換器。
pub struct C2k {
    inner: BaseE2k<String, char>,
}

impl std::fmt::Debug for C2k {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("C2k").finish()
    }
}

impl C2k {
    /// 新しいインスタンスを生成する。
    ///
    /// # Arguments
    ///
    /// - `max_length`: 読みの最大長。
    pub fn new(max_length: usize) -> Self {
        static MODEL: std::sync::LazyLock<Vec<u8>> = std::sync::LazyLock::new(|| {
            cfg_elif::expr::cfg!(if (docsrs) {
                Vec::new()
            } else if (feature == "compress_model") {
                {
                    use std::io::Read;
                    let model = include_bytes!(concat!(
                        env!("E2K_MODEL_ROOT"),
                        "/model-c2k.safetensors.br"
                    ));
                    let mut input = brotli_decompressor::Decompressor::new(model.as_slice(), 4096);
                    let mut buf = Vec::new();
                    input.read_to_end(&mut buf).expect("Model is corrupted");
                    buf
                }
            } else {
                include_bytes!(concat!(env!("E2K_MODEL_ROOT"), "/model-c2k.safetensors")).to_vec()
            })
        });
        let weights = safetensors::SafeTensors::deserialize(&MODEL).expect("Model is corrupted");
        let inner = BaseE2k::new(
            weights,
            constants::ASCII_ENTRIES
                .iter()
                .enumerate()
                .map(|(i, &c)| (c.to_string(), i))
                .collect(),
            constants::KANAS
                .iter()
                .enumerate()
                .map(|(i, &c)| {
                    (
                        i,
                        c.chars()
                            .next()
                            .expect("Unreachable: There should be no empty string"),
                    )
                })
                .collect(),
            max_length,
        );
        Self { inner }
    }

    /// 推論を行う。
    pub fn infer(&self, input: &str) -> String {
        let input = input.chars().map(|c| c.to_string()).collect::<Vec<_>>();
        self.inner.infer(&input).into_iter().collect()
    }

    /// アルゴリズムを設定する。
    pub fn set_decode_strategy(&mut self, strategy: Strategy) {
        self.inner.set_decode_strategy(strategy);
    }
}
