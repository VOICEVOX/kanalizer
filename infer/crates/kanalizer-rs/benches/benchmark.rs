#[rustfmt::skip] // TODO: Rust 2024に移行したらこの抑制をやめる
use criterion::{Criterion, criterion_group, criterion_main};

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("Kanalizer", |b| {
        let kanalizer = kanalizer::Kanalizer::new();
        b.iter(|| {
            std::hint::black_box(kanalizer.convert("kanalizer", &Default::default()).unwrap())
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
