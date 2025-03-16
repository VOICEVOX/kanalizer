use clap::{Parser, ValueEnum};

#[derive(Parser)]
struct Args {
    /// 読みの最大長。
    #[clap(short, long, default_value = "32")]
    max_length: usize,

    /// アルゴリズム。
    #[clap(short, long, default_value = "greedy")]
    strategy: StrategyArg,

    /// Top-KのK。
    #[clap(short = 'k', long, default_value = "3")]
    top_k: usize,

    /// Top-PのP。
    #[clap(short = 'p', long, default_value = "0.9")]
    top_p: f32,

    /// Top-Pの温度。
    #[clap(short = 't', long, default_value = "1.0")]
    temperature: f32,
}

#[derive(ValueEnum, Debug, Clone)]
enum StrategyArg {
    Greedy,
    TopK,
    TopP,
}

fn main() {
    let args = Args::parse();

    let mut c2k = e2k::C2k::new(args.max_length);
    match args.strategy {
        StrategyArg::Greedy => {
            c2k.set_decode_strategy(e2k::Strategy::Greedy);
            println!("アルゴリズム：Greedy");
        }
        StrategyArg::TopK => {
            c2k.set_decode_strategy(e2k::Strategy::TopK(e2k::StrategyTopK { k: args.top_k }));
            println!("アルゴリズム：Top-K, K={}", args.top_k);
        }
        StrategyArg::TopP => {
            c2k.set_decode_strategy(e2k::Strategy::TopP(e2k::StrategyTopP {
                top_p: args.top_p,
                temperature: args.temperature,
            }));
            println!(
                "アルゴリズム：Top-P, P={}, T={}",
                args.top_p, args.temperature
            );
        }
    }
    println!("Ctrl-C で終了します。");
    loop {
        let line = dialoguer::Input::<String>::new()
            .with_prompt("Input")
            .interact()
            .unwrap();
        let dst = c2k.infer(&line);
        println!("{} -> {}", line, dst);
    }
}
