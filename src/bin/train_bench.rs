//! Training benchmark for the CPU `hist` path.
//!
//! Generates a deterministic synthetic dataset, trains, and reports wall clock
//! time. `tools/bench_xgb.py` runs the identical configuration through the
//! reference XGBoost so the two numbers are directly comparable.
//!
//! ```text
//! cargo run --release --no-default-features --bin train_bench -- \
//!     --rows 100000 --features 50 --rounds 20 --depth 6 --max-bin 256
//! ```

use std::time::Instant;
use xgboost_rs::parameters::{
    BoosterParameters, BoosterType, LearningTaskParameters, TrainingParameters,
    TreeBoosterParameters,
};
use xgboost_rs::{DMatrix, api};

/// xorshift64*: a tiny deterministic generator so Rust and Python can agree on
/// the dataset without shipping one.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `[0, 1)`.
    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u32 << 24) as f32)
    }
}

struct Args {
    rows: usize,
    features: usize,
    rounds: u32,
    depth: u32,
    max_bin: u32,
    sparsity: f32,
    subsample: f32,
    colsample: f32,
    threads: usize,
    repeats: usize,
    /// When set, write the generated dataset here so the Python harness can
    /// train the reference implementation on identical bytes.
    dump: Option<String>,
    /// Report setup versus per-round cost.
    breakdown: bool,
    /// Print a checksum of the trained model.
    hash: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            rows: 100_000,
            features: 50,
            rounds: 20,
            depth: 6,
            max_bin: 256,
            sparsity: 0.0,
            subsample: 1.0,
            colsample: 1.0,
            threads: 0,
            repeats: 1,
            dump: None,
            breakdown: false,
            hash: false,
        }
    }
}

fn parse_args() -> Args {
    let mut args = Args::default();
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let value = || -> String {
            argv.get(i + 1).cloned().unwrap_or_else(|| panic!("{} needs a value", argv[i]))
        };
        match argv[i].as_str() {
            "--rows" => args.rows = value().parse().unwrap(),
            "--features" => args.features = value().parse().unwrap(),
            "--rounds" => args.rounds = value().parse().unwrap(),
            "--depth" => args.depth = value().parse().unwrap(),
            "--max-bin" => args.max_bin = value().parse().unwrap(),
            "--sparsity" => args.sparsity = value().parse().unwrap(),
            "--subsample" => args.subsample = value().parse().unwrap(),
            "--colsample" => args.colsample = value().parse().unwrap(),
            "--threads" => args.threads = value().parse().unwrap(),
            "--repeats" => args.repeats = value().parse().unwrap(),
            "--dump" => args.dump = Some(value()),
            "--breakdown" => {
                args.breakdown = true;
                i -= 1;
            }
            "--hash" => {
                args.hash = true;
                i -= 1;
            }
            other => panic!("unknown argument `{other}`"),
        }
        i += 2;
    }
    args
}

/// Deterministic synthetic regression data, matched by the Python harness.
fn make_data(args: &Args) -> DMatrix {
    let mut rng = Rng(0x1234_5678_9abc_def0);
    let n = args.rows * args.features;
    let mut x = Vec::with_capacity(n);
    for _ in 0..n {
        let v = rng.next_f32();
        if args.sparsity > 0.0 && rng.next_f32() < args.sparsity {
            x.push(f32::NAN);
        } else {
            x.push(v);
        }
    }
    let mut y = Vec::with_capacity(args.rows);
    for r in 0..args.rows {
        let row = &x[r * args.features..(r + 1) * args.features];
        // A non-linear target so the trees have something to fit.
        let a = row[0];
        let b = row[1 % args.features];
        let c = row[2 % args.features];
        let signal = if a.is_nan() { 0.0 } else { a * 2.0 }
            + if b.is_nan() { 0.0 } else { b * b * 3.0 }
            + if c.is_nan() { 0.0 } else { (c > 0.5) as i32 as f32 };
        y.push(signal + rng.next_f32() * 0.1);
    }

    let mut d = DMatrix::from_dense(&x, args.rows, args.features, f32::NAN).unwrap();
    d.set_labels(&y).unwrap();
    d
}

fn main() {
    let args = parse_args();
    if args.threads > 0 {
        xgboost_rs::set_num_threads(args.threads);
    }

    let t = Instant::now();
    let dtrain = make_data(&args);
    let build = t.elapsed();

    if let Some(path) = &args.dump {
        dump_dataset(&dtrain, path);
        println!("wrote {path}");
    }

    let params = TrainingParameters {
        booster: BoosterParameters {
            booster: BoosterType::Gbtree(TreeBoosterParameters {
                eta: 0.3,
                max_depth: args.depth,
                max_bin: args.max_bin,
                subsample: args.subsample,
                colsample_bytree: args.colsample,
                ..Default::default()
            }),
            learning: LearningTaskParameters::default(),
            ..Default::default()
        },
        num_boost_round: args.rounds,
        ..Default::default()
    };

    println!(
        "rows={} features={} rounds={} depth={} max_bin={} sparsity={} \
         subsample={} colsample={} threads={}",
        args.rows,
        args.features,
        args.rounds,
        args.depth,
        args.max_bin,
        args.sparsity,
        args.subsample,
        args.colsample,
        xgboost_rs::num_threads(),
    );
    println!("data build: {:.3}s", build.as_secs_f64());

    let mut best = f64::INFINITY;
    let mut last_rmse = 0.0;
    for _ in 0..args.repeats {
        let t = Instant::now();
        let (_booster, history) = api::train(&params, &dtrain, &[(&dtrain, "train")]).unwrap();
        let elapsed = t.elapsed().as_secs_f64();
        best = best.min(elapsed);
        last_rmse = history.last().unwrap()[0].1;
    }
    println!("train: {best:.3}s  final train-rmse: {last_rmse:.6}");

    if args.hash {
        // A checksum of the serialised model: identical output across thread
        // counts is what "deterministic" has to mean in practice.
        let (booster, _) = api::train(&params, &dtrain, &[]).unwrap();
        let model = booster.save_model();
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in model.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        println!("model-hash: {h:016x}");
    }

    if args.breakdown {
        // Setup splits into the quantile sketch and the binning pass.
        let t = Instant::now();
        let cuts = xgboost_rs::data::cuts::build_cuts(&dtrain, args.max_bin).unwrap();
        let sketch = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let gi = xgboost_rs::data::gradient_index::build_gradient_index(&dtrain, &cuts).unwrap();
        let binning = t.elapsed().as_secs_f64();
        println!(
            "  cuts: {sketch:.3}s   binning: {binning:.3}s   index: {} MiB",
            gi.size_bytes() >> 20
        );

        // Setup (quantile sketch + binning) is paid once; the difference
        // between a 1-round and an N-round fit isolates the per-round cost.
        let one = time_rounds(&params, &dtrain, 1);
        let many = time_rounds(&params, &dtrain, args.rounds.max(2));
        let per_round = (many - one) / (args.rounds.max(2) - 1) as f64;
        println!(
            "  setup: {:.3}s   per round: {:.4}s",
            one - per_round,
            per_round
        );
    }
}

/// Write the dataset as raw little-endian `f32`: `rows * features` values
/// followed by `rows` labels. NaN marks a missing value.
fn dump_dataset(d: &DMatrix, path: &str) {
    use std::io::Write;
    let (rows, cols) = (d.num_row(), d.num_col());
    let mut buf: Vec<u8> = Vec::with_capacity((rows * cols + rows) * 4);
    for r in 0..rows {
        let mut dense = vec![f32::NAN; cols];
        let (idx, val) = d.row(r);
        for (&c, &v) in idx.iter().zip(val) {
            dense[c as usize] = v;
        }
        for v in dense {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    for v in &d.info().labels {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::File::create(path).unwrap().write_all(&buf).unwrap();
}

/// Wall clock for a fit of exactly `rounds` rounds.
fn time_rounds(params: &TrainingParameters, dtrain: &DMatrix, rounds: u32) -> f64 {
    let mut params = params.clone();
    params.num_boost_round = rounds;
    let t = Instant::now();
    api::train(&params, dtrain, &[]).unwrap();
    t.elapsed().as_secs_f64()
}
