//! Performance sweep over the fit parameters that change what the CPU `hist`
//! path has to do.
//!
//! Not every parameter costs anything — `eta` and `lambda` are arithmetic on
//! numbers already loaded — so this covers the ones that change the *work*:
//! the histogram size, the tree size, how many trees there are, how many rows
//! and columns each split considers, and how many threads do it.
//!
//! ```text
//! cargo run --release --no-default-features --bin param_bench -- \
//!     --rows 200000 --features 40 --rounds 10
//! ```
//!
//! Each configuration is timed best-of-`--repeats` whole fits, so every one
//! pays its own quantile sketch and binning pass and none benefits from a
//! cache another cannot use. `rel` is against the first row of the group.

use std::time::Instant;

use xgboost_rs::parameters::{
    BoosterParameters, BoosterType, GeneralParameters, GrowPolicy, LearningTaskParameters,
    MonotoneConstraint, SamplingMethod, TrainingParameters, TreeBoosterParameters, VerboseEval,
    Verbosity,
};
use xgboost_rs::{DMatrix, api};

struct Args {
    rows: usize,
    features: usize,
    rounds: u32,
    repeats: usize,
    threads: u32,
    /// Only run groups whose name contains this.
    filter: Option<String>,
}

impl Default for Args {
    fn default() -> Self {
        Self { rows: 200_000, features: 40, rounds: 10, repeats: 3, threads: 0, filter: None }
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
            "--repeats" => args.repeats = value().parse().unwrap(),
            "--threads" => args.threads = value().parse().unwrap(),
            "--filter" => args.filter = Some(value()),
            other => panic!("unknown argument `{other}`"),
        }
        i += 2;
    }
    args
}

/// xorshift64*, so the dataset is the same on every machine and every run.
struct Rng(u64);

impl Rng {
    fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        ((x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32) / ((1u32 << 24) as f32)
    }
}

fn make_data(rows: usize, features: usize) -> DMatrix {
    let mut rng = Rng(0x1234_5678_9abc_def0);
    let x: Vec<f32> = (0..rows * features).map(|_| rng.next_f32()).collect();
    let y: Vec<f32> = (0..rows)
        .map(|r| {
            let row = &x[r * features..(r + 1) * features];
            let a = row[0];
            let b = row[1 % features];
            let c = row[2 % features];
            a * 2.0 + b * b * 3.0 + f32::from(c > 0.5)
        })
        .collect();
    let mut d = DMatrix::from_dense(&x, rows, features, f32::NAN).unwrap();
    d.set_labels(&y).unwrap();
    d
}

fn base_params(tree: TreeBoosterParameters, args: &Args) -> TrainingParameters {
    TrainingParameters {
        booster: BoosterParameters {
            booster: BoosterType::Gbtree(tree),
            general: GeneralParameters {
                nthread: args.threads,
                verbosity: Verbosity::Silent,
                ..Default::default()
            },
            learning: LearningTaskParameters::default(),
        },
        num_boost_round: args.rounds,
        verbose_eval: VerboseEval::Silent,
        ..Default::default()
    }
}

/// Best-of-N wall clock for a whole fit, with the final train RMSE so two rows
/// can be checked for doing comparable work.
fn time_fit(params: &TrainingParameters, d: &DMatrix, repeats: usize) -> (f64, f64) {
    let mut best = f64::INFINITY;
    let mut rmse = f64::NAN;
    for _ in 0..repeats {
        let t = Instant::now();
        let (_, history) = api::train(params, d, &[(d, "train")]).unwrap();
        best = best.min(t.elapsed().as_secs_f64());
        rmse = history.last().unwrap()[0].1;
    }
    (best, rmse)
}

/// One group of configurations, all compared against the group's first row.
struct Group {
    name: &'static str,
    cases: Vec<(String, TrainingParameters)>,
}

fn main() {
    let args = parse_args();
    let d = make_data(args.rows, args.features);

    println!(
        "rows={} features={} rounds={} repeats={} threads={}",
        args.rows,
        args.features,
        args.rounds,
        args.repeats,
        if args.threads == 0 { xgboost_rs::num_threads() as u32 } else { args.threads },
    );
    println!();

    for group in groups(&args) {
        if let Some(filter) = &args.filter
            && !group.name.contains(filter.as_str())
        {
            continue;
        }
        println!("== {} ==", group.name);
        println!("{:<28} {:>9} {:>9} {:>8}   train-rmse", "case", "s/fit", "ms/round", "rel");
        let mut baseline = f64::NAN;
        for (label, params) in &group.cases {
            let (seconds, rmse) = time_fit(params, &d, args.repeats);
            if baseline.is_nan() {
                baseline = seconds;
            }
            println!(
                "{:<28} {:>9.3} {:>9.1} {:>7.2}x   {:.6}",
                label,
                seconds,
                seconds * 1000.0 / args.rounds as f64,
                seconds / baseline,
                rmse
            );
        }
        println!();
    }
}

fn groups(args: &Args) -> Vec<Group> {
    let tree = TreeBoosterParameters::default;
    let mut groups = Vec::new();

    // Histogram width: bins per feature drive both the binning pass and the
    // per-node histogram reduction.
    groups.push(Group {
        name: "max_bin",
        cases: [16u32, 64, 256, 512]
            .into_iter()
            .map(|max_bin| {
                (
                    format!("max_bin={max_bin}"),
                    base_params(TreeBoosterParameters { max_bin, ..tree() }, args),
                )
            })
            .collect(),
    });

    // Tree size: each extra level doubles the nodes to evaluate.
    groups.push(Group {
        name: "max_depth",
        cases: [4u32, 6, 8, 10]
            .into_iter()
            .map(|max_depth| {
                (
                    format!("max_depth={max_depth}"),
                    base_params(TreeBoosterParameters { max_depth, ..tree() }, args),
                )
            })
            .collect(),
    });

    // Growth order at a matched leaf budget: depthwise expands a level at a
    // time, lossguide one node at a time.
    groups.push(Group {
        name: "grow_policy",
        cases: [GrowPolicy::DepthWise, GrowPolicy::LossGuide]
            .into_iter()
            .flat_map(|grow_policy| {
                [16u32, 64].into_iter().map(move |max_leaves| {
                    (
                        format!("{grow_policy}/max_leaves={max_leaves}"),
                        base_params(
                            TreeBoosterParameters {
                                grow_policy,
                                max_leaves,
                                max_depth: 0,
                                ..TreeBoosterParameters::default()
                            },
                            args,
                        ),
                    )
                })
            })
            .collect(),
    });

    // Row sampling. Unsampled rows keep their place in the row set with a
    // zeroed gradient, so this measures what sampling *costs* rather than what
    // it saves — the histogram pass still visits every row.
    let mut sampling = vec![("subsample=1.0 (none)".to_owned(), base_params(tree(), args))];
    for method in [SamplingMethod::Uniform, SamplingMethod::GradientBased] {
        for subsample in [0.5f32, 0.1] {
            sampling.push((
                format!("{method}/subsample={subsample}"),
                base_params(
                    TreeBoosterParameters { subsample, sampling_method: method, ..tree() },
                    args,
                ),
            ));
        }
    }
    groups.push(Group { name: "subsample", cases: sampling });

    // Column sampling: fewer candidate features per node means less split
    // evaluation, though the histograms are still built over every column.
    let mut colsample = vec![("no column sampling".to_owned(), base_params(tree(), args))];
    for ratio in [0.5f32, 0.25] {
        colsample.push((
            format!("colsample_bytree={ratio}"),
            base_params(TreeBoosterParameters { colsample_bytree: ratio, ..tree() }, args),
        ));
        colsample.push((
            format!("colsample_bylevel={ratio}"),
            base_params(TreeBoosterParameters { colsample_bylevel: ratio, ..tree() }, args),
        ));
        colsample.push((
            format!("colsample_bynode={ratio}"),
            base_params(TreeBoosterParameters { colsample_bynode: ratio, ..tree() }, args),
        ));
    }
    groups.push(Group { name: "colsample", cases: colsample });

    // Forest width: a round grows this many trees, so the cost is linear.
    groups.push(Group {
        name: "num_parallel_tree",
        cases: [1u32, 2, 4]
            .into_iter()
            .map(|num_parallel_tree| {
                (
                    format!("num_parallel_tree={num_parallel_tree}"),
                    base_params(
                        TreeBoosterParameters {
                            num_parallel_tree,
                            subsample: 0.5,
                            ..TreeBoosterParameters::default()
                        },
                        args,
                    ),
                )
            })
            .collect(),
    });

    // Constraints: both add a check per candidate split.
    let mut constraints = vec![("unconstrained".to_owned(), base_params(tree(), args))];
    constraints.push((
        "monotone (all features)".to_owned(),
        base_params(
            TreeBoosterParameters {
                monotone_constraints: vec![MonotoneConstraint::Increasing; args.features],
                ..tree()
            },
            args,
        ),
    ));
    // Interleaved groups, so the signal features (0, 1, 2) land on both sides
    // and the constraint actually binds — a constraint that never rejects a
    // candidate would measure nothing.
    constraints.push((
        "interaction (2 groups)".to_owned(),
        base_params(
            TreeBoosterParameters {
                interaction_constraints: Some(vec![
                    (0..args.features as u32).filter(|f| f % 2 == 0).collect(),
                    (0..args.features as u32).filter(|f| f % 2 == 1).collect(),
                ]),
                ..tree()
            },
            args,
        ),
    ));
    groups.push(Group { name: "constraints", cases: constraints });

    // Thread scaling. Runs last: it is the only group that ignores `--threads`.
    groups.push(Group {
        name: "nthread",
        cases: [1u32, 2, 4, 8]
            .into_iter()
            .map(|nthread| {
                let mut params = base_params(tree(), args);
                params.booster.general.nthread = nthread;
                (format!("nthread={nthread}"), params)
            })
            .collect(),
    });

    groups
}
