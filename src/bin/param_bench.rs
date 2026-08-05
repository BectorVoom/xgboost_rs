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
//!
//! # Measured
//!
//! At `--rows 200000 --features 40 --rounds 10` on 8 threads, relative to the
//! group's first row. Absolute times are machine-dependent; the ratios are the
//! point, and they are what the implementation notes elsewhere refer to.
//!
//! | Group | What moves | Ratio |
//! |---|---|---|
//! | `tree_method` | `hist` -> `approx` (constant hessian) | 1.00x |
//! | | `hist` -> `approx` (varying hessian) | 4.6x |
//! | | `hist` -> `exact` | 10.5x |
//! | `opt_dense_col` | `1.0` -> `0.5` on a 25%-sparse matrix | 0.57x |
//! | `gblinear` | `cyclic` -> `thrifty` | 1.2x |
//! | | `cyclic` -> `greedy` | 8.5x |
//! | | `gbtree` -> `gblinear/cyclic` | 0.88x |
//! | `max_depth` | 4 -> 10 | 3.2x |
//! | `num_class` | 2 -> 8 | 2.9x |
//! | `dart` | `rate_drop` 0 -> 0.5 | 5.1x |
//! | `nthread` | 1 -> 8 | 0.48x |
//!
//! Three of those deserve a note, because the number is the *point* of the
//! parameter rather than an artefact:
//!
//! * **`approx` costs nothing extra under a constant hessian.** Its sketch is
//!   weighted by the hessian, and `reg:squarederror`'s hessian is `1` for
//!   every row in every round — so the sketch is the one `hist` already built
//!   and it is not rebuilt. Give the hessian something to vary and the full
//!   per-round re-sketch and re-bin appears, at 4.6x.
//! * **`exact` is an order of magnitude slower**, which is the trade it makes:
//!   every distinct value is a split candidate instead of one per bin, and
//!   there is no binned matrix to compress the scan.
//! * **`greedy` is quadratic in the feature count.** It re-scans the whole
//!   matrix once per feature it selects; `thrifty` ranks every feature in a
//!   single pass and then cycles that order, which is why it lands at 1.2x
//!   rather than 8.5x. (The `gblinear` group's `rel` column is against a
//!   `gbtree` reference row, so read those two ratios off the `cyclic` row:
//!   0.88x, 1.02x and 7.46x of the tree fit respectively.)

use std::time::Instant;

use xgboost_rs::parameters::{
    BoosterParameters, BoosterType, DartParameters, EvalMetric, FeatureSelector,
    GeneralParameters, GrowPolicy, LearningTaskParameters, LinearBoosterParameters, LinearUpdater,
    MonotoneConstraint, Objective, SamplingMethod, TrainingParameters, TreeBoosterParameters,
    TreeMethod, VerboseEval, Verbosity,
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

/// The same features, with labels in `[0, 1]` — the range the logistic and
/// log-link objectives accept.
fn unit_label_data(args: &Args) -> DMatrix {
    let mut d = make_data(args.rows, args.features);
    let labels: Vec<f32> = (0..args.rows).map(|i| f32::from(i % 3 == 0)).collect();
    d.set_labels(&labels).unwrap();
    d
}

/// The same relationship with a quarter of the values missing.
///
/// `opt_dense_col` only bites on a column that *has* missing values, so the
/// dense matrix the rest of the sweep uses would measure nothing there.
fn sparse_data(args: &Args) -> DMatrix {
    let mut rng = Rng(0x1234_5678_9abc_def0);
    let (rows, features) = (args.rows, args.features);
    let mut x: Vec<f32> = (0..rows * features).map(|_| rng.next_f32()).collect();
    let y: Vec<f32> = (0..rows)
        .map(|r| {
            let row = &x[r * features..(r + 1) * features];
            row[0] * 2.0 + row[1 % features] * row[1 % features] * 3.0
        })
        .collect();
    // Drawn, not strided: a stride that shares a factor with the row length
    // would blank whole columns and leave every other one fully dense, which
    // is exactly the case `opt_dense_col` cannot distinguish.
    let mut mask = Rng(0x9e37_79b9_7f4a_7c15);
    for v in x.iter_mut() {
        if mask.next_f32() < 0.25 {
            *v = f32::NAN;
        }
    }
    let mut d = DMatrix::from_dense(&x, rows, features, f32::NAN).unwrap();
    d.set_labels(&y).unwrap();
    d
}

/// The same features, with class indices in `[0, num_class)`.
fn class_label_data(args: &Args, num_class: u32) -> DMatrix {
    let mut d = make_data(args.rows, args.features);
    let labels: Vec<f32> = (0..args.rows).map(|i| (i % num_class as usize) as f32).collect();
    d.set_labels(&labels).unwrap();
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

/// One configuration to time. `data` overrides the shared matrix for the
/// objectives whose label range the shared one does not satisfy.
struct Case {
    label: String,
    params: TrainingParameters,
    data: Option<DMatrix>,
}

impl Case {
    fn new(label: impl Into<String>, params: TrainingParameters) -> Self {
        Self { label: label.into(), params, data: None }
    }

    fn with_data(mut self, data: DMatrix) -> Self {
        self.data = Some(data);
        self
    }
}

/// One group of configurations, all compared against the group's first row.
struct Group {
    name: &'static str,
    cases: Vec<Case>,
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
        // The score column is whatever metric the fit reports, which is the
        // objective's default unless the case chose one.
        println!("{:<28} {:>9} {:>9} {:>8}   train-metric", "case", "s/fit", "ms/round", "rel");
        let mut baseline = f64::NAN;
        for case in &group.cases {
            let matrix = case.data.as_ref().unwrap_or(&d);
            let (seconds, score) = time_fit(&case.params, matrix, args.repeats);
            if baseline.is_nan() {
                baseline = seconds;
            }
            println!(
                "{:<28} {:>9.3} {:>9.1} {:>7.2}x   {:.6}",
                case.label,
                seconds,
                seconds * 1000.0 / args.rounds as f64,
                seconds / baseline,
                score
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
                Case::new(
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
                Case::new(
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
                    Case::new(
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
    let mut sampling = vec![Case::new("subsample=1.0 (none)", base_params(tree(), args))];
    for method in [SamplingMethod::Uniform, SamplingMethod::GradientBased] {
        for subsample in [0.5f32, 0.1] {
            sampling.push(Case::new(
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
    let mut colsample = vec![Case::new("no column sampling", base_params(tree(), args))];
    for ratio in [0.5f32, 0.25] {
        colsample.push(Case::new(
            format!("colsample_bytree={ratio}"),
            base_params(TreeBoosterParameters { colsample_bytree: ratio, ..tree() }, args),
        ));
        colsample.push(Case::new(
            format!("colsample_bylevel={ratio}"),
            base_params(TreeBoosterParameters { colsample_bylevel: ratio, ..tree() }, args),
        ));
        colsample.push(Case::new(
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
                Case::new(
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
    let mut constraints = vec![Case::new("unconstrained", base_params(tree(), args))];
    constraints.push(Case::new(
        "monotone (all features)",
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
    constraints.push(Case::new(
        "interaction (2 groups)",
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

    // Histogram cache size. Bounds how many released histogram buffers are
    // held for reuse; below the tree's width every level re-allocates.
    groups.push(Group {
        name: "max_cached_hist_node",
        cases: [65536u64, 64, 8, 1]
            .into_iter()
            .map(|nodes| {
                Case::new(
                    format!("max_cached_hist_node={nodes}"),
                    base_params(
                        TreeBoosterParameters {
                            max_cached_hist_node: Some(nodes),
                            max_depth: 12,
                            max_bin: 512,
                            ..TreeBoosterParameters::default()
                        },
                        args,
                    ),
                )
            })
            .collect(),
    });

    // The objective is the other half of a round: every one visits every row,
    // but they do very different amounts of work there.
    let mut objectives = Vec::new();
    for objective in [
        Objective::RegSquaredError,
        Objective::RegAbsoluteError,
        Objective::RegPseudoHuberError { huber_slope: 1.0 },
        Objective::RegLogistic,
        Objective::CountPoisson { max_delta_step: 0.7 },
        Objective::RegTweedie { tweedie_variance_power: 1.5 },
        Objective::RegQuantileError { quantile_alpha: vec![0.5] },
        Objective::SurvivalCox,
    ] {
        let mut params = base_params(tree(), args);
        params.booster.learning.objective = objective.clone();
        // The shared labels are unbounded reals; the logistic and log-link
        // objectives need their own range, so those cases bring their own
        // matrix rather than being dropped from the sweep.
        objectives.push(
            Case::new(objective.name(), params).with_data(unit_label_data(args)),
        );
    }
    groups.push(Group { name: "objective", cases: objectives });

    // Output groups: a round grows one tree per class, so the cost is linear
    // in `num_class` and this is the single most expensive knob a
    // classification fit has.
    groups.push(Group {
        name: "num_class",
        cases: [2u32, 4, 8]
            .into_iter()
            .map(|num_class| {
                let mut params = base_params(tree(), args);
                params.booster.learning.objective = Objective::MultiSoftprob { num_class };
                Case::new(format!("num_class={num_class}"), params)
                    .with_data(class_label_data(args, num_class))
            })
            .collect(),
    });

    // Metrics are evaluated once per round on the whole watchlist; the ones
    // that sort or sweep the predictions cost more than the elementwise ones.
    let mut metrics = Vec::new();
    for metric in [
        EvalMetric::Rmse,
        EvalMetric::Mae,
        EvalMetric::Logloss,
        EvalMetric::Auc,
        EvalMetric::Aucpr,
        EvalMetric::Ndcg { top_n: None, minus: false },
    ] {
        let mut params = base_params(tree(), args);
        params.booster.learning.eval_metric = vec![metric.clone()];
        metrics.push(Case::new(metric.to_string(), params).with_data(unit_label_data(args)));
    }
    groups.push(Group { name: "eval_metric", cases: metrics });

    // DART: every round drops trees, which costs a prediction pass over the
    // dropped ones and grows with the ensemble.
    let mut dart = vec![Case::new("gbtree (no dropout)", base_params(tree(), args))];
    for rate_drop in [0.1f32, 0.5] {
        let mut params = base_params(tree(), args);
        params.booster.booster = BoosterType::Dart(DartParameters {
            rate_drop,
            tree: tree(),
            ..Default::default()
        });
        dart.push(Case::new(format!("dart/rate_drop={rate_drop}"), params));
    }
    groups.push(Group { name: "dart", cases: dart });

    // Tree method. The three do fundamentally different amounts of work per
    // level: `hist` bins once up front and reduces histograms, `approx`
    // re-sketches and re-bins the whole matrix every round, and `exact` scans
    // every distinct value of every column with no binning at all.
    let mut methods = Vec::new();
    for tree_method in [TreeMethod::Hist, TreeMethod::Approx, TreeMethod::Exact] {
        methods.push(Case::new(
            tree_method.to_string(),
            base_params(TreeBoosterParameters { tree_method, ..tree() }, args),
        ));
    }
    // `approx` only re-sketches when the hessian moves, so the same sweep on a
    // logistic objective is the one that pays the full per-round cost.
    for tree_method in [TreeMethod::Hist, TreeMethod::Approx] {
        let mut params =
            base_params(TreeBoosterParameters { tree_method, ..tree() }, args);
        params.booster.learning.objective = Objective::RegLogistic;
        methods.push(
            Case::new(format!("{tree_method} (varying hessian)"), params)
                .with_data(unit_label_data(args)),
        );
    }
    groups.push(Group { name: "tree_method", cases: methods });

    // `opt_dense_col` is the `exact` updater's one speed knob: a column
    // sparser than this threshold gets a *second*, forward scan on top of the
    // backward one, which is what lets the fit learn a per-split missing-value
    // direction. Lowering it trades that freedom for half the column work, so
    // it only measures anything on a matrix that has missing values — hence
    // the sparse matrix here.
    groups.push(Group {
        name: "opt_dense_col",
        cases: [1.0f32, 0.5, 0.0]
            .into_iter()
            .map(|opt_dense_col| {
                Case::new(
                    format!("exact/opt_dense_col={opt_dense_col}"),
                    base_params(
                        TreeBoosterParameters {
                            tree_method: TreeMethod::Exact,
                            opt_dense_col,
                            ..tree()
                        },
                        args,
                    ),
                )
                .with_data(sparse_data(args))
            })
            .collect(),
    });

    // The linear booster, and the feature selectors that decide how much work
    // one coordinate sweep does. `greedy` re-scans the whole matrix per
    // feature, so it is quadratic in the feature count; `thrifty` is its
    // linear approximation.
    let mut linear = vec![Case::new("gbtree (reference)", base_params(tree(), args))];
    for (updater, selector) in [
        (LinearUpdater::Shotgun, FeatureSelector::Cyclic),
        (LinearUpdater::Shotgun, FeatureSelector::Shuffle),
        (LinearUpdater::CoordDescent, FeatureSelector::Cyclic),
        (LinearUpdater::CoordDescent, FeatureSelector::Random),
        (LinearUpdater::CoordDescent, FeatureSelector::Thrifty),
        (LinearUpdater::CoordDescent, FeatureSelector::Greedy),
    ] {
        let mut params = base_params(tree(), args);
        params.booster.booster = BoosterType::Gblinear(LinearBoosterParameters {
            updater,
            feature_selector: selector,
            ..Default::default()
        });
        linear.push(Case::new(format!("gblinear/{updater}/{selector}"), params));
    }
    groups.push(Group { name: "gblinear", cases: linear });

    // Thread scaling. Runs last: it is the only group that ignores `--threads`.
    groups.push(Group {
        name: "nthread",
        cases: [1u32, 2, 4, 8]
            .into_iter()
            .map(|nthread| {
                let mut params = base_params(tree(), args);
                params.booster.general.nthread = nthread;
                Case::new(format!("nthread={nthread}"), params)
            })
            .collect(),
    });

    groups
}
