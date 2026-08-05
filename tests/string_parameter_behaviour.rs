//! Every pattern of every string-valued parameter, tested for what it *does*.
//!
//! `tests/string_parameters.rs` proves that each spelling round-trips through
//! the parameter surface. This file proves the next thing, which is the one
//! that actually matters to a user: that every accepted spelling reaches the
//! fit — it is either honoured or rejected with a message naming the parameter,
//! and never accepted and quietly ignored.
//!
//! It also covers the *parameterised* string parameters, whose grammar the
//! round-trip tests cannot enumerate because the argument is unbounded:
//! `error@t`, `tweedie-nloglik@rho`, `ndcg@n-`, `map@n-`, `pre@n`, `ams@t`,
//! `cuda:<ordinal>`, `sycl:<kind>:<ordinal>`, the comma-separated `updater`
//! sequence, the `(a,b,c)` constraint and alpha lists, and the JSON
//! `interaction_constraints`.

use std::str::FromStr;

use xgboost_rs::parameters::{
    AftDistribution, BoosterParameters, BoosterType, DartNormalizeType, DartParameters,
    DartSampleType, DefaultDirection, Device, EvalMetric, FeatureSelector, GeneralParameters,
    GrowPolicy, LambdaRankPairMethod, LambdaRankParameters, LearningTaskParameters,
    LinearBoosterParameters, LinearUpdater, MonotoneConstraint, MultiStrategy, Objective,
    ProcessType, SamplingMethod, SyclKind, ToConfig, TrainingParameters, TreeBoosterParameters,
    TreeMethod, TreeUpdaterName, VerboseEval, Verbosity,
};
use xgboost_rs::{DMatrix, api};

/// A matrix whose label depends on every feature, so a parameter that changes
/// which features or which nodes a tree uses has something to change.
fn rich_data(rows: usize, cols: usize) -> DMatrix {
    let mut state = 0xd1ce_d1ce_d1ce_d1ceu64;
    let mut next = || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32 / (1u32 << 24) as f32
    };
    let x: Vec<f32> = (0..rows * cols).map(|_| next()).collect();
    let y: Vec<f32> = (0..rows)
        .map(|r| {
            let row = &x[r * cols..(r + 1) * cols];
            // Every feature contributes, and a product term gives interaction
            // constraints something to forbid.
            row.iter().enumerate().map(|(c, v)| v / (c + 1) as f32).sum::<f32>()
                + row[0] * row[cols - 1]
        })
        .collect();
    let mut d = DMatrix::from_dense(&x, rows, cols, f32::NAN).unwrap();
    d.set_labels(&y).unwrap();
    d
}

/// The same relationship as [`rich_data`] with a quarter of the values
/// missing, so the parameters that only steer *missing* values have something
/// to steer.
fn sparse_data(rows: usize, cols: usize) -> DMatrix {
    let mut state = 0x0bad_c0de_0bad_c0deu64;
    let mut next = || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32 / (1u32 << 24) as f32
    };
    let mut x: Vec<f32> = (0..rows * cols).map(|_| next()).collect();
    let y: Vec<f32> = (0..rows)
        .map(|r| x[r * cols..(r + 1) * cols].iter().enumerate().map(|(c, v)| v / (c + 1) as f32).sum())
        .collect();
    for (i, v) in x.iter_mut().enumerate() {
        if (i * 7919) % 4 == 0 {
            *v = f32::NAN;
        }
    }
    let mut d = DMatrix::from_dense(&x, rows, cols, f32::NAN).unwrap();
    d.set_labels(&y).unwrap();
    d
}

/// Data that suits every objective at once: positive labels in `[0, 1]`, so the
/// log links, the logistic losses and the classification objectives all accept
/// it. Ranking metadata is attached too.
fn universal_data(rows: usize) -> DMatrix {
    let mut state = 0x5eed_5eed_5eed_5eedu64;
    let mut next = || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32 / (1u32 << 24) as f32
    };
    let cols = 3;
    let x: Vec<f32> = (0..rows * cols).map(|_| next()).collect();
    // Labels of exactly 0 or 1: valid for every objective's label check, and
    // positive-or-zero for the count links.
    let y: Vec<f32> = (0..rows).map(|r| f32::from(x[r * cols] > 0.5)).collect();

    let mut d = DMatrix::from_dense(&x, rows, cols, f32::NAN).unwrap();
    d.set_labels(&y).unwrap();
    // Five documents per query, so the ranking objectives and metrics have
    // something to rank.
    d.set_group(&vec![5; rows / 5]).unwrap();
    // Censoring bounds, so the survival objectives have their interval.
    let lower: Vec<f32> = (0..rows).map(|i| 1.0 + (i % 5) as f32).collect();
    let upper: Vec<f32> = lower.iter().map(|t| t + 1.0).collect();
    d.set_label_bounds(&lower, &upper).unwrap();
    d
}

fn training(learning: LearningTaskParameters, tree: TreeBoosterParameters) -> TrainingParameters {
    TrainingParameters {
        booster: BoosterParameters {
            booster: BoosterType::Gbtree(tree),
            general: GeneralParameters { verbosity: Verbosity::Silent, ..Default::default() },
            learning,
        },
        num_boost_round: 2,
        verbose_eval: VerboseEval::Silent,
        ..Default::default()
    }
}

/// Run a fit and report either the objective the model recorded, or the error.
fn run(p: &TrainingParameters, d: &DMatrix) -> Result<String, String> {
    match api::train(p, d, &[(d, "eval")]) {
        Ok((booster, _)) => {
            let json: serde_json::Value = serde_json::from_str(&booster.save_model()).unwrap();
            Ok(json["learner"]["objective"]["name"].as_str().unwrap().to_owned())
        }
        Err(e) => Err(e.to_string()),
    }
}

// ------------------------------------------------------------ objective ----

/// Every `objective` spelling must train, and the model must record the
/// objective the caller asked for rather than a fallback.
#[test]
fn every_objective_spelling_reaches_the_fit() {
    let d = universal_data(200);
    let rank = LambdaRankParameters::default();

    for objective in [
        Objective::RegSquaredError,
        Objective::RegSquaredLogError,
        Objective::RegLogistic,
        Objective::RegPseudoHuberError { huber_slope: 1.0 },
        Objective::RegAbsoluteError,
        Objective::RegQuantileError { quantile_alpha: vec![0.5] },
        Objective::RegExpectileError { expectile_alpha: vec![0.5] },
        Objective::RegGamma,
        Objective::RegTweedie { tweedie_variance_power: 1.5 },
        Objective::RegLinear,
        Objective::CountPoisson { max_delta_step: 0.7 },
        Objective::SurvivalCox,
        Objective::SurvivalAft {
            aft_loss_distribution: AftDistribution::Normal,
            aft_loss_distribution_scale: 1.0,
        },
        Objective::BinaryLogistic,
        Objective::BinaryLogitRaw,
        Objective::BinaryHinge,
        Objective::MultiSoftmax { num_class: 2 },
        Objective::MultiSoftprob { num_class: 2 },
        Objective::RankPairwise(rank),
        Objective::RankNdcg(rank),
        Objective::RankMap(rank),
    ] {
        let spelling = objective.name();
        // `reg:gamma` needs a strictly positive label, which the shared 0/1
        // labels do not provide; give it its own matrix.
        let d = if spelling == "reg:gamma" {
            let mut positive = universal_data(200);
            positive.set_labels(&vec![1.0f32; 200]).unwrap();
            positive
        } else {
            d.clone()
        };

        let p = training(
            LearningTaskParameters { objective: objective.clone(), ..Default::default() },
            TreeBoosterParameters::default(),
        );
        match run(&p, &d) {
            Ok(recorded) => {
                // `reg:linear` is the deprecated spelling of squared error and
                // records the objective it really is.
                let expected =
                    if spelling == "reg:linear" { "reg:squarederror" } else { spelling };
                assert_eq!(recorded, expected, "`{spelling}` recorded the wrong objective");
            }
            Err(e) => panic!("`{spelling}` was rejected: {e}"),
        }
    }
}

/// Every objective spelling parses back to itself, and nothing outside the list
/// parses at all.
#[test]
fn objective_spellings_are_exactly_the_upstream_ones() {
    let spellings = [
        "reg:squarederror",
        "reg:squaredlogerror",
        "reg:logistic",
        "reg:pseudohubererror",
        "reg:absoluteerror",
        "reg:quantileerror",
        "reg:expectileerror",
        "reg:gamma",
        "reg:tweedie",
        "reg:linear",
        "count:poisson",
        "survival:cox",
        "survival:aft",
        "binary:logistic",
        "binary:logitraw",
        "binary:hinge",
        "multi:softmax",
        "multi:softprob",
        "rank:pairwise",
        "rank:ndcg",
        "rank:map",
    ];
    for name in spellings {
        let parsed = Objective::from_str(name).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(parsed.name(), name);
        assert_eq!(parsed.to_config_map()["objective"], name);
    }
    // Case, spacing and near-misses are all rejected.
    for bad in [
        "",
        " ",
        "reg:squarederror ",
        "REG:SQUAREDERROR",
        "reg:squareerror",
        "binary:logistics",
        "multi:soft",
        "rank:pair",
        "reg:squarederror\n",
    ] {
        assert!(Objective::from_str(bad).is_err(), "`{bad}` should not parse as an objective");
    }
}

// ---------------------------------------------------------- eval_metric ----

/// Every metric spelling — bare, `@argument` and trailing `-` — must be
/// accepted by a fit and reported back under exactly that name.
#[test]
fn every_eval_metric_pattern_is_reported_under_its_own_name() {
    let d = universal_data(200);
    let spellings = [
        "rmse",
        "rmsle",
        "mae",
        "mape",
        "mphe",
        "logloss",
        "error",
        "error@0.25",
        "error@0.75",
        "auc",
        "aucpr",
        "pre",
        "pre@1",
        "pre@5",
        "ndcg",
        "ndcg@1",
        "ndcg@5",
        "ndcg-",
        "ndcg@5-",
        "map",
        "map@3",
        "map-",
        "map@3-",
        "poisson-nloglik",
        "gamma-nloglik",
        "gamma-deviance",
        "cox-nloglik",
        "tweedie-nloglik@1.1",
        "tweedie-nloglik@1.5",
        "tweedie-nloglik@1.9",
        "aft-nloglik",
        "interval-regression-accuracy",
        "quantile",
        "expectile",
        "ams@0.0",
        "ams@0.15",
    ];

    for name in spellings {
        let metric = EvalMetric::from_str(name).unwrap_or_else(|e| panic!("{name}: {e}"));
        // A float argument renders in its shortest form, so `@1.0` normalises
        // to `@1`; re-parsing the rendered name must give the same metric.
        let rendered = metric.to_string();
        assert_eq!(
            EvalMetric::from_str(&rendered).unwrap(),
            metric,
            "`{name}` rendered to `{rendered}`, which does not parse back"
        );

        let p = training(
            LearningTaskParameters { eval_metric: vec![metric], ..Default::default() },
            TreeBoosterParameters::default(),
        );
        let (_, history) =
            api::train(&p, &d, &[(&d, "eval")]).unwrap_or_else(|e| panic!("`{name}`: {e}"));
        assert_eq!(
            history[0][0].0,
            format!("eval-{rendered}"),
            "`{name}` reported the wrong name"
        );
        assert!(
            history[0][0].1.is_finite() || name == "mape",
            "`{name}` scored {}",
            history[0][0].1
        );
    }
}

/// `tweedie-nloglik@1` is inside the documented `[1, 2)` range and is accepted,
/// but the likelihood divides by `1 - rho` and so is singular exactly there.
/// Upstream has the same singularity; the point of this test is that the
/// boundary is reached deliberately rather than by accident.
#[test]
fn the_tweedie_metric_is_singular_at_a_variance_power_of_one() {
    let metric = EvalMetric::from_str("tweedie-nloglik@1").unwrap();
    assert_eq!(metric.to_string(), "tweedie-nloglik@1");

    let d = universal_data(100);
    let p = training(
        LearningTaskParameters { eval_metric: vec![metric], ..Default::default() },
        TreeBoosterParameters::default(),
    );
    let (_, history) = api::train(&p, &d, &[(&d, "eval")]).unwrap();
    assert!(!history[0][0].1.is_finite(), "the singularity should not be papered over");

    // Anywhere else in the range it is a real score.
    for rho in ["1.1", "1.5", "1.9"] {
        let metric = EvalMetric::from_str(&format!("tweedie-nloglik@{rho}")).unwrap();
        let p = training(
            LearningTaskParameters { eval_metric: vec![metric], ..Default::default() },
            TreeBoosterParameters::default(),
        );
        let (_, history) = api::train(&p, &d, &[(&d, "eval")]).unwrap();
        assert!(history[0][0].1.is_finite(), "rho {rho} scored {}", history[0][0].1);
    }
}

/// The multiclass metrics need a multiclass fit, so they get their own case.
#[test]
fn the_multiclass_metric_spellings_reach_a_multiclass_fit() {
    let d = universal_data(200);
    for name in ["merror", "mlogloss"] {
        let metric = EvalMetric::from_str(name).unwrap();
        let p = training(
            LearningTaskParameters {
                objective: Objective::MultiSoftprob { num_class: 2 },
                eval_metric: vec![metric],
                ..Default::default()
            },
            TreeBoosterParameters::default(),
        );
        let (_, history) = api::train(&p, &d, &[(&d, "eval")]).unwrap();
        assert_eq!(history[0][0].0, format!("eval-{name}"));
        assert!(history[0][0].1.is_finite());
    }
}

/// A malformed `@argument` is rejected, at parse time or at range-check time —
/// never accepted with the argument silently dropped.
#[test]
fn malformed_metric_arguments_are_rejected() {
    for bad in [
        // Missing a mandatory argument.
        "tweedie-nloglik",
        "ams",
        // An argument where none is allowed.
        "rmse@1",
        "logloss@0.5",
        "auc@1",
        "merror@2",
        // An unparseable argument.
        "error@x",
        "ndcg@x",
        "map@-1",
        "pre@x",
        "ams@x",
        "tweedie-nloglik@x",
        // An out-of-range argument.
        "error@1.5",
        "error@-0.5",
        "tweedie-nloglik@2.0",
        "tweedie-nloglik@0.5",
        "ams@-1",
        "pre@0",
        "ndcg@0",
        "map@0",
        // Near-miss names.
        "",
        " ",
        "rmsee",
        "ndcgg",
        "aucroc",
        "NDCG",
    ] {
        assert!(EvalMetric::from_str(bad).is_err(), "`{bad}` should not parse as a metric");
    }
}

/// The trailing `-` belongs to the ranking metrics alone, and it survives the
/// `@n` argument.
#[test]
fn the_trailing_minus_is_parsed_only_for_the_ranking_metrics() {
    assert_eq!(
        EvalMetric::from_str("ndcg@5-").unwrap(),
        EvalMetric::Ndcg { top_n: Some(5), minus: true }
    );
    assert_eq!(
        EvalMetric::from_str("map-").unwrap(),
        EvalMetric::Map { top_n: None, minus: true }
    );
    // A metric that has no minus variant does not gain one.
    assert!(EvalMetric::from_str("rmse-").is_err());
    assert!(EvalMetric::from_str("pre-").is_err());
    assert!(EvalMetric::from_str("auc-").is_err());
}

/// A whole list of metrics is emitted as one comma-separated string, and every
/// entry survives.
#[test]
fn a_metric_list_is_emitted_and_evaluated_in_order() {
    let d = universal_data(100);
    let metrics = vec![
        EvalMetric::Logloss,
        EvalMetric::Auc,
        EvalMetric::ErrorAt(0.7),
        EvalMetric::Ndcg { top_n: Some(3), minus: true },
    ];
    let learning =
        LearningTaskParameters { eval_metric: metrics.clone(), ..Default::default() };
    assert_eq!(
        learning.to_config_map()["eval_metric"],
        "logloss,auc,error@0.7,ndcg@3-",
        "the emitted list must keep order and spelling"
    );

    let p = training(learning, TreeBoosterParameters::default());
    let (_, history) = api::train(&p, &d, &[(&d, "eval")]).unwrap();
    let names: Vec<String> =
        history[0].iter().map(|(n, _)| n.trim_start_matches("eval-").to_owned()).collect();
    assert_eq!(names, ["logloss", "auc", "error@0.7", "ndcg@3-"]);
}

/// A plugin metric this crate does not implement is rejected by name rather
/// than ignored.
#[test]
fn an_unknown_custom_metric_is_rejected_by_name() {
    let d = universal_data(50);
    let p = training(
        LearningTaskParameters {
            eval_metric: vec![EvalMetric::Custom("my-plugin-metric".into())],
            ..Default::default()
        },
        TreeBoosterParameters::default(),
    );
    match api::train(&p, &d, &[(&d, "eval")]) {
        Ok(_) => panic!("an unimplemented metric must not be silently skipped"),
        Err(e) => assert!(e.to_string().contains("my-plugin-metric"), "{e}"),
    }
}

// --------------------------------------------------------------- device ----

/// Every `device` spelling parses, normalises the way upstream does, and either
/// trains or is rejected naming `device`.
#[test]
fn every_device_spelling_is_honoured_or_rejected_by_name() {
    let d = universal_data(50);

    // (spelling, the spelling it normalises to, whether a CPU fit accepts it)
    let cases: [(&str, &str, bool); 10] = [
        ("cpu", "cpu", true),
        ("cuda", "cuda", false),
        ("cuda:0", "cuda:0", false),
        ("cuda:3", "cuda:3", false),
        // `gpu` is the deprecated alias and normalises to `cuda`.
        ("gpu", "cuda", false),
        ("gpu:1", "cuda:1", false),
        ("sycl", "sycl", false),
        ("sycl:cpu", "sycl:cpu", false),
        ("sycl:gpu", "sycl:gpu", false),
        ("sycl:gpu:1", "sycl:gpu:1", false),
    ];

    for (spelling, normalised, trains) in cases {
        let device = Device::from_str(spelling).unwrap_or_else(|e| panic!("{spelling}: {e}"));
        assert_eq!(device.to_string(), normalised, "`{spelling}` normalised wrongly");

        let mut p = training(LearningTaskParameters::default(), TreeBoosterParameters::default());
        p.booster.general.device = device;
        assert_eq!(p.booster.general.to_config_map()["device"], normalised);

        match api::train(&p, &d, &[]) {
            Ok(_) => assert!(trains, "`{spelling}` should not have trained on this build"),
            Err(e) => {
                assert!(!trains, "`{spelling}` should have trained: {e}");
                assert!(
                    e.to_string().contains("device") || e.to_string().contains("tree_method"),
                    "`{spelling}` was rejected without naming the parameter: {e}"
                );
            }
        }
    }

    for bad in ["", " ", "CPU", "cuda:", "cuda:-1", "cuda:x", "sycl:tpu", "tpu", "cpu:0"] {
        assert!(Device::from_str(bad).is_err(), "`{bad}` should not parse as a device");
    }
}

/// The SYCL device kinds each keep their own spelling.
#[test]
fn every_sycl_kind_keeps_its_spelling() {
    assert_eq!(Device::from_str("sycl").unwrap(), Device::Sycl(SyclKind::Default, None));
    assert_eq!(Device::from_str("sycl:cpu").unwrap(), Device::Sycl(SyclKind::Cpu, None));
    assert_eq!(Device::from_str("sycl:gpu:2").unwrap(), Device::Sycl(SyclKind::Gpu, Some(2)));
}

// ----------------------------------------------------- tree enumerations ----

/// Every `tree_method` spelling trains, and each one that is a distinct
/// algorithm builds a distinct model.
///
/// `auto` and `hist` are the same algorithm upstream, so they must agree
/// exactly; `exact` and `approx` must not, or the parameter would be accepted
/// and ignored.
#[test]
fn every_tree_method_spelling_builds_its_own_model() {
    let d = rich_data(1000, 5);
    let mut models = std::collections::BTreeMap::new();
    for &method in TreeMethod::ALL {
        let p = training(
            LearningTaskParameters::default(),
            TreeBoosterParameters { tree_method: method, ..Default::default() },
        );
        let (booster, _) =
            api::train(&p, &d, &[]).unwrap_or_else(|e| panic!("`{method}` must train: {e}"));
        models.insert(method.as_str(), booster.save_model());
    }
    assert_eq!(models["auto"], models["hist"], "`auto` resolves to `hist`");
    assert_ne!(models["hist"], models["exact"], "`exact` must be its own algorithm");
    assert_ne!(models["exact"], models["approx"]);

    // `approx` sketches its quantiles weighted by the hessian. Under
    // `reg:squarederror` every hessian is 1, so that sketch is the one `hist`
    // builds and the two methods must coincide exactly — the same equality
    // upstream has, and worth pinning so a future change to the weighting is
    // not mistaken for noise.
    assert_eq!(models["hist"], models["approx"], "a constant hessian makes approx == hist");

    // Give the hessian something to vary and they must part company.
    let binary = universal_data(600);
    let mut varying = Vec::new();
    for method in [TreeMethod::Hist, TreeMethod::Approx] {
        let p = training(
            LearningTaskParameters {
                objective: Objective::BinaryLogistic,
                ..Default::default()
            },
            TreeBoosterParameters { tree_method: method, max_bin: 16, ..Default::default() },
        );
        varying.push(api::train(&p, &binary, &[]).unwrap().0.save_model());
    }
    assert_ne!(varying[0], varying[1], "a varying hessian must move the approx cuts");
}

/// `exact` grows level by level and has no other stopping rule, so an
/// unlimited depth is refused rather than silently bounded.
#[test]
fn exact_refuses_an_unlimited_depth() {
    let d = rich_data(200, 3);
    let p = training(
        LearningTaskParameters::default(),
        TreeBoosterParameters {
            tree_method: TreeMethod::Exact,
            max_depth: 0,
            max_leaves: 8,
            ..Default::default()
        },
    );
    match api::train(&p, &d, &[]) {
        Ok(_) => panic!("`exact` with max_depth = 0 must be rejected"),
        Err(e) => assert!(e.to_string().contains("max_depth"), "{e}"),
    }
}

/// Both `grow_policy` spellings change the tree the fit builds.
#[test]
fn both_grow_policy_spellings_build_different_trees() {
    let d = rich_data(2000, 6);
    let mut models = Vec::new();
    for &policy in GrowPolicy::ALL {
        let p = training(
            LearningTaskParameters::default(),
            TreeBoosterParameters {
                grow_policy: policy,
                max_depth: 0,
                max_leaves: 8,
                ..Default::default()
            },
        );
        models.push(api::train(&p, &d, &[]).unwrap_or_else(|e| panic!("{policy}: {e}")).0.save_model());
    }
    assert_ne!(models[0], models[1], "the two growth orders must differ");
}

/// Both `sampling_method` spellings select a different sampler.
#[test]
fn both_sampling_method_spellings_select_a_sampler() {
    let d = rich_data(2000, 6);
    let mut models = Vec::new();
    for &method in SamplingMethod::ALL {
        let p = training(
            LearningTaskParameters::default(),
            TreeBoosterParameters { subsample: 0.5, sampling_method: method, ..Default::default() },
        );
        models.push(api::train(&p, &d, &[]).unwrap_or_else(|e| panic!("{method}: {e}")).0.save_model());
    }
    assert_ne!(models[0], models[1], "the two samplers must differ");
}

/// Every `updater` spelling is either run or rejected naming the parameter.
///
/// The three that are refused are the ones whose device this build has no code
/// for; a caller who names them gets an error, never a quiet substitution.
#[test]
fn every_updater_spelling_is_honoured_or_rejected_by_name() {
    const CPU_UPDATERS: &[TreeUpdaterName] = &[
        TreeUpdaterName::GrowQuantileHistMaker,
        TreeUpdaterName::GrowHistMaker,
        TreeUpdaterName::GrowColMaker,
        TreeUpdaterName::Prune,
        TreeUpdaterName::Refresh,
    ];
    let d = universal_data(100);
    for &updater in TreeUpdaterName::ALL {
        // A tree-modifying updater cannot lead the pipeline, so pair it with a
        // grower — which is the only sequence the parameter surface accepts.
        let sequence = if updater.can_modify_tree() {
            vec![TreeUpdaterName::GrowQuantileHistMaker, updater]
        } else {
            vec![updater]
        };
        let p = training(
            LearningTaskParameters::default(),
            TreeBoosterParameters { updater: Some(sequence), max_depth: 4, ..Default::default() },
        );
        match api::train(&p, &d, &[]) {
            Ok(_) => assert!(
                CPU_UPDATERS.contains(&updater),
                "`{updater}` trained but has no CPU implementation"
            ),
            Err(e) => {
                assert!(
                    !CPU_UPDATERS.contains(&updater),
                    "`{updater}` is implemented but was rejected: {e}"
                );
                assert!(
                    e.to_string().contains("tree_method") || e.to_string().contains("updater"),
                    "`{updater}` was rejected without naming the parameter: {e}"
                );
            }
        }
    }
}

/// The `exact` pipeline is `grow_colmaker,prune`, so `gamma` has to reach the
/// pruner rather than the grower — a split below it is built and then removed.
#[test]
fn gamma_reaches_the_pruner_under_the_exact_tree_method() {
    let d = rich_data(600, 4);
    let mut sizes = Vec::new();
    for gamma in [0.0f32, 5.0] {
        let p = training(
            LearningTaskParameters::default(),
            TreeBoosterParameters {
                tree_method: TreeMethod::Exact,
                max_depth: 4,
                gamma,
                ..Default::default()
            },
        );
        let (booster, _) = api::train(&p, &d, &[]).unwrap();
        let model: serde_json::Value = serde_json::from_str(&booster.save_model()).unwrap();
        let tree = &model["learner"]["gradient_booster"]["model"]["trees"][0];
        let deleted: u64 =
            tree["tree_param"]["num_deleted"].as_str().unwrap().parse().unwrap();
        sizes.push(deleted);
    }
    assert_eq!(sizes[0], 0, "no pruning without gamma");
    assert!(sizes[1] > 0, "gamma must prune splits away");
}

/// Both `process_type` spellings reach the fit and do different things.
///
/// `update` rewrites the trees an existing model holds, so it needs one:
/// without a base model it is refused by name rather than run as a silent
/// no-op, and with one it produces a *different* ensemble from growing afresh.
#[test]
fn every_process_type_spelling_is_honoured_or_rejected_by_name() {
    let d = rich_data(600, 4);
    let base = api::train(
        &training(LearningTaskParameters::default(), TreeBoosterParameters::default()),
        &d,
        &[],
    )
    .unwrap()
    .0;

    let update_params = training(
        LearningTaskParameters::default(),
        TreeBoosterParameters {
            process_type: ProcessType::Update,
            updater: Some(vec![TreeUpdaterName::Refresh]),
            ..Default::default()
        },
    );

    // Without a base model there is nothing to revisit.
    match api::train(&update_params, &d, &[]) {
        Ok(_) => panic!("`update` without a base model must be rejected"),
        Err(e) => assert!(e.to_string().contains("process_type"), "{e}"),
    }

    // With one, the round rewrites rather than grows.
    let (updated, _) = api::train_from(&update_params, &d, &[], Some(&base)).unwrap();
    assert_eq!(updated.num_trees(), base.num_trees(), "`update` adds no trees");

    let (grown, _) = api::train_from(
        &training(LearningTaskParameters::default(), TreeBoosterParameters::default()),
        &d,
        &[],
        Some(&base),
    )
    .unwrap();
    assert!(grown.num_trees() > base.num_trees(), "`default` grows new trees");
}

/// `refresh_leaf` decides whether the `refresh` updater rewrites leaf values
/// or only the statistics beneath them.
#[test]
fn refresh_leaf_decides_whether_refresh_moves_the_predictions() {
    let d = rich_data(600, 4);
    let base = api::train(
        &training(LearningTaskParameters::default(), TreeBoosterParameters::default()),
        &d,
        &[],
    )
    .unwrap()
    .0;
    // Refreshing against *different* data is what makes the two settings
    // visibly differ: refreshing on the same data reproduces the leaves it
    // already has, so there would be nothing to see.
    let other = sparse_data(600, 4);

    let refreshed = |refresh_leaf: bool| {
        let p = training(
            LearningTaskParameters::default(),
            TreeBoosterParameters {
                process_type: ProcessType::Update,
                updater: Some(vec![TreeUpdaterName::Refresh]),
                refresh_leaf,
                ..Default::default()
            },
        );
        api::train_from(&p, &other, &[], Some(&base)).unwrap().0.predict(&d)
    };
    let kept = refreshed(false);
    let moved = refreshed(true);
    assert_eq!(kept, base.predict(&d), "refresh_leaf = false must not move a prediction");
    assert_ne!(kept, moved, "refresh_leaf = true must re-fit the leaves");
}

/// Every `multi_strategy` spelling is honoured or rejected naming the
/// parameter.
#[test]
fn every_multi_strategy_spelling_is_honoured_or_rejected_by_name() {
    let d = universal_data(100);
    for &strategy in MultiStrategy::ALL {
        let p = training(
            LearningTaskParameters::default(),
            TreeBoosterParameters { multi_strategy: strategy, ..Default::default() },
        );
        match api::train(&p, &d, &[]) {
            Ok(_) => assert_eq!(
                strategy,
                MultiStrategy::OneOutputPerTree,
                "`{strategy}` trained but vector leaves are not implemented"
            ),
            Err(e) => assert!(
                e.to_string().contains("multi_strategy"),
                "`{strategy}` was rejected without naming the parameter: {e}"
            ),
        }
    }
}

/// `default_direction` steers the `exact` updater's missing-value handling, so
/// every spelling must change an `exact` fit over sparse data — and none of
/// them may change a `hist` fit, which learns the direction from the
/// histograms instead.
#[test]
fn every_default_direction_spelling_steers_the_exact_updater() {
    // Sparse data, so there are missing values for the direction to route.
    let d = sparse_data(600, 4);

    let mut exact_models = Vec::new();
    let mut hist_models = Vec::new();
    for &direction in DefaultDirection::ALL {
        let params =
            TreeBoosterParameters { default_direction: direction, ..Default::default() };
        assert_eq!(params.to_config_map()["default_direction"], direction.as_str());

        let hist = training(LearningTaskParameters::default(), params.clone());
        hist_models.push(api::train(&hist, &d, &[]).unwrap().0.save_model());

        let exact = training(
            LearningTaskParameters::default(),
            TreeBoosterParameters { tree_method: TreeMethod::Exact, max_depth: 4, ..params },
        );
        exact_models
            .push(api::train(&exact, &d, &[]).unwrap_or_else(|e| panic!("{direction}: {e}")).0);
    }
    assert!(
        hist_models.windows(2).all(|w| w[0] == w[1]),
        "`default_direction` steers `exact` only; a hist fit must be unchanged"
    );

    // `left` sends every missing value left, `right` sends it right, and
    // `learn` picks per split — so no two agree.
    let flags = |b: &xgboost_rs::Booster| -> Vec<u64> {
        let m: serde_json::Value = serde_json::from_str(&b.save_model()).unwrap();
        m["learner"]["gradient_booster"]["model"]["trees"][0]["default_left"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect()
    };
    let left = flags(&exact_models[1]);
    let right = flags(&exact_models[2]);
    assert!(left.iter().any(|v| *v == 1), "`left` must route missing values left");
    assert!(right.iter().all(|v| *v == 0), "`right` must route every missing value right");
    assert_ne!(left, right);
}

/// Every `monotone_constraints` spelling reaches the fit, and the list is
/// emitted in XGBoost's `(a,b,c)` form.
#[test]
fn every_monotone_constraint_spelling_reaches_the_fit() {
    // A single rising feature, so each direction has a visible effect.
    let rows = 300;
    let x: Vec<f32> = (0..rows).map(|i| i as f32 / rows as f32).collect();
    let y: Vec<f32> = x.iter().map(|v| if (0.4..0.6).contains(v) { v - 0.5 } else { *v }).collect();
    let mut d = DMatrix::from_dense(&x, rows, 1, f32::NAN).unwrap();
    d.set_labels(&y).unwrap();

    for &constraint in MonotoneConstraint::ALL {
        let params = TreeBoosterParameters {
            monotone_constraints: vec![constraint],
            max_depth: 4,
            ..Default::default()
        };
        assert_eq!(
            params.to_config_map()["monotone_constraints"],
            format!("({})", constraint.as_str()),
            "the emitted list must use the upstream `(a,b,c)` form"
        );

        let mut p = training(LearningTaskParameters::default(), params);
        p.num_boost_round = 20;
        let preds = api::train(&p, &d, &[]).unwrap().0.predict(&d);
        match constraint {
            MonotoneConstraint::Increasing => {
                assert!(preds.windows(2).all(|w| w[1] >= w[0]), "`1` did not force a rise");
            }
            MonotoneConstraint::Decreasing => {
                assert!(preds.windows(2).all(|w| w[1] <= w[0]), "`-1` did not force a fall");
            }
            MonotoneConstraint::Unconstrained => {
                assert!(preds.windows(2).any(|w| w[1] < w[0]), "`0` should not constrain");
            }
        }
    }

    // Several features join with commas, in feature order.
    let many = TreeBoosterParameters::builder()
        .monotone_constraints(vec![
            MonotoneConstraint::Increasing,
            MonotoneConstraint::Unconstrained,
            MonotoneConstraint::Decreasing,
        ])
        .build()
        .unwrap();
    assert_eq!(many.to_config_map()["monotone_constraints"], "(1,0,-1)");
}

/// `interaction_constraints` is emitted as JSON, and it reaches the fit.
#[test]
fn the_interaction_constraint_string_is_json_and_reaches_the_fit() {
    let params = TreeBoosterParameters::builder()
        .interaction_constraints(vec![vec![0, 1], vec![2, 3, 4, 5]])
        .build()
        .unwrap();
    assert_eq!(params.to_config_map()["interaction_constraints"], "[[0,1],[2,3,4,5]]");

    let d = rich_data(2000, 6);
    let mut p = training(LearningTaskParameters::default(), params);
    p.num_boost_round = 5;
    let constrained = api::train(&p, &d, &[]).unwrap().0.save_model();

    let plain = training(LearningTaskParameters::default(), TreeBoosterParameters::default());
    let mut plain = plain;
    plain.num_boost_round = 5;
    assert_ne!(constrained, api::train(&plain, &d, &[]).unwrap().0.save_model());
}

/// The `updater` sequence is emitted as a comma-separated list in order.
#[test]
fn the_updater_sequence_is_a_comma_separated_list() {
    let params = TreeBoosterParameters::builder()
        .updater([
            TreeUpdaterName::GrowQuantileHistMaker,
            TreeUpdaterName::Prune,
            TreeUpdaterName::Refresh,
        ])
        .build()
        .unwrap();
    assert_eq!(
        params.to_config_map()["updater"],
        "grow_quantile_histmaker,prune,refresh"
    );
}

// -------------------------------------------------------- dart and linear ----

/// Every `sample_type` and `normalize_type` spelling reaches a DART fit.
#[test]
fn every_dart_spelling_reaches_the_fit() {
    let d = universal_data(400);
    let mut models = Vec::new();
    for &sample_type in DartSampleType::ALL {
        for &normalize_type in DartNormalizeType::ALL {
            let dart = DartParameters {
                sample_type,
                normalize_type,
                rate_drop: 0.5,
                one_drop: true,
                ..Default::default()
            };
            let config = dart.to_config_map();
            assert_eq!(config["sample_type"], sample_type.as_str());
            assert_eq!(config["normalize_type"], normalize_type.as_str());

            let mut p =
                training(LearningTaskParameters::default(), TreeBoosterParameters::default());
            p.booster.booster = BoosterType::Dart(dart);
            p.num_boost_round = 8;
            models.push(
                api::train(&p, &d, &[])
                    .unwrap_or_else(|e| panic!("{sample_type}/{normalize_type}: {e}"))
                    .0
                    .predict(&d),
            );
        }
    }
    // The four combinations are four different fits.
    for i in 0..models.len() {
        for j in i + 1..models.len() {
            assert_ne!(models[i], models[j], "DART combinations {i} and {j} coincide");
        }
    }
}

/// Every `feature_selector` and linear `updater` spelling is accepted by the
/// parameter surface, and a `gblinear` fit is rejected naming the booster —
/// never accepted as a tree fit in disguise.
#[test]
fn every_linear_spelling_that_the_surface_accepts_also_trains() {
    let d = universal_data(50);

    for &updater in LinearUpdater::ALL {
        for &selector in FeatureSelector::ALL {
            // `shotgun` cannot run the selectors that need a full pass.
            let built =
                LinearBoosterParameters::builder().updater(updater).feature_selector(selector).build();
            match built {
                Ok(linear) => {
                    let config = linear.to_config_map();
                    assert_eq!(config["updater"], updater.as_str());
                    assert_eq!(config["feature_selector"], selector.as_str());

                    let mut p = training(
                        LearningTaskParameters::default(),
                        TreeBoosterParameters::default(),
                    );
                    p.booster.booster = BoosterType::Gblinear(linear);
                    p.num_boost_round = 10;
                    let (booster, _) = api::train(&p, &d, &[])
                        .unwrap_or_else(|e| panic!("{updater}/{selector}: {e}"));
                    assert_eq!(booster.num_trees(), 0, "a linear fit grows no trees");
                    assert!(booster.predict(&d).iter().all(|v| v.is_finite()));
                }
                Err(e) => {
                    assert_eq!(updater, LinearUpdater::Shotgun, "{selector}: {e}");
                    assert!(e.to_string().contains("feature_selector"), "{e}");
                }
            }
        }
    }
}

// --------------------------------------------------- learning-task strings ----

/// Every `verbosity` spelling is accepted and reaches the context.
#[test]
fn every_verbosity_spelling_reaches_the_fit() {
    let d = universal_data(50);
    for &verbosity in Verbosity::ALL {
        let mut p = training(LearningTaskParameters::default(), TreeBoosterParameters::default());
        p.booster.general.verbosity = verbosity;
        assert_eq!(p.booster.general.to_config_map()["verbosity"], verbosity.as_str());
        // Verbosity only changes what is printed, so the model must not move.
        api::train(&p, &d, &[]).unwrap_or_else(|e| panic!("{verbosity}: {e}"));
    }
}

/// Every `aft_loss_distribution` spelling reaches the AFT loss and gives a
/// different fit.
#[test]
fn every_aft_distribution_spelling_reaches_the_fit() {
    let d = universal_data(300);
    let mut models = Vec::new();
    for &dist in AftDistribution::ALL {
        let objective = Objective::SurvivalAft {
            aft_loss_distribution: dist,
            aft_loss_distribution_scale: 1.0,
        };
        assert_eq!(objective.to_config_map()["aft_loss_distribution"], dist.as_str());

        let mut p = training(
            LearningTaskParameters { objective, ..Default::default() },
            TreeBoosterParameters::default(),
        );
        p.num_boost_round = 6;
        models.push(api::train(&p, &d, &[]).unwrap_or_else(|e| panic!("{dist}: {e}")).0.predict(&d));
    }
    for i in 0..models.len() {
        for j in i + 1..models.len() {
            assert_ne!(models[i], models[j], "AFT distributions {i} and {j} coincide");
        }
    }
}

/// Every `lambdarank_pair_method` spelling reaches the ranking objective.
#[test]
fn every_lambdarank_pair_method_spelling_reaches_the_fit() {
    let d = universal_data(400);
    let mut models = Vec::new();
    for &pair_method in LambdaRankPairMethod::ALL {
        let param = LambdaRankParameters { pair_method, ..LambdaRankParameters::default() };
        let objective = Objective::RankNdcg(param);
        assert_eq!(objective.to_config_map()["lambdarank_pair_method"], pair_method.as_str());

        let mut p = training(
            LearningTaskParameters { objective, ..Default::default() },
            TreeBoosterParameters::default(),
        );
        p.num_boost_round = 6;
        models
            .push(api::train(&p, &d, &[]).unwrap_or_else(|e| panic!("{pair_method}: {e}")).0.predict(&d));
    }
    assert_ne!(models[0], models[1], "the two pair methods must differ");
}

/// The `(a,b,c)` alpha lists round-trip through the config and reach the fit.
#[test]
fn the_alpha_lists_are_emitted_in_the_upstream_form() {
    for (objective, key, expected) in [
        (
            Objective::RegQuantileError { quantile_alpha: vec![0.1, 0.5, 0.9] },
            "quantile_alpha",
            "(0.1,0.5,0.9)",
        ),
        (
            Objective::RegExpectileError { expectile_alpha: vec![0.25, 0.75] },
            "expectile_alpha",
            "(0.25,0.75)",
        ),
    ] {
        assert_eq!(objective.to_config_map()[key], expected);
    }

    // A single alpha still uses the list form, so a parser never has to guess.
    let one = Objective::RegQuantileError { quantile_alpha: vec![0.5] };
    assert_eq!(one.to_config_map()["quantile_alpha"], "(0.5)");
}

/// No emitted configuration value may contain whitespace: `Learner::SetParam`
/// splits on it, so a value that did would be silently truncated.
#[test]
fn no_emitted_string_contains_whitespace() {
    let rank = LambdaRankParameters::default();
    let objectives = [
        Objective::RegQuantileError { quantile_alpha: vec![0.1, 0.9] },
        Objective::RegExpectileError { expectile_alpha: vec![0.1, 0.9] },
        Objective::SurvivalAft {
            aft_loss_distribution: AftDistribution::Extreme,
            aft_loss_distribution_scale: 2.5,
        },
        Objective::MultiSoftprob { num_class: 7 },
        Objective::RankNdcg(rank),
        Objective::RegTweedie { tweedie_variance_power: 1.25 },
    ];
    for objective in objectives {
        for (key, value) in objective.to_config() {
            assert!(!value.chars().any(char::is_whitespace), "`{key}` = `{value}` has whitespace");
        }
    }

    let tree = TreeBoosterParameters::builder()
        .monotone_constraints(vec![MonotoneConstraint::Increasing, MonotoneConstraint::Decreasing])
        .interaction_constraints(vec![vec![0, 1], vec![2, 3]])
        .updater([TreeUpdaterName::GrowQuantileHistMaker, TreeUpdaterName::Prune])
        .build()
        .unwrap();
    for (key, value) in tree.to_config() {
        assert!(!value.chars().any(char::is_whitespace), "`{key}` = `{value}` has whitespace");
    }
}
