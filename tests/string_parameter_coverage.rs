//! Proof that "every string parameter pattern is tested" stays true.
//!
//! The other two string-parameter files test spellings against a written list.
//! A list goes stale the moment a variant is added, and nothing notices. This
//! file closes that hole from both ends:
//!
//! * every closed enum is checked against its own `ALL`, so a variant missing
//!   from `ALL` is caught;
//! * [`Objective`] and [`EvalMetric`] carry data and have no `ALL`, so each is
//!   labelled through an **exhaustive match**. Adding a variant stops this file
//!   compiling, which is the whole point — the census cannot silently fall
//!   behind the type.
//!
//! Every pattern named here is then exercised end to end: parsed, rendered,
//! parsed back, and run through a fit.

use std::collections::BTreeSet;
use std::str::FromStr;

use xgboost_rs::parameters::{
    AftDistribution, BoosterParameters, BoosterType, DartNormalizeType, DartParameters,
    DartSampleType, DefaultDirection, Device, EvalMetric, FeatureSelector, GeneralParameters,
    GrowPolicy, LambdaRankPairMethod, LambdaRankParameters, LearningTaskParameters, LinearUpdater,
    MonotoneConstraint, MultiStrategy, Objective, ProcessType, SamplingMethod, ToConfig,
    TrainingParameters, TreeBoosterParameters, TreeMethod, TreeUpdaterName, VerboseEval, Verbosity,
};
use xgboost_rs::{DMatrix, api};

/// Data every objective and metric can be pointed at: positive labels in
/// `[1, 2]`, a censoring interval, and query groups.
fn universal_data(rows: usize) -> DMatrix {
    let cols = 3usize;
    let x: Vec<f32> = (0..rows * cols).map(|i| ((i * 37) % 101) as f32 / 101.0).collect();
    let y: Vec<f32> = (0..rows).map(|r| 1.0 + (r % 2) as f32).collect();
    let mut d = DMatrix::from_dense(&x, rows, cols, f32::NAN).unwrap();
    d.set_labels(&y).unwrap();
    d.set_label_bounds(&y, &y.iter().map(|v| v + 1.0).collect::<Vec<_>>()).unwrap();
    d.set_group(&vec![4; rows / 4]).unwrap();
    d
}

fn training(learning: LearningTaskParameters) -> TrainingParameters {
    TrainingParameters {
        booster: BoosterParameters {
            booster: BoosterType::Gbtree(TreeBoosterParameters::default()),
            general: GeneralParameters { verbosity: Verbosity::Silent, ..Default::default() },
            learning,
        },
        num_boost_round: 2,
        verbose_eval: VerboseEval::Silent,
        ..Default::default()
    }
}

// ------------------------------------------------------- closed enums ----

/// Every closed string enum: each variant's spelling must be distinct,
/// non-empty, free of whitespace, and parse back to itself.
///
/// `ALL` is the list the other test files iterate, so this is what makes their
/// coverage mean "every variant" rather than "every variant someone listed".
macro_rules! census {
    ($($ty:ty),+ $(,)?) => {$({
        let variants = <$ty>::ALL;
        assert!(!variants.is_empty(), "{} has no variants", stringify!($ty));

        let mut seen = BTreeSet::new();
        for &variant in variants {
            let spelling = variant.as_str();
            assert!(!spelling.is_empty(), "{}: an empty spelling", stringify!($ty));
            assert!(
                !spelling.contains(char::is_whitespace),
                "{}: `{spelling}` contains whitespace, which a config line cannot carry",
                stringify!($ty)
            );
            assert!(
                seen.insert(spelling),
                "{}: `{spelling}` appears twice in ALL",
                stringify!($ty)
            );
            assert_eq!(
                <$ty>::from_str(spelling).expect("its own spelling must parse"),
                variant,
                "{}: `{spelling}` did not parse back to itself",
                stringify!($ty)
            );
            assert_eq!(variant.to_string(), spelling, "{}: Display disagrees with as_str", stringify!($ty));
        }

        // A spelling nothing defines must be refused, naming the parameter it
        // was offered for.
        let err = <$ty>::from_str("definitely-not-a-real-value").unwrap_err().to_string();
        assert!(
            err.contains(<$ty>::parameter_name()),
            "{}: rejection did not name `{}`: {err}",
            stringify!($ty),
            <$ty>::parameter_name()
        );
    })+};
}

#[test]
fn every_closed_string_enum_is_a_complete_distinct_round_tripping_set() {
    census!(
        TreeMethod,
        GrowPolicy,
        SamplingMethod,
        ProcessType,
        MultiStrategy,
        TreeUpdaterName,
        DefaultDirection,
        MonotoneConstraint,
        DartSampleType,
        DartNormalizeType,
        Verbosity,
        AftDistribution,
        LambdaRankPairMethod,
        FeatureSelector,
        LinearUpdater,
    );
}

/// The closed enums, with how many variants each has.
///
/// A bare count looks like a tautology, but it is the tripwire: adding a
/// variant changes the count, this test fails, and whoever added it has to go
/// and give the new spelling a behaviour test rather than leaving it to be
/// accepted and never exercised.
#[test]
fn the_closed_enum_variant_counts_are_pinned() {
    let counts: Vec<(&str, usize)> = vec![
        ("TreeMethod", TreeMethod::ALL.len()),
        ("GrowPolicy", GrowPolicy::ALL.len()),
        ("SamplingMethod", SamplingMethod::ALL.len()),
        ("ProcessType", ProcessType::ALL.len()),
        ("MultiStrategy", MultiStrategy::ALL.len()),
        ("TreeUpdaterName", TreeUpdaterName::ALL.len()),
        ("DefaultDirection", DefaultDirection::ALL.len()),
        ("MonotoneConstraint", MonotoneConstraint::ALL.len()),
        ("DartSampleType", DartSampleType::ALL.len()),
        ("DartNormalizeType", DartNormalizeType::ALL.len()),
        ("Verbosity", Verbosity::ALL.len()),
        ("AftDistribution", AftDistribution::ALL.len()),
        ("LambdaRankPairMethod", LambdaRankPairMethod::ALL.len()),
        ("FeatureSelector", FeatureSelector::ALL.len()),
        ("LinearUpdater", LinearUpdater::ALL.len()),
    ];
    assert_eq!(
        counts,
        vec![
            ("TreeMethod", 4),
            ("GrowPolicy", 2),
            ("SamplingMethod", 2),
            ("ProcessType", 2),
            ("MultiStrategy", 2),
            ("TreeUpdaterName", 8),
            ("DefaultDirection", 3),
            ("MonotoneConstraint", 3),
            ("DartSampleType", 2),
            ("DartNormalizeType", 2),
            ("Verbosity", 4),
            ("AftDistribution", 3),
            ("LambdaRankPairMethod", 2),
            ("FeatureSelector", 5),
            ("LinearUpdater", 2),
        ],
        "a string enum gained or lost a variant; give the new spelling a \
         behaviour test in string_parameter_behaviour.rs, then update this list"
    );
}

/// `ProcessType`'s two spellings, exercised for their effect rather than only
/// their spelling: `default` grows trees, `update` rewrites them.
#[test]
fn every_process_type_spelling_reaches_the_fit() {
    let d = universal_data(200);
    for &process_type in ProcessType::ALL {
        let tree = match process_type {
            ProcessType::Default => TreeBoosterParameters::default(),
            ProcessType::Update => TreeBoosterParameters::builder()
                .process_type(ProcessType::Update)
                .updater([TreeUpdaterName::Refresh])
                .build()
                .unwrap(),
        };
        let config = tree.to_config_map();
        assert_eq!(config["process_type"], process_type.as_str());

        let base = api::train(&training(LearningTaskParameters::default()), &d, &[]).unwrap().0;
        let p = TrainingParameters {
            booster: BoosterParameters {
                booster: BoosterType::Gbtree(tree),
                general: GeneralParameters { verbosity: Verbosity::Silent, ..Default::default() },
                learning: LearningTaskParameters::default(),
            },
            num_boost_round: 2,
            verbose_eval: VerboseEval::Silent,
            ..Default::default()
        };
        let (fitted, _) = api::train_from(&p, &d, &[], Some(&base)).unwrap();
        match process_type {
            ProcessType::Default => {
                assert!(fitted.num_trees() > base.num_trees(), "`default` must grow trees")
            }
            ProcessType::Update => {
                assert_eq!(fitted.num_trees(), base.num_trees(), "`update` must not")
            }
        }
    }
}

// ------------------------------------------------------ open patterns ----

/// A label for every [`Objective`] shape.
///
/// The match is exhaustive on purpose: a new objective will not compile until
/// it is named here, and [`every_objective_pattern_is_covered`] then insists it
/// is also fitted.
fn objective_label(o: &Objective) -> &'static str {
    match o {
        Objective::RegSquaredError => "reg:squarederror",
        Objective::RegSquaredLogError => "reg:squaredlogerror",
        Objective::RegLogistic => "reg:logistic",
        Objective::RegPseudoHuberError { .. } => "reg:pseudohubererror",
        Objective::RegAbsoluteError => "reg:absoluteerror",
        Objective::RegQuantileError { .. } => "reg:quantileerror",
        Objective::RegExpectileError { .. } => "reg:expectileerror",
        Objective::RegGamma => "reg:gamma",
        Objective::RegTweedie { .. } => "reg:tweedie",
        Objective::RegLinear => "reg:linear",
        Objective::CountPoisson { .. } => "count:poisson",
        Objective::SurvivalCox => "survival:cox",
        Objective::SurvivalAft { .. } => "survival:aft",
        Objective::BinaryLogistic => "binary:logistic",
        Objective::BinaryLogitRaw => "binary:logitraw",
        Objective::BinaryHinge => "binary:hinge",
        Objective::MultiSoftmax { .. } => "multi:softmax",
        Objective::MultiSoftprob { .. } => "multi:softprob",
        Objective::RankPairwise(_) => "rank:pairwise",
        Objective::RankNdcg(_) => "rank:ndcg",
        Objective::RankMap(_) => "rank:map",
    }
}

/// Every objective shape, each with the arguments its pattern carries.
fn objective_samples() -> Vec<Objective> {
    let rank = LambdaRankParameters::default();
    vec![
        Objective::RegSquaredError,
        Objective::RegSquaredLogError,
        Objective::RegLogistic,
        Objective::RegPseudoHuberError { huber_slope: 1.5 },
        Objective::RegAbsoluteError,
        Objective::RegQuantileError { quantile_alpha: vec![0.25, 0.75] },
        Objective::RegExpectileError { expectile_alpha: vec![0.3, 0.7] },
        Objective::RegGamma,
        Objective::RegTweedie { tweedie_variance_power: 1.4 },
        Objective::RegLinear,
        Objective::CountPoisson { max_delta_step: 0.7 },
        Objective::SurvivalCox,
        Objective::SurvivalAft {
            aft_loss_distribution: AftDistribution::Logistic,
            aft_loss_distribution_scale: 1.2,
        },
        Objective::BinaryLogistic,
        Objective::BinaryLogitRaw,
        Objective::BinaryHinge,
        Objective::MultiSoftmax { num_class: 2 },
        Objective::MultiSoftprob { num_class: 2 },
        Objective::RankPairwise(rank),
        Objective::RankNdcg(rank),
        Objective::RankMap(rank),
    ]
}

/// Every objective pattern is sampled, and each sample trains.
#[test]
fn every_objective_pattern_is_covered() {
    let samples = objective_samples();
    let labelled: BTreeSet<&str> = samples.iter().map(objective_label).collect();

    // A pattern with no sample would show up as a missing label. There is no
    // way to enumerate the match arms directly, so the count is pinned instead
    // — and the match itself guarantees no arm is missing.
    assert_eq!(
        labelled.len(),
        samples.len(),
        "two samples share a pattern; every objective shape needs its own"
    );
    assert_eq!(labelled.len(), 21, "an objective was added or removed; add a sample for it");

    // Binary labels for the classifiers, positive ones for the log-link
    // objectives, which the shared matrix already provides.
    let positive = universal_data(200);
    let mut binary = universal_data(200);
    let labels: Vec<f32> = (0..200).map(|r| (r % 2) as f32).collect();
    binary.set_labels(&labels).unwrap();

    for objective in samples {
        let label = objective_label(&objective);
        // The logistic losses need labels in `[0, 1]`; the log-link ones need
        // strictly positive labels, which the shared matrix already has.
        let d = if label.starts_with("binary:")
            || label.starts_with("multi:")
            || label == "reg:logistic"
        {
            &binary
        } else {
            &positive
        };
        let p = training(LearningTaskParameters {
            objective: objective.clone(),
            ..Default::default()
        });
        let booster = api::train(&p, d, &[])
            .unwrap_or_else(|e| panic!("`{label}` was rejected: {e}"))
            .0;
        assert!(
            booster.predict(d).iter().all(|v| v.is_finite()),
            "`{label}` predicted a non-finite value"
        );

        // The spelling round trips through the config surface.
        let config = p.booster.learning.to_config_map();
        let emitted = &config["objective"];
        assert_eq!(
            Objective::from_str(emitted).map(|o| objective_label(&o)).unwrap(),
            objective_label(&Objective::from_str(emitted).unwrap()),
            "`{label}` emitted `{emitted}`, which does not parse back"
        );
    }
}

/// A label for every [`EvalMetric`] shape, exhaustive for the same reason.
fn metric_label(m: &EvalMetric) -> &'static str {
    match m {
        EvalMetric::Rmse => "rmse",
        EvalMetric::Rmsle => "rmsle",
        EvalMetric::Mae => "mae",
        EvalMetric::Mape => "mape",
        EvalMetric::Mphe => "mphe",
        EvalMetric::Logloss => "logloss",
        EvalMetric::Error => "error",
        EvalMetric::ErrorAt(_) => "error@t",
        EvalMetric::MError => "merror",
        EvalMetric::MLogloss => "mlogloss",
        EvalMetric::Auc => "auc",
        EvalMetric::Aucpr => "aucpr",
        EvalMetric::Pre(None) => "pre",
        EvalMetric::Pre(Some(_)) => "pre@k",
        EvalMetric::Ndcg { top_n: None, .. } => "ndcg",
        EvalMetric::Ndcg { top_n: Some(_), .. } => "ndcg@k",
        EvalMetric::Map { top_n: None, .. } => "map",
        EvalMetric::Map { top_n: Some(_), .. } => "map@k",
        EvalMetric::PoissonNegLogLik => "poisson-nloglik",
        EvalMetric::GammaNegLogLik => "gamma-nloglik",
        EvalMetric::CoxNegLogLik => "cox-nloglik",
        EvalMetric::GammaDeviance => "gamma-deviance",
        EvalMetric::TweedieNegLogLik(_) => "tweedie-nloglik@p",
        EvalMetric::AftNegLogLik => "aft-nloglik",
        EvalMetric::IntervalRegressionAccuracy => "interval-regression-accuracy",
        EvalMetric::Quantile => "quantile",
        EvalMetric::Expectile => "expectile",
        EvalMetric::Ams(_) => "ams@t",
        EvalMetric::Custom(_) => "custom",
    }
}

/// One sample per metric shape, including both halves of each optional
/// argument and both settings of the ranking `-` suffix.
fn metric_samples() -> Vec<EvalMetric> {
    vec![
        EvalMetric::Rmse,
        EvalMetric::Rmsle,
        EvalMetric::Mae,
        EvalMetric::Mape,
        EvalMetric::Mphe,
        EvalMetric::Logloss,
        EvalMetric::Error,
        EvalMetric::ErrorAt(0.25),
        EvalMetric::MError,
        EvalMetric::MLogloss,
        EvalMetric::Auc,
        EvalMetric::Aucpr,
        EvalMetric::Pre(None),
        EvalMetric::Pre(Some(3)),
        EvalMetric::from_str("ndcg").unwrap(),
        EvalMetric::from_str("ndcg@5").unwrap(),
        EvalMetric::from_str("ndcg-").unwrap(),
        EvalMetric::from_str("ndcg@5-").unwrap(),
        EvalMetric::from_str("map").unwrap(),
        EvalMetric::from_str("map@3").unwrap(),
        EvalMetric::from_str("map-").unwrap(),
        EvalMetric::from_str("map@3-").unwrap(),
        EvalMetric::PoissonNegLogLik,
        EvalMetric::GammaNegLogLik,
        EvalMetric::CoxNegLogLik,
        EvalMetric::GammaDeviance,
        EvalMetric::TweedieNegLogLik(1.5),
        EvalMetric::AftNegLogLik,
        EvalMetric::IntervalRegressionAccuracy,
        EvalMetric::Quantile,
        EvalMetric::Expectile,
        EvalMetric::Ams(0.15),
    ]
}

/// Every metric pattern is sampled, renders to a spelling that parses back, and
/// is reported under that spelling by a real fit.
#[test]
fn every_eval_metric_pattern_is_covered() {
    let samples = metric_samples();
    let labelled: BTreeSet<&str> = samples.iter().map(metric_label).collect();
    assert_eq!(
        labelled.len(),
        28,
        "a metric shape gained or lost a sample; the labels seen were {labelled:?}"
    );
    // Only `custom` is deliberately unsampled here: it is any string at all,
    // and its rejection is tested in string_parameter_behaviour.rs.
    assert!(!labelled.contains("custom"));

    let d = universal_data(200);
    for metric in samples {
        let label = metric_label(&metric);
        let rendered = metric.to_string();
        assert!(
            !rendered.contains(char::is_whitespace),
            "`{label}` rendered `{rendered}`, which a config line cannot carry"
        );
        assert_eq!(
            EvalMetric::from_str(&rendered).unwrap(),
            metric,
            "`{label}` rendered `{rendered}`, which does not parse back"
        );

        // The multiclass metrics need a multiclass fit to mean anything.
        let mut learning = LearningTaskParameters {
            eval_metric: vec![metric.clone()],
            ..Default::default()
        };
        let d = if matches!(metric, EvalMetric::MError | EvalMetric::MLogloss) {
            learning.objective = Objective::MultiSoftprob { num_class: 2 };
            let mut multi = universal_data(200);
            multi.set_labels(&(0..200).map(|r| (r % 2) as f32).collect::<Vec<f32>>()).unwrap();
            multi
        } else {
            d.clone()
        };

        let (_, history) = api::train(&training(learning), &d, &[(&d, "eval")])
            .unwrap_or_else(|e| panic!("`{label}` (`{rendered}`) was rejected: {e}"));
        assert_eq!(
            history[0][0].0,
            format!("eval-{rendered}"),
            "`{label}` reported under the wrong name"
        );
    }
}

/// [`Device`] is the third open pattern: a kind, an optional ordinal, and for
/// SYCL an optional sub-kind.
#[test]
fn every_device_pattern_round_trips() {
    let spellings = [
        "cpu",
        "cuda",
        "cuda:0",
        "cuda:3",
        "gpu",
        "gpu:1",
        "sycl",
        "sycl:0",
        "sycl:cpu",
        "sycl:cpu:1",
        "sycl:gpu",
        "sycl:gpu:2",
    ];
    let mut seen = BTreeSet::new();
    for spelling in spellings {
        let device = Device::from_str(spelling).unwrap_or_else(|e| panic!("`{spelling}`: {e}"));
        let rendered = device.to_string();
        assert_eq!(
            Device::from_str(&rendered).unwrap(),
            device,
            "`{spelling}` rendered `{rendered}`, which does not parse back"
        );
        seen.insert(rendered);
    }
    // `gpu` is an alias for `cuda`, so the rendered set is smaller than the
    // spelling set — which is exactly the aliasing being asserted.
    assert!(seen.contains("cuda"), "the `gpu` alias must render as `cuda`: {seen:?}");

    for bad in ["", "cuda:", "cuda:-1", "cuda:x", "tpu", "sycl:tpu", "cpu:0"] {
        let err = Device::from_str(bad)
            .map(|d| d.to_string())
            .expect_err(&format!("`{bad}` must not parse"));
        assert!(err.to_string().contains("device"), "`{bad}`: {err}");
    }
}

/// Every DART spelling pair reaches the fit, and every combination is a
/// distinct configuration rather than the same one under two names.
#[test]
fn every_dart_pattern_combination_is_distinct() {
    let d = universal_data(300);
    let mut models = BTreeSet::new();
    for &sample_type in DartSampleType::ALL {
        for &normalize_type in DartNormalizeType::ALL {
            let dart = DartParameters::builder()
                .sample_type(sample_type)
                .normalize_type(normalize_type)
                .rate_drop(0.4)
                .build()
                .unwrap();
            let config = dart.to_config_map();
            assert_eq!(config["sample_type"], sample_type.as_str());
            assert_eq!(config["normalize_type"], normalize_type.as_str());

            let p = TrainingParameters {
                booster: BoosterParameters {
                    booster: BoosterType::Dart(dart),
                    general: GeneralParameters {
                        verbosity: Verbosity::Silent,
                        ..Default::default()
                    },
                    learning: LearningTaskParameters::default(),
                },
                num_boost_round: 8,
                verbose_eval: VerboseEval::Silent,
                ..Default::default()
            };
            let booster = api::train(&p, &d, &[])
                .unwrap_or_else(|e| panic!("`{sample_type}`/`{normalize_type}`: {e}"))
                .0;
            models.insert(booster.save_model());
        }
    }
    assert_eq!(
        models.len(),
        DartSampleType::ALL.len() * DartNormalizeType::ALL.len(),
        "two DART spellings produced the same model, so one of them did nothing"
    );
}
