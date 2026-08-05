//! Every pattern of every string-valued parameter.
//!
//! XGBoost's enumerated parameters are strings on the wire, and a value that
//! serialises to a spelling the parser would reject is a silent
//! misconfiguration: the fit runs, with a different setting than the caller
//! asked for. These tests walk *every* variant of *every* such parameter
//! through the whole round trip —
//!
//! ```text
//! variant -> as_str -> Display -> FromStr -> variant
//!         -> serde JSON -> variant
//!         -> ToConfig entry under the upstream parameter name
//! ```
//!
//! — and check that nothing outside that list is accepted.
//!
//! `Objective` and `EvalMetric` are not `str_enum!` types (their variants carry
//! per-objective parameters), so they get the same treatment written out
//! against their own `name()` / `to_string()`.

use std::collections::BTreeSet;
use std::str::FromStr;

use xgboost_rs::parameters::{
    AftDistribution, DartNormalizeType, DartParameters, DartSampleType, DefaultDirection, Device,
    EvalMetric, GeneralParameters, GrowPolicy, LambdaRankPairMethod, LambdaRankParameters,
    LearningTaskParameters,
    LinearBoosterParameters, LinearUpdater, FeatureSelector, MonotoneConstraint, MultiStrategy,
    Objective, ProcessType, SamplingMethod, SyclKind, ToConfig, TreeBoosterParameters, TreeMethod,
    TreeUpdaterName, Verbosity,
};

/// Walk every variant of one `str_enum!` parameter through the full round trip.
///
/// The macro takes the type and the parameter's upstream name; both are
/// available on the type itself, which is the point — a spelling can only be
/// declared once.
macro_rules! check_enum {
    ($ty:ty) => {{
        let param = <$ty>::parameter_name();
        let variants = <$ty>::ALL;
        assert!(!variants.is_empty(), "{param} declares no values");

        let mut spellings = BTreeSet::new();
        for &variant in variants {
            let text = variant.as_str();

            assert!(!text.is_empty(), "{param}: a value has an empty spelling");
            assert!(
                !text.chars().any(char::is_whitespace),
                "{param}: `{text}` contains whitespace, which XGBoost's parser would split"
            );
            assert!(
                spellings.insert(text),
                "{param}: `{text}` is used by two values"
            );

            // Display agrees with the wire spelling.
            assert_eq!(variant.to_string(), text, "{param}: Display differs from as_str");

            // Every spelling parses back to the value that produced it.
            assert_eq!(
                <$ty>::from_str(text).unwrap(),
                variant,
                "{param}: `{text}` does not parse back"
            );

            // serde uses the same spelling in both directions.
            let json = serde_json::to_string(&variant).unwrap();
            assert_eq!(json, format!("\"{text}\""), "{param}: serde spelling differs");
            let back: $ty = serde_json::from_str(&json).unwrap();
            assert_eq!(back, variant, "{param}: serde round trip lost the value");
        }

        // Nothing outside the list is accepted, and the error names the
        // parameter and lists what is.
        for bad in ["", " ", "nope", "AUTO", "0nonsense"] {
            if spellings.contains(bad) {
                continue;
            }
            let err = <$ty>::from_str(bad).unwrap_err().to_string();
            assert!(err.contains(param), "{param}: error for `{bad}` omits the name: {err}");
        }

        spellings
    }};
}

#[test]
fn every_tree_booster_string_parameter_round_trips() {
    check_enum!(TreeMethod);
    check_enum!(GrowPolicy);
    check_enum!(SamplingMethod);
    check_enum!(ProcessType);
    check_enum!(MultiStrategy);
    check_enum!(TreeUpdaterName);
    check_enum!(DefaultDirection);
    check_enum!(MonotoneConstraint);
    check_enum!(DartSampleType);
    check_enum!(DartNormalizeType);
}

#[test]
fn every_general_and_learning_string_parameter_round_trips() {
    check_enum!(Verbosity);
    check_enum!(AftDistribution);
    check_enum!(LambdaRankPairMethod);
}

#[test]
fn every_linear_booster_string_parameter_round_trips() {
    check_enum!(FeatureSelector);
    check_enum!(LinearUpdater);
}

/// The spellings are the ones the upstream docs use, so a config emitted here
/// is a config a real XGBoost accepts. Pinned literally: a typo here would
/// round-trip perfectly and still be wrong.
#[test]
fn the_spellings_are_the_upstream_ones() {
    assert_eq!(
        TreeMethod::ALL.iter().map(|v| v.as_str()).collect::<Vec<_>>(),
        ["auto", "exact", "approx", "hist"]
    );
    assert_eq!(
        GrowPolicy::ALL.iter().map(|v| v.as_str()).collect::<Vec<_>>(),
        ["depthwise", "lossguide"]
    );
    assert_eq!(
        SamplingMethod::ALL.iter().map(|v| v.as_str()).collect::<Vec<_>>(),
        ["uniform", "gradient_based"]
    );
    assert_eq!(
        ProcessType::ALL.iter().map(|v| v.as_str()).collect::<Vec<_>>(),
        ["default", "update"]
    );
    assert_eq!(
        MultiStrategy::ALL.iter().map(|v| v.as_str()).collect::<Vec<_>>(),
        ["one_output_per_tree", "multi_output_tree"]
    );
    assert_eq!(
        TreeUpdaterName::ALL.iter().map(|v| v.as_str()).collect::<Vec<_>>(),
        [
            "grow_colmaker",
            "grow_histmaker",
            "grow_quantile_histmaker",
            "grow_quantile_histmaker_sycl",
            "grow_gpu_hist",
            "grow_gpu_approx",
            "prune",
            "refresh",
        ]
    );
    assert_eq!(
        DefaultDirection::ALL.iter().map(|v| v.as_str()).collect::<Vec<_>>(),
        ["learn", "left", "right"]
    );
    assert_eq!(
        MonotoneConstraint::ALL.iter().map(|v| v.as_str()).collect::<Vec<_>>(),
        ["-1", "0", "1"]
    );
    assert_eq!(
        DartSampleType::ALL.iter().map(|v| v.as_str()).collect::<Vec<_>>(),
        ["uniform", "weighted"]
    );
    assert_eq!(
        DartNormalizeType::ALL.iter().map(|v| v.as_str()).collect::<Vec<_>>(),
        ["tree", "forest"]
    );
    assert_eq!(
        Verbosity::ALL.iter().map(|v| v.as_str()).collect::<Vec<_>>(),
        ["0", "1", "2", "3"]
    );
    assert_eq!(
        AftDistribution::ALL.iter().map(|v| v.as_str()).collect::<Vec<_>>(),
        ["normal", "logistic", "extreme"]
    );
    assert_eq!(
        LambdaRankPairMethod::ALL.iter().map(|v| v.as_str()).collect::<Vec<_>>(),
        ["mean", "topk"]
    );
    assert_eq!(
        FeatureSelector::ALL.iter().map(|v| v.as_str()).collect::<Vec<_>>(),
        ["cyclic", "shuffle", "random", "greedy", "thrifty"]
    );
    assert_eq!(
        LinearUpdater::ALL.iter().map(|v| v.as_str()).collect::<Vec<_>>(),
        ["shotgun", "coord_descent"]
    );
}

/// The parameter names are the upstream ones too, and each enum knows its own.
#[test]
fn each_enum_reports_the_parameter_it_configures() {
    assert_eq!(TreeMethod::parameter_name(), "tree_method");
    assert_eq!(GrowPolicy::parameter_name(), "grow_policy");
    assert_eq!(SamplingMethod::parameter_name(), "sampling_method");
    assert_eq!(ProcessType::parameter_name(), "process_type");
    assert_eq!(MultiStrategy::parameter_name(), "multi_strategy");
    assert_eq!(TreeUpdaterName::parameter_name(), "updater");
    assert_eq!(DefaultDirection::parameter_name(), "default_direction");
    assert_eq!(MonotoneConstraint::parameter_name(), "monotone_constraints");
    assert_eq!(DartSampleType::parameter_name(), "sample_type");
    assert_eq!(DartNormalizeType::parameter_name(), "normalize_type");
    assert_eq!(Verbosity::parameter_name(), "verbosity");
    assert_eq!(AftDistribution::parameter_name(), "aft_loss_distribution");
    assert_eq!(LambdaRankPairMethod::parameter_name(), "lambdarank_pair_method");
    assert_eq!(FeatureSelector::parameter_name(), "feature_selector");
    assert_eq!(LinearUpdater::parameter_name(), "updater");
}

/// Every variant must survive being set on the struct it belongs to and coming
/// back out of the emitted config under its own name.
#[test]
fn every_variant_survives_config_emission() {
    for &tree_method in TreeMethod::ALL {
        // `exact` forbids a few combinations, so build the plainest possible
        // parameters around the one value under test.
        let params =
            TreeBoosterParameters::builder().tree_method(tree_method).build().unwrap();
        assert_eq!(params.to_config_map()["tree_method"], tree_method.as_str());
    }
    for &grow_policy in GrowPolicy::ALL {
        let params = TreeBoosterParameters::builder().grow_policy(grow_policy).build().unwrap();
        assert_eq!(params.to_config_map()["grow_policy"], grow_policy.as_str());
    }
    for &method in SamplingMethod::ALL {
        let params = TreeBoosterParameters::builder().sampling_method(method).build().unwrap();
        assert_eq!(params.to_config_map()["sampling_method"], method.as_str());
    }
    for &strategy in MultiStrategy::ALL {
        let params = TreeBoosterParameters::builder()
            .multi_strategy(strategy)
            .tree_method(TreeMethod::Hist)
            .build()
            .unwrap();
        assert_eq!(params.to_config_map()["multi_strategy"], strategy.as_str());
    }
    for &direction in DefaultDirection::ALL {
        let params =
            TreeBoosterParameters::builder().default_direction(direction).build().unwrap();
        assert_eq!(params.to_config_map()["default_direction"], direction.as_str());
    }
    for &updater in TreeUpdaterName::ALL {
        // A tree-modifying updater cannot lead the pipeline under the default
        // process type, so pair it with a grower.
        let sequence = if updater.can_modify_tree() {
            vec![TreeUpdaterName::GrowQuantileHistMaker, updater]
        } else {
            vec![updater]
        };
        let params = TreeBoosterParameters::builder().updater(sequence).build().unwrap();
        assert!(
            params.to_config_map()["updater"].split(',').any(|u| u == updater.as_str()),
            "{updater} missing from the emitted updater sequence"
        );
    }
    for &process_type in ProcessType::ALL {
        let builder = TreeBoosterParameters::builder().process_type(process_type);
        // `update` requires an explicit pipeline of tree-modifying updaters.
        let params = match process_type {
            ProcessType::Update => {
                builder.updater([TreeUpdaterName::Refresh, TreeUpdaterName::Prune])
            }
            ProcessType::Default => builder,
        }
        .build()
        .unwrap();
        assert_eq!(params.to_config_map()["process_type"], process_type.as_str());
    }

    // One constraint per feature, emitted in XGBoost's parenthesised form.
    let all: Vec<MonotoneConstraint> = MonotoneConstraint::ALL.to_vec();
    let params = TreeBoosterParameters::builder().monotone_constraints(all).build().unwrap();
    assert_eq!(params.to_config_map()["monotone_constraints"], "(-1,0,1)");

    for &sample_type in DartSampleType::ALL {
        let params = DartParameters::builder().sample_type(sample_type).build().unwrap();
        assert_eq!(params.to_config_map()["sample_type"], sample_type.as_str());
    }
    for &normalize_type in DartNormalizeType::ALL {
        let params = DartParameters::builder().normalize_type(normalize_type).build().unwrap();
        assert_eq!(params.to_config_map()["normalize_type"], normalize_type.as_str());
    }
    for &verbosity in Verbosity::ALL {
        let params = GeneralParameters::builder().verbosity(verbosity).build().unwrap();
        assert_eq!(params.to_config_map()["verbosity"], verbosity.as_str());
    }
    for &selector in FeatureSelector::ALL {
        // `shotgun` only supports the two stateless selectors, so pair anything
        // else with the coordinate-descent updater it requires.
        let updater = match selector {
            FeatureSelector::Cyclic | FeatureSelector::Shuffle => LinearUpdater::Shotgun,
            _ => LinearUpdater::CoordDescent,
        };
        let params = LinearBoosterParameters::builder()
            .feature_selector(selector)
            .updater(updater)
            .build()
            .unwrap();
        assert_eq!(params.to_config_map()["feature_selector"], selector.as_str());
    }
    for &updater in LinearUpdater::ALL {
        let params = LinearBoosterParameters::builder().updater(updater).build().unwrap();
        assert_eq!(params.to_config_map()["updater"], updater.as_str());
    }
    for &pair_method in LambdaRankPairMethod::ALL {
        let objective = Objective::RankNdcg(LambdaRankParameters {
            pair_method,
            ..Default::default()
        });
        let params =
            LearningTaskParameters::builder().objective(objective).build().unwrap();
        assert_eq!(params.to_config_map()["lambdarank_pair_method"], pair_method.as_str());
    }
    for &distribution in AftDistribution::ALL {
        let objective = Objective::SurvivalAft {
            aft_loss_distribution: distribution,
            aft_loss_distribution_scale: 1.0,
        };
        let params =
            LearningTaskParameters::builder().objective(objective).build().unwrap();
        assert_eq!(params.to_config_map()["aft_loss_distribution"], distribution.as_str());
    }
}

/// `device` is a string parameter with structure rather than a fixed list, so
/// every shape it can take is checked instead of every value.
#[test]
fn every_device_spelling_round_trips() {
    let cases: Vec<(Device, &str)> = vec![
        (Device::Cpu, "cpu"),
        (Device::Cuda(None), "cuda"),
        (Device::cuda(0), "cuda:0"),
        (Device::cuda(7), "cuda:7"),
        (Device::Sycl(SyclKind::Default, None), "sycl"),
        (Device::Sycl(SyclKind::Cpu, None), "sycl:cpu"),
        (Device::Sycl(SyclKind::Gpu, None), "sycl:gpu"),
        (Device::Sycl(SyclKind::Default, Some(1)), "sycl:1"),
        (Device::Sycl(SyclKind::Gpu, Some(2)), "sycl:gpu:2"),
        (Device::Sycl(SyclKind::Cpu, Some(-1)), "sycl:cpu:-1"),
    ];
    for (device, text) in &cases {
        assert_eq!(device.to_string(), *text);
        assert_eq!(Device::from_str(text).unwrap(), *device);
        let json = serde_json::to_string(device).unwrap();
        assert_eq!(json, format!("\"{text}\""));
        assert_eq!(serde_json::from_str::<Device>(&json).unwrap(), *device);

        let params = GeneralParameters::builder().device(*device).build().unwrap();
        assert_eq!(params.to_config_map()["device"], *text);
    }

    // `gpu` is XGBoost's legacy alias for `cuda`; it parses, and normalises so
    // that nothing downstream has to know about the second spelling.
    assert_eq!(Device::from_str("gpu").unwrap(), Device::Cuda(None));
    assert_eq!(Device::from_str("gpu:2").unwrap(), Device::cuda(2));
    assert_eq!(Device::from_str("gpu:2").unwrap().to_string(), "cuda:2");

    for bad in ["", "cuda:", "cuda:-1", "cuda:x", "sycl:tpu", "sycl:cpu:-2", "CPU", " cpu"] {
        assert!(Device::from_str(bad).is_err(), "`{bad}` should not parse as a device");
    }
}

/// Objectives are strings too, and each one's name has to survive the trip
/// through the config and back through serde.
#[test]
fn every_objective_spelling_round_trips() {
    let objectives = vec![
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
        Objective::MultiSoftmax { num_class: 3 },
        Objective::MultiSoftprob { num_class: 3 },
        Objective::RankPairwise(LambdaRankParameters::default()),
        Objective::RankNdcg(LambdaRankParameters::default()),
        Objective::RankMap(LambdaRankParameters::default()),
        Objective::BinaryLogistic,
        Objective::BinaryLogitRaw,
        Objective::BinaryHinge,
    ];

    let mut names = BTreeSet::new();
    for objective in &objectives {
        let name = objective.name();
        assert!(names.insert(name), "`{name}` is claimed by two objectives");
        assert!(!name.chars().any(char::is_whitespace), "`{name}` contains whitespace");

        let params =
            LearningTaskParameters::builder().objective(objective.clone()).build().unwrap();
        assert_eq!(params.to_config_map()["objective"], name);

        let json = serde_json::to_string(objective).unwrap();
        let back: Objective = serde_json::from_str(&json).unwrap();
        assert_eq!(back, *objective, "`{name}` did not survive serde");
    }
    assert!(names.contains("reg:squarederror"));
    assert!(names.contains("multi:softprob"));
    assert!(names.contains("rank:ndcg"));
}

/// Metrics are strings, and several are parameterised — `error@0.6`,
/// `ndcg@5-`, `tweedie-nloglik@1.5`. Every shape is covered, including all
/// four combinations of the optional `@n` and trailing `-` on the ranking
/// metrics, which is where a spelling is easiest to get wrong.
#[test]
fn every_eval_metric_spelling_round_trips() {
    let mut metrics: Vec<(EvalMetric, String)> = vec![
        (EvalMetric::Rmse, "rmse".into()),
        (EvalMetric::Rmsle, "rmsle".into()),
        (EvalMetric::Mae, "mae".into()),
        (EvalMetric::Mape, "mape".into()),
        (EvalMetric::Mphe, "mphe".into()),
        (EvalMetric::Logloss, "logloss".into()),
        (EvalMetric::Error, "error".into()),
        (EvalMetric::ErrorAt(0.6), "error@0.6".into()),
        (EvalMetric::MError, "merror".into()),
        (EvalMetric::MLogloss, "mlogloss".into()),
        (EvalMetric::Auc, "auc".into()),
        (EvalMetric::Aucpr, "aucpr".into()),
        (EvalMetric::Pre(None), "pre".into()),
        (EvalMetric::Pre(Some(4)), "pre@4".into()),
        (EvalMetric::PoissonNegLogLik, "poisson-nloglik".into()),
        (EvalMetric::GammaNegLogLik, "gamma-nloglik".into()),
        (EvalMetric::CoxNegLogLik, "cox-nloglik".into()),
        (EvalMetric::GammaDeviance, "gamma-deviance".into()),
        (EvalMetric::TweedieNegLogLik(1.5), "tweedie-nloglik@1.5".into()),
        (EvalMetric::AftNegLogLik, "aft-nloglik".into()),
        (EvalMetric::IntervalRegressionAccuracy, "interval-regression-accuracy".into()),
        (EvalMetric::Quantile, "quantile".into()),
        (EvalMetric::Expectile, "expectile".into()),
        (EvalMetric::Ams(0.15), "ams@0.15".into()),
    ];

    // The ranking metrics: every combination of truncation and the `-` form.
    for top_n in [None, Some(5u32)] {
        for minus in [false, true] {
            let suffix = format!(
                "{}{}",
                top_n.map(|n| format!("@{n}")).unwrap_or_default(),
                if minus { "-" } else { "" }
            );
            metrics.push((EvalMetric::Ndcg { top_n, minus }, format!("ndcg{suffix}")));
            metrics.push((EvalMetric::Map { top_n, minus }, format!("map{suffix}")));
        }
    }

    for (metric, text) in &metrics {
        assert_eq!(metric.to_string(), *text, "Display spelling");
        assert_eq!(EvalMetric::from_str(text).unwrap(), *metric, "`{text}` did not parse back");
        metric.validate().unwrap_or_else(|e| panic!("`{text}` failed validation: {e}"));
        let json = serde_json::to_string(metric).unwrap();
        assert_eq!(json, format!("\"{text}\""), "serde spelling for `{text}`");
        assert_eq!(serde_json::from_str::<EvalMetric>(&json).unwrap(), *metric);
    }

    // `Custom` exists for plugin metrics: it never comes out of `FromStr`, but
    // it does survive serde, which is what makes a saved config reloadable.
    let custom = EvalMetric::Custom("my-metric".into());
    assert_eq!(custom.to_string(), "my-metric");
    let json = serde_json::to_string(&custom).unwrap();
    assert_eq!(serde_json::from_str::<EvalMetric>(&json).unwrap(), custom);
    assert!(
        EvalMetric::from_str("my-metric").is_err(),
        "an unknown name is a typo, not a plugin"
    );

    // Several metrics in one fit are emitted as one comma-joined entry, which
    // is how XGBoost's repeated `eval_metric` argument is spelled in a map.
    let params = LearningTaskParameters::builder()
        .eval_metric([EvalMetric::Rmse, EvalMetric::Mae, EvalMetric::Ndcg {
            top_n: Some(3),
            minus: false,
        }])
        .build()
        .unwrap();
    assert_eq!(params.to_config_map()["eval_metric"], "rmse,mae,ndcg@3");

    for bad in ["", "nope", "ndcg@", "ndcg@x", "error@", "tweedie-nloglik@", "rmse "] {
        assert!(EvalMetric::from_str(bad).is_err(), "`{bad}` should not parse as a metric");
    }

    // Out-of-range arguments are rejected even though the spelling parses.
    for bad in [
        EvalMetric::ErrorAt(1.5),
        EvalMetric::TweedieNegLogLik(2.0),
        EvalMetric::Ndcg { top_n: Some(0), minus: false },
        EvalMetric::Custom("has space".into()),
    ] {
        assert!(bad.validate().is_err(), "{bad} should fail validation");
    }
}

/// Nothing in a config may carry whitespace: XGBoost's parser splits on it, so
/// a value with a space in it silently becomes a different value.
#[test]
fn no_emitted_value_contains_whitespace() {
    let configs = vec![
        TreeBoosterParameters::builder()
            .monotone_constraints(MonotoneConstraint::ALL.to_vec())
            .interaction_constraints(vec![vec![0, 1], vec![2]])
            .updater([TreeUpdaterName::GrowQuantileHistMaker, TreeUpdaterName::Prune])
            .build()
            .unwrap()
            .to_config(),
        DartParameters::builder().rate_drop(0.5).build().unwrap().to_config(),
        LinearBoosterParameters::builder().build().unwrap().to_config(),
        GeneralParameters::builder().device(Device::cuda(3)).build().unwrap().to_config(),
        LearningTaskParameters::builder()
            .objective(Objective::RegQuantileError { quantile_alpha: vec![0.1, 0.5, 0.9] })
            .eval_metric([EvalMetric::Rmse, EvalMetric::Ndcg { top_n: Some(3), minus: false }])
            .build()
            .unwrap()
            .to_config(),
    ];
    for config in configs {
        for (name, value) in config {
            assert!(
                !name.chars().any(char::is_whitespace),
                "parameter name `{name}` contains whitespace"
            );
            assert!(
                !value.chars().any(char::is_whitespace),
                "`{name}` = `{value}` contains whitespace"
            );
        }
    }
}
