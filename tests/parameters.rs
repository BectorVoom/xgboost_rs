//! End-to-end tests for the fit parameter surface: the public API as a caller
//! sees it, serde round-trips, and the XGBoost-compatible config emission that
//! makes the oracle harness possible.

use std::collections::BTreeMap;

use xgboost_rs::parameters::{
    BoosterParameters, BoosterType, DartParameters, DartSampleType, Device, EvalMetric,
    GeneralParameters, GrowPolicy, LearningTaskParameters, LinearBoosterParameters, LinearUpdater,
    MonotoneConstraint, Objective, SamplingMethod, ToConfig, TrainingParameters,
    TreeBoosterParameters, TreeMethod, TreeUpdaterName, VerboseEval, Verbosity,
};

/// A non-trivial GPU configuration exercising most of the surface.
fn gpu_params() -> BoosterParameters {
    BoosterParameters::builder()
        .general(
            GeneralParameters::builder()
                .device(Device::cuda(0))
                .nthread(4)
                .verbosity(Verbosity::Info)
                .validate_parameters(true)
                .build()
                .unwrap(),
        )
        .tree(
            TreeBoosterParameters::builder()
                .tree_method(TreeMethod::Hist)
                .eta(0.05)
                .max_depth(10)
                .max_bin(512)
                .subsample(0.8)
                .sampling_method(SamplingMethod::GradientBased)
                .colsample_bytree(0.9)
                .colsample_bylevel(0.8)
                .colsample_bynode(0.7)
                .grow_policy(GrowPolicy::LossGuide)
                .max_leaves(64)
                .lambda(2.0)
                .alpha(0.5)
                .gamma(0.1)
                .min_child_weight(5.0)
                .monotone_constraints(vec![
                    MonotoneConstraint::Increasing,
                    MonotoneConstraint::Unconstrained,
                ])
                .interaction_constraints(vec![vec![0, 1]])
                .max_cached_hist_node(2048)
                .build()
                .unwrap(),
        )
        .learning(
            LearningTaskParameters::builder()
                .objective(Objective::MultiSoftprob { num_class: 3 })
                .eval_metric([EvalMetric::MLogloss, EvalMetric::MError])
                .base_score(0.5)
                .seed(7)
                .build()
                .unwrap(),
        )
        .build()
        .unwrap()
}

#[test]
fn cpu_and_gpu_configs_differ_only_where_they_should() {
    let cpu = BoosterParameters::default();
    let gpu = BoosterParameters::builder()
        .general(GeneralParameters::builder().device(Device::cuda(0)).build().unwrap())
        .build()
        .unwrap();

    let cpu_config = cpu.to_config_map();
    let gpu_config = gpu.to_config_map();

    let differing: Vec<_> = cpu_config
        .iter()
        .filter(|(key, value)| gpu_config.get(*key) != Some(*value))
        .map(|(key, _)| key.as_str())
        .collect();
    assert_eq!(differing, vec!["device"]);

    // The device does change what actually runs, though.
    assert_eq!(cpu.resolved_updaters().unwrap(), Some(vec![TreeUpdaterName::GrowQuantileHistMaker]));
    assert_eq!(gpu.resolved_updaters().unwrap(), Some(vec![TreeUpdaterName::GrowGpuHist]));
}

#[test]
fn full_gpu_config_uses_upstream_names_and_values() {
    let config = gpu_params().to_config_map();
    let expected: BTreeMap<&str, &str> = BTreeMap::from([
        ("device", "cuda:0"),
        ("nthread", "4"),
        ("verbosity", "2"),
        ("validate_parameters", "1"),
        ("booster", "gbtree"),
        ("tree_method", "hist"),
        ("eta", "0.05"),
        ("max_depth", "10"),
        ("max_bin", "512"),
        ("subsample", "0.8"),
        ("sampling_method", "gradient_based"),
        ("colsample_bytree", "0.9"),
        ("colsample_bylevel", "0.8"),
        ("colsample_bynode", "0.7"),
        ("grow_policy", "lossguide"),
        ("max_leaves", "64"),
        ("lambda", "2"),
        ("alpha", "0.5"),
        ("gamma", "0.1"),
        ("min_child_weight", "5"),
        ("monotone_constraints", "(1,0)"),
        ("interaction_constraints", "[[0,1]]"),
        ("max_cached_hist_node", "2048"),
        ("objective", "multi:softprob"),
        ("num_class", "3"),
        ("eval_metric", "mlogloss,merror"),
        ("base_score", "0.5"),
        ("seed", "7"),
    ]);
    for (key, value) in expected {
        assert_eq!(config.get(key).map(String::as_str), Some(value), "parameter `{key}`");
    }
}

#[test]
fn every_config_value_is_whitespace_free() {
    // `Learner::SetParam` rejects configuration values containing whitespace.
    for (key, value) in gpu_params().to_config() {
        assert!(!key.chars().any(char::is_whitespace), "key `{key}`");
        assert!(!value.chars().any(char::is_whitespace), "`{key}` = `{value}`");
        assert!(!value.is_empty(), "`{key}` is empty");
    }
}

#[test]
fn json_config_is_a_flat_string_map() {
    let json = gpu_params().to_config_json();
    let parsed: BTreeMap<String, String> = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed["device"], "cuda:0");
    assert_eq!(parsed, gpu_params().to_config_map());
}

#[test]
fn parameters_round_trip_through_serde() {
    for params in [
        BoosterParameters::default(),
        gpu_params(),
        BoosterParameters::builder()
            .dart(DartParameters::builder().sample_type(DartSampleType::Weighted).rate_drop(0.3).build().unwrap())
            .build()
            .unwrap(),
        BoosterParameters::builder()
            .linear(
                LinearBoosterParameters::builder()
                    .updater(LinearUpdater::CoordDescent)
                    .build()
                    .unwrap(),
            )
            .build()
            .unwrap(),
    ] {
        let json = serde_json::to_string(&params).unwrap();
        let restored: BoosterParameters = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, params);
        restored.validate().unwrap();
    }
}

#[test]
fn objectives_and_metrics_round_trip_through_serde() {
    let objectives = [
        Objective::RegSquaredError,
        Objective::RegTweedie { tweedie_variance_power: 1.3 },
        Objective::RegQuantileError { quantile_alpha: vec![0.25, 0.75] },
        Objective::MultiSoftmax { num_class: 4 },
        Objective::RankNdcg(Default::default()),
    ];
    for objective in objectives {
        let json = serde_json::to_string(&objective).unwrap();
        assert_eq!(serde_json::from_str::<Objective>(&json).unwrap(), objective);
    }

    let metrics =
        [EvalMetric::Rmse, EvalMetric::Ndcg { top_n: Some(5), minus: true }, EvalMetric::Ams(0.15)];
    for metric in metrics {
        let json = serde_json::to_string(&metric).unwrap();
        assert_eq!(serde_json::from_str::<EvalMetric>(&json).unwrap(), metric);
    }

    // A plugin metric survives the round trip even though `FromStr` rejects it.
    let custom = EvalMetric::Custom("my-plugin-metric".to_owned());
    let json = serde_json::to_string(&custom).unwrap();
    assert_eq!(serde_json::from_str::<EvalMetric>(&json).unwrap(), custom);
}

#[test]
fn unknown_json_fields_are_rejected() {
    let err = serde_json::from_str::<BoosterParameters>(r#"{"nonsense": 1}"#).unwrap_err();
    assert!(err.to_string().contains("nonsense"), "{err}");
}

#[test]
fn training_parameters_wrap_the_model_parameters() {
    let training = TrainingParameters::builder()
        .booster(gpu_params())
        .num_boost_round(200)
        .early_stopping_rounds(10)
        .verbose_eval(VerboseEval::Every(25))
        .maximize(false)
        .build()
        .unwrap();

    assert_eq!(training.num_boost_round, 200);
    assert!(matches!(training.booster.booster, BoosterType::Gbtree(_)));

    let json = serde_json::to_string(&training).unwrap();
    assert_eq!(serde_json::from_str::<TrainingParameters>(&json).unwrap(), training);
}

#[test]
fn validation_errors_name_the_offending_parameter() {
    let cases: Vec<(&str, xgboost_rs::Error)> = vec![
        ("eta", TreeBoosterParameters::builder().eta(-1.0).build().unwrap_err()),
        ("subsample", TreeBoosterParameters::builder().subsample(0.0).build().unwrap_err()),
        (
            "num_class",
            LearningTaskParameters::builder()
                .objective(Objective::MultiSoftmax { num_class: 0 })
                .build()
                .unwrap_err(),
        ),
        (
            "device",
            BoosterParameters::builder()
                .general(GeneralParameters::builder().device(Device::cuda(0)).build().unwrap())
                .linear(LinearBoosterParameters::default())
                .build()
                .unwrap_err(),
        ),
        ("num_boost_round", TrainingParameters::builder().num_boost_round(0).build().unwrap_err()),
    ];
    for (name, err) in cases {
        let message = err.to_string();
        assert!(message.contains(name), "expected `{name}` in `{message}`");
    }
}
