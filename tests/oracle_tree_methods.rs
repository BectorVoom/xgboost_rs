//! Oracle for the tree methods, boosters and updaters beyond CPU `hist`.
//!
//! `tests/oracle_train.rs` pins the `hist` path against committed fixtures.
//! This file does the same for everything added around it — `exact`, `approx`,
//! `gblinear`, continued training and `process_type=update` — against fixtures
//! from the same pinned XGBoost 3.0.5.
//!
//! The bar is the one `SPEC §1.5` sets for `hist`: **tree structure identical**
//! (every child link, split index and default-direction flag, and the deleted
//! node slots pruning leaves behind), predictions and metrics within `1e-5`.
//!
//! What is deliberately *not* pinned here is anything that consumes the random
//! engine — `subsample` and the `colsample_*` family. Their draw sequence goes
//! through `std::shuffle` and `std::uniform_int_distribution`, whose output is
//! defined by the C++ standard library the reference binary was linked
//! against rather than by XGBoost, so no single answer is "correct" across
//! platforms. `tests/fit_parameters.rs` covers those by behaviour instead.

mod common;

use common::{f32_array, load_data, load_json};
use serde_json::Value;
use xgboost_rs::parameters::{
    BoosterParameters, BoosterType, EvalMetric, GeneralParameters, LearningTaskParameters,
    LinearBoosterParameters, LinearUpdater, Objective, ProcessType, TrainingParameters,
    TreeBoosterParameters, TreeMethod, TreeUpdaterName, VerboseEval, Verbosity,
};
use xgboost_rs::{Booster, DMatrix, api};

/// Cases whose fixture holds a tree ensemble, as `(case name, dataset name)`.
const TREE_CASES: &[&str] = &[
    "exact_dense_small",
    "exact_missing",
    "exact_gamma",
    "exact_sparse",
    "exact_logistic",
    "exact_weighted",
    "approx_dense_small",
    "approx_missing",
    "approx_sparse",
    "approx_logistic",
    "approx_weighted",
    "update_base",
    "update_continued",
    "update_refresh",
    "update_prune",
    "update_refresh_prune",
];

/// Cases whose fixture holds a `gblinear` weight vector.
const LINEAR_CASES: &[&str] = &["linear_shotgun", "linear_coord_descent"];

/// Rebuild the training configuration a fixture was generated with.
///
/// Only the parameters the fixtures actually set are read; an unrecognised one
/// would silently train something else, so anything new must be added here
/// alongside the generator.
fn params_from(fixture: &Value) -> TrainingParameters {
    let p = &fixture["params"];
    let num = |k: &str| p.get(k).and_then(Value::as_f64);
    let text = |k: &str| p.get(k).and_then(Value::as_str);

    let objective = match text("objective").unwrap_or("reg:squarederror") {
        "binary:logistic" => Objective::BinaryLogistic,
        "reg:squarederror" => Objective::RegSquaredError,
        other => panic!("fixture uses an objective this test does not build: {other}"),
    };
    let metric: EvalMetric = text("eval_metric").unwrap_or("rmse").parse().unwrap();
    let learning =
        LearningTaskParameters { objective, eval_metric: vec![metric], ..Default::default() };

    let booster = if text("booster") == Some("gblinear") {
        let updater = match text("updater").unwrap_or("shotgun") {
            "coord_descent" => LinearUpdater::CoordDescent,
            _ => LinearUpdater::Shotgun,
        };
        BoosterType::Gblinear(LinearBoosterParameters {
            eta: num("eta").unwrap_or(0.5) as f32,
            lambda: num("lambda").unwrap_or(0.0) as f32,
            alpha: num("alpha").unwrap_or(0.0) as f32,
            updater,
            ..Default::default()
        })
    } else {
        // An explicit `updater` wins over `tree_method`, exactly as upstream.
        let updater = text("updater").map(|u| {
            u.split(',')
                .map(|name| name.parse::<TreeUpdaterName>().expect("fixture updater"))
                .collect::<Vec<_>>()
        });
        let tree_method = match text("tree_method").unwrap_or("hist") {
            "exact" => TreeMethod::Exact,
            "approx" => TreeMethod::Approx,
            _ => TreeMethod::Hist,
        };
        let process_type = match text("process_type") {
            Some("update") => ProcessType::Update,
            _ => ProcessType::Default,
        };
        BoosterType::Gbtree(TreeBoosterParameters {
            eta: num("eta").unwrap_or(0.3) as f32,
            gamma: num("gamma").unwrap_or(0.0) as f32,
            max_depth: num("max_depth").unwrap_or(6.0) as u32,
            min_child_weight: num("min_child_weight").unwrap_or(1.0) as f32,
            lambda: num("lambda").unwrap_or(1.0) as f32,
            alpha: num("alpha").unwrap_or(0.0) as f32,
            max_bin: num("max_bin").unwrap_or(256.0) as u32,
            tree_method,
            updater,
            process_type,
            ..Default::default()
        })
    };

    TrainingParameters {
        booster: BoosterParameters {
            booster,
            general: GeneralParameters {
                nthread: 1,
                verbosity: Verbosity::Silent,
                ..Default::default()
            },
            learning,
        },
        num_boost_round: fixture["num_round"].as_u64().unwrap() as u32,
        verbose_eval: VerboseEval::Silent,
        ..Default::default()
    }
}

/// Train a case, first training its base model when the fixture names one.
fn train_case(case: &str) -> (Booster, api::EvalHistory, Value, DMatrix) {
    let fixture = load_json(case);
    let dmat = load_data(fixture["data"].as_str().unwrap());
    let params = params_from(&fixture);

    let base = fixture["base"].as_str().map(|name| {
        let base_fixture = load_json(name);
        let base_params = params_from(&base_fixture);
        api::train(&base_params, &dmat, &[])
            .unwrap_or_else(|e| panic!("{case}: base model {name} failed: {e}"))
            .0
    });

    let (booster, history) = api::train_from(&params, &dmat, &[(&dmat, "train")], base.as_ref())
        .unwrap_or_else(|e| panic!("{case}: training failed: {e}"));
    (booster, history, fixture, dmat)
}

fn trees_of(booster: &Booster) -> Vec<Value> {
    let model: Value = serde_json::from_str(&booster.save_model()).unwrap();
    model["learner"]["gradient_booster"]["model"]["trees"].as_array().unwrap().clone()
}

#[test]
fn tree_structure_matches_xgboost_exactly() {
    for &case in TREE_CASES {
        let (booster, _, fixture, _) = train_case(case);
        let got = trees_of(&booster);
        let want = fixture["trees"].as_array().unwrap();
        assert_eq!(got.len(), want.len(), "{case}: tree count differs");

        for (t, (g, w)) in got.iter().zip(want).enumerate() {
            for key in
                ["left_children", "right_children", "parents", "split_indices", "default_left"]
            {
                assert_eq!(g[key], w[key], "{case}: tree {t} `{key}` differs");
            }
            // Pruning retires node slots rather than renumbering around them,
            // so the count of retired slots is part of the structure.
            assert_eq!(
                g["tree_param"]["num_deleted"], w["tree_param"]["num_deleted"],
                "{case}: tree {t} deleted-node count differs"
            );

            let got_cond = f32_array(&g["split_conditions"]);
            let want_cond = f32_array(&w["split_conditions"]);
            for (n, (a, b)) in got_cond.iter().zip(&want_cond).enumerate() {
                assert!(
                    (a - b).abs() <= 1e-5 * b.abs().max(1.0),
                    "{case}: tree {t} node {n} split/leaf value {a} != {b}"
                );
            }
        }
    }
}

#[test]
fn predictions_match_xgboost_within_tolerance() {
    for &case in TREE_CASES.iter().chain(LINEAR_CASES) {
        let (booster, _, fixture, dmat) = train_case(case);
        for (kind, got) in
            [("prediction", booster.predict(&dmat)), ("margin", booster.predict_margin(&dmat))]
        {
            let want = f32_array(&fixture[if kind == "margin" { "margins" } else { "predictions" }]);
            assert_eq!(got.len(), want.len(), "{case}: {kind} count differs");
            for (i, (a, b)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (a - b).abs() <= 1e-5 * b.abs().max(1.0),
                    "{case}: {kind} {i}: {a} != {b}"
                );
            }
        }
    }
}

#[test]
fn the_metric_history_matches_xgboost_round_by_round() {
    for &case in TREE_CASES.iter().chain(LINEAR_CASES) {
        let (_, history, fixture, _) = train_case(case);
        let want: Vec<f64> =
            fixture["metric_history"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        assert_eq!(history.len(), want.len(), "{case}: round count differs");
        for (i, (round, w)) in history.iter().zip(&want).enumerate() {
            let got = round[0].1;
            assert!(
                (got - w).abs() <= 1e-5 * w.abs().max(1.0),
                "{case}: round {i} metric {got} != {w}"
            );
        }
    }
}

#[test]
fn base_score_matches_xgboost() {
    for &case in TREE_CASES.iter().chain(LINEAR_CASES) {
        let (booster, _, fixture, _) = train_case(case);
        let want = fixture["base_score"].as_f64().unwrap() as f32;
        let got = booster.base_score();
        assert!(
            (got - want).abs() <= 1e-6 * want.abs().max(1.0),
            "{case}: base_score {got} != {want}"
        );
    }
}

#[test]
fn linear_weights_match_xgboost() {
    for &case in LINEAR_CASES {
        let (booster, _, fixture, _) = train_case(case);
        let model: Value = serde_json::from_str(&booster.save_model()).unwrap();
        let got = f32_array(&model["learner"]["gradient_booster"]["model"]["weights"]);
        let want = f32_array(&fixture["weights"]);
        assert_eq!(got.len(), want.len(), "{case}: weight count differs");
        for (i, (a, b)) in got.iter().zip(&want).enumerate() {
            assert!((a - b).abs() <= 1e-4 * b.abs().max(1.0), "{case}: weight {i}: {a} != {b}");
        }
    }
}

/// `process_type=update` must leave the ensemble the same size it found — it
/// rewrites rounds rather than adding them.
#[test]
fn an_update_round_rewrites_rather_than_grows() {
    let (base, _, _, _) = train_case("update_base");
    for case in ["update_refresh", "update_prune", "update_refresh_prune"] {
        let (updated, _, _, _) = train_case(case);
        assert_eq!(updated.num_trees(), base.num_trees(), "{case} changed the tree count");
    }
    let (continued, _, _, _) = train_case("update_continued");
    assert!(continued.num_trees() > base.num_trees(), "continued training must add trees");
}
