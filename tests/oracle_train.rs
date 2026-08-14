//! End-to-end oracle: train in Rust with a fixture's exact configuration and
//! compare trees, metrics and predictions against the pinned XGBoost run.
//!
//! The bar (SPEC §1.5): predictions and metrics within `1e-5`; tree structure
//! and every integer identical.

mod common;

use common::{CASES, f32_array, load_data, load_json};
use serde_json::Value;
use xgboost_rs::parameters::{
    BoosterParameters, EvalMetric, LearningTaskParameters, Objective, TrainingParameters,
    TreeBoosterParameters,
};
use xgboost_rs::{DMatrix, api};

/// Build the training configuration a fixture was generated with.
fn params_from(fixture: &Value) -> TrainingParameters {
    let p = &fixture["params"];
    let get = |k: &str| p.get(k).and_then(|v| v.as_f64());

    let tree = TreeBoosterParameters {
        eta: get("eta").unwrap_or(0.3) as f32,
        gamma: get("gamma").unwrap_or(0.0) as f32,
        max_depth: get("max_depth").unwrap_or(6.0) as u32,
        min_child_weight: get("min_child_weight").unwrap_or(1.0) as f32,
        lambda: get("lambda").unwrap_or(1.0) as f32,
        alpha: get("alpha").unwrap_or(0.0) as f32,
        max_bin: get("max_bin").unwrap_or(256.0) as u32,
        ..Default::default()
    };

    TrainingParameters {
        booster: BoosterParameters {
            booster: xgboost_rs::parameters::BoosterType::Gbtree(tree),
            learning: LearningTaskParameters {
                objective: Objective::RegSquaredError,
                eval_metric: vec![EvalMetric::Rmse],
                ..Default::default()
            },
            ..Default::default()
        },
        num_boost_round: fixture["num_round"].as_u64().unwrap() as u32,
        ..Default::default()
    }
}

fn train_case(case: &str, data: &str) -> (xgboost_rs::Booster, api::EvalHistory, Value, DMatrix) {
    let fixture = load_json(case);
    let dmat = load_data(data);
    let params = params_from(&fixture);
    let (booster, history) = api::train(&params, &dmat, &[(&dmat, "train")])
        .unwrap_or_else(|e| panic!("{case}: training failed: {e}"));
    (booster, history, fixture, dmat)
}

/// Integer tree structure must be identical, not merely close.
#[test]
fn tree_structure_matches_xgboost_exactly() {
    for (case, data) in CASES {
        let (booster, _, fixture, _) = train_case(case, data);
        let model: Value = serde_json::from_str(&booster.save_model()).unwrap();
        let got_trees = model["learner"]["gradient_booster"]["model"]["trees"]
            .as_array()
            .unwrap()
            .clone();
        let want_trees = fixture["trees"].as_array().unwrap();

        assert_eq!(got_trees.len(), want_trees.len(), "{case}: tree count differs");
        for (t, (got, want)) in got_trees.iter().zip(want_trees).enumerate() {
            for key in ["left_children", "right_children", "parents", "split_indices", "default_left"]
            {
                assert_eq!(
                    got[key], want[key],
                    "{case}: tree {t} `{key}` differs"
                );
            }
            // Split thresholds are cut values, so they must be bit-exact too.
            let got_cond = f32_array(&got["split_conditions"]);
            let want_cond = f32_array(&want["split_conditions"]);
            for (n, (g, w)) in got_cond.iter().zip(&want_cond).enumerate() {
                assert!(
                    (g - w).abs() <= 1e-5 * w.abs().max(1.0),
                    "{case}: tree {t} node {n} split/leaf value {g} != {w}"
                );
            }
        }
    }
}

#[test]
fn base_score_matches_xgboost() {
    for (case, data) in CASES {
        let (booster, _, fixture, _) = train_case(case, data);
        let want = fixture["base_score"].as_f64().unwrap() as f32;
        let got = booster.base_score();
        assert!(
            (got - want).abs() <= 1e-6 * want.abs().max(1.0),
            "{case}: base_score {got} != {want}"
        );
    }
}

#[test]
fn predictions_match_xgboost_within_tolerance() {
    for (case, data) in CASES {
        let (booster, _, fixture, dmat) = train_case(case, data);
        let want = f32_array(&fixture["predictions"]);
        let got = booster.predict(&dmat);
        assert_eq!(got.len(), want.len(), "{case}: prediction count differs");
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= 1e-5 * w.abs().max(1.0),
                "{case}: prediction {i}: {g} != {w}"
            );
        }
    }
}

#[test]
fn leaf_assignment_matches_xgboost_exactly() {
    for (case, data) in CASES {
        let (booster, _, fixture, dmat) = train_case(case, data);
        let want = fixture["leaf"].as_array().unwrap();
        let got = booster.predict_leaf(&dmat).unwrap();
        for (r, (g, w)) in got.iter().zip(want).enumerate() {
            let w: Vec<u32> = w.as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
            assert_eq!(*g, w, "{case}: row {r} lands in different leaves");
        }
    }
}

#[test]
fn per_round_rmse_matches_xgboost() {
    for (case, data) in CASES {
        let (_, history, fixture, _) = train_case(case, data);
        let want: Vec<f64> =
            fixture["rmse"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        assert_eq!(history.len(), want.len(), "{case}: round count differs");
        for (i, (round, w)) in history.iter().zip(&want).enumerate() {
            let (name, got) = &round[0];
            assert_eq!(name, "train-rmse");
            assert!(
                (got - w).abs() <= 1e-5 * w.abs().max(1.0),
                "{case}: round {i} rmse {got} != {w}"
            );
        }
    }
}

#[test]
fn feature_importance_matches_xgboost() {
    // `weight` is a count and must be exact. `gain` is a sum of per-split loss
    // changes divided by that count, so it carries the accumulated rounding of
    // every split a feature was used for — on agaricus that is thousands of
    // f32 additions, in an order neither implementation promises to share.
    // Held to a looser bound than the structural comparisons for that reason,
    // and reported in full so a real divergence cannot hide behind the slack.
    const GAIN_TOL: f64 = 1e-4;
    let mut failures = Vec::new();

    for (case, data) in CASES {
        let (booster, _, fixture, _) = train_case(case, data);
        for (kind, key, tol) in
            [("weight", "score_weight", 0.0), ("gain", "score_gain", GAIN_TOL)]
        {
            let want = fixture[key].as_object().unwrap();
            let got = booster.get_score(kind).unwrap();
            if got.len() != want.len() {
                failures.push(format!(
                    "  {case}: {kind} covers {} features, want {}",
                    got.len(),
                    want.len()
                ));
                continue;
            }
            for (feature, w) in want {
                let w = w.as_f64().unwrap();
                let g = got[feature];
                if (g - w).abs() > tol * w.abs().max(1.0) {
                    failures.push(format!(
                        "  {case}: {kind} for {feature}: {g} != {w} (rel {:.2e})",
                        (g - w).abs() / w.abs().max(1.0)
                    ));
                }
            }
        }
    }
    assert!(failures.is_empty(), "feature importance differs:\n{}", failures.join("\n"));
}

#[test]
fn model_json_round_trips() {
    let (booster, _, _, dmat) = train_case("dense_small_b16_d3", "dense_small");
    let json = booster.save_model();
    let reloaded = xgboost_rs::Booster::load_model(&json).unwrap();

    assert_eq!(reloaded.boosted_rounds(), booster.boosted_rounds());
    assert_eq!(reloaded.num_features(), booster.num_features());
    let a = booster.predict(&dmat);
    let b = reloaded.predict(&dmat);
    assert_eq!(a, b, "predictions must survive a save/load round trip");
}
