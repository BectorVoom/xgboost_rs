//! The `gblinear` booster, end to end through the public API.
//!
//! `src/linear` unit-tests the solvers against gradients supplied directly.
//! This file goes through [`api::train`] instead, so it covers the things only
//! the whole stack can get wrong: the intercept the objective estimates, the
//! prediction cache the training loop carries, model IO, and the prediction
//! kinds a model with no trees has to refuse.

use xgboost_rs::parameters::{
    BoosterParameters, BoosterType, EvalMetric, FeatureSelector, GeneralParameters,
    IterationRange, LearningTaskParameters, LinearBoosterParameters, LinearUpdater, Objective,
    PredictParameters, PredictionType, TrainingParameters, VerboseEval, Verbosity,
};
use xgboost_rs::{DMatrix, FeatureType, api};

/// `y = 2 * x0 - 3 * x1 + 0.5 * x2 + 1`, which a linear model fits exactly.
fn linear_data(rows: usize) -> DMatrix {
    let mut state = 0xa5a5_5a5a_a5a5_5a5au64;
    let mut next = || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        ((state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32) / ((1u32 << 24) as f32)
    };
    let cols = 3;
    let x: Vec<f32> = (0..rows * cols).map(|_| next()).collect();
    let y: Vec<f32> = (0..rows)
        .map(|r| {
            let row = &x[r * cols..(r + 1) * cols];
            2.0 * row[0] - 3.0 * row[1] + 0.5 * row[2] + 1.0
        })
        .collect();
    let mut d = DMatrix::from_dense(&x, rows, cols, f32::NAN).unwrap();
    d.set_labels(&y).unwrap();
    d
}

fn params(linear: LinearBoosterParameters, rounds: u32) -> TrainingParameters {
    TrainingParameters {
        booster: BoosterParameters {
            booster: BoosterType::Gblinear(linear),
            general: GeneralParameters { verbosity: Verbosity::Silent, ..Default::default() },
            learning: LearningTaskParameters::default(),
        },
        num_boost_round: rounds,
        verbose_eval: VerboseEval::Silent,
        ..Default::default()
    }
}

fn rmse(preds: &[f32], d: &DMatrix) -> f32 {
    let y = &d.info().labels;
    (preds.iter().zip(y).map(|(p, t)| (p - t).powi(2)).sum::<f32>() / y.len() as f32).sqrt()
}

#[test]
fn a_linear_fit_recovers_a_linear_target() {
    let d = linear_data(500);
    let p = params(LinearBoosterParameters::builder().eta(0.5).build().unwrap(), 200);
    let (booster, history) = api::train(&p, &d, &[(&d, "train")]).unwrap();

    assert_eq!(history.len(), 200);
    assert_eq!(booster.num_trees(), 0, "a linear model has no trees");
    assert_eq!(booster.boosted_rounds(), 200);
    assert!(rmse(&booster.predict(&d), &d) < 1e-2);
    // The reported metric agrees with a fresh prediction, i.e. the training
    // loop's prediction cache tracked the model.
    let (_, reported) = history.last().unwrap()[0].clone();
    assert!((reported as f32 - rmse(&booster.predict(&d), &d)).abs() < 1e-4);
}

#[test]
fn every_updater_and_selector_pairing_trains() {
    let d = linear_data(300);
    for (updater, selector) in [
        (LinearUpdater::Shotgun, FeatureSelector::Cyclic),
        (LinearUpdater::Shotgun, FeatureSelector::Shuffle),
        (LinearUpdater::CoordDescent, FeatureSelector::Cyclic),
        (LinearUpdater::CoordDescent, FeatureSelector::Shuffle),
        (LinearUpdater::CoordDescent, FeatureSelector::Random),
        (LinearUpdater::CoordDescent, FeatureSelector::Greedy),
        (LinearUpdater::CoordDescent, FeatureSelector::Thrifty),
    ] {
        let linear = LinearBoosterParameters::builder()
            .updater(updater)
            .feature_selector(selector)
            .eta(0.5)
            .build()
            .unwrap();
        let (booster, _) = api::train(&params(linear, 300), &d, &[]).unwrap();
        let error = rmse(&booster.predict(&d), &d);
        assert!(error < 0.1, "{updater}/{selector}: rmse {error}");
    }
}

#[test]
fn a_fit_is_reproducible_even_with_the_randomised_selectors() {
    let d = linear_data(200);
    for selector in [FeatureSelector::Shuffle, FeatureSelector::Random] {
        let linear = LinearBoosterParameters::builder()
            .updater(LinearUpdater::CoordDescent)
            .feature_selector(selector)
            .build()
            .unwrap();
        let p = params(linear, 20);
        let a = api::train(&p, &d, &[]).unwrap().0;
        let b = api::train(&p, &d, &[]).unwrap().0;
        assert_eq!(a.save_model(), b.save_model(), "{selector} must be seed-reproducible");
    }
}

#[test]
fn the_seed_changes_the_randomised_selectors_and_nothing_else() {
    let d = linear_data(200);
    let build = |selector, seed| {
        let linear = LinearBoosterParameters::builder()
            .updater(LinearUpdater::CoordDescent)
            .feature_selector(selector)
            .build()
            .unwrap();
        let mut p = params(linear, 5);
        p.booster.learning.seed = seed;
        api::train(&p, &d, &[]).unwrap().0.save_model()
    };
    assert_ne!(build(FeatureSelector::Shuffle, 0), build(FeatureSelector::Shuffle, 1));
    assert_eq!(build(FeatureSelector::Cyclic, 0), build(FeatureSelector::Cyclic, 1));
}

#[test]
fn a_saved_linear_model_round_trips() {
    let d = linear_data(200);
    let p = params(LinearBoosterParameters::builder().eta(0.5).build().unwrap(), 50);
    let (booster, _) = api::train(&p, &d, &[]).unwrap();

    let json: serde_json::Value = serde_json::from_str(&booster.save_model()).unwrap();
    assert_eq!(json["learner"]["gradient_booster"]["name"], "gblinear");
    let weights = json["learner"]["gradient_booster"]["model"]["weights"].as_array().unwrap();
    assert_eq!(weights.len(), 4, "three features plus the intercept");
    assert_eq!(json["learner"]["gradient_booster"]["model"]["boosted_rounds"], 50);

    let loaded = xgboost_rs::Booster::load_model(&booster.save_model()).unwrap();
    let before = booster.predict(&d);
    let after = loaded.predict(&d);
    for (a, b) in before.iter().zip(&after) {
        assert!((a - b).abs() < 1e-5, "{a} vs {b}");
    }
    assert_eq!(loaded.num_features(), 3);
    assert_eq!(loaded.boosted_rounds(), 50);
}

#[test]
fn a_malformed_weight_array_is_rejected() {
    let d = linear_data(50);
    let p = params(LinearBoosterParameters::default(), 2);
    let (booster, _) = api::train(&p, &d, &[]).unwrap();
    let mut json: serde_json::Value = serde_json::from_str(&booster.save_model()).unwrap();
    json["learner"]["gradient_booster"]["model"]["weights"] = serde_json::json!([1.0, 2.0]);
    let err = match xgboost_rs::Booster::load_model(&json.to_string()) {
        Ok(_) => panic!("a weight array of the wrong length must be rejected"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("weights"), "{err}");
}

#[test]
fn feature_importance_is_the_weight_vector() {
    let d = linear_data(300);
    let p = params(LinearBoosterParameters::builder().eta(0.5).build().unwrap(), 200);
    let (booster, _) = api::train(&p, &d, &[]).unwrap();

    let score = booster.get_score("weight").unwrap();
    assert_eq!(score.len(), 3, "one entry per feature, never the intercept");
    assert!((score["f0"] - 2.0).abs() < 0.05);
    assert!((score["f1"] + 3.0).abs() < 0.05);
    assert!((score["f2"] - 0.5).abs() < 0.05);

    // `gain` counts split loss reductions, which a linear model does not have.
    let err = booster.get_score("gain").unwrap_err().to_string();
    assert!(err.contains("gblinear"), "{err}");
}

/// A named matrix keys the coefficient dictionary by name, and the names
/// survive the model file — the tree path's behaviour, on the booster that
/// builds its keys separately.
#[test]
fn feature_importance_uses_the_matrix_feature_names() {
    let mut d = linear_data(300);
    d.set_feature_names(&["slope", "drop", "nudge"]).unwrap();
    let p = params(LinearBoosterParameters::builder().eta(0.5).build().unwrap(), 200);
    let (booster, _) = api::train(&p, &d, &[]).unwrap();

    let score = booster.get_score("weight").unwrap();
    assert_eq!(score.len(), 3);
    assert!((score["slope"] - 2.0).abs() < 0.05);
    assert!((score["drop"] + 3.0).abs() < 0.05);
    assert!((score["nudge"] - 0.5).abs() < 0.05);

    let reloaded = xgboost_rs::Booster::load_model(&booster.save_model()).unwrap();
    assert_eq!(reloaded.feature_names(), ["slope", "drop", "nudge"]);
    assert_eq!(reloaded.get_score("weight").unwrap(), score);
}

#[test]
fn the_prediction_kinds_a_linear_model_cannot_answer_are_refused() {
    let d = linear_data(100);
    let p = params(LinearBoosterParameters::default(), 5);
    let (booster, _) = api::train(&p, &d, &[]).unwrap();

    let err = match booster.predict_leaf(&d) {
        Ok(_) => panic!("a linear model has no leaves to report"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("gblinear"), "{err}");

    let leaf =
        PredictParameters::builder().predict_type(PredictionType::Leaf).build().unwrap();
    let err = booster.predict_with(&leaf, &d).unwrap_err().to_string();
    assert!(err.contains("leaves"), "{err}");

    // There are no rounds to slice, so a range is an error rather than a
    // silently ignored argument.
    let ranged = PredictParameters::builder()
        .iteration_range(IterationRange::new(0, 2).unwrap())
        .build()
        .unwrap();
    let err = booster.predict_with(&ranged, &d).unwrap_err().to_string();
    assert!(err.contains("iteration_range"), "{err}");

    let full = PredictParameters::builder()
        .iteration_range(IterationRange::default())
        .build()
        .unwrap();
    booster.predict_with(&full, &d).expect("the full range is the only accepted one");
}

#[test]
fn contributions_are_exact_and_interactions_are_diagonal() {
    let d = linear_data(60);
    let p = params(LinearBoosterParameters::builder().eta(0.5).build().unwrap(), 100);
    let (booster, _) = api::train(&p, &d, &[]).unwrap();

    let contribs = booster
        .predict_with(
            &PredictParameters::builder()
                .predict_type(PredictionType::Contribution)
                .build()
                .unwrap(),
            &d,
        )
        .unwrap();
    assert_eq!(contribs.shape, vec![60, 4]);
    let margins = booster.predict_margin(&d);
    for r in 0..60 {
        let sum: f32 = contribs.values[r * 4..(r + 1) * 4].iter().sum();
        assert!((sum - margins[r]).abs() < 1e-4, "row {r}: {sum} vs {}", margins[r]);
    }

    let inter = booster
        .predict_with(
            &PredictParameters::builder()
                .predict_type(PredictionType::Interaction)
                .build()
                .unwrap(),
            &d,
        )
        .unwrap();
    assert_eq!(inter.shape, vec![60, 4, 4]);
    for r in 0..60 {
        let block = &inter.values[r * 16..(r + 1) * 16];
        for i in 0..4 {
            for j in 0..4 {
                if i == j {
                    assert!((block[i * 4 + j] - contribs.values[r * 4 + i]).abs() < 1e-5);
                } else {
                    assert_eq!(block[i * 4 + j], 0.0, "a linear model has no interactions");
                }
            }
        }
    }
}

#[test]
fn a_linear_multiclass_fit_gets_one_weight_set_per_class() {
    let mut d = linear_data(300);
    let labels: Vec<f32> = (0..300).map(|i| (i % 3) as f32).collect();
    d.set_labels(&labels).unwrap();

    let mut p = params(LinearBoosterParameters::builder().eta(0.5).build().unwrap(), 20);
    p.booster.learning.objective = Objective::MultiSoftprob { num_class: 3 };
    p.booster.learning.eval_metric = vec![EvalMetric::MLogloss];
    let (booster, history) = api::train(&p, &d, &[(&d, "train")]).unwrap();

    assert_eq!(booster.num_output_group(), 3);
    let preds = booster.predict(&d);
    assert_eq!(preds.len(), 900, "one probability per class per row");
    for r in 0..300 {
        let sum: f32 = preds[r * 3..(r + 1) * 3].iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "row {r} probabilities sum to {sum}");
    }
    assert!(history.last().unwrap()[0].1.is_finite());

    let json: serde_json::Value = serde_json::from_str(&booster.save_model()).unwrap();
    let weights = json["learner"]["gradient_booster"]["model"]["weights"].as_array().unwrap();
    assert_eq!(weights.len(), (3 + 1) * 3);
    assert_eq!(json["learner"]["learner_model_param"]["num_class"], "3");

    let loaded = xgboost_rs::Booster::load_model(&booster.save_model()).unwrap();
    assert_eq!(loaded.num_output_group(), 3);
}

#[test]
fn categorical_columns_are_rejected_rather_than_read_as_numbers() {
    let mut d = linear_data(50);
    d.set_feature_types(&[FeatureType::Categorical, FeatureType::Numerical, FeatureType::Numerical])
        .unwrap();
    let p = params(LinearBoosterParameters::default(), 1);
    let err = match api::train(&p, &d, &[]) {
        Ok(_) => panic!("a categorical column must not be fitted as a number"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("categorical"), "{err}");
}

#[test]
fn early_stopping_works_against_a_linear_fit() {
    let d = linear_data(200);
    let mut p = params(LinearBoosterParameters::builder().eta(0.5).build().unwrap(), 500);
    p.early_stopping_rounds = Some(5);
    let (booster, history) = api::train(&p, &d, &[(&d, "train")]).unwrap();
    assert!(history.len() < 500, "the fit converged and stopped early");
    assert!(booster.best_iteration().is_some());
}
