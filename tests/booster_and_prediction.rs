//! The `dart` booster's dropout parameters, and the whole prediction
//! parameter surface, tested through the public API by their effect.

use xgboost_rs::parameters::{
    BoosterParameters, BoosterType, DartNormalizeType, DartParameters, DartSampleType,
    GeneralParameters, IterationRange, LearningTaskParameters, PredictParameters, PredictionType,
    TrainingParameters, TreeBoosterParameters, VerboseEval, Verbosity,
};
use xgboost_rs::{Booster, DMatrix, api};

fn data(rows: usize, cols: usize) -> DMatrix {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32 / (1u32 << 24) as f32
    };
    let x: Vec<f32> = (0..rows * cols).map(|_| next()).collect();
    let y: Vec<f32> = (0..rows)
        .map(|r| x[r * cols..(r + 1) * cols].iter().enumerate().map(|(c, v)| v / (c + 1) as f32).sum())
        .collect();
    let mut d = DMatrix::from_dense(&x, rows, cols, f32::NAN).unwrap();
    d.set_labels(&y).unwrap();
    d
}

fn training(booster: BoosterType, rounds: u32) -> TrainingParameters {
    TrainingParameters {
        booster: BoosterParameters {
            booster,
            general: GeneralParameters { verbosity: Verbosity::Silent, ..Default::default() },
            learning: LearningTaskParameters::default(),
        },
        num_boost_round: rounds,
        verbose_eval: VerboseEval::Silent,
        ..Default::default()
    }
}

fn dart(params: DartParameters, rounds: u32) -> TrainingParameters {
    training(BoosterType::Dart(params), rounds)
}

fn tree(rounds: u32) -> TrainingParameters {
    training(BoosterType::Gbtree(TreeBoosterParameters::default()), rounds)
}

fn train(p: &TrainingParameters, d: &DMatrix) -> Booster {
    api::train(p, d, &[]).expect("training failed").0
}

fn mse(booster: &Booster, d: &DMatrix) -> f64 {
    let preds = booster.predict(d);
    preds
        .iter()
        .zip(&d.info().labels)
        .map(|(p, y)| {
            let e = (p - y) as f64;
            e * e
        })
        .sum::<f64>()
        / preds.len() as f64
}

// ------------------------------------------------------------------ DART --

/// With no dropout at all, `dart` and `gbtree` must produce the same model:
/// DART's only difference *is* the dropout.
#[test]
fn dart_without_dropout_is_gbtree() {
    let d = data(500, 4);
    let plain = train(&tree(5), &d);
    let no_drop = train(&dart(DartParameters::default(), 5), &d);
    assert_eq!(plain.predict(&d), no_drop.predict(&d));
}

/// A non-zero `rate_drop` changes the fit, and the model records it.
#[test]
fn rate_drop_changes_the_fit_and_the_tree_weights() {
    let d = data(1000, 5);
    let plain = train(&tree(10), &d);

    let dropped = train(
        &dart(DartParameters { rate_drop: 0.3, ..Default::default() }, 10),
        &d,
    );
    assert_ne!(plain.predict(&d), dropped.predict(&d), "`rate_drop` must reach the fit");
    // Every tree is still there, but they no longer count equally.
    assert_eq!(dropped.num_trees(), plain.num_trees());

    let json: serde_json::Value = serde_json::from_str(&dropped.save_model()).unwrap();
    assert_eq!(json["learner"]["gradient_booster"]["name"], "dart");
    let weights = json["learner"]["gradient_booster"]["weight_drop"].as_array().unwrap();
    assert_eq!(weights.len(), 10);
    assert!(
        weights.iter().any(|w| (w.as_f64().unwrap() - 1.0).abs() > 1e-6),
        "dropout should have rescaled at least one tree: {weights:?}"
    );
}

/// A DART model round-trips through JSON with its weights, so predictions after
/// a reload match.
#[test]
fn a_dart_model_round_trips_with_its_weights() {
    let d = data(400, 3);
    let booster = train(&dart(DartParameters { rate_drop: 0.4, ..Default::default() }, 8), &d);
    let reloaded = Booster::load_model(&booster.save_model()).unwrap();
    assert_eq!(reloaded.predict(&d), booster.predict(&d));
}

/// `one_drop` guarantees a drop even when `rate_drop` would have chosen none.
#[test]
fn one_drop_forces_a_dropout() {
    let d = data(400, 3);
    // A zero rate would normally drop nothing at all.
    let none = train(&dart(DartParameters { rate_drop: 0.0, ..Default::default() }, 6), &d);
    let forced = train(
        &dart(DartParameters { rate_drop: 0.0, one_drop: true, ..Default::default() }, 6),
        &d,
    );
    assert_ne!(none.predict(&d), forced.predict(&d), "`one_drop` must force a drop");
}

/// `skip_drop` of 1 skips every dropout, which brings DART back to `gbtree`.
#[test]
fn skip_drop_of_one_disables_dropout_entirely() {
    let d = data(400, 3);
    let skipped = train(
        &dart(
            DartParameters { rate_drop: 0.9, one_drop: true, skip_drop: 1.0, ..Default::default() },
            6,
        ),
        &d,
    );
    // `skip_drop` consumes a random draw per round, but no tree is ever
    // dropped, so every weight stays at 1 and the fit matches `gbtree`.
    let json: serde_json::Value = serde_json::from_str(&skipped.save_model()).unwrap();
    let weights = json["learner"]["gradient_booster"]["weight_drop"].as_array().unwrap();
    assert!(weights.iter().all(|w| (w.as_f64().unwrap() - 1.0).abs() < 1e-9), "{weights:?}");
    assert_eq!(skipped.predict(&d), train(&tree(6), &d).predict(&d));
}

/// The two `normalize_type` values rescale differently, so they give different
/// models.
#[test]
fn the_two_normalize_types_differ() {
    let d = data(600, 4);
    let by_tree = train(
        &dart(
            DartParameters {
                rate_drop: 0.5,
                one_drop: true,
                normalize_type: DartNormalizeType::Tree,
                ..Default::default()
            },
            8,
        ),
        &d,
    );
    let by_forest = train(
        &dart(
            DartParameters {
                rate_drop: 0.5,
                one_drop: true,
                normalize_type: DartNormalizeType::Forest,
                ..Default::default()
            },
            8,
        ),
        &d,
    );
    assert_ne!(by_tree.predict(&d), by_forest.predict(&d), "`normalize_type` must reach the fit");
}

/// The two `sample_type` values select trees differently.
#[test]
fn the_two_sample_types_differ() {
    let d = data(600, 4);
    let uniform = train(
        &dart(
            DartParameters {
                rate_drop: 0.4,
                sample_type: DartSampleType::Uniform,
                ..Default::default()
            },
            10,
        ),
        &d,
    );
    let weighted = train(
        &dart(
            DartParameters {
                rate_drop: 0.4,
                sample_type: DartSampleType::Weighted,
                ..Default::default()
            },
            10,
        ),
        &d,
    );
    assert_ne!(uniform.predict(&d), weighted.predict(&d), "`sample_type` must reach the fit");
}

/// Dropout is a regulariser, not a bug: a DART fit must still learn.
#[test]
fn a_dart_fit_still_learns() {
    let d = data(1000, 4);
    let booster = train(&dart(DartParameters { rate_drop: 0.2, ..Default::default() }, 30), &d);
    let mean = d.info().labels.iter().sum::<f32>() / d.num_row() as f32;
    let baseline: f64 = d
        .info()
        .labels
        .iter()
        .map(|y| {
            let e = (mean - y) as f64;
            e * e
        })
        .sum::<f64>()
        / d.num_row() as f64;
    assert!(mse(&booster, &d) < baseline, "a DART fit must beat the intercept");
}

/// A DART fit is reproducible, and the seed reaches its dropout.
#[test]
fn dart_dropout_is_reproducible_and_seed_dependent() {
    let d = data(500, 4);
    let p = dart(DartParameters { rate_drop: 0.3, ..Default::default() }, 8);
    assert_eq!(train(&p, &d).predict(&d), train(&p, &d).predict(&d));

    let mut other = p.clone();
    other.booster.learning.seed = 99;
    assert_ne!(train(&p, &d).predict(&d), train(&other, &d).predict(&d));
}

// ------------------------------------------------- prediction parameters --

#[test]
fn every_prediction_type_returns_its_documented_shape() {
    let d = data(60, 3);
    let booster = train(&tree(4), &d);
    let n = d.num_row();
    let f = d.num_col();

    for (kind, expected) in [
        (PredictionType::Value, vec![n]),
        (PredictionType::Margin, vec![n]),
        (PredictionType::Leaf, vec![n, 4]),
        (PredictionType::Contribution, vec![n, f + 1]),
        (PredictionType::ApproxContribution, vec![n, f + 1]),
        (PredictionType::Interaction, vec![n, f + 1, f + 1]),
        (PredictionType::ApproxInteraction, vec![n, f + 1, f + 1]),
    ] {
        let params = PredictParameters::builder().predict_type(kind).build().unwrap();
        let out = booster.predict_with(&params, &d).unwrap_or_else(|e| panic!("{kind}: {e}"));
        assert_eq!(out.shape, expected, "{kind}");
        assert_eq!(out.values.len(), expected.iter().product::<usize>(), "{kind}");
        assert!(out.values.iter().all(|v| v.is_finite()), "{kind} produced a non-finite value");
    }
}

/// SHAP contributions must sum to the margin — the guarantee that makes them
/// attributions rather than arbitrary numbers.
#[test]
fn shap_contributions_sum_to_the_margin() {
    let d = data(40, 3);
    let booster = train(&tree(6), &d);
    let margin = booster.predict_margin(&d);
    let width = d.num_col() + 1;

    for kind in [PredictionType::Contribution, PredictionType::ApproxContribution] {
        let params = PredictParameters::builder().predict_type(kind).build().unwrap();
        let out = booster.predict_with(&params, &d).unwrap();
        for r in 0..d.num_row() {
            let sum: f32 = out.values[r * width..(r + 1) * width].iter().sum();
            assert!(
                (sum - margin[r]).abs() < 1e-3,
                "{kind} row {r}: contributions sum to {sum}, margin is {}",
                margin[r]
            );
        }
    }
}

/// SHAP interaction values are symmetric and their rows sum back to the
/// contributions.
#[test]
fn shap_interactions_are_symmetric_and_sum_to_the_contributions() {
    let d = data(20, 3);
    let booster = train(&tree(4), &d);
    let width = d.num_col() + 1;

    let contrib = booster
        .predict_with(
            &PredictParameters::builder().predict_type(PredictionType::Contribution).build().unwrap(),
            &d,
        )
        .unwrap();
    let inter = booster
        .predict_with(
            &PredictParameters::builder().predict_type(PredictionType::Interaction).build().unwrap(),
            &d,
        )
        .unwrap();

    for r in 0..d.num_row() {
        let block = &inter.values[r * width * width..(r + 1) * width * width];
        for i in 0..width {
            for j in 0..width {
                assert!(
                    (block[i * width + j] - block[j * width + i]).abs() < 1e-4,
                    "row {r} is not symmetric at ({i}, {j})"
                );
            }
            let row_sum: f32 = (0..width).map(|j| block[i * width + j]).sum();
            let want = contrib.values[r * width + i];
            assert!(
                (row_sum - want).abs() < 1e-3,
                "row {r} feature {i}: interactions sum to {row_sum}, contribution is {want}"
            );
        }
    }
}

/// `iteration_range` really truncates the model.
#[test]
fn the_iteration_range_selects_boosting_rounds() {
    let d = data(200, 3);
    let full = train(&tree(10), &d);

    let first_three = PredictParameters::builder()
        .iteration_range(IterationRange::new(0, 3).unwrap())
        .build()
        .unwrap();
    let truncated = full.predict_with(&first_three, &d).unwrap();

    // The same as training only three rounds.
    let short = train(&tree(3), &d);
    for (a, b) in truncated.values.iter().zip(short.predict(&d)) {
        assert!((a - b).abs() < 1e-5, "{a} vs {b}");
    }

    // The whole model differs from its own prefix.
    let all = full.predict_with(&PredictParameters::default(), &d).unwrap();
    assert_ne!(all.values, truncated.values);

    // A range past the end of the model is an error, not a silent clamp.
    let too_far = PredictParameters::builder()
        .iteration_range(IterationRange::new(0, 99).unwrap())
        .build()
        .unwrap();
    assert!(full.predict_with(&too_far, &d).is_err());
}

/// `strict_shape` gives every kind its full rank, whatever the model.
#[test]
fn strict_shape_reports_the_full_rank() {
    let d = data(30, 3);
    let booster = train(&tree(4), &d);

    let plain = PredictParameters::default();
    let strict = PredictParameters::builder().strict_shape(true).build().unwrap();
    assert_eq!(booster.predict_with(&plain, &d).unwrap().shape, vec![30]);
    assert_eq!(booster.predict_with(&strict, &d).unwrap().shape, vec![30, 1]);

    // The values are the same either way; only the shape changes.
    assert_eq!(
        booster.predict_with(&plain, &d).unwrap().values,
        booster.predict_with(&strict, &d).unwrap().values
    );
}

/// `validate_features` catches a matrix with the wrong number of columns
/// before it produces nonsense.
#[test]
fn validate_features_rejects_a_mismatched_matrix() {
    let d = data(50, 3);
    let booster = train(&tree(3), &d);
    let other = data(50, 5);

    let checking = PredictParameters::default();
    assert!(checking.validate_features, "the check is on by default");
    match booster.predict_with(&checking, &other) {
        Ok(_) => panic!("a 5-column matrix should not predict from a 3-column model"),
        Err(e) => assert!(e.to_string().contains("validate_features"), "{e}"),
    }

    // Turning it off is a deliberate choice the caller can make.
    let unchecked = PredictParameters::builder().validate_features(false).build().unwrap();
    assert!(booster.predict_with(&unchecked, &other).is_ok());
}

// --------------------------------------------- feature names and types --

/// Names travel from the training matrix into the model, key the importance
/// dictionary, and survive a save/load round trip.
#[test]
fn feature_names_reach_the_model_and_its_importances() {
    const NAMES: [&str; 3] = ["age", "income", "tenure"];
    let mut d = data(200, 3);
    d.set_feature_names(&NAMES).unwrap();

    let booster = train(&tree(6), &d);
    assert_eq!(booster.feature_names(), NAMES);

    let score = booster.get_score("gain").unwrap();
    assert!(!score.is_empty(), "the model must split on something");
    for key in score.keys() {
        assert!(NAMES.contains(&key.as_str()), "unexpected importance key `{key}`");
    }

    // An unnamed fit keeps XGBoost's `f{index}` fallback.
    let unnamed = train(&tree(6), &data(200, 3));
    assert!(unnamed.feature_names().is_empty());
    assert!(unnamed.get_score("gain").unwrap().keys().all(|k| k.starts_with('f')));

    let reloaded = Booster::load_model(&booster.save_model()).unwrap();
    assert_eq!(reloaded.feature_names(), NAMES);
    assert_eq!(reloaded.get_score("weight").unwrap(), booster.get_score("weight").unwrap());
}

/// `validate_features` compares names, not just the column count — a matrix
/// whose columns are named differently is a different matrix.
#[test]
fn validate_features_rejects_mismatched_names() {
    let mut d = data(80, 3);
    d.set_feature_names(&["a", "b", "c"]).unwrap();
    let booster = train(&tree(3), &d);

    let checking = PredictParameters::default();
    booster.predict_with(&checking, &d).unwrap();

    let mut renamed = data(80, 3);
    renamed.set_feature_names(&["a", "z", "c"]).unwrap();
    match booster.predict_with(&checking, &renamed) {
        Ok(_) => panic!("a matrix with a different feature name should be rejected"),
        Err(e) => {
            let msg = e.to_string();
            assert!(msg.contains("feature 1") && msg.contains('b') && msg.contains('z'), "{msg}");
        }
    }

    // An unnamed matrix makes no claim to contradict, so it still predicts.
    booster.predict_with(&checking, &data(80, 3)).unwrap();

    // And the check can be turned off, as for the column-count case.
    let unchecked = PredictParameters::builder().validate_features(false).build().unwrap();
    booster.predict_with(&unchecked, &renamed).unwrap();
}

#[test]
fn malformed_feature_names_are_rejected() {
    let mut d = data(20, 3);

    let err = d.set_feature_names(&["a", "b"]).unwrap_err().to_string();
    assert!(err.contains('3') && err.contains('2'), "{err}");

    let err = d.set_feature_names(&["a", "b", "a"]).unwrap_err().to_string();
    assert!(err.contains("unique"), "{err}");

    for bad in ["x[0]", "a<b", "]"] {
        let err = d.set_feature_names(&["a", "b", bad]).unwrap_err().to_string();
        assert!(err.contains("feature_names"), "{bad}: {err}");
    }

    assert!(d.info().feature_names.is_empty(), "a rejection must not half-apply");

    // An empty slice clears rather than fails.
    d.set_feature_names(&["a", "b", "c"]).unwrap();
    d.set_feature_names::<&str>(&[]).unwrap();
    assert!(d.info().feature_names.is_empty());
}

/// Column types reach the model and survive its file, in the spelling the
/// model format uses.
#[test]
fn feature_types_reach_the_model_and_its_file() {
    use xgboost_rs::FeatureType::{Categorical, Numerical};

    let mut d = data(200, 3);
    d.set_feature_types(&[Numerical, Categorical, Numerical]).unwrap();
    // Column 1 is read as category codes now, so it must hold whole codes.
    let booster = train(&tree(4), &d);
    assert_eq!(booster.feature_types(), ["q", "c", "q"]);

    let text = booster.save_model();
    let json: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(json["learner"]["feature_types"], serde_json::json!(["q", "c", "q"]));

    let reloaded = Booster::load_model(&text).unwrap();
    assert_eq!(reloaded.feature_types(), ["q", "c", "q"]);
    assert_eq!(text, reloaded.save_model(), "the round trip is stable");

    // A fit that declared no types records none, rather than inventing `q`s.
    assert!(train(&tree(4), &data(200, 3)).feature_types().is_empty());
}

/// XGBoost records the type string it was *given*, so a model may say `int`,
/// `float` or `i` where a fit here would say `q`. Those must come back out
/// unchanged rather than normalised — a re-saved model should still be the
/// model that was loaded.
#[test]
fn a_loaded_model_keeps_the_type_spellings_it_came_with() {
    let booster = train(&tree(3), &data(120, 3));
    let mut json: serde_json::Value = serde_json::from_str(&booster.save_model()).unwrap();
    json["learner"]["feature_types"] = serde_json::json!(["int", "float", "i"]);

    let loaded = Booster::load_model(&json.to_string()).unwrap();
    assert_eq!(loaded.feature_types(), ["int", "float", "i"]);

    let resaved: serde_json::Value = serde_json::from_str(&loaded.save_model()).unwrap();
    assert_eq!(resaved["learner"]["feature_types"], serde_json::json!(["int", "float", "i"]));
}

#[test]
fn a_model_naming_an_unreadable_feature_type_is_rejected() {
    let booster = train(&tree(3), &data(120, 3));
    let mut json: serde_json::Value = serde_json::from_str(&booster.save_model()).unwrap();
    json["learner"]["feature_types"] = serde_json::json!(["q", "categorical", "q"]);

    let err = match Booster::load_model(&json.to_string()) {
        Ok(_) => panic!("`categorical` is not one of the five spellings"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("categorical") && err.contains('c'), "{err}");

    json["learner"]["feature_types"] = serde_json::json!(["q", 3, "q"]);
    assert!(Booster::load_model(&json.to_string()).is_err(), "a non-string is malformed");
}

/// Leaf prediction cannot start part-way through the model, and says so.
#[test]
fn leaf_prediction_rejects_a_non_zero_range_start() {
    let err = PredictParameters::builder()
        .predict_type(PredictionType::Leaf)
        .iteration_range(IterationRange::new(2, 5).unwrap())
        .build()
        .unwrap_err();
    assert!(err.to_string().contains("iteration_range"), "{err}");
}

/// A multi-class model predicts one column per class through every kind.
#[test]
fn multi_class_prediction_keeps_its_groups() {
    use xgboost_rs::parameters::Objective;
    let mut d = data(120, 3);
    let y: Vec<f32> = (0..120).map(|i| (i % 3) as f32).collect();
    d.set_labels(&y).unwrap();

    let mut p = tree(4);
    p.booster.learning.objective = Objective::MultiSoftprob { num_class: 3 };
    let booster = train(&p, &d);

    let value = booster.predict_with(&PredictParameters::default(), &d).unwrap();
    assert_eq!(value.shape, vec![120, 3]);

    let contrib = booster
        .predict_with(
            &PredictParameters::builder().predict_type(PredictionType::Contribution).build().unwrap(),
            &d,
        )
        .unwrap();
    assert_eq!(contrib.shape, vec![120, 3, 4]);

    // `multi:softmax` collapses to one class index per row.
    p.booster.learning.objective = Objective::MultiSoftmax { num_class: 3 };
    let hard = train(&p, &d);
    assert_eq!(hard.predict_with(&PredictParameters::default(), &d).unwrap().shape, vec![120]);
}

/// `training = true` asks for the prediction a DART fit sees *while* it is
/// training: one made from a thinned ensemble, with this round's dropout
/// applied.
#[test]
fn the_training_flag_applies_dart_dropout() {
    let d = data(300, 4);
    let booster = train(
        &dart(DartParameters { rate_drop: 0.5, ..Default::default() }, 12),
        &d,
    );

    let inference = PredictParameters::default();
    let training_time =
        PredictParameters::builder().training(true).build().unwrap();

    let full = booster.predict_with(&inference, &d).unwrap().values;
    let thinned = booster.predict_with(&training_time, &d).unwrap().values;
    assert_ne!(full, thinned, "dropout must actually remove trees from the prediction");

    // Predicting is not a training round: it must be repeatable, and must not
    // disturb what a later prediction sees.
    assert_eq!(thinned, booster.predict_with(&training_time, &d).unwrap().values);
    assert_eq!(full, booster.predict_with(&inference, &d).unwrap().values);
}

/// The flag reaches margin prediction too, not just the transformed values.
#[test]
fn the_training_flag_applies_to_margins() {
    let d = data(300, 4);
    let booster = train(&dart(DartParameters { rate_drop: 0.5, ..Default::default() }, 12), &d);

    let margin = |training: bool| {
        let p = PredictParameters::builder()
            .predict_type(PredictionType::Margin)
            .training(training)
            .build()
            .unwrap();
        booster.predict_with(&p, &d).unwrap().values
    };
    assert_ne!(margin(false), margin(true));
}

/// `gbtree` drops nothing, so the flag is accepted and changes nothing — which
/// is exactly what upstream does with it.
#[test]
fn the_training_flag_is_inert_without_dropout() {
    let d = data(200, 3);
    let booster = train(&tree(8), &d);

    let inference = PredictParameters::default();
    let training_time = PredictParameters::builder().training(true).build().unwrap();
    assert_eq!(
        booster.predict_with(&inference, &d).unwrap().values,
        booster.predict_with(&training_time, &d).unwrap().values
    );

    // And a `dart` fit that never drops is in the same position.
    let no_drop = train(&dart(DartParameters { rate_drop: 0.0, ..Default::default() }, 8), &d);
    assert_eq!(
        no_drop.predict_with(&inference, &d).unwrap().values,
        no_drop.predict_with(&training_time, &d).unwrap().values
    );
}
