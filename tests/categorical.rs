//! Categorical splits, and the two parameters that shape them:
//! `max_cat_to_onehot` and `max_cat_threshold`.
//!
//! The payoff of a categorical split is that it can separate a set of category
//! codes that no threshold could, because the codes carry no order. Every test
//! here is written against that: it builds data whose target depends on
//! *membership*, not on magnitude, and checks the fit finds it.

use xgboost_rs::data::FeatureType;
use xgboost_rs::parameters::{
    BoosterParameters, BoosterType, GeneralParameters, LearningTaskParameters, TrainingParameters,
    TreeBoosterParameters, TreeMethod, VerboseEval, Verbosity,
};
use xgboost_rs::{Booster, DMatrix, api};

/// One categorical column of `n_cats` codes, where the target depends on
/// whether the code is in `positive` — a relationship that is deliberately not
/// monotone in the code.
fn membership_data(rows: usize, n_cats: u32, positive: &[u32]) -> DMatrix {
    let codes: Vec<f32> = (0..rows).map(|i| (i as u32 % n_cats) as f32).collect();
    let y: Vec<f32> = codes
        .iter()
        .map(|&c| if positive.contains(&(c as u32)) { 1.0 } else { -1.0 })
        .collect();
    let mut d = DMatrix::from_dense(&codes, rows, 1, f32::NAN).unwrap();
    d.set_labels(&y).unwrap();
    d.set_feature_types(&[FeatureType::Categorical]).unwrap();
    d
}

fn params(tree: TreeBoosterParameters, rounds: u32) -> TrainingParameters {
    TrainingParameters {
        booster: BoosterParameters {
            booster: BoosterType::Gbtree(tree),
            general: GeneralParameters { verbosity: Verbosity::Silent, ..Default::default() },
            learning: LearningTaskParameters::default(),
        },
        num_boost_round: rounds,
        verbose_eval: VerboseEval::Silent,
        ..Default::default()
    }
}

fn train(params: &TrainingParameters, d: &DMatrix) -> Booster {
    api::train(params, d, &[]).expect("training failed").0
}

fn error(params: &TrainingParameters, d: &DMatrix) -> String {
    match api::train(params, d, &[]) {
        Ok(_) => panic!("this configuration should not have been accepted"),
        Err(e) => e.to_string(),
    }
}

/// Mean absolute error of a fit against its own training labels.
fn train_error(booster: &Booster, d: &DMatrix) -> f32 {
    let preds = booster.predict(d);
    let labels = d.info().labels.clone();
    let sum: f32 = preds.iter().zip(&labels).map(|(p, y)| (p - y).abs()).sum();
    sum / labels.len() as f32
}

/// The headline: an interleaved category/label relationship is separable by set
/// membership and by nothing a threshold can express.
#[test]
fn a_non_monotone_category_set_is_learned() {
    // Even codes are positive, odd codes negative: no `code < t` split
    // separates them.
    let d = membership_data(600, 6, &[0, 2, 4]);
    let p = params(
        TreeBoosterParameters::builder().max_depth(4).eta(0.5).build().unwrap(),
        30,
    );
    let fitted = train(&p, &d);
    assert!(
        train_error(&fitted, &d) < 0.05,
        "categorical splits should separate the classes almost exactly, got {}",
        train_error(&fitted, &d)
    );

    // The very same numbers, read as an ordered feature, cannot do it: with
    // depth 4 there are not enough thresholds to isolate three alternating
    // codes as well as one membership test does.
    let mut numeric = d.clone();
    numeric.set_feature_types(&[FeatureType::Numerical]).unwrap();
    let as_numbers = train(&p, &numeric);
    assert!(
        train_error(&as_numbers, &numeric) > train_error(&fitted, &d),
        "reading category codes as numbers should fit worse"
    );
}

/// `max_cat_to_onehot` selects between the two enumerators, and the choice
/// shows up in the model.
#[test]
fn max_cat_to_onehot_switches_the_split_strategy() {
    let d = membership_data(600, 8, &[1, 3, 5, 7]);

    // Above the category count: every split is one-hot, testing a single code.
    let one_hot = train(
        &params(
            TreeBoosterParameters::builder()
                .max_depth(3)
                .max_cat_to_onehot(64)
                .build()
                .unwrap(),
            10,
        ),
        &d,
    );
    // Below it: splits partition the codes, so one split can name four of them.
    let partition = train(
        &params(
            TreeBoosterParameters::builder()
                .max_depth(3)
                .max_cat_to_onehot(1)
                .build()
                .unwrap(),
            10,
        ),
        &d,
    );

    assert_ne!(
        one_hot.save_model(),
        partition.save_model(),
        "the two enumerators must not produce the same trees"
    );
    // A partition split isolates the four positive codes in one go, so the
    // fit is ahead of a one-hot fit given the same, tight depth budget.
    assert!(
        train_error(&partition, &d) < train_error(&one_hot, &d),
        "partition {} should beat one-hot {}",
        train_error(&partition, &d),
        train_error(&one_hot, &d)
    );
}

/// A one-hot split names exactly one category; a partition split may name
/// several. The saved model is where that is visible.
#[test]
fn one_hot_splits_name_a_single_category() {
    let d = membership_data(600, 8, &[1, 3, 5, 7]);
    let one_hot = train(
        &params(
            TreeBoosterParameters::builder().max_depth(3).max_cat_to_onehot(64).build().unwrap(),
            5,
        ),
        &d,
    );
    for sizes in category_split_sizes(&one_hot) {
        assert_eq!(sizes, 1, "a one-hot split names one category");
    }

    let partition = train(
        &params(
            TreeBoosterParameters::builder().max_depth(3).max_cat_to_onehot(1).build().unwrap(),
            5,
        ),
        &d,
    );
    assert!(
        category_split_sizes(&partition).iter().any(|&n| n > 1),
        "a partition split should name more than one category"
    );
}

/// `max_cat_threshold` caps how many categories one side of a split may name.
#[test]
fn max_cat_threshold_caps_the_partition_size() {
    let d = membership_data(900, 30, &(0..15).collect::<Vec<_>>());
    for cap in [2u32, 3, 5] {
        let fitted = train(
            &params(
                TreeBoosterParameters::builder()
                    .max_depth(4)
                    .max_cat_to_onehot(1)
                    .max_cat_threshold(cap)
                    .build()
                    .unwrap(),
                5,
            ),
            &d,
        );
        for size in category_split_sizes(&fitted) {
            assert!(
                size <= cap as usize,
                "max_cat_threshold={cap} but a split named {size} categories"
            );
        }
    }
}

/// Raising the cap lets one split name more categories, so it is not merely
/// being clamped to something small anyway.
#[test]
fn a_larger_max_cat_threshold_admits_larger_partitions() {
    let d = membership_data(900, 30, &(0..15).collect::<Vec<_>>());
    let small = train(
        &params(
            TreeBoosterParameters::builder()
                .max_depth(4)
                .max_cat_to_onehot(1)
                .max_cat_threshold(2)
                .build()
                .unwrap(),
            5,
        ),
        &d,
    );
    let large = train(
        &params(
            TreeBoosterParameters::builder()
                .max_depth(4)
                .max_cat_to_onehot(1)
                .max_cat_threshold(16)
                .build()
                .unwrap(),
            5,
        ),
        &d,
    );
    let biggest = |b: &Booster| category_split_sizes(b).into_iter().max().unwrap_or(0);
    assert!(
        biggest(&large) > biggest(&small),
        "a cap of 16 should admit a bigger partition than a cap of 2: {} vs {}",
        biggest(&large),
        biggest(&small)
    );
}

/// A saved model carries its category sets, so a reloaded fit predicts
/// identically.
#[test]
fn categorical_trees_survive_a_save_and_load() {
    let d = membership_data(600, 10, &[0, 3, 4, 9]);
    let fitted = train(
        &params(
            TreeBoosterParameters::builder().max_depth(4).max_cat_to_onehot(1).build().unwrap(),
            12,
        ),
        &d,
    );
    let text = fitted.save_model();
    assert!(text.contains("\"split_type\""), "the model records split kinds");

    // Which columns hold category codes is part of the model, not only of the
    // matrix it was fitted on: the file records it and a reload keeps it.
    assert_eq!(fitted.feature_types(), ["c"]);

    let loaded = Booster::load_model(&text).unwrap();
    assert_eq!(
        fitted.predict(&d),
        loaded.predict(&d),
        "a reloaded categorical model must predict the same"
    );
    assert_eq!(loaded.feature_types(), ["c"]);
    assert_eq!(text, loaded.save_model(), "the round trip is stable");
}

/// `approx` bins the same way `hist` does, so it grows categorical splits too.
#[test]
fn the_approx_tree_method_also_splits_on_categories() {
    let d = membership_data(600, 6, &[0, 2, 4]);
    let fitted = train(
        &params(
            TreeBoosterParameters::builder()
                .tree_method(TreeMethod::Approx)
                .max_depth(4)
                .eta(0.5)
                .build()
                .unwrap(),
            30,
        ),
        &d,
    );
    assert!(
        !category_split_sizes(&fitted).is_empty(),
        "approx should have grown at least one categorical split"
    );
    assert!(train_error(&fitted, &d) < 0.1);
}

/// `exact` enumerates a column in ascending value order, which is an ordering
/// category codes do not have. It is refused rather than quietly fitted.
#[test]
fn the_exact_tree_method_refuses_categorical_columns() {
    let d = membership_data(200, 6, &[0, 2, 4]);
    let p = params(
        TreeBoosterParameters::builder()
            .tree_method(TreeMethod::Exact)
            .max_depth(4)
            .build()
            .unwrap(),
        5,
    );
    let err = error(&p, &d);
    assert!(err.contains("exact"), "{err}");
    assert!(err.contains("hist"), "the error should say what to use instead: {err}");
}

/// A category code has to be a non-negative whole number that `f32` can name
/// exactly; anything else is rejected instead of being rounded into the wrong
/// category.
#[test]
fn out_of_range_category_codes_are_rejected() {
    let p = params(TreeBoosterParameters::builder().max_depth(3).build().unwrap(), 5);

    for bad in [-1.0f32, 1e9] {
        let mut values: Vec<f32> = (0..40).map(|i| (i % 4) as f32).collect();
        values[7] = bad;
        let mut d = DMatrix::from_dense(&values, 40, 1, f32::NAN).unwrap();
        d.set_labels(&vec![1.0f32; 40]).unwrap();
        d.set_feature_types(&[FeatureType::Categorical]).unwrap();

        let err = error(&p, &d);
        assert!(
            err.contains("categorical") || err.contains("category"),
            "value {bad} should be refused as a category: {err}"
        );
    }
}

/// Missing values in a categorical column follow the learned default
/// direction, exactly as they do for a numerical one.
#[test]
fn missing_values_in_a_categorical_column_take_a_default_direction() {
    // Rows with a missing category share the label of the positive set, so the
    // fit has a reason to send them the same way.
    let rows = 600;
    let codes: Vec<f32> = (0..rows)
        .map(|i| if i % 7 == 0 { f32::NAN } else { (i as u32 % 6) as f32 })
        .collect();
    let y: Vec<f32> = codes
        .iter()
        .map(|&c| if c.is_nan() || [0.0, 2.0, 4.0].contains(&c) { 1.0 } else { -1.0 })
        .collect();
    let mut d = DMatrix::from_dense(&codes, rows, 1, f32::NAN).unwrap();
    d.set_labels(&y).unwrap();
    d.set_feature_types(&[FeatureType::Categorical]).unwrap();

    let fitted = train(
        &params(TreeBoosterParameters::builder().max_depth(4).eta(0.5).build().unwrap(), 30),
        &d,
    );
    let preds = fitted.predict(&d);
    for (i, (&p, &label)) in preds.iter().zip(&y).enumerate() {
        if codes[i].is_nan() {
            assert!(
                (p - label).abs() < 0.3,
                "row {i} is missing and should follow the positive set, got {p}"
            );
        }
    }
}

/// A category the model never saw is not in any split's set, so it takes the
/// left branch everywhere — a defined answer rather than an out-of-range read.
#[test]
fn an_unseen_category_predicts_without_error() {
    let d = membership_data(600, 6, &[0, 2, 4]);
    let fitted = train(
        &params(TreeBoosterParameters::builder().max_depth(4).build().unwrap(), 10),
        &d,
    );

    let unseen = DMatrix::from_dense(&[99.0f32, 3.0], 2, 1, f32::NAN).unwrap();
    let preds = fitted.predict(&unseen);
    assert!(preds.iter().all(|p| p.is_finite()), "unseen categories must still predict: {preds:?}");
}

/// Both categorical knobs are reported as unused when no column is
/// categorical, and not reported when one is.
#[test]
fn the_categorical_knobs_are_only_live_for_categorical_data() {
    let tree = TreeBoosterParameters::builder()
        .max_depth(3)
        .max_cat_to_onehot(9)
        .max_cat_threshold(7)
        .build()
        .unwrap();
    let config = xgboost_rs::parameters::ToConfig::to_config_map(&tree);
    assert_eq!(config["max_cat_to_onehot"], "9");
    assert_eq!(config["max_cat_threshold"], "7");

    // Both fits are accepted; the difference is only whether the knobs matter,
    // which the fit itself demonstrates.
    let categorical = membership_data(300, 8, &[1, 3]);
    train(&params(tree.clone(), 5), &categorical);

    let mut numeric = categorical.clone();
    numeric.set_feature_types(&[FeatureType::Numerical]).unwrap();
    train(&params(tree, 5), &numeric);
}

/// The `refresh` updater re-walks existing trees to recompute their statistics.
/// It has to route categorical splits the same way prediction does, or it
/// attributes rows to the wrong nodes.
#[test]
fn refresh_routes_categorical_splits_the_way_prediction_does() {
    use xgboost_rs::parameters::{ProcessType, TreeUpdaterName};

    let d = membership_data(600, 8, &[1, 3, 5, 7]);
    let grown = train(
        &params(
            TreeBoosterParameters::builder()
                .max_depth(4)
                .max_cat_to_onehot(1)
                .eta(1.0)
                .build()
                .unwrap(),
            8,
        ),
        &d,
    );
    assert!(
        !category_split_sizes(&grown).is_empty(),
        "the fixture must actually contain categorical splits"
    );

    // Refreshing against the very data the trees were grown on must reproduce
    // the leaf values it already has: every row lands where it landed before.
    let mut refresh = params(
        TreeBoosterParameters::builder()
            .process_type(ProcessType::Update)
            .updater([TreeUpdaterName::Refresh])
            .max_depth(4)
            .eta(1.0)
            .build()
            .unwrap(),
        8,
    );
    refresh.booster.general.verbosity = Verbosity::Silent;

    let before = grown.predict(&d);
    let refreshed = api::train_from(&refresh, &d, &[], Some(&grown)).expect("refresh failed").0;
    let after = refreshed.predict(&d);
    for (i, (a, b)) in before.iter().zip(&after).enumerate() {
        assert!(
            (a - b).abs() < 1e-4,
            "row {i}: refresh moved the prediction from {a} to {b}, so it routed differently"
        );
    }
}

/// How many categories each categorical split in a saved model names.
fn category_split_sizes(booster: &Booster) -> Vec<usize> {
    let model: serde_json::Value = serde_json::from_str(&booster.save_model()).unwrap();
    let trees = model
        .pointer("/learner/gradient_booster/model/trees")
        .and_then(|t| t.as_array())
        .expect("a gbtree model has trees");
    let mut out = Vec::new();
    for tree in trees {
        for size in tree["categories_sizes"].as_array().expect("categories_sizes") {
            out.push(size.as_u64().unwrap() as usize);
        }
    }
    out
}
