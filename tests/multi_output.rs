//! `multi_strategy`, the choice between modelling several outputs as separate
//! problems and modelling them together.
//!
//! `one_output_per_tree` grows one tree per output each round; a tree's split
//! serves that output alone. `multi_output_tree` grows one tree whose leaves
//! carry a vector, so every split is a decision all the outputs share. The
//! tests below pin the observable differences: the ensemble size, the leaf
//! shape, and that both still learn.

use xgboost_rs::parameters::{
    BoosterParameters, BoosterType, GeneralParameters, LearningTaskParameters, MultiStrategy,
    Objective, TrainingParameters, TreeBoosterParameters, TreeMethod, VerboseEval, Verbosity,
};
use xgboost_rs::{Booster, DMatrix, api};

/// Two-target regression where the targets share a driver and each has its own,
/// so a shared split is useful but not sufficient.
fn multi_target_data(rows: usize) -> DMatrix {
    let cols = 4usize;
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut next = || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        (state.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 40) as f32 / (1u32 << 24) as f32
    };
    let x: Vec<f32> = (0..rows * cols).map(|_| next()).collect();
    let mut y = Vec::with_capacity(rows * 2);
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let shared = row[0];
        y.push(shared + 0.5 * row[1]);
        y.push(shared - 0.5 * row[2]);
    }
    let mut d = DMatrix::from_dense(&x, rows, cols, f32::NAN).unwrap();
    d.set_labels_multi(&y, 2).unwrap();
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

fn strategy(s: MultiStrategy) -> TreeBoosterParameters {
    TreeBoosterParameters::builder().multi_strategy(s).max_depth(4).eta(0.3).build().unwrap()
}

fn train(p: &TrainingParameters, d: &DMatrix) -> Booster {
    api::train(p, d, &[]).expect("training failed").0
}

fn error(p: &TrainingParameters, d: &DMatrix) -> String {
    match api::train(p, d, &[]) {
        Ok(_) => panic!("this configuration should not have been accepted"),
        Err(e) => e.to_string(),
    }
}

fn model(booster: &Booster) -> serde_json::Value {
    serde_json::from_str(&booster.save_model()).unwrap()
}

fn trees(booster: &Booster) -> Vec<serde_json::Value> {
    model(booster)
        .pointer("/learner/gradient_booster/model/trees")
        .and_then(|t| t.as_array())
        .expect("a gbtree model has trees")
        .clone()
}

/// Mean absolute error across both targets.
fn train_error(booster: &Booster, d: &DMatrix) -> f32 {
    let preds = booster.predict(d);
    let labels = &d.info().labels;
    assert_eq!(preds.len(), labels.len());
    preds.iter().zip(labels).map(|(p, y)| (p - y).abs()).sum::<f32>() / labels.len() as f32
}

/// The headline difference: a vector leaf means one tree per round instead of
/// one per output.
#[test]
fn a_vector_leaf_grows_one_tree_per_round_not_one_per_output() {
    let d = multi_target_data(400);
    let per_output = train(&params(strategy(MultiStrategy::OneOutputPerTree), 10), &d);
    let vector = train(&params(strategy(MultiStrategy::MultiOutputTree), 10), &d);

    assert_eq!(trees(&per_output).len(), 20, "two targets x ten rounds");
    assert_eq!(trees(&vector).len(), 10, "one tree a round, covering both targets");
}

/// Each leaf of a vector-leaf tree carries one value per target, and the model
/// records that in `size_leaf_vector`.
#[test]
fn a_vector_leaf_tree_records_its_leaf_size() {
    let d = multi_target_data(400);
    let vector = train(&params(strategy(MultiStrategy::MultiOutputTree), 5), &d);
    for tree in trees(&vector) {
        assert_eq!(tree["tree_param"]["size_leaf_vector"], "2");
        let n_nodes: usize = tree["tree_param"]["num_nodes"].as_str().unwrap().parse().unwrap();
        assert_eq!(
            tree["leaf_values"].as_array().unwrap().len(),
            n_nodes * 2,
            "one value per node per target"
        );
    }

    // The ordinary strategy is unchanged: scalar leaves, no vector array.
    let per_output = train(&params(strategy(MultiStrategy::OneOutputPerTree), 5), &d);
    for tree in trees(&per_output) {
        assert_eq!(tree["tree_param"]["size_leaf_vector"], "1");
        assert!(tree["leaf_values"].as_array().unwrap().is_empty());
    }
}

/// Both strategies have to actually learn both targets — a smaller ensemble is
/// only interesting if it still fits.
#[test]
fn both_strategies_learn_every_target() {
    let d = multi_target_data(600);
    for s in [MultiStrategy::OneOutputPerTree, MultiStrategy::MultiOutputTree] {
        let booster = train(&params(strategy(s), 40), &d);
        let preds = booster.predict(&d);
        assert_eq!(preds.len(), d.num_row() * 2, "{s}: one prediction per (row, target)");
        assert!(train_error(&booster, &d) < 0.1, "{s} did not fit: {}", train_error(&booster, &d));

        // The two targets differ, so the predictions must too.
        let first: Vec<f32> = preds.iter().step_by(2).copied().collect();
        let second: Vec<f32> = preds.iter().skip(1).step_by(2).copied().collect();
        assert_ne!(first, second, "{s} predicted both targets identically");
    }
}

/// A vector-leaf model round trips: the leaf vectors are part of the file.
#[test]
fn a_vector_leaf_model_survives_a_save_and_load() {
    let d = multi_target_data(400);
    let fitted = train(&params(strategy(MultiStrategy::MultiOutputTree), 12), &d);
    let text = fitted.save_model();
    let loaded = Booster::load_model(&text).unwrap();

    assert_eq!(fitted.predict(&d), loaded.predict(&d), "a reloaded vector-leaf model must agree");
    assert_eq!(text, loaded.save_model(), "the round trip is stable");
}

/// The strategy is a real modelling choice, not a relabelling: it changes the
/// splits, so it changes the predictions.
#[test]
fn the_two_strategies_fit_different_models() {
    let d = multi_target_data(400);
    let per_output = train(&params(strategy(MultiStrategy::OneOutputPerTree), 15), &d);
    let vector = train(&params(strategy(MultiStrategy::MultiOutputTree), 15), &d);
    assert_ne!(per_output.predict(&d), vector.predict(&d));
}

/// A vector-leaf fit is reproducible, like every other.
#[test]
fn a_vector_leaf_fit_is_reproducible() {
    let d = multi_target_data(300);
    let p = params(strategy(MultiStrategy::MultiOutputTree), 8);
    assert_eq!(train(&p, &d).save_model(), train(&p, &d).save_model());
}

/// `multi:softprob` is the other way a fit gets several outputs, and a vector
/// leaf covers its classes too.
#[test]
fn a_vector_leaf_covers_multiclass_outputs() {
    let rows = 450usize;
    let x: Vec<f32> = (0..rows * 3).map(|i| ((i * 53) % 97) as f32 / 97.0).collect();
    let y: Vec<f32> = (0..rows).map(|r| (r % 3) as f32).collect();
    let mut d = DMatrix::from_dense(&x, rows, 3, f32::NAN).unwrap();
    d.set_labels(&y).unwrap();

    let mut p = params(strategy(MultiStrategy::MultiOutputTree), 10);
    p.booster.learning.objective = Objective::MultiSoftprob { num_class: 3 };
    let booster = train(&p, &d);

    assert_eq!(trees(&booster).len(), 10, "one tree a round for all three classes");
    let preds = booster.predict(&d);
    assert_eq!(preds.len(), rows * 3);
    // `multi:softprob` returns a distribution per row.
    for r in 0..rows {
        let total: f32 = preds[r * 3..(r + 1) * 3].iter().sum();
        assert!((total - 1.0).abs() < 1e-4, "row {r} probabilities sum to {total}");
    }
}

/// A single-output fit is unaffected: there is nothing for a vector leaf to
/// gather, so the strategy has no effect rather than a surprising one.
#[test]
fn a_single_output_fit_is_unchanged_by_the_strategy() {
    let rows = 300usize;
    let x: Vec<f32> = (0..rows * 3).map(|i| ((i * 29) % 83) as f32 / 83.0).collect();
    let y: Vec<f32> = (0..rows).map(|r| x[r * 3]).collect();
    let mut d = DMatrix::from_dense(&x, rows, 3, f32::NAN).unwrap();
    d.set_labels(&y).unwrap();

    let per_output = train(&params(strategy(MultiStrategy::OneOutputPerTree), 8), &d);
    let vector = train(&params(strategy(MultiStrategy::MultiOutputTree), 8), &d);
    assert_eq!(per_output.save_model(), vector.save_model());
}

/// The vector-leaf grower builds a histogram per target, which only `hist`
/// does here. The other tree methods are refused — at build time, before a fit
/// is even attempted — rather than quietly growing one tree per output.
#[test]
fn the_other_tree_methods_are_refused_rather_than_ignored() {
    for method in [TreeMethod::Approx, TreeMethod::Exact] {
        let err = TreeBoosterParameters::builder()
            .multi_strategy(MultiStrategy::MultiOutputTree)
            .tree_method(method)
            .max_depth(4)
            .build()
            .expect_err("`{method}` must not build with a vector leaf")
            .to_string();
        assert!(
            err.contains("multi_strategy") || err.contains("multi_output_tree"),
            "`{method}` was rejected without naming the parameter: {err}"
        );
    }

    // An explicit updater pipeline that is not `hist` is refused at fit time,
    // where the pipeline is resolved.
    let d = multi_target_data(200);
    let tree = TreeBoosterParameters::builder()
        .multi_strategy(MultiStrategy::MultiOutputTree)
        .updater([xgboost_rs::parameters::TreeUpdaterName::GrowQuantileHistMaker,
                  xgboost_rs::parameters::TreeUpdaterName::Prune])
        .max_depth(4)
        .build()
        .unwrap();
    let err = error(&params(tree, 5), &d);
    assert!(err.contains("multi_strategy"), "{err}");
}

/// Row sampling drops whole rows, so a sampled vector-leaf fit still predicts
/// every target of every row.
#[test]
fn row_sampling_still_covers_every_output() {
    let d = multi_target_data(500);
    let tree = TreeBoosterParameters::builder()
        .multi_strategy(MultiStrategy::MultiOutputTree)
        .subsample(0.6)
        .max_depth(4)
        .build()
        .unwrap();
    let p = params(tree, 20);
    let booster = train(&p, &d);
    let preds = booster.predict(&d);
    assert_eq!(preds.len(), d.num_row() * 2);
    assert!(preds.iter().all(|v| v.is_finite()));
    assert_eq!(booster.save_model(), train(&p, &d).save_model(), "sampling stays reproducible");
}

/// `num_parallel_tree` still multiplies the round's trees — it is just that
/// each is now one vector-leaf tree rather than one per output.
#[test]
fn num_parallel_tree_multiplies_vector_leaf_trees() {
    let d = multi_target_data(300);
    let tree = TreeBoosterParameters::builder()
        .multi_strategy(MultiStrategy::MultiOutputTree)
        .num_parallel_tree(3)
        .subsample(0.8)
        .max_depth(3)
        .build()
        .unwrap();
    let booster = train(&params(tree, 4), &d);
    assert_eq!(trees(&booster).len(), 12, "three trees a round, four rounds");
}
