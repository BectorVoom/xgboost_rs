//! Every fit parameter the CPU `hist` path implements, tested through the
//! public `train` API by its *effect on the model*, not by reading it back.
//!
//! A parameter that is accepted and then ignored is worse than one that is
//! rejected, so each test here shows the fit changing: a different tree, a
//! different feature used, a different number of trees, a different stopping
//! round.

use std::collections::BTreeSet;

use xgboost_rs::parameters::{
    BoosterParameters, BoosterType, EvalMetric, GeneralParameters, GrowPolicy,
    LearningTaskParameters, MonotoneConstraint, Objective, SamplingMethod, TrainingParameters,
    TreeBoosterParameters, TreeMethod, VerboseEval, Verbosity,
};
use xgboost_rs::{Booster, DMatrix, api};

/// Deterministic synthetic regression data with a usable signal in every
/// column, so column sampling has something to change.
fn data(rows: usize, cols: usize) -> DMatrix {
    let mut state = 0x1234_5678_9abc_def0u64;
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
            // Every feature contributes, with weights that fall off slowly.
            row.iter().enumerate().map(|(c, v)| v * (1.0 / (c + 1) as f32)).sum::<f32>()
        })
        .collect();

    let mut d = DMatrix::from_dense(&x, rows, cols, f32::NAN).unwrap();
    d.set_labels(&y).unwrap();
    d
}

/// A single feature with a monotone relationship, plus noise that tempts the
/// fit into violating it.
fn wiggly_data(rows: usize) -> DMatrix {
    let x: Vec<f32> = (0..rows).map(|i| i as f32 / rows as f32).collect();
    // Rising overall, but with a dip in the middle that an unconstrained fit
    // will happily follow.
    let y: Vec<f32> = x
        .iter()
        .map(|&v| if (0.4..0.6).contains(&v) { v - 0.5 } else { v })
        .collect();
    let mut d = DMatrix::from_dense(&x, rows, 1, f32::NAN).unwrap();
    d.set_labels(&y).unwrap();
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

/// The error message from a fit that must not be accepted.
///
/// `Booster` is deliberately not `Debug` — a model is not something to print —
/// so `unwrap_err` is unavailable and the match is spelled out.
fn train_error(params: &TrainingParameters, d: &DMatrix, evals: &[(&DMatrix, &str)]) -> String {
    match api::train(params, d, evals) {
        Ok(_) => panic!("this configuration should have been rejected"),
        Err(e) => e.to_string(),
    }
}

/// Every feature used as a split anywhere in the model.
fn split_features(booster: &Booster) -> BTreeSet<u32> {
    let model: serde_json::Value = serde_json::from_str(&booster.save_model()).unwrap();
    let trees = model["learner"]["gradient_booster"]["model"]["trees"].as_array().unwrap().clone();
    let mut used = BTreeSet::new();
    for tree in trees {
        let left = tree["left_children"].as_array().unwrap();
        let indices = tree["split_indices"].as_array().unwrap();
        for (nid, l) in left.iter().enumerate() {
            if l.as_i64().unwrap() != -1 {
                used.insert(indices[nid].as_u64().unwrap() as u32);
            }
        }
    }
    used
}

/// The fraction of the model's splits that fall on `features`.
///
/// A count rather than a set, because what column weighting changes is *how
/// often* a column is a candidate, not whether it can ever be one.
fn split_share(booster: &Booster, features: &[u32]) -> f64 {
    let counts = booster.get_score("weight").unwrap();
    let total: f64 = counts.values().sum();
    if total == 0.0 {
        return 0.0;
    }
    let hit: f64 = features.iter().filter_map(|f| counts.get(&format!("f{f}"))).sum();
    hit / total
}

/// The leaf values of the first tree, in node order.
fn leaves(booster: &Booster, tree_idx: usize) -> Vec<f32> {
    let model: serde_json::Value = serde_json::from_str(&booster.save_model()).unwrap();
    let tree = &model["learner"]["gradient_booster"]["model"]["trees"][tree_idx];
    let left = tree["left_children"].as_array().unwrap();
    let cond = tree["split_conditions"].as_array().unwrap();
    left.iter()
        .enumerate()
        .filter(|(_, l)| l.as_i64().unwrap() == -1)
        .map(|(nid, _)| cond[nid].as_f64().unwrap() as f32)
        .collect()
}

// ---------------------------------------------------------------- sampling --

#[test]
fn subsample_changes_the_model_and_is_reproducible() {
    let d = data(2000, 8);
    let full = train(&params(TreeBoosterParameters::default(), 4), &d);

    let sampled_params = params(
        TreeBoosterParameters { subsample: 0.5, ..Default::default() },
        4,
    );
    let sampled = train(&sampled_params, &d);
    assert_ne!(
        full.save_model(),
        sampled.save_model(),
        "subsample must change what the fit sees"
    );

    // The same seed gives the same model; a different seed does not.
    let again = train(&sampled_params, &d);
    assert_eq!(sampled.save_model(), again.save_model(), "sampling must be reproducible");

    let mut other_seed = sampled_params.clone();
    other_seed.booster.learning.seed = 99;
    assert_ne!(
        sampled.save_model(),
        train(&other_seed, &d).save_model(),
        "the seed must reach the row sampler"
    );
}

#[test]
fn subsample_still_predicts_every_row() {
    // Unsampled rows keep their place in the tree, so predictions cover the
    // whole matrix rather than only the drawn rows.
    let d = data(1000, 5);
    let booster = train(
        &params(TreeBoosterParameters { subsample: 0.3, ..Default::default() }, 5),
        &d,
    );
    let preds = booster.predict(&d);
    assert_eq!(preds.len(), d.num_row());
    assert!(preds.iter().all(|p| p.is_finite()), "every row must get a real prediction");
    // And the fit still learns something.
    let baseline: f32 = d.info().labels.iter().sum::<f32>() / d.num_row() as f32;
    let err_model: f32 = preds
        .iter()
        .zip(&d.info().labels)
        .map(|(p, y)| (p - y).powi(2))
        .sum();
    let err_mean: f32 = d.info().labels.iter().map(|y| (baseline - y).powi(2)).sum();
    assert!(err_model < err_mean, "a sampled fit must still beat the intercept");
}

#[test]
fn the_two_sampling_methods_differ() {
    let d = data(2000, 6);
    let uniform = train(
        &params(
            TreeBoosterParameters {
                subsample: 0.5,
                sampling_method: SamplingMethod::Uniform,
                ..Default::default()
            },
            3,
        ),
        &d,
    );
    let gradient = train(
        &params(
            TreeBoosterParameters {
                subsample: 0.5,
                sampling_method: SamplingMethod::GradientBased,
                ..Default::default()
            },
            3,
        ),
        &d,
    );
    assert_ne!(
        uniform.save_model(),
        gradient.save_model(),
        "`sampling_method` must select a different sampler"
    );
}

#[test]
fn subsample_of_one_is_identical_to_no_sampling() {
    // The row-count rule means `subsample = 1` draws no randomness at all, so
    // it must give the model an unsampled fit gives, bit for bit.
    let d = data(500, 4);
    let a = train(&params(TreeBoosterParameters::default(), 3), &d);
    let b = train(
        &params(TreeBoosterParameters { subsample: 1.0, ..Default::default() }, 3),
        &d,
    );
    assert_eq!(a.save_model(), b.save_model());
}

// ------------------------------------------------------- column subsampling --

#[test]
fn colsample_bytree_restricts_the_features_a_tree_can_use() {
    let d = data(2000, 10);
    // One feature per tree: each tree can split on exactly one column.
    let booster = train(
        &params(TreeBoosterParameters { colsample_bytree: 0.1, ..Default::default() }, 1),
        &d,
    );
    assert_eq!(
        split_features(&booster).len(),
        1,
        "a single tree limited to one column cannot split on more"
    );

    // Over many trees the sampling still reaches more than one column.
    let many = train(
        &params(TreeBoosterParameters { colsample_bytree: 0.1, ..Default::default() }, 20),
        &d,
    );
    assert!(split_features(&many).len() > 1, "different trees should draw different columns");
}

#[test]
fn colsample_bylevel_and_bynode_change_the_fit() {
    let d = data(2000, 10);
    let base = train(&params(TreeBoosterParameters::default(), 3), &d);

    for tree in [
        TreeBoosterParameters { colsample_bylevel: 0.4, ..Default::default() },
        TreeBoosterParameters { colsample_bynode: 0.4, ..Default::default() },
    ] {
        let sampled = train(&params(tree.clone(), 3), &d);
        assert_ne!(base.save_model(), sampled.save_model(), "{tree:?} had no effect");
    }
}

#[test]
fn the_column_ratios_compose() {
    // 0.5 * 0.5 * 0.4 of ten columns leaves one candidate per node, so a
    // single tree can still only split on the columns it was handed.
    let d = data(2000, 10);
    let booster = train(
        &params(
            TreeBoosterParameters {
                colsample_bytree: 0.5,
                colsample_bylevel: 0.5,
                colsample_bynode: 0.4,
                max_depth: 3,
                ..Default::default()
            },
            1,
        ),
        &d,
    );
    let used = split_features(&booster);
    assert!(!used.is_empty(), "the tree must still grow");
    assert!(used.len() <= 5, "nothing outside the per-tree sample may be used: {used:?}");
}

/// `feature_weights` steers *which* columns a sample keeps. Every column here
/// carries real signal, so an unweighted fit spreads its splits over all of
/// them; weighting a handful heavily should concentrate the splits there.
#[test]
fn feature_weights_steer_the_column_sample() {
    const COLS: usize = 12;
    let heavy: [u32; 3] = [1, 5, 9];

    let mut weighted = data(2000, COLS);
    let w: Vec<f32> =
        (0..COLS).map(|c| if heavy.contains(&(c as u32)) { 100.0 } else { 1.0 }).collect();
    weighted.set_feature_weights(&w).unwrap();

    let tree = TreeBoosterParameters { colsample_bynode: 0.25, ..Default::default() };
    let p = params(tree, 6);

    let plain_share = split_share(&train(&p, &data(2000, COLS)), &heavy);
    let weighted_share = split_share(&train(&p, &weighted), &heavy);

    assert!(
        weighted_share > plain_share + 0.25,
        "weighted fit used the heavy columns for {weighted_share:.2} of its splits, \
         unweighted {plain_share:.2}"
    );
}

/// A weight of zero is a strong preference, not a mask: upstream floors every
/// weight at `kRtEps`, so the column stays reachable — just barely. What must
/// hold is that the fit still trains, and stops leaning on it.
#[test]
fn a_zero_feature_weight_all_but_removes_a_column() {
    const COLS: usize = 8;
    let mut d = data(2000, COLS);
    // Column 0 is the strongest signal (`data` weights features by 1/(c+1)),
    // so an unweighted fit leans on it hardest — the clearest thing to starve.
    let mut w = vec![1.0f32; COLS];
    w[0] = 0.0;
    d.set_feature_weights(&w).unwrap();

    let tree = TreeBoosterParameters { colsample_bynode: 0.5, ..Default::default() };
    let p = params(tree, 6);

    let plain = split_share(&train(&p, &data(2000, COLS)), &[0]);
    let starved = split_share(&train(&p, &d), &[0]);
    assert!(plain > 0.2, "the unweighted fit should favour column 0, got {plain:.2}");
    assert!(starved < 0.02, "a zero-weight column should all but vanish, got {starved:.2}");
    assert!(!split_features(&train(&p, &d)).is_empty(), "the fit must still grow trees");
}

/// Weights are a *sampling* preference, so with nothing being sampled they can
/// change nothing — and the fit must be the identical model, not merely a
/// similar one.
#[test]
fn feature_weights_do_nothing_without_column_sampling() {
    let mut d = data(1000, 6);
    let plain = train(&params(TreeBoosterParameters::default(), 4), &data(1000, 6));
    d.set_feature_weights(&[1.0, 50.0, 1.0, 50.0, 1.0, 50.0]).unwrap();
    let weighted = train(&params(TreeBoosterParameters::default(), 4), &d);
    assert_eq!(plain.save_model(), weighted.save_model());
}

#[test]
fn weighted_column_sampling_is_reproducible_and_seed_dependent() {
    let mut d = data(1000, 10);
    d.set_feature_weights(&[1.0, 9.0, 2.0, 8.0, 3.0, 7.0, 4.0, 6.0, 5.0, 5.0]).unwrap();

    let tree = TreeBoosterParameters { colsample_bytree: 0.5, ..Default::default() };
    let p = params(tree, 5);
    assert_eq!(train(&p, &d).save_model(), train(&p, &d).save_model());

    let mut other = p.clone();
    other.booster.learning.seed = 11;
    assert_ne!(train(&p, &d).save_model(), train(&other, &d).save_model());
}

#[test]
fn malformed_feature_weights_are_rejected() {
    let mut d = data(100, 4);

    let err = d.set_feature_weights(&[1.0, 1.0, 1.0]).unwrap_err().to_string();
    assert!(err.contains('4') && err.contains('3'), "{err}");

    let err = d.set_feature_weights(&[1.0, -1.0, 1.0, 1.0]).unwrap_err().to_string();
    assert!(err.contains("feature_weights") && err.contains('1'), "{err}");

    let err = d.set_feature_weights(&[1.0, 1.0, f32::NAN, 1.0]).unwrap_err().to_string();
    assert!(err.contains("feature_weights"), "{err}");

    // None of the rejections may have half-applied.
    assert!(d.info().feature_weights.is_empty());

    // An empty slice clears rather than fails, which is how a caller undoes it.
    d.set_feature_weights(&[1.0, 2.0, 3.0, 4.0]).unwrap();
    d.set_feature_weights(&[]).unwrap();
    assert!(d.info().feature_weights.is_empty());
}

#[test]
fn column_sampling_is_reproducible_and_seed_dependent() {
    let d = data(1000, 12);
    let tree = TreeBoosterParameters { colsample_bytree: 0.5, ..Default::default() };
    let p = params(tree, 5);
    assert_eq!(train(&p, &d).save_model(), train(&p, &d).save_model());

    let mut other = p.clone();
    other.booster.learning.seed = 7;
    assert_ne!(train(&p, &d).save_model(), train(&other, &d).save_model());
}

// ------------------------------------------------------------ forest fits --

#[test]
fn num_parallel_tree_grows_a_forest_per_round() {
    let d = data(1000, 6);
    let rounds = 4;
    let booster = train(
        &params(
            TreeBoosterParameters { num_parallel_tree: 3, subsample: 0.8, ..Default::default() },
            rounds,
        ),
        &d,
    );
    assert_eq!(booster.num_trees(), (rounds * 3) as usize);
    // A round still counts once, as it does upstream.
    assert_eq!(booster.boosted_rounds(), rounds as usize);

    // The model records the forest width, so a reload agrees about the shape.
    let json: serde_json::Value = serde_json::from_str(&booster.save_model()).unwrap();
    let param = &json["learner"]["gradient_booster"]["model"]["gbtree_model_param"];
    assert_eq!(param["num_parallel_tree"], "3");
    let indptr = json["learner"]["gradient_booster"]["model"]["iteration_indptr"]
        .as_array()
        .unwrap();
    assert_eq!(indptr.len(), rounds as usize + 1, "one interval per round");
    assert_eq!(indptr[1].as_u64().unwrap(), 3);

    let reloaded = Booster::load_model(&booster.save_model()).unwrap();
    assert_eq!(reloaded.boosted_rounds(), rounds as usize);
    assert_eq!(reloaded.predict(&d), booster.predict(&d));
}

#[test]
fn parallel_trees_in_a_round_differ_from_each_other() {
    // Without sampling every tree in a round would be identical, because they
    // are grown from the same gradients. Sampling is what makes a forest.
    let d = data(2000, 8);
    let booster = train(
        &params(
            TreeBoosterParameters {
                num_parallel_tree: 2,
                subsample: 0.5,
                colsample_bytree: 0.5,
                ..Default::default()
            },
            1,
        ),
        &d,
    );
    assert_eq!(booster.num_trees(), 2);
    assert_ne!(leaves(&booster, 0), leaves(&booster, 1), "the two trees must not coincide");
}

// -------------------------------------------------------------- constraints --

#[test]
fn a_monotone_constraint_makes_predictions_monotone() {
    let d = wiggly_data(400);
    let unconstrained = train(
        &params(TreeBoosterParameters { max_depth: 4, ..Default::default() }, 20),
        &d,
    );
    let preds = unconstrained.predict(&d);
    assert!(
        preds.windows(2).any(|w| w[1] < w[0]),
        "the unconstrained fit should follow the dip; the test proves nothing otherwise"
    );

    let increasing = train(
        &params(
            TreeBoosterParameters {
                max_depth: 4,
                monotone_constraints: vec![MonotoneConstraint::Increasing],
                ..Default::default()
            },
            20,
        ),
        &d,
    );
    let preds = increasing.predict(&d);
    assert!(
        preds.windows(2).all(|w| w[1] >= w[0]),
        "an increasing constraint must not allow predictions to fall"
    );

    let decreasing = train(
        &params(
            TreeBoosterParameters {
                max_depth: 4,
                monotone_constraints: vec![MonotoneConstraint::Decreasing],
                ..Default::default()
            },
            20,
        ),
        &d,
    );
    let preds = decreasing.predict(&d);
    assert!(
        preds.windows(2).all(|w| w[1] <= w[0]),
        "a decreasing constraint must not allow predictions to rise"
    );
}

#[test]
fn an_unconstrained_entry_leaves_a_feature_alone() {
    let d = data(1000, 3);
    let base = train(&params(TreeBoosterParameters { max_depth: 3, ..Default::default() }, 5), &d);
    let all_zero = train(
        &params(
            TreeBoosterParameters {
                max_depth: 3,
                monotone_constraints: vec![MonotoneConstraint::Unconstrained; 3],
                ..Default::default()
            },
            5,
        ),
        &d,
    );
    assert_eq!(
        base.save_model(),
        all_zero.save_model(),
        "all-zero constraints must cost nothing and change nothing"
    );
}

#[test]
fn interaction_constraints_keep_features_out_of_each_others_subtrees() {
    let d = data(2000, 6);
    // Two groups: nothing from {0,1,2} may appear below a split on {3,4,5}.
    let groups = vec![vec![0, 1, 2], vec![3, 4, 5]];
    let booster = train(
        &params(
            TreeBoosterParameters {
                max_depth: 4,
                interaction_constraints: Some(groups),
                ..Default::default()
            },
            10,
        ),
        &d,
    );

    // Walk every root-to-leaf path and check no path mixes the groups.
    let model: serde_json::Value = serde_json::from_str(&booster.save_model()).unwrap();
    let trees = model["learner"]["gradient_booster"]["model"]["trees"].as_array().unwrap();
    let group_of = |f: u64| if f < 3 { 0 } else { 1 };

    for (t, tree) in trees.iter().enumerate() {
        let left = tree["left_children"].as_array().unwrap();
        let right = tree["right_children"].as_array().unwrap();
        let indices = tree["split_indices"].as_array().unwrap();

        // Depth-first walk carrying the groups seen so far.
        let mut stack = vec![(0usize, BTreeSet::<u8>::new())];
        while let Some((nid, seen)) = stack.pop() {
            if left[nid].as_i64().unwrap() == -1 {
                continue;
            }
            let feature = indices[nid].as_u64().unwrap();
            let mut seen = seen.clone();
            seen.insert(group_of(feature));
            assert!(
                seen.len() == 1,
                "tree {t}: a path mixes interaction groups at node {nid} (feature {feature})"
            );
            stack.push((left[nid].as_i64().unwrap() as usize, seen.clone()));
            stack.push((right[nid].as_i64().unwrap() as usize, seen));
        }
    }
}

#[test]
fn constraints_naming_missing_features_are_rejected() {
    let d = data(100, 3);
    let too_many = params(
        TreeBoosterParameters {
            monotone_constraints: vec![MonotoneConstraint::Increasing; 4],
            ..Default::default()
        },
        1,
    );
    let err = train_error(&too_many, &d, &[]);
    assert!(err.contains("monotone_constraints"), "{err}");

    let out_of_range = params(
        TreeBoosterParameters {
            interaction_constraints: Some(vec![vec![0, 9]]),
            ..Default::default()
        },
        1,
    );
    let err = train_error(&out_of_range, &d, &[]);
    assert!(err.contains("interaction_constraints"), "{err}");
}

// -------------------------------------------------------------- growth ----

#[test]
fn grow_policy_selects_a_different_tree_shape() {
    let d = data(2000, 6);
    let depthwise = train(
        &params(
            TreeBoosterParameters {
                grow_policy: GrowPolicy::DepthWise,
                max_depth: 0,
                max_leaves: 8,
                ..Default::default()
            },
            1,
        ),
        &d,
    );
    let lossguide = train(
        &params(
            TreeBoosterParameters {
                grow_policy: GrowPolicy::LossGuide,
                max_depth: 0,
                max_leaves: 8,
                ..Default::default()
            },
            1,
        ),
        &d,
    );
    assert_ne!(
        depthwise.save_model(),
        lossguide.save_model(),
        "the two growth orders must build different trees"
    );
}

// -------------------------------------------------------------- context ----

#[test]
fn nthread_does_not_change_the_model() {
    // Thread count is a performance knob; a fit must be bit-identical whatever
    // it is set to, including with sampling in play.
    let d = data(4000, 6);
    let tree = TreeBoosterParameters {
        subsample: 0.6,
        colsample_bytree: 0.6,
        ..Default::default()
    };
    let mut single = params(tree.clone(), 4);
    single.booster.general.nthread = 1;
    let mut many = params(tree, 4);
    many.booster.general.nthread = 4;

    assert_eq!(
        train(&single, &d).save_model(),
        train(&many, &d).save_model(),
        "the thread count must not reach the model"
    );
}

#[test]
fn seed_per_iteration_changes_the_rounds_but_stays_reproducible() {
    let d = data(1000, 6);
    let tree = TreeBoosterParameters { subsample: 0.5, ..Default::default() };
    let mut fixed = params(tree.clone(), 5);
    fixed.booster.learning.seed = 4;
    let mut per_iter = params(tree, 5);
    per_iter.booster.learning.seed = 4;
    per_iter.booster.learning.seed_per_iteration = true;

    assert_ne!(train(&fixed, &d).save_model(), train(&per_iter, &d).save_model());
    assert_eq!(train(&per_iter, &d).save_model(), train(&per_iter, &d).save_model());
}

#[test]
fn scale_pos_weight_reaches_the_objective() {
    // Labels of exactly 1 are reweighted, which moves the intercept and the fit.
    let x: Vec<f32> = (0..200).map(|i| (i % 10) as f32).collect();
    let y: Vec<f32> = (0..200).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
    let mut d = DMatrix::from_dense(&x, 200, 1, f32::NAN).unwrap();
    d.set_labels(&y).unwrap();

    let plain = train(&params(TreeBoosterParameters::default(), 1), &d);
    let mut weighted = params(TreeBoosterParameters::default(), 1);
    weighted.booster.learning.scale_pos_weight = 4.0;
    let weighted = train(&weighted, &d);

    assert!(
        (plain.base_score() - 0.5).abs() < 1e-6,
        "half the labels are 1, so the plain intercept is 0.5, got {}",
        plain.base_score()
    );
    assert!(
        (weighted.base_score() - 0.8).abs() < 1e-6,
        "quadrupling the positives moves the intercept to 4/5, got {}",
        weighted.base_score()
    );
}

// --------------------------------------------------------- training loop ----

#[test]
fn every_configured_metric_is_reported() {
    let d = data(500, 4);
    let mut p = params(TreeBoosterParameters::default(), 3);
    p.booster.learning.eval_metric = vec![EvalMetric::Rmse, EvalMetric::Rmse];

    let (_, history) = api::train(&p, &d, &[(&d, "train"), (&d, "valid")]).unwrap();
    assert_eq!(history.len(), 3);
    let names: Vec<&str> = history[0].iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, ["train-rmse", "train-rmse", "valid-rmse", "valid-rmse"]);
}

#[test]
fn the_default_metric_can_be_disabled() {
    let d = data(200, 3);
    let mut p = params(TreeBoosterParameters::default(), 2);
    p.booster.general.disable_default_eval_metric = true;

    let (_, history) = api::train(&p, &d, &[(&d, "train")]).unwrap();
    assert_eq!(history.len(), 2);
    assert!(history[0].is_empty(), "no metric was configured, so none may be reported");

    // An explicit metric survives the flag, as it does upstream.
    p.booster.learning.eval_metric = vec![EvalMetric::Rmse];
    let (_, history) = api::train(&p, &d, &[(&d, "train")]).unwrap();
    assert_eq!(history[0].len(), 1);
}

#[test]
fn early_stopping_halts_a_fit_that_stops_improving() {
    let d = data(400, 4);
    // A tiny learning rate on a fit already at its floor: after the first few
    // rounds the training metric stops moving.
    let mut p = params(
        TreeBoosterParameters { eta: 1.0, max_depth: 8, ..Default::default() },
        50,
    );
    p.early_stopping_rounds = Some(2);

    let (booster, history) = api::train(&p, &d, &[(&d, "train")]).unwrap();
    assert!(history.len() < 50, "early stopping should have fired, ran {} rounds", history.len());
    assert_eq!(booster.boosted_rounds(), history.len(), "the model keeps every round it ran");

    let best = booster.best_iteration().expect("early stopping records its best round");
    assert!(best < history.len());
    assert_eq!(booster.best_score().unwrap(), history[best][0].1);
    // Stopping happened `rounds` rounds after the best one.
    assert_eq!(history.len() - 1 - best, 2);
}

#[test]
fn early_stopping_watches_the_last_metric_of_the_last_set() {
    let d = data(300, 4);
    let mut p = params(TreeBoosterParameters { eta: 1.0, ..Default::default() }, 30);
    p.early_stopping_rounds = Some(3);

    let (booster, history) = api::train(&p, &d, &[(&d, "train"), (&d, "valid")]).unwrap();
    let best = booster.best_iteration().unwrap();
    let watched = history[0].len() - 1;
    assert_eq!(history[0][watched].0, "valid-rmse");
    assert_eq!(booster.best_score().unwrap(), history[best][watched].1);
}

#[test]
fn early_stopping_can_be_told_to_maximise() {
    let d = data(300, 4);
    // RMSE falls, so asking to maximise it means round 0 is never beaten and
    // the fit stops as soon as the patience runs out.
    let mut p = params(TreeBoosterParameters::default(), 30);
    p.early_stopping_rounds = Some(2);
    p.maximize = Some(true);

    let (booster, history) = api::train(&p, &d, &[(&d, "train")]).unwrap();
    assert_eq!(booster.best_iteration(), Some(0));
    assert_eq!(history.len(), 3, "two rounds without improvement after the first");
}

#[test]
fn early_stopping_needs_something_to_watch() {
    let d = data(100, 3);
    let mut p = params(TreeBoosterParameters::default(), 5);
    p.early_stopping_rounds = Some(2);
    let err = train_error(&p, &d, &[]);
    assert!(err.contains("early_stopping_rounds"), "{err}");
}

#[test]
fn a_fit_without_early_stopping_reports_no_best_round() {
    let d = data(100, 3);
    let (booster, history) =
        api::train(&params(TreeBoosterParameters::default(), 4), &d, &[(&d, "train")]).unwrap();
    assert_eq!(history.len(), 4);
    assert_eq!(booster.best_iteration(), None);
    assert_eq!(booster.best_score(), None);
}

// ----------------------------------------------------------- rejections ----

#[test]
fn unimplemented_algorithm_choices_are_rejected_not_ignored() {
    let d = data(100, 3);

    // Every tree method now trains; only a device this build has no code for
    // and the algorithm choices below are refused.
    for method in [TreeMethod::Auto, TreeMethod::Hist, TreeMethod::Exact, TreeMethod::Approx] {
        let p = params(TreeBoosterParameters { tree_method: method, ..Default::default() }, 1);
        api::train(&p, &d, &[]).unwrap_or_else(|e| panic!("{method} must train: {e}"));
    }

    // `device=cuda` now trains: the `grow_gpu_hist` updater is implemented.
    // `tests/gpu_training.rs` checks it against the CPU fit; here it only has
    // to be accepted rather than refused.
    let mut gpu = params(TreeBoosterParameters::default(), 1);
    gpu.booster.general.device = xgboost_rs::parameters::Device::cuda(0);
    #[cfg(feature = "gpu")]
    api::train(&gpu, &d, &[]).unwrap_or_else(|e| panic!("device=cuda must train: {e}"));
    // Without the GPU kernels compiled in there is nothing to run it on, and
    // the fit says so rather than falling back to the CPU.
    #[cfg(not(feature = "gpu"))]
    {
        let err = train_error(&gpu, &d, &[]);
        assert!(err.contains("device"), "{err}");
    }

    // SYCL has no updater in this build either way.
    let mut sycl = params(TreeBoosterParameters::default(), 1);
    sycl.booster.general.device = xgboost_rs::parameters::Device::Sycl(
        xgboost_rs::parameters::SyclKind::Gpu,
        None,
    );
    let err = train_error(&sycl, &d, &[]);
    assert!(err.contains("SYCL"), "{err}");
}

/// The objectives and metrics that used to be rejected now train, so the
/// rejection list must not silently grow back over them.
#[test]
fn implemented_objectives_and_metrics_are_accepted() {
    let x: Vec<f32> = (0..200).map(|i| (i % 17) as f32 / 17.0).collect();
    let y: Vec<f32> = (0..200).map(|i| f32::from(i % 3 == 0)).collect();
    let mut d = DMatrix::from_dense(&x, 200, 1, f32::NAN).unwrap();
    d.set_labels(&y).unwrap();

    let mut p = params(TreeBoosterParameters::default(), 2);
    p.booster.learning.objective = Objective::BinaryLogistic;
    p.booster.learning.eval_metric = vec![EvalMetric::Auc, EvalMetric::Logloss];
    let (_, history) = api::train(&p, &d, &[(&d, "train")]).expect("binary:logistic must train");
    let names: Vec<&str> = history[0].iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, ["train-auc", "train-logloss"]);
}

/// `auto` and `hist` are the same algorithm, so both must be accepted and give
/// the same model.
#[test]
fn auto_and_hist_are_the_same_fit() {
    let d = data(500, 4);
    let auto = train(&params(TreeBoosterParameters::default(), 3), &d);
    let hist = train(
        &params(TreeBoosterParameters { tree_method: TreeMethod::Hist, ..Default::default() }, 3),
        &d,
    );
    assert_eq!(auto.save_model(), hist.save_model());
}

/// `sparse_threshold` picks how each column is stored in the transposed copy
/// the row partitioner reads. It is a memory/speed trade, so every setting must
/// reach the fit and produce *the same* model.
#[test]
fn sparse_threshold_changes_the_layout_but_not_the_model() {
    // A matrix with columns of very different density, so the threshold has
    // something to choose between.
    let (rows, cols) = (600usize, 8usize);
    let x: Vec<f32> = (0..rows * cols)
        .map(|i| {
            let (r, c) = (i / cols, i % cols);
            if (r * 7 + c) % (c + 2) == 0 { ((i * 31) % 89) as f32 / 89.0 } else { f32::NAN }
        })
        .collect();
    let mut d = DMatrix::from_dense(&x, rows, cols, f32::NAN).unwrap();
    let y: Vec<f32> = (0..rows).map(|r| ((r % 13) as f32) / 13.0).collect();
    d.set_labels(&y).unwrap();

    let fit = |threshold: f64| {
        let tree = TreeBoosterParameters::builder()
            .sparse_threshold(threshold)
            .max_depth(5)
            .build()
            .unwrap();
        train(&params(tree, 10), &d).save_model()
    };

    let baseline = fit(0.0);
    for threshold in [0.2f64, 0.5, 0.9, 1.0] {
        assert_eq!(
            baseline,
            fit(threshold),
            "sparse_threshold={threshold} changed the model; it must only change storage"
        );
    }
}
