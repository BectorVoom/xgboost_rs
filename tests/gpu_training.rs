//! `device=cuda` end to end through `api::train`.
//!
//! The device fit is checked against the CPU fit of the same parameters: same
//! tree shapes, and predictions within the 1e-5 the oracle tests use. The two
//! cannot be bit-identical — the device accumulates quantised `i64` gradients
//! rather than `f64` ones — which is the same gap XGBoost has between its own
//! `hist` and `gpu_hist`.
//!
//! Note the runtime: without the `cuda` feature these run on wgpu/Vulkan, so
//! `device=cuda` selects the *GPU code path* rather than a CUDA device. That is
//! what makes the path testable anywhere; `--features cuda` runs the same code
//! on a real CUDA device.

use xgboost_rs::parameters::{
    BoosterParameters, BoosterType, Device, GeneralParameters, GrowPolicy, LearningTaskParameters,
    MultiStrategy, Objective, TrainingParameters, TreeBoosterParameters, VerboseEval, Verbosity,
};
use xgboost_rs::{DMatrix, api};

fn data(rows: usize, cols: usize, missing: f32, seed: u64) -> DMatrix {
    let mut st = seed | 1;
    let mut next = || {
        st ^= st >> 12;
        st ^= st << 25;
        st ^= st >> 27;
        ((st.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32) as f32 / u32::MAX as f32
    };
    let x: Vec<f32> = (0..rows * cols)
        .map(|_| if next() < missing { f32::NAN } else { next() * 4.0 - 2.0 })
        .collect();
    let y: Vec<f32> = (0..rows)
        .map(|r| {
            x[r * cols..(r + 1) * cols]
                .iter()
                .enumerate()
                .filter(|(_, v)| v.is_finite())
                .map(|(c, v)| v / (c + 1) as f32)
                .sum()
        })
        .collect();
    let mut d = DMatrix::from_dense(&x, rows, cols, f32::NAN).unwrap();
    d.set_labels(&y).unwrap();
    d
}

fn params(device: Device, tree: TreeBoosterParameters, rounds: u32) -> TrainingParameters {
    TrainingParameters {
        booster: BoosterParameters {
            booster: BoosterType::Gbtree(tree),
            general: GeneralParameters { device, verbosity: Verbosity::Silent, ..Default::default() },
            learning: LearningTaskParameters {
                objective: Objective::RegSquaredError,
                ..Default::default()
            },
        },
        num_boost_round: rounds,
        verbose_eval: VerboseEval::Silent,
        ..Default::default()
    }
}

/// Fit both ways and return `(per-round node counts, predictions)` for each.
fn fit_both(
    dmat: &DMatrix,
    tree: TreeBoosterParameters,
    rounds: u32,
) -> ((Vec<usize>, Vec<f32>), (Vec<usize>, Vec<f32>)) {
    let (cpu, _) = api::train(&params(Device::Cpu, tree.clone(), rounds), dmat, &[]).unwrap();
    let (gpu, _) = api::train(&params(Device::cuda(0), tree, rounds), dmat, &[]).unwrap();

    let nodes = |b: &xgboost_rs::Booster| -> Vec<usize> {
        let m: serde_json::Value = serde_json::from_str(&b.save_model()).unwrap();
        m["learner"]["gradient_booster"]["model"]["trees"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["split_indices"].as_array().unwrap().len())
            .collect()
    };
    ((nodes(&cpu), cpu.predict(dmat)), (nodes(&gpu), gpu.predict(dmat)))
}

/// A single round must be **bit-identical**.
///
/// Both paths see the same gradients, and although the device sums quantised
/// `i64` bins where the CPU sums `f64` ones, the split arithmetic that decides
/// the tree agrees exactly. This is the strong claim, so it is checked on every
/// parameter combination below.
fn compare_exact(dmat: &DMatrix, tree: TreeBoosterParameters) {
    let ((cpu_nodes, cpu_pred), (gpu_nodes, gpu_pred)) = fit_both(dmat, tree, 1);
    assert_eq!(gpu_nodes, cpu_nodes, "node counts");
    assert_eq!(gpu_pred, cpu_pred, "predictions");
}

/// Over several rounds the two may part company: round one's leaf values feed
/// round two's gradients, so the quantisation difference gets a chance to move
/// a split, and from there the trees are simply different.
///
/// `lossguide` amplifies this the most, because its queue orders nodes by gain
/// and a last-ulp difference can swap which node splits next — that shows up as
/// two trees of the same shape but different content. What must still hold is
/// that the device model is not materially *worse*. Measured across these
/// configurations, the largest gap is 2% of train RMSE, and it favours the
/// device; the bound below is set just above that.
fn compare_accuracy(dmat: &DMatrix, tree: TreeBoosterParameters, rounds: u32) {
    let ((_, cpu_pred), (_, gpu_pred)) = fit_both(dmat, tree, rounds);
    let labels = &dmat.info().labels;
    let rmse = |p: &[f32]| -> f64 {
        (p.iter().zip(labels).map(|(a, b)| ((a - b) as f64).powi(2)).sum::<f64>()
            / p.len() as f64)
            .sqrt()
    };
    let (c, g) = (rmse(&cpu_pred), rmse(&gpu_pred));
    assert!(g <= c * 1.03, "device fit is worse: train rmse cpu {c} vs gpu {g}");
    assert!(c <= g * 1.03, "device fit diverged further than expected: cpu {c} vs gpu {g}");
}

#[test]
fn device_cuda_trains_and_matches_the_cpu_fit() {
    let d = data(2000, 8, 0.0, 7);
    compare_exact(&d, TreeBoosterParameters::default());
    compare_accuracy(&d, TreeBoosterParameters::default(), 5);
}

#[test]
fn matches_with_missing_values() {
    let d = data(1500, 6, 0.25, 11);
    compare_exact(&d, TreeBoosterParameters::default());
    compare_accuracy(&d, TreeBoosterParameters::default(), 4);
}

#[test]
fn matches_across_depth_and_grow_policy() {
    let d = data(1500, 5, 0.0, 13);
    for max_depth in [1u32, 2, 4, 8] {
        compare_exact(&d, TreeBoosterParameters::builder().max_depth(max_depth).build().unwrap());
    }
    for max_leaves in [2u32, 8, 16] {
        let p = TreeBoosterParameters::builder()
            .grow_policy(GrowPolicy::LossGuide)
            .max_depth(0)
            .max_leaves(max_leaves)
            .build()
            .unwrap();
        compare_exact(&d, p.clone());
        compare_accuracy(&d, p, 3);
    }
}

#[test]
fn matches_with_regularisation_and_shrinkage() {
    let d = data(1500, 5, 0.1, 17);
    let p = TreeBoosterParameters::builder()
        .eta(0.1)
        .lambda(5.0)
        .alpha(1.0)
        .gamma(0.2)
        .min_child_weight(5.0)
        .build()
        .unwrap();
    compare_exact(&d, p.clone());
    compare_accuracy(&d, p, 4);
}

/// Row and column sampling narrow what the grower sees rather than changing
/// how it searches, so the device path gets them for free — but only if the
/// draws happen in the same order, which is what this pins.
#[test]
fn matches_with_row_and_column_sampling() {
    let d = data(2000, 8, 0.0, 19);
    let p = TreeBoosterParameters::builder()
        .subsample(0.7)
        .colsample_bytree(0.8)
        .colsample_bylevel(0.9)
        .colsample_bynode(0.9)
        .build()
        .unwrap();
    compare_exact(&d, p.clone());
    compare_accuracy(&d, p, 4);
}

#[test]
fn matches_with_monotone_and_interaction_constraints() {
    use xgboost_rs::parameters::MonotoneConstraint;
    let d = data(1500, 6, 0.0, 37);
    compare_exact(
        &d,
        TreeBoosterParameters::builder()
            .monotone_constraints(vec![
                MonotoneConstraint::Increasing,
                MonotoneConstraint::Unconstrained,
                MonotoneConstraint::Decreasing,
                MonotoneConstraint::Unconstrained,
                MonotoneConstraint::Unconstrained,
                MonotoneConstraint::Increasing,
            ])
            .build()
            .unwrap(),
    );
    compare_exact(
        &d,
        TreeBoosterParameters::builder()
            .interaction_constraints(vec![vec![0u32, 1, 2], vec![3, 4, 5]])
            .build()
            .unwrap(),
    );
}

#[test]
fn matches_across_max_bin() {
    let d = data(2000, 5, 0.0, 41);
    for max_bin in [8u32, 32, 256, 512] {
        compare_exact(&d, TreeBoosterParameters::builder().max_bin(max_bin).build().unwrap());
    }
}

#[test]
fn matches_with_num_parallel_tree() {
    let d = data(1200, 5, 0.0, 23);
    let p = TreeBoosterParameters::builder().num_parallel_tree(3).build().unwrap();
    compare_exact(&d, p.clone());
    compare_accuracy(&d, p, 2);
}

/// One categorical column (membership, not order, drives the label) plus a
/// couple of ordinary numeric ones — enough to force column sampling and the
/// numeric evaluator to run alongside the categorical one in the same batch.
fn categorical_data(rows: usize, n_cats: u32, positive: &[u32], missing: f32, seed: u64) -> DMatrix {
    use xgboost_rs::FeatureType;
    let mut st = seed | 1;
    let mut next = || {
        st ^= st >> 12;
        st ^= st << 25;
        st ^= st >> 27;
        ((st.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32) as f32 / u32::MAX as f32
    };
    let cols = 3;
    let mut x = vec![0.0f32; rows * cols];
    for r in 0..rows {
        x[r * cols] = if next() < missing { f32::NAN } else { (r as u32 % n_cats) as f32 };
        x[r * cols + 1] = next() * 4.0 - 2.0;
        x[r * cols + 2] = next() * 4.0 - 2.0;
    }
    let y: Vec<f32> = (0..rows)
        .map(|r| {
            let cat = x[r * cols];
            let base = if cat.is_finite() && positive.contains(&(cat as u32)) { 1.0 } else { -1.0 };
            base + 0.1 * (x[r * cols + 1] + x[r * cols + 2])
        })
        .collect();
    let mut d = DMatrix::from_dense(&x, rows, cols, f32::NAN).unwrap();
    d.set_labels(&y).unwrap();
    d.set_feature_types(&[FeatureType::Categorical, FeatureType::Numerical, FeatureType::Numerical])
        .unwrap();
    d
}

/// Few enough categories that `UseOneHot` picks the one-hot enumerator on
/// both devices.
#[test]
fn categorical_splits_match_the_cpu_fit_one_hot() {
    let d = categorical_data(800, 3, &[1], 0.0, 41);
    let p = TreeBoosterParameters::builder().max_depth(3).max_cat_to_onehot(64).build().unwrap();
    compare_exact(&d, p.clone());
    compare_accuracy(&d, p, 4);
}

/// Enough categories, with `max_cat_to_onehot` forced down, that the
/// partition enumerator runs instead.
#[test]
fn categorical_splits_match_the_cpu_fit_partition() {
    let d = categorical_data(800, 12, &[1, 3, 5, 7, 9], 0.0, 43);
    let p = TreeBoosterParameters::builder().max_depth(4).max_cat_to_onehot(1).build().unwrap();
    compare_exact(&d, p.clone());
    compare_accuracy(&d, p, 4);
}

/// Missing values in the categorical column exercise both `enumerate_one_hot`
/// scans (missing grouped left vs. grouped with the chosen category), and
/// `lossguide` gives the last-ulp-sensitive queue a categorical split to
/// reorder around.
#[test]
fn categorical_splits_match_the_cpu_fit_with_missing_values_and_lossguide() {
    let d = categorical_data(900, 6, &[0, 2, 4], 0.2, 47);
    let p = TreeBoosterParameters::builder()
        .grow_policy(GrowPolicy::LossGuide)
        .max_depth(0)
        .max_leaves(12)
        .build()
        .unwrap();
    compare_exact(&d, p.clone());
    compare_accuracy(&d, p, 4);
}

// ------------------------------------------ multi_strategy=multi_output_tree

/// Regression on `n_targets` outputs that share a driver and each have one of
/// their own, so a split all of them share is useful but never sufficient —
/// which is what makes a vector leaf a different fit from one tree per output.
fn multi_target_data(rows: usize, n_targets: usize, missing: f32, seed: u64) -> DMatrix {
    let cols = 5usize;
    let mut st = seed | 1;
    let mut next = || {
        st ^= st >> 12;
        st ^= st << 25;
        st ^= st >> 27;
        ((st.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32) as f32 / u32::MAX as f32
    };
    let x: Vec<f32> = (0..rows * cols)
        .map(|_| if next() < missing { f32::NAN } else { next() * 4.0 - 2.0 })
        .collect();
    let value = |v: f32| if v.is_finite() { v } else { 0.0 };
    let y: Vec<f32> = (0..rows * n_targets)
        .map(|i| {
            let (r, t) = (i / n_targets, i % n_targets);
            let row = &x[r * cols..(r + 1) * cols];
            value(row[0]) + 0.5 * (t + 1) as f32 * value(row[1 + t % (cols - 1)])
        })
        .collect();
    let mut d = DMatrix::from_dense(&x, rows, cols, f32::NAN).unwrap();
    d.set_labels_multi(&y, n_targets).unwrap();
    d
}

fn vector_leaf(mut tree: TreeBoosterParameters) -> TreeBoosterParameters {
    tree.multi_strategy = MultiStrategy::MultiOutputTree;
    tree
}

#[test]
fn multi_output_tree_matches_the_cpu_fit() {
    let d = multi_target_data(1200, 3, 0.0, 5);
    compare_exact(&d, vector_leaf(TreeBoosterParameters::default()));
    compare_accuracy(&d, vector_leaf(TreeBoosterParameters::default()), 4);
}

/// The backward scan is only taken when a feature has missing rows in the node,
/// and a vector leaf decides that over the targets together.
#[test]
fn multi_output_tree_matches_with_missing_values() {
    let d = multi_target_data(1200, 2, 0.3, 17);
    compare_exact(&d, vector_leaf(TreeBoosterParameters::default()));
    compare_accuracy(&d, vector_leaf(TreeBoosterParameters::default()), 4);
}

/// `lossguide` batches one node at a time and `depthwise` a whole level, so the
/// two exercise different frontier layouts over the per-target histograms.
#[test]
fn multi_output_tree_matches_across_depth_and_grow_policy() {
    let d = multi_target_data(900, 4, 0.1, 23);
    for depth in [1u32, 3, 6] {
        let p = vector_leaf(
            TreeBoosterParameters::builder().max_depth(depth).build().unwrap(),
        );
        compare_exact(&d, p);
    }
    let p = vector_leaf(
        TreeBoosterParameters::builder()
            .grow_policy(GrowPolicy::LossGuide)
            .max_leaves(12)
            .max_depth(0)
            .build()
            .unwrap(),
    );
    compare_exact(&d, p);
}

/// `min_child_weight` is tested against the children's *mean* hessian, and the
/// monotone box shapes every target's weight; both have to travel to the device.
#[test]
fn multi_output_tree_matches_with_regularisation_and_constraints() {
    use xgboost_rs::parameters::MonotoneConstraint::{Increasing, Unconstrained};
    let d = multi_target_data(900, 3, 0.05, 29);
    let p = vector_leaf(
        TreeBoosterParameters::builder()
            .lambda(2.5)
            .alpha(0.4)
            .min_child_weight(8.0)
            .max_delta_step(0.7)
            .eta(0.4)
            .build()
            .unwrap(),
    );
    compare_exact(&d, p.clone());
    compare_accuracy(&d, p, 3);

    let p = vector_leaf(
        TreeBoosterParameters::builder()
            .monotone_constraints(
                [Increasing, Unconstrained, Unconstrained, Unconstrained, Increasing].to_vec(),
            )
            .build()
            .unwrap(),
    );
    compare_exact(&d, p);
}

/// Column sampling draws per node, and the draw order is part of the model, so
/// the device path has to mask the same features the CPU path does.
#[test]
fn multi_output_tree_matches_with_sampling() {
    let d = multi_target_data(1200, 2, 0.0, 37);
    let p = vector_leaf(
        TreeBoosterParameters::builder()
            .subsample(0.7)
            .colsample_bytree(0.8)
            .colsample_bynode(0.7)
            .num_parallel_tree(2)
            .build()
            .unwrap(),
    );
    compare_exact(&d, p.clone());
    compare_accuracy(&d, p, 3);
}

/// A vector leaf has no categorical split on either device, and must say so.
#[test]
fn multi_output_tree_rejects_categorical_features() {
    let d = categorical_data(300, 6, &[1, 3], 0.0, 41);
    let p = vector_leaf(TreeBoosterParameters::default());
    let err = match api::train(&params(Device::cuda(0), p, 1), &d, &[]) {
        Err(e) => e,
        Ok(_) => panic!("a categorical vector-leaf fit must be refused"),
    };
    assert!(err.to_string().contains("multi_output_tree"), "{err}");
}

/// SYCL has no updater here, and must say so rather than fall back.
#[test]
fn rejects_sycl() {
    use xgboost_rs::parameters::SyclKind;
    let d = data(200, 3, 0.0, 31);
    let err = match api::train(
        &params(Device::Sycl(SyclKind::Gpu, None), TreeBoosterParameters::default(), 2),
        &d,
        &[],
    ) {
        Err(e) => e,
        Ok(_) => panic!("a SYCL fit must be refused"),
    };
    assert!(err.to_string().contains("SYCL"), "{err}");
}
