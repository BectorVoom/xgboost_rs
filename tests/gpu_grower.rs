//! The GPU tree driver against the CPU one, tree for tree.
//!
//! The two are not required to be bit-identical and cannot be: the CPU grower
//! accumulates `f64` sums from the raw `f32` gradients, while the device
//! accumulates the *quantised* `i64` gradients so that atomics commit in any
//! order. That is the same divergence XGBoost has between its own `hist` and
//! `gpu_hist`. What must hold is that they grow the *same tree* — same splits,
//! same directions, same shape — with leaf values agreeing to the 1e-5 the
//! oracle tests use.

use cubecl::Runtime;
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

use xgboost_rs::data::cuts::build_cuts;
use xgboost_rs::data::gradient_index::build_gradient_index;
use xgboost_rs::objective::GradientPair;
use xgboost_rs::gpu::grower::GpuHistGrower;
use xgboost_rs::parameters::{GeneralParameters, LearningTaskParameters};
use xgboost_rs::tree::hist::HistGrower;
use xgboost_rs::tree::model::RegTree;
use xgboost_rs::tree::param::{GrowPolicy, TrainParam};
use xgboost_rs::{Context, DMatrix};

type R = WgpuRuntime;

fn client() -> ComputeClient<R> {
    R::client(&WgpuDevice::default())
}

struct Rng(u64);

impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        ((self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32) as f32 / u32::MAX as f32
    }
}

/// A matrix whose label depends on every feature, so a parameter that changes
/// which features or nodes a tree uses has something to change.
fn data(rows: usize, cols: usize, missing: f32, seed: u64) -> (DMatrix, Vec<GradientPair>) {
    let mut rng = Rng(seed);
    let x: Vec<f32> = (0..rows * cols)
        .map(|_| if rng.next_f32() < missing { f32::NAN } else { rng.next_f32() * 4.0 - 2.0 })
        .collect();
    let y: Vec<f32> = (0..rows)
        .map(|r| {
            let row = &x[r * cols..(r + 1) * cols];
            row.iter().enumerate().filter(|(_, v)| v.is_finite()).map(|(c, v)| v / (c + 1) as f32).sum()
        })
        .collect();
    let mut d = DMatrix::from_dense(&x, rows, cols, f32::NAN).unwrap();
    d.set_labels(&y).unwrap();

    // Squared-error gradients around a zero base score.
    let gpair: Vec<GradientPair> =
        y.iter().map(|&t| GradientPair { grad: -t, hess: 1.0 }).collect();
    (d, gpair)
}

fn context() -> Context {
    Context::new(&GeneralParameters::default(), &LearningTaskParameters::default())
}

/// Grow the same tree both ways and compare.
fn compare(dmat: &DMatrix, gpair: &[GradientPair], param: TrainParam, max_bin: u32) {
    let cuts = build_cuts(dmat, max_bin).unwrap();

    let gi = build_gradient_index(dmat, &cuts).unwrap();
    let mut cpu_tree = RegTree::new(dmat.num_col());
    let mut cpu_ctx = context();
    let mut cpu = HistGrower::new(&param, &gi, dmat);
    cpu.grow(&mut cpu_ctx, gpair, &mut cpu_tree);

    let mut gpu_tree = RegTree::new(dmat.num_col());
    let mut gpu_ctx = context();
    let mut gpu =
        GpuHistGrower::<R>::new(client(), dmat, cuts.clone(), param.clone()).unwrap();
    gpu.grow(gpair, &mut gpu_tree, gpu_ctx.rng()).unwrap();

    assert_eq!(gpu_tree.num_nodes(), cpu_tree.num_nodes(), "node count");

    for nid in 0..cpu_tree.num_nodes() {
        let (c, g) = (cpu_tree.nodes[nid], gpu_tree.nodes[nid]);
        assert_eq!(g.is_leaf(), c.is_leaf(), "node {nid} leaf-ness");
        if c.is_leaf() {
            let (cv, gv) = (cpu_tree.leaf_value(nid)[0], gpu_tree.leaf_value(nid)[0]);
            assert!(
                (cv - gv).abs() <= 1e-5 * cv.abs().max(1.0),
                "node {nid} leaf value: cpu {cv} vs gpu {gv}"
            );
        } else {
            assert_eq!(g.split_index, c.split_index, "node {nid} split feature");
            assert_eq!(g.default_left, c.default_left, "node {nid} default direction");
            assert_eq!(g.value, c.value, "node {nid} threshold");
            assert_eq!(g.left, c.left, "node {nid} left child");
            assert_eq!(g.right, c.right, "node {nid} right child");
        }
    }
}

#[test]
fn dense_depthwise_matches_the_cpu_grower() {
    let (d, gpair) = data(2000, 8, 0.0, 7);
    compare(&d, &gpair, TrainParam::default(), 256);
}

#[test]
fn missing_values_match() {
    let (d, gpair) = data(2000, 6, 0.25, 11);
    compare(&d, &gpair, TrainParam::default(), 64);
}

#[test]
fn honours_max_depth() {
    let (d, gpair) = data(1500, 5, 0.0, 13);
    for max_depth in [1i32, 2, 3, 8] {
        compare(&d, &gpair, TrainParam { max_depth, ..Default::default() }, 32);
    }
}

#[test]
fn honours_lossguide_and_max_leaves() {
    let (d, gpair) = data(1500, 5, 0.0, 17);
    for max_leaves in [2i32, 4, 16] {
        compare(
            &d,
            &gpair,
            TrainParam {
                grow_policy: GrowPolicy::LossGuide,
                max_depth: 0,
                max_leaves,
                ..Default::default()
            },
            32,
        );
    }
}

#[test]
fn honours_the_regularisation_knobs() {
    let (d, gpair) = data(1500, 5, 0.1, 19);
    for (lambda, alpha, gamma) in
        [(0.0f32, 0.0f32, 0.0f32), (5.0, 0.0, 0.0), (1.0, 2.0, 0.0), (1.0, 0.0, 0.5)]
    {
        compare(
            &d,
            &gpair,
            TrainParam {
                reg_lambda: lambda,
                reg_alpha: alpha,
                min_split_loss: gamma,
                ..Default::default()
            },
            32,
        );
    }
}

#[test]
fn honours_min_child_weight_and_max_delta_step() {
    let (d, gpair) = data(1500, 5, 0.0, 23);
    for mcw in [0.0f32, 1.0, 20.0, 200.0] {
        compare(&d, &gpair, TrainParam { min_child_weight: mcw, ..Default::default() }, 32);
    }
    for mds in [0.5f32, 2.0] {
        compare(&d, &gpair, TrainParam { max_delta_step: mds, ..Default::default() }, 32);
    }
}

#[test]
fn honours_monotone_constraints() {
    use xgboost_rs::parameters::MonotoneConstraint;
    let (d, gpair) = data(1500, 4, 0.0, 29);
    let c = vec![
        MonotoneConstraint::Increasing,
        MonotoneConstraint::Unconstrained,
        MonotoneConstraint::Decreasing,
        MonotoneConstraint::Unconstrained,
    ];
    compare(&d, &gpair, TrainParam { monotone_constraints: c, ..Default::default() }, 32);
}

#[test]
fn honours_interaction_constraints() {
    let (d, gpair) = data(1500, 6, 0.0, 31);
    compare(
        &d,
        &gpair,
        TrainParam {
            interaction_constraints: Some(vec![vec![0, 1, 2], vec![3, 4, 5]]),
            ..Default::default()
        },
        32,
    );
}

#[test]
fn honours_max_bin() {
    let (d, gpair) = data(2000, 5, 0.0, 37);
    for max_bin in [8u32, 16, 64, 256, 512] {
        compare(&d, &gpair, TrainParam::default(), max_bin);
    }
}

/// The leaf segments must partition every row exactly once, and agree with
/// walking the tree.
#[test]
fn leaf_segments_cover_every_row_once() {
    let (d, gpair) = data(3000, 6, 0.2, 41);
    let cuts = build_cuts(&d, 64).unwrap();
    let param = TrainParam::default();

    let mut tree = RegTree::new(d.num_col());
    let mut ctx = context();
    let mut gpu = GpuHistGrower::<R>::new(client(), &d, cuts, param).unwrap();
    let grown = gpu.grow(&gpair, &mut tree, ctx.rng()).unwrap();

    let mut seen = vec![0u32; d.num_row()];
    for &(nid, begin, len) in &grown.leaf_segments {
        assert!(tree.nodes[nid].is_leaf(), "segment for internal node {nid}");
        for &rid in &grown.ridx[begin as usize..(begin + len) as usize] {
            seen[rid as usize] += 1;
            // Walking the tree must land the row in the same leaf.
            let (indices, values) = d.row(rid as usize);
            let walked = tree.leaf_index(|f| {
                indices.iter().position(|&i| i == f).map(|k| values[k])
            });
            assert_eq!(walked, nid, "row {rid} walks to {walked}, segment says {nid}");
        }
    }
    assert!(seen.iter().all(|&c| c == 1), "every row belongs to exactly one leaf");
}
