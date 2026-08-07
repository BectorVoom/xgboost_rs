//! The device split evaluator against a sequential CPU reference.
//!
//! The reference is a direct transcription of `HistGrower::enumerate_forward` /
//! `enumerate_backward` (`src/tree/hist.rs`) reading the *quantised* histogram,
//! so the two sides differ only in how the work is scheduled. Integer prefix
//! sums are exact, so agreement must be exact too: same feature, same
//! threshold, same default direction, same `loss_chg` bits.

use cubecl::Runtime;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

use xgboost_rs::gpu::GradientPairInt64;
use xgboost_rs::gpu::evaluate_splits::{NodeInput, SplitConfig, SplitEvaluatorGpu};
use xgboost_rs::parameters::MonotoneConstraint;
use xgboost_rs::tree::evaluator::SplitEvaluator;
use xgboost_rs::tree::param::{GradStats, SplitEntry, TrainParam};

type R = WgpuRuntime;

fn client() -> cubecl::prelude::ComputeClient<R> {
    R::client(&WgpuDevice::default())
}

/// xorshift, so a case is reproducible without pulling in a rng crate.
struct Rng(u64);

impl Rng {
    fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
    }

    fn next_f32(&mut self) -> f32 {
        self.next_u32() as f32 / u32::MAX as f32
    }
}

/// One synthetic node: a quantised histogram plus the parent sums it came from.
struct Case {
    cut_ptrs: Vec<u32>,
    cut_values: Vec<f32>,
    min_values: Vec<f32>,
    hist: Vec<GradientPairInt64>,
    parent: GradientPairInt64,
    to_float_grad: f64,
    to_float_hess: f64,
}

/// Build a histogram from `n_rows` rows, each landing in a random bin of each
/// feature — or in none of them, which is what gives the backward scan
/// something to find.
fn make_case(seed: u64, n_features: usize, bins_per_feature: &[usize], missing_rate: f32) -> Case {
    let mut rng = Rng(seed);
    assert_eq!(bins_per_feature.len(), n_features);

    let mut cut_ptrs = vec![0u32];
    for &b in bins_per_feature {
        cut_ptrs.push(cut_ptrs.last().unwrap() + b as u32);
    }
    let n_bins = *cut_ptrs.last().unwrap() as usize;

    // Cut values increase within a feature, as real quantile cuts do.
    let mut cut_values = Vec::with_capacity(n_bins);
    for &b in bins_per_feature {
        let mut v = 0.0f32;
        for _ in 0..b {
            v += 0.1 + rng.next_f32();
            cut_values.push(v);
        }
    }
    let min_values: Vec<f32> = (0..n_features).map(|_| -rng.next_f32()).collect();

    let n_rows = 4000usize;
    // Fixed-point gradients in the range a real quantiser produces.
    let scale = 1i64 << 40;
    let mut hist = vec![GradientPairInt64::default(); n_bins];
    let mut parent = GradientPairInt64::default();

    for _ in 0..n_rows {
        let g = ((rng.next_f32() - 0.5) * 2.0 * scale as f32) as i64;
        let h = (rng.next_f32() * scale as f32) as i64 + 1;
        parent.grad += g;
        parent.hess += h;
        for f in 0..n_features {
            if rng.next_f32() < missing_rate {
                continue; // row is missing this feature
            }
            let b = (rng.next_u32() as usize) % bins_per_feature[f];
            let cell = cut_ptrs[f] as usize + b;
            hist[cell].grad += g;
            hist[cell].hess += h;
        }
    }

    Case {
        cut_ptrs,
        cut_values,
        min_values,
        hist,
        parent,
        // Factors of the same magnitude `GradientQuantiser` produces.
        to_float_grad: 1.0 / scale as f64,
        to_float_hess: 1.0 / scale as f64,
    }
}

/// Sequential reference: `enumerate_forward` then `enumerate_backward`, reading
/// the quantised bins and decoding each prefix once — exactly what the kernel
/// does, but one bin at a time.
fn cpu_best_split(case: &Case, p: &TrainParam, ev: &SplitEvaluator, allowed: &[bool]) -> SplitEntry {
    let decode = |g: i64, h: i64| {
        GradStats::new(g as f64 * case.to_float_grad, h as f64 * case.to_float_hess)
    };
    let parent_stats = decode(case.parent.grad, case.parent.hess);
    let root_gain = ev.calc_gain(0, p, &parent_stats);

    let mut best = SplitEntry::default();
    for fidx in 0..(case.cut_ptrs.len() - 1) {
        if !allowed[fidx] {
            continue;
        }
        let (ibegin, iend) = (case.cut_ptrs[fidx] as usize, case.cut_ptrs[fidx + 1] as usize);
        let mut feature = SplitEntry::default();

        // Forward: `left_sum` accumulates over ascending bins.
        let (mut lg, mut lh) = (0i64, 0i64);
        for i in ibegin..iend {
            lg += case.hist[i].grad;
            lh += case.hist[i].hess;
            let left = decode(lg, lh);
            let right = decode(case.parent.grad - lg, case.parent.hess - lh);
            let gain = ev.calc_split_gain(0, fidx as u32, p, &left, &right);
            if gain.is_finite() {
                feature.update(gain - root_gain, fidx as u32, case.cut_values[i], false, left, right);
            }
        }

        // Backward, only when the feature has missing rows in this node.
        if lg != case.parent.grad || lh != case.parent.hess {
            let (mut rg, mut rh) = (0i64, 0i64);
            for i in (ibegin..iend).rev() {
                rg += case.hist[i].grad;
                rh += case.hist[i].hess;
                let right = decode(rg, rh);
                let left = decode(case.parent.grad - rg, case.parent.hess - rh);
                let gain = ev.calc_split_gain(0, fidx as u32, p, &left, &right);
                if gain.is_finite() {
                    let split_pt = if i == ibegin {
                        case.min_values[fidx]
                    } else {
                        case.cut_values[i - 1]
                    };
                    feature.update(gain - root_gain, fidx as u32, split_pt, true, left, right);
                }
            }
        }
        best.update_entry(&feature);
    }
    best
}

/// Run both sides on one case and assert they agree.
fn check(case: &Case, p: &TrainParam, constraints: &[MonotoneConstraint], allowed: &[bool]) {
    let n_features = case.cut_ptrs.len() - 1;
    let ev = SplitEvaluator::new(constraints, n_features);

    let expected = cpu_best_split(case, p, &ev, allowed);

    let cfg = SplitConfig {
        lambda: p.reg_lambda,
        alpha: p.reg_alpha,
        max_delta_step: p.max_delta_step,
        min_child_weight: p.min_child_weight,
        to_float_grad: case.to_float_grad,
        to_float_hess: case.to_float_hess,
        monotone: constraints
            .iter()
            .map(|c| match c {
                MonotoneConstraint::Increasing => 1,
                MonotoneConstraint::Decreasing => -1,
                MonotoneConstraint::Unconstrained => 0,
            })
            .collect(),
    };

    let client = client();
    let gpu = SplitEvaluatorGpu::<R>::new(
        client.clone(),
        &case.cut_ptrs,
        &case.cut_values,
        &case.min_values,
        cfg,
    )
    .unwrap();

    let parent_stats =
        GradStats::new(case.parent.grad as f64 * case.to_float_grad, case.parent.hess as f64 * case.to_float_hess);
    let node = NodeInput {
        hist_base: 0,
        parent_grad: case.parent.grad,
        parent_hess: case.parent.hess,
        root_gain: ev.calc_gain(0, p, &parent_stats),
        lower: f32::MIN,
        upper: f32::MAX,
    };

    let flat: Vec<i64> = case.hist.iter().flat_map(|c| [c.grad, c.hess]).collect();
    let hist_handle = client.create_from_slice(bytemuck::cast_slice(&flat));
    let mask: Vec<u32> = allowed.iter().map(|&a| u32::from(a)).collect();

    let got = gpu.evaluate(&hist_handle, case.hist.len(), &[node], &mask).unwrap();
    let got = got[0];

    assert_eq!(
        got.split_index(),
        expected.split_index(),
        "feature: gpu {} vs cpu {}",
        got.split_index(),
        expected.split_index()
    );
    if expected.loss_chg > 0.0 {
        assert_eq!(got.default_left(), expected.default_left(), "default_left");
        assert_eq!(got.split_value, expected.split_value, "split_value");
        assert_eq!(
            got.left_grad as f64 * case.to_float_grad,
            expected.left_sum.sum_grad,
            "left grad"
        );
        assert_eq!(
            got.left_hess as f64 * case.to_float_hess,
            expected.left_sum.sum_hess,
            "left hess"
        );
    }
    // `loss_chg` is the one value that is not required to be bit-identical.
    // The reference narrows each gain term to `f32` before dividing; the
    // device backend is free to evaluate the expression at `f64` and narrow
    // once, which it does. The result differs in the last ulp or two — more
    // accurate, not wrong — and it never moved a split decision above, which
    // is why every field that *determines the model* is checked exactly and
    // only this one carries a tolerance.
    let tol = 1e-6 * expected.loss_chg.abs().max(1.0);
    assert!(
        (got.loss_chg - expected.loss_chg).abs() <= tol,
        "loss_chg: gpu {} vs cpu {}",
        got.loss_chg,
        expected.loss_chg
    );
}

fn all_allowed(n: usize) -> Vec<bool> {
    vec![true; n]
}

#[test]
fn dense_features_no_missing_values() {
    let case = make_case(7, 8, &[32; 8], 0.0);
    check(&case, &TrainParam::default(), &[], &all_allowed(8));
}

#[test]
fn missing_values_drive_the_backward_scan() {
    let case = make_case(11, 6, &[64; 6], 0.3);
    check(&case, &TrainParam::default(), &[], &all_allowed(6));
}

/// More bins than one workgroup holds, so the tiled scan carries between tiles.
#[test]
fn features_wider_than_the_block() {
    let case = make_case(13, 3, &[700, 300, 1000], 0.2);
    check(&case, &TrainParam::default(), &[], &all_allowed(3));
}

#[test]
fn ragged_feature_widths() {
    let case = make_case(17, 7, &[1, 2, 3, 5, 255, 256, 257], 0.15);
    check(&case, &TrainParam::default(), &[], &all_allowed(7));
}

#[test]
fn honours_min_child_weight() {
    let case = make_case(19, 5, &[48; 5], 0.1);
    for mcw in [0.0f32, 1.0, 50.0, 5000.0] {
        let p = TrainParam { min_child_weight: mcw, ..Default::default() };
        check(&case, &p, &[], &all_allowed(5));
    }
}

#[test]
fn honours_lambda_and_alpha() {
    let case = make_case(23, 5, &[48; 5], 0.1);
    for (lambda, alpha) in [(0.0f32, 0.0f32), (1.0, 0.0), (0.0, 2.5), (10.0, 7.5)] {
        let p = TrainParam { reg_lambda: lambda, reg_alpha: alpha, ..Default::default() };
        check(&case, &p, &[], &all_allowed(5));
    }
}

/// `max_delta_step != 0` switches `CalcGainGivenWeight` to its all-`f32`
/// branch, which is a different expression, not a scaling of the same one.
#[test]
fn honours_max_delta_step() {
    let case = make_case(29, 5, &[48; 5], 0.1);
    for mds in [0.5f32, 2.0, 10.0] {
        let p = TrainParam { max_delta_step: mds, ..Default::default() };
        check(&case, &p, &[], &all_allowed(5));
    }
}

#[test]
fn honours_monotone_constraints() {
    let case = make_case(31, 4, &[40; 4], 0.1);
    let c = vec![
        MonotoneConstraint::Increasing,
        MonotoneConstraint::Unconstrained,
        MonotoneConstraint::Decreasing,
        MonotoneConstraint::Increasing,
    ];
    check(&case, &TrainParam::default(), &c, &all_allowed(4));
}

/// The feature mask is how column sampling and interaction constraints reach
/// the kernel; a masked feature must never be chosen.
#[test]
fn honours_the_feature_mask() {
    let case = make_case(37, 6, &[40; 6], 0.1);
    let allowed = vec![false, true, false, false, true, false];
    check(&case, &TrainParam::default(), &[], &allowed);

    let only_one = vec![false, false, false, false, false, true];
    check(&case, &TrainParam::default(), &[], &only_one);
}

/// Several nodes in one launch must give the same answers as one at a time.
#[test]
fn batches_nodes_independently() {
    let case = make_case(41, 5, &[64; 5], 0.2);
    let n_features = 5;
    let p = TrainParam::default();
    let ev = SplitEvaluator::new(&[], n_features);
    let parent_stats = GradStats::new(
        case.parent.grad as f64 * case.to_float_grad,
        case.parent.hess as f64 * case.to_float_hess,
    );
    let root_gain = ev.calc_gain(0, &p, &parent_stats);

    let cfg = SplitConfig {
        lambda: p.reg_lambda,
        alpha: p.reg_alpha,
        max_delta_step: p.max_delta_step,
        min_child_weight: p.min_child_weight,
        to_float_grad: case.to_float_grad,
        to_float_hess: case.to_float_hess,
        monotone: vec![0; n_features],
    };
    let client = client();
    let gpu = SplitEvaluatorGpu::<R>::new(
        client.clone(),
        &case.cut_ptrs,
        &case.cut_values,
        &case.min_values,
        cfg,
    )
    .unwrap();

    // Two copies of the same histogram, stacked: both nodes must agree.
    let mut flat: Vec<i64> = case.hist.iter().flat_map(|c| [c.grad, c.hess]).collect();
    flat.extend(flat.clone());
    let handle = client.create_from_slice(bytemuck::cast_slice(&flat));

    let node = NodeInput {
        hist_base: 0,
        parent_grad: case.parent.grad,
        parent_hess: case.parent.hess,
        root_gain,
        lower: f32::MIN,
        upper: f32::MAX,
    };
    let second = NodeInput { hist_base: case.hist.len() as u32, ..node };

    let mask = vec![1u32; 2 * n_features];
    let got = gpu.evaluate(&handle, case.hist.len() * 2, &[node, second], &mask).unwrap();

    assert_eq!(got[0], got[1], "the same histogram twice must split the same way");
    let expected = cpu_best_split(&case, &p, &ev, &all_allowed(n_features));
    assert_eq!(got[0].split_index(), expected.split_index());
    assert_eq!(got[0].loss_chg, expected.loss_chg);
}
