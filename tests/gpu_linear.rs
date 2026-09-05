//! `gblinear` with `device=cuda`, end to end through `api::train`.
//!
//! Unlike the tree path, the bar here is **bit-identical weights**, not a
//! tolerance. Coordinate descent's only parallel step is a sum over a column,
//! and both devices fold that sum in the same fixed 4096-entry blocks and add
//! the block totals in block order — so the device is not merely close to the
//! CPU, it computes the same `f64`. Anything less would mean `device` silently
//! changes the model.
//!
//! Note the runtime: without the `cuda` feature these run on wgpu/Vulkan, so
//! `device=cuda` selects the *GPU code path* rather than a CUDA device.

use xgboost_rs::parameters::{
    BoosterParameters, BoosterType, Device, FeatureSelector, GeneralParameters,
    LearningTaskParameters, LinearBoosterParameters, LinearUpdater, Objective, TrainingParameters,
    VerboseEval, Verbosity,
};
use xgboost_rs::{DMatrix, api};

/// A matrix wide enough that several features compete, and long enough to span
/// more than one 4096-entry reduction block — which is the case where the fold
/// order could differ if either side got it wrong.
fn data(rows: usize, cols: usize, sparsity: f32, seed: u64) -> DMatrix {
    let mut st = seed | 1;
    let mut next = || {
        st ^= st >> 12;
        st ^= st << 25;
        st ^= st >> 27;
        ((st.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32) as f32 / u32::MAX as f32
    };
    let x: Vec<f32> = (0..rows * cols)
        .map(|_| if next() < sparsity { f32::NAN } else { next() * 4.0 - 2.0 })
        .collect();
    let y: Vec<f32> = (0..rows)
        .map(|r| {
            x[r * cols..(r + 1) * cols]
                .iter()
                .enumerate()
                .filter(|(_, v)| v.is_finite())
                .map(|(c, v)| v * (1.0 + c as f32) / 3.0)
                .sum()
        })
        .collect();
    let mut d = DMatrix::from_dense(&x, rows, cols, f32::NAN).unwrap();
    d.set_labels(&y).unwrap();
    d
}

fn params(device: Device, linear: LinearBoosterParameters, rounds: u32) -> TrainingParameters {
    TrainingParameters {
        booster: BoosterParameters {
            booster: BoosterType::Gblinear(linear),
            general: GeneralParameters {
                device,
                verbosity: Verbosity::Silent,
                ..Default::default()
            },
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

/// The model's weight vector, straight out of the saved model.
fn weights(booster: &xgboost_rs::Booster) -> Vec<f32> {
    let m: serde_json::Value = serde_json::from_str(&booster.save_model()).unwrap();
    m["learner"]["gradient_booster"]["model"]["weights"]
        .as_array()
        .expect("a gblinear model saves a flat weight vector")
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}

/// Fit `linear` both ways and require the two models to be identical.
fn compare_exact(dmat: &DMatrix, linear: LinearBoosterParameters, rounds: u32) {
    if !has_f64() {
        return;
    }
    let (cpu, _) = api::train(&params(Device::Cpu, linear.clone(), rounds), dmat, &[]).unwrap();
    let (gpu, _) = api::train(&params(Device::cuda(0), linear, rounds), dmat, &[]).unwrap();

    let (cw, gw) = (weights(&cpu), weights(&gpu));
    assert_eq!(gw.len(), cw.len(), "weight vector length");
    for (i, (g, c)) in gw.iter().zip(&cw).enumerate() {
        assert_eq!(g.to_bits(), c.to_bits(), "weight[{i}]: gpu {g} != cpu {c}");
    }
    assert_eq!(gpu.predict(dmat), cpu.predict(dmat), "predictions");
}

fn coord(selector: FeatureSelector) -> LinearBoosterParameters {
    LinearBoosterParameters::builder()
        .updater(LinearUpdater::CoordDescent)
        .feature_selector(selector)
        .build()
        .unwrap()
}

/// The one reduction the whole claim rests on, isolated: a column's
/// `(g·x, h·x²)` must come back as the same `f64` the CPU computes, for every
/// column of a matrix wide and long enough to span several blocks.
///
/// Checked separately from a fit because a fit only ever shows the *sum* of
/// every difference; this says which side of the round is wrong when one is.
/// The split gain arithmetic, the quantiser and the linear solver are ports of
/// XGBoost's `double`, so a backend with no `f64` cannot run them and refuses
/// at construction with `Error::NoF64Support`. Metal is that backend: MSL has
/// no `double` at all. Nothing to compare there — see
/// `xgboost_rs::gpu::supports_f64`.
fn has_f64() -> bool {
    xgboost_rs::gpu::supports_f64(&xgboost_rs::gpu::default_client(0))
}

#[test]
fn the_column_reduction_is_bit_identical() {
    if !has_f64() {
        return;
    }
    use xgboost_rs::data::csc::CscPages;
    use xgboost_rs::gpu::linear::GpuLinear;
    use xgboost_rs::linear::coordinate::column_gradient;
    use xgboost_rs::objective::GradientPair;

    for (rows, cols, sparsity, seed) in [(9000usize, 4usize, 0.0f32, 23u64), (3000, 6, 0.3, 29)] {
        let d = data(rows, cols, sparsity, seed);
        let pages = CscPages::build(&d, None, false);

        // Gradients with the same shape a real round has: signed, and a few
        // rows excluded by a negative hessian.
        let gpair: Vec<GradientPair> = (0..rows)
            .map(|r| {
                let g = ((r % 97) as f32 - 48.0) / 7.0;
                let h = if r % 251 == 0 { -1.0 } else { 1.0 + (r % 13) as f32 / 4.0 };
                GradientPair { grad: g, hess: h }
            })
            .collect();

        let mut gpu = GpuLinear::new(xgboost_rs::gpu::default_client(0), &pages, rows, 1);
        gpu.upload_gpair(&gpair);

        for fidx in 0..cols {
            let want = column_gradient(pages.iter().next().unwrap(), fidx, 0, 1, &gpair);
            let got = gpu.column_gradient(0, fidx, 0);
            assert_eq!(
                (got.0.to_bits(), got.1.to_bits()),
                (want.0.to_bits(), want.1.to_bits()),
                "{rows}x{cols} sparsity {sparsity}, feature {fidx}: gpu {got:?} != cpu {want:?}"
            );
        }
    }
}

#[test]
#[ignore = "diagnostic, not an assertion"]
fn probe_f32_rounding() {
    use cubecl::prelude::*;
    use xgboost_rs::gpu::DefaultRuntime;
    use xgboost_rs::gpu::linear::probe_kernel;

    let client = xgboost_rs::gpu::default_client(0);
    let (h, v, dw, g) = (1.25f32, 0.820_967_7f32, 0.37f32, -7.0f32);
    let out = client.empty(6 * size_of::<f32>());
    let scratch = client.empty(2 * size_of::<f32>());
    probe_kernel::launch::<DefaultRuntime>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { ArrayArg::from_raw_parts(out.clone(), 6) },
        unsafe { ArrayArg::from_raw_parts(scratch, 2) },
        h,
        v,
        dw,
        g,
    );
    let bytes = client.read_one_unchecked(out);
    let got: &[f32] = bytemuck::cast_slice(&bytes);

    let cpu = g + h * v * dw;
    let wide = (g as f64 + h as f64 * v as f64 * dw as f64) as f32;
    println!("cpu   {cpu:>14} {:>12}", cpu.to_bits());
    println!("wide  {wide:>14} {:>12}", wide.to_bits());
    for (i, x) in got.iter().enumerate() {
        println!("out[{i}] {x:>14} {:>12}  {}", x.to_bits(), if *x == cpu { "== cpu" } else { "" });
    }
}

/// The other half of a coordinate step: correcting a column's residuals must
/// leave every gradient bit-identical to the CPU's, in `f32`.
#[test]
fn the_residual_update_is_bit_identical() {
    if !has_f64() {
        return;
    }
    use xgboost_rs::data::csc::CscPages;
    use xgboost_rs::gpu::linear::GpuLinear;
    use xgboost_rs::linear::coordinate::update_residual;
    use xgboost_rs::objective::GradientPair;

    let (rows, cols) = (5000usize, 4usize);
    let d = data(rows, cols, 0.2, 31);
    let pages = CscPages::build(&d, None, false);
    let page = pages.iter().next().unwrap();

    let mut want: Vec<GradientPair> = (0..rows)
        .map(|r| GradientPair {
            grad: ((r % 89) as f32 - 44.0) / 3.0,
            hess: if r % 311 == 0 { -1.0 } else { 0.5 + (r % 17) as f32 / 8.0 },
        })
        .collect();

    let mut gpu = GpuLinear::new(xgboost_rs::gpu::default_client(0), &pages, rows, 1);
    gpu.upload_gpair(&want);

    // One column at a time, so a difference is attributed to the step that
    // caused it rather than to the accumulation of four.
    for (fidx, dw) in (0..cols).zip([0.37f32, -1.25, 0.008_137, 12.5]) {
        let before = want.clone();
        update_residual(page, fidx, 0, 1, dw, &mut want);
        gpu.update_residual(0, fidx, 0, dw);

        let (col_rows, col_values) = page.column(fidx);
        let mut got = vec![GradientPair::default(); rows];
        gpu.download_gpair(&mut got);
        for (r, (g, w)) in got.iter().zip(&want).enumerate() {
            let v = col_rows.iter().position(|&x| x as usize == r).map(|k| col_values[k]);
            assert_eq!(
                g.grad.to_bits(),
                w.grad.to_bits(),
                "feature {fidx}, dw {dw}, row {r}: gpu {} != cpu {} \
                 (from g {} h {} v {v:?})",
                g.grad,
                w.grad,
                before[r].grad,
                before[r].hess
            );
        }
    }
}

#[test]
fn coord_descent_matches_the_cpu_fit_exactly() {
    let d = data(2000, 8, 0.0, 3);
    compare_exact(&d, coord(FeatureSelector::Cyclic), 1);
    compare_exact(&d, coord(FeatureSelector::Cyclic), 6);
}

/// More than one reduction block per column, which is what the fixed block
/// size is *for*: with 12000 rows every column spans three blocks, so a device
/// that folded them in any other order would show up here.
#[test]
fn matches_when_a_column_spans_several_reduction_blocks() {
    let d = data(12_000, 4, 0.0, 5);
    compare_exact(&d, coord(FeatureSelector::Cyclic), 3);
}

#[test]
fn matches_with_missing_values() {
    let d = data(3000, 6, 0.3, 7);
    compare_exact(&d, coord(FeatureSelector::Cyclic), 4);
}

/// Every feature selector, including the two that score features by the
/// residual gradients and so make the device hand them back mid-round.
#[test]
fn every_feature_selector_matches() {
    let d = data(1500, 6, 0.0, 11);
    for selector in [
        FeatureSelector::Cyclic,
        FeatureSelector::Shuffle,
        FeatureSelector::Random,
        FeatureSelector::Greedy,
        FeatureSelector::Thrifty,
    ] {
        compare_exact(&d, coord(selector), 3);
    }
}

#[test]
fn matches_with_regularisation() {
    let d = data(2000, 5, 0.0, 13);
    for (alpha, lambda) in [(0.0f32, 1.0f32), (0.5, 0.0), (0.5, 2.0)] {
        let p = LinearBoosterParameters::builder()
            .updater(LinearUpdater::CoordDescent)
            .alpha(alpha)
            .lambda(lambda)
            .build()
            .unwrap();
        compare_exact(&d, p, 4);
    }
}

/// Several output groups: the gradients interleave `(row, group)` and each
/// group's descent must read its own stride.
#[test]
fn matches_with_several_output_groups() {
    if !has_f64() {
        return;
    }
    let d = {
        let mut d = data(1200, 5, 0.0, 17);
        let y: Vec<f32> = d.info().labels.iter().map(|v| (v.abs() as u32 % 3) as f32).collect();
        d.set_labels(&y).unwrap();
        d
    };
    let mut p = params(Device::Cpu, coord(FeatureSelector::Cyclic), 3);
    p.booster.learning.objective = Objective::MultiSoftprob { num_class: 3 };
    let (cpu, _) = api::train(&p, &d, &[]).unwrap();

    p.booster.general.device = Device::cuda(0);
    let (gpu, _) = api::train(&p, &d, &[]).unwrap();

    let (cw, gw) = (weights(&cpu), weights(&gpu));
    assert_eq!(gw.len(), cw.len());
    for (i, (g, c)) in gw.iter().zip(&cw).enumerate() {
        assert_eq!(g.to_bits(), c.to_bits(), "weight[{i}]: gpu {g} != cpu {c}");
    }
}

/// `shotgun` has no device implementation. It is accepted — upstream accepts
/// it too — and runs on the CPU, but the configuration says so rather than
/// leaving `device` looking as though it did something.
#[test]
fn shotgun_runs_on_the_cpu_and_says_so() {
    let d = data(800, 4, 0.0, 19);
    let shotgun = LinearBoosterParameters::builder()
        .updater(LinearUpdater::Shotgun)
        .build()
        .unwrap();

    let cuda = params(Device::cuda(0), shotgun.clone(), 3);
    let warnings = cuda.booster.warnings();
    assert!(
        warnings.iter().any(|w| w.contains("shotgun") && w.contains("CPU")),
        "a device `shotgun` fit must warn that it runs on the CPU, got {warnings:?}"
    );

    let (cpu, _) = api::train(&params(Device::Cpu, shotgun, 3), &d, &[]).unwrap();
    let (gpu, _) = api::train(&cuda, &d, &[]).unwrap();
    assert_eq!(weights(&gpu), weights(&cpu), "it is the same CPU fit either way");
}
