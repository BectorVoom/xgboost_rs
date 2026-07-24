//! Gradient quantisation, ported from `xgboost/src/tree/gpu_hist/quantiser.{cu,cuh}`
//! and `xgboost/src/common/deterministic.cuh`.
//!
//! Gradients are converted to a fixed-point `i64` representation so that the
//! histogram sum is deterministic regardless of the order in which atomics
//! commit. The scale factor is derived from a reproducible rounding factor
//! (Demmel & Nguyen, "Fast Reproducible Floating-Point Summation", alg. 5).

use cubecl::prelude::*;

use super::{GradientPair, GradientPairInt64, GradientPairPrecise};

/// Port of `common::CreateRoundingFactor<T>` (deterministic.cuh).
///
/// Returns `M = 2 ^ ceil(log2(delta))` where
/// `delta = max_abs / (1 - 2 * n * eps)`.
pub fn create_rounding_factor(max_abs: f64, n: u64) -> f64 {
    let delta = max_abs / (1.0 - 2.0 * n as f64 * f64::EPSILON);
    2.0f64.powi(frexp_exp(delta))
}

/// Exponent `exp` such that `x = m * 2^exp` with `|m| in [0.5, 1)`, i.e. the
/// exponent output of C `frexp`. Returns 0 for zero (as `frexp` does).
fn frexp_exp(x: f64) -> i32 {
    if x == 0.0 || !x.is_finite() {
        return 0;
    }
    let bits = x.abs().to_bits();
    let biased = ((bits >> 52) & 0x7ff) as i32;
    if biased != 0 {
        biased - 1022
    } else {
        // Subnormal: highest set mantissa bit p gives x in [2^(p-1074), 2^(p-1073)).
        let p = 63 - (bits & ((1u64 << 52) - 1)).leading_zeros() as i32;
        p - 1073
    }
}

/// Port of `tree::GradientQuantiser` (quantiser.cuh): per-target conversion
/// factors between floating-point and fixed-point gradients.
#[derive(Clone, Copy, Debug)]
pub struct GradientQuantiser {
    pub to_fixed_point: GradientPairPrecise,
    pub to_floating_point: GradientPairPrecise,
}

impl GradientQuantiser {
    /// Port of `BuildQuantiserFromPair` (quantiser.cu). The bound used for the
    /// rounding factor is `max(sum of positive values, |sum of negative values|)`
    /// per channel, rather than `max|v| * n`, to avoid outlier sensitivity.
    pub fn new(gpairs: &[GradientPair], total_rows: u64) -> Self {
        // Port of the `Clip` functor reduction (positive / negative sums).
        let mut pos = GradientPairPrecise::default();
        let mut neg = GradientPairPrecise::default();
        for g in gpairs {
            pos.grad += f64::from(g.grad.max(0.0));
            pos.hess += f64::from(g.hess.max(0.0));
            neg.grad += f64::from((-g.grad).max(0.0));
            neg.hess += f64::from((-g.hess).max(0.0));
        }

        let rounding_grad = create_rounding_factor(pos.grad.max(neg.grad), total_rows);
        let rounding_hess = create_rounding_factor(pos.hess.max(neg.hess), total_rows);

        // Keep 1 bit for the sign: scale the rounding factor down by 2^62.
        let divisor = (1i64 << 62) as f64;
        let to_floating_point =
            GradientPairPrecise { grad: rounding_grad / divisor, hess: rounding_hess / divisor };
        let to_fixed_point = GradientPairPrecise {
            grad: 1.0 / to_floating_point.grad,
            hess: 1.0 / to_floating_point.hess,
        };
        Self { to_fixed_point, to_floating_point }
    }

    /// Port of `GradientQuantiser::ToFixedPoint` (host reference).
    pub fn to_fixed_point(&self, g: GradientPair) -> GradientPairInt64 {
        let grad = (f64::from(g.grad) * self.to_fixed_point.grad) as i64;
        let mut hess = (f64::from(g.hess) * self.to_fixed_point.hess) as i64;
        // Preserve positive curvature through fixed-point truncation.
        if g.hess > 0.0 && hess == 0 {
            hess = 1;
        }
        GradientPairInt64 { grad, hess }
    }

    /// Port of `GradientQuantiser::ToFloatingPoint`.
    pub fn to_floating_point(&self, g: GradientPairInt64) -> GradientPairPrecise {
        GradientPairPrecise {
            grad: g.grad as f64 * self.to_floating_point.grad,
            hess: g.hess as f64 * self.to_floating_point.hess,
        }
    }
}

/// Device kernel: port of the elementwise conversion in `CalcQuantizedGpairs`
/// (quantiser.cu), i.e. `GradientQuantiser::ToFixedPointImpl` applied per row.
///
/// `gpair` holds interleaved `[grad, hess]` `f32` values, `out` interleaved
/// `[grad, hess]` `i64` values.
#[cube(launch)]
pub fn quantise_gpair_kernel(
    gpair: &Array<f32>,
    out: &mut Array<i64>,
    to_fixed_grad: f64,
    to_fixed_hess: f64,
    n: u32,
) {
    let i = ABSOLUTE_POS as u32;
    if i < n {
        let grad_f = f64::cast_from(gpair[(2 * i) as usize]);
        let hess_f = f64::cast_from(gpair[(2 * i + 1) as usize]);

        let grad = i64::cast_from(grad_f * to_fixed_grad);
        let mut hess = i64::cast_from(hess_f * to_fixed_hess);
        // Preserve positive curvature through fixed-point truncation.
        if hess_f > 0.0 && hess == 0 {
            hess = 1;
        }

        out[(2 * i) as usize] = grad;
        out[(2 * i + 1) as usize] = hess;
    }
}

/// Launch [`quantise_gpair_kernel`] over `n` gradient pairs, leaving the
/// result on device (ready to feed `HistogramEngine::build_to_device`).
pub fn quantise_to_device<R: Runtime>(
    client: &ComputeClient<R>,
    gpairs: &[GradientPair],
    quantiser: &GradientQuantiser,
) -> super::DeviceGpairs {
    let n = gpairs.len();
    let interleaved: Vec<f32> = gpairs.iter().flat_map(|g| [g.grad, g.hess]).collect();

    let in_handle = client.create_from_slice(bytemuck::cast_slice(&interleaved));
    let out_handle = client.empty(n * core::mem::size_of::<GradientPairInt64>());

    let cube_dim = 256;
    let cube_count = (n as u32).div_ceil(cube_dim).max(1);
    quantise_gpair_kernel::launch::<R>(
        client,
        CubeCount::Static(cube_count, 1, 1),
        CubeDim::new_1d(cube_dim),
        unsafe { ArrayArg::from_raw_parts(in_handle, 2 * n) },
        unsafe { ArrayArg::from_raw_parts(out_handle.clone(), 2 * n) },
        quantiser.to_fixed_point.grad,
        quantiser.to_fixed_point.hess,
        n as u32,
    );

    super::DeviceGpairs { handle: out_handle, n }
}

/// Convenience wrapper around [`quantise_to_device`] that reads the result
/// back to the host.
pub fn quantise<R: Runtime>(
    client: &ComputeClient<R>,
    gpairs: &[GradientPair],
    quantiser: &GradientQuantiser,
) -> Vec<GradientPairInt64> {
    let device = quantise_to_device(client, gpairs, quantiser);
    let bytes = client.read_one_unchecked(device.handle);
    bytemuck::cast_slice(&bytes).to_vec()
}
