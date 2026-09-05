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
        Self::new_multi(&[gpairs], total_rows)
    }

    /// The quantiser a vector-leaf tree shares across its targets.
    ///
    /// One scale has to serve every target, because a histogram bin is decoded
    /// by a single pair of factors whichever target it belongs to.
    ///
    /// The bound is the *sum* of the per-column bounds, not the largest of
    /// them, even though a bin only ever accumulates one target's gradients.
    /// That is what keeps the fixed point addable across targets, which a
    /// vector-leaf tree needs: the bookkeeping it still states per node — the
    /// node's cover, the summed child sums a candidate carries — adds the
    /// targets together in `i64`, and each target on its own is already scaled
    /// to fill the 62 bits below the sign. Bounding by the largest column
    /// overflows that sum from three targets up. The cost is `log2(n_targets)`
    /// bits of a 62-bit fixed point.
    ///
    /// With one column this is exactly [`Self::new`], which is why that is
    /// written in terms of it.
    pub fn new_multi(columns: &[&[GradientPair]], total_rows: u64) -> Self {
        // Port of the `Clip` functor reduction (positive / negative sums).
        let mut bound = GradientPairPrecise::default();
        for gpairs in columns {
            let mut pos = GradientPairPrecise::default();
            let mut neg = GradientPairPrecise::default();
            for g in *gpairs {
                pos.grad += f64::from(g.grad.max(0.0));
                pos.hess += f64::from(g.hess.max(0.0));
                neg.grad += f64::from((-g.grad).max(0.0));
                neg.hess += f64::from((-g.hess).max(0.0));
            }
            bound.grad += pos.grad.max(neg.grad);
            bound.hess += pos.hess.max(neg.hess);
        }
        Self::from_bound(bound, total_rows)
    }

    /// The factors for a given `Clip` bound: the tail of `new_multi`, shared
    /// with the device reduction.
    fn from_bound(bound: GradientPairPrecise, total_rows: u64) -> Self {
        let rounding_grad = create_rounding_factor(bound.grad, total_rows);
        let rounding_hess = create_rounding_factor(bound.hess, total_rows);

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
#[cube(launch_unchecked)]
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

    let (cube_count, cube_dim) = super::launch::elementwise(client, n);
    // SAFETY: the kernel guards every index against the lengths it is
    // given; see the `gpu` module docs on unchecked launches.
    unsafe {
        quantise_gpair_kernel::launch_unchecked::<R>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(in_handle, 2 * n),
            ArrayArg::from_raw_parts(out_handle.clone(), 2 * n),
            quantiser.to_fixed_point.grad,
            quantiser.to_fixed_point.hess,
            n as u32,
        );
    }

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

// ------------------------------------------------------ device quantiser ----
//
// The host quantiser is three passes over every gradient — the bound, the
// conversion, the sum — and an 8 MB upload of the `i64` result. On a cloud
// VM's cores that was 9 ms of a 30 ms round at 500 000 rows and 44 ms of 90
// at a million: a third of the tree, before a histogram was built. Upstream
// quantises on the device (`CalcQuantizedGpairs`), and so does this: 4 MB of
// `f32` goes up, and the three passes are three kernels plus two readbacks
// of a few kilobytes of partial sums.

/// Cubes a device reduction is spread over. Enough that a million rows are
/// a few hundred per unit, few enough that the partials read back in one
/// page.
const REDUCE_CUBES: u32 = 1024;

/// Per-cube partial sums of the `Clip` reduction behind
/// [`GradientQuantiser::new`]: `[+grad, +hess, -grad, -hess]` over the cube's
/// rows, positive parts and negated negative parts, in `f64`.
///
/// Rows `cube * chunk ..` are the cube's; a unit walks them from its own
/// index in strides of the cube width and the units' sums join in a tree, so
/// the partial is a function of the geometry alone and the host, adding the
/// partials in cube order, gets the same bound run after run.
#[cube(launch_unchecked)]
pub fn gpair_bounds_kernel(
    gpair: &Array<f32>,
    partials: &mut Array<f64>,
    n: u32,
    chunk: u32,
    #[comptime] block: usize,
) {
    let mut s = SharedMemory::<f64>::new(block * 4usize);
    let t = UNIT_POS_X as usize;
    let begin = CUBE_POS_X * chunk;
    let mut end = begin + chunk;
    if end > n {
        end = n;
    }
    let pg = RuntimeCell::<f64>::new(0.0f64);
    let ph = RuntimeCell::<f64>::new(0.0f64);
    let ng = RuntimeCell::<f64>::new(0.0f64);
    let nh = RuntimeCell::<f64>::new(0.0f64);
    let i = RuntimeCell::<u32>::new(begin + UNIT_POS_X);
    while i.read() < end {
        let g = f64::cast_from(gpair[(2u32 * i.read()) as usize]);
        let h = f64::cast_from(gpair[(2u32 * i.read() + 1u32) as usize]);
        if g > 0.0 {
            pg.store(pg.read() + g);
        } else {
            ng.store(ng.read() - g);
        }
        if h > 0.0 {
            ph.store(ph.read() + h);
        } else {
            nh.store(nh.read() - h);
        }
        i.store(i.read() + CUBE_DIM_X);
    }
    s[t * 4usize] = pg.read();
    s[t * 4usize + 1] = ph.read();
    s[t * 4usize + 2] = ng.read();
    s[t * 4usize + 3] = nh.read();
    let half = RuntimeCell::<u32>::new((block / 2usize) as u32);
    while half.read() > 0u32 {
        let d = half.read();
        sync_cube();
        if UNIT_POS_X < d {
            let b = (UNIT_POS_X + d) as usize;
            s[t * 4usize] += s[b * 4usize];
            s[t * 4usize + 1] += s[b * 4usize + 1];
            s[t * 4usize + 2] += s[b * 4usize + 2];
            s[t * 4usize + 3] += s[b * 4usize + 3];
        }
        half.store(d / 2u32);
    }
    sync_cube();
    if UNIT_POS_X == 0u32 {
        let c = CUBE_POS_X as usize;
        partials[c * 4usize] = s[0usize];
        partials[c * 4usize + 1] = s[1usize];
        partials[c * 4usize + 2] = s[2usize];
        partials[c * 4usize + 3] = s[3usize];
    }
    // A runtime that runs cubes one after another on one shared buffer must
    // not let a unit start the next cube's sums under this one's read.
    sync_cube();
}

/// Per-cube partial sums of quantised pairs, exact in `i64`: the node sum of
/// the root, which the host used to fold before the upload.
#[cube(launch_unchecked)]
pub fn gpair_sum_kernel(
    gpair: &Array<i64>,
    partials: &mut Array<i64>,
    n: u32,
    chunk: u32,
    #[comptime] block: usize,
) {
    let mut s = SharedMemory::<i64>::new(block * 2usize);
    let t = UNIT_POS_X as usize;
    let begin = CUBE_POS_X * chunk;
    let mut end = begin + chunk;
    if end > n {
        end = n;
    }
    let g = RuntimeCell::<i64>::new(0i64);
    let h = RuntimeCell::<i64>::new(0i64);
    let i = RuntimeCell::<u32>::new(begin + UNIT_POS_X);
    while i.read() < end {
        g.store(g.read() + gpair[(2u32 * i.read()) as usize]);
        h.store(h.read() + gpair[(2u32 * i.read() + 1u32) as usize]);
        i.store(i.read() + CUBE_DIM_X);
    }
    s[t * 2usize] = g.read();
    s[t * 2usize + 1] = h.read();
    let half = RuntimeCell::<u32>::new((block / 2usize) as u32);
    while half.read() > 0u32 {
        let d = half.read();
        sync_cube();
        if UNIT_POS_X < d {
            let b = (UNIT_POS_X + d) as usize;
            s[t * 2usize] += s[b * 2usize];
            s[t * 2usize + 1] += s[b * 2usize + 1];
        }
        half.store(d / 2u32);
    }
    sync_cube();
    if UNIT_POS_X == 0u32 {
        let c = CUBE_POS_X as usize;
        partials[c * 2usize] = s[0usize];
        partials[c * 2usize + 1] = s[1usize];
    }
    sync_cube();
}

/// Geometry of the two reductions: a plane-aligned cube (one unit on a
/// plane-less runtime, where the tree reduction then degenerates to the
/// unit's own loop), and rows per cube.
fn reduce_geometry<R: Runtime>(client: &ComputeClient<R>, n: usize) -> (u32, u32, u32) {
    let block = super::launch::scan_block_1d(client, 256);
    let cubes = (n as u32).div_ceil(block * 16).clamp(1, REDUCE_CUBES);
    let chunk = (n as u32).div_ceil(cubes).max(1);
    (block, cubes, chunk)
}

/// A gradient column on the device as interleaved `[grad, hess]` `f32`.
#[derive(Clone, Debug)]
pub struct DeviceGpairsF32 {
    pub(crate) handle: cubecl::server::Handle,
    pub(crate) n: usize,
}

impl DeviceGpairsF32 {
    /// Upload one column. A gradient pair is two `f32`s, which is the
    /// interleaved layout the kernels read, so the slice goes up as it is.
    pub fn upload<R: Runtime>(client: &ComputeClient<R>, gpairs: &[GradientPair]) -> Self {
        let pairs: Vec<[f32; 2]> = gpairs.iter().map(|g| [g.grad, g.hess]).collect();
        Self::upload_pairs(client, pairs)
    }

    /// [`Self::upload`] for the objective's pair type, which is the same two
    /// floats.
    pub fn upload_objective<R: Runtime>(
        client: &ComputeClient<R>,
        gpairs: &[crate::objective::GradientPair],
    ) -> Self {
        let pairs: Vec<[f32; 2]> = gpairs.iter().map(|g| [g.grad, g.hess]).collect();
        Self::upload_pairs(client, pairs)
    }

    /// Through the pinned pool (`tables::upload_vec`): 4 MB a round.
    fn upload_pairs<R: Runtime>(client: &ComputeClient<R>, pairs: Vec<[f32; 2]>) -> Self {
        let n = pairs.len();
        Self { handle: super::tables::upload_vec(client, pairs), n }
    }
}

impl GradientQuantiser {
    /// [`Self::new_multi`] with the bound reduced on the device: the same
    /// `Clip` sums, taken by [`gpair_bounds_kernel`] and joined on the host in
    /// cube order.
    pub fn new_on_device<R: Runtime>(
        client: &ComputeClient<R>,
        columns: &[DeviceGpairsF32],
        total_rows: u64,
    ) -> Self {
        let mut bound = GradientPairPrecise::default();
        for col in columns {
            let (block, cubes, chunk) = reduce_geometry(client, col.n);
            let partials = client.empty(cubes as usize * 4 * size_of::<f64>());
            // SAFETY: the kernel's row loop is bounded by `n` and its cube
            // index by the grid, which is `cubes` wide.
            unsafe {
                gpair_bounds_kernel::launch_unchecked::<R>(
                    client,
                    CubeCount::Static(cubes, 1, 1),
                    CubeDim::new_1d(block),
                    ArrayArg::from_raw_parts(col.handle.clone(), col.n * 2),
                    ArrayArg::from_raw_parts(partials.clone(), cubes as usize * 4),
                    col.n as u32,
                    chunk,
                    block as usize,
                );
            }
            let bytes = client.read_one_unchecked(partials);
            let words: &[f64] = bytemuck::cast_slice(&bytes);
            let mut pos = GradientPairPrecise::default();
            let mut neg = GradientPairPrecise::default();
            for w in words.chunks_exact(4) {
                pos.grad += w[0];
                pos.hess += w[1];
                neg.grad += w[2];
                neg.hess += w[3];
            }
            bound.grad += pos.grad.max(neg.grad);
            bound.hess += pos.hess.max(neg.hess);
        }
        Self::from_bound(bound, total_rows)
    }
}

/// [`quantise_to_device`] for a column already on the device.
pub fn quantise_device_column<R: Runtime>(
    client: &ComputeClient<R>,
    column: &DeviceGpairsF32,
    quantiser: &GradientQuantiser,
) -> super::DeviceGpairs {
    let n = column.n;
    let out_handle = client.empty(n.max(1) * core::mem::size_of::<GradientPairInt64>());
    let (cube_count, cube_dim) = super::launch::elementwise(client, n);
    // SAFETY: the kernel guards every index against `n`.
    unsafe {
        quantise_gpair_kernel::launch_unchecked::<R>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(column.handle.clone(), 2 * n),
            ArrayArg::from_raw_parts(out_handle.clone(), 2 * n),
            quantiser.to_fixed_point.grad,
            quantiser.to_fixed_point.hess,
            n as u32,
        );
    }
    super::DeviceGpairs { handle: out_handle, n }
}

/// The exact sum of a quantised column, reduced on the device.
pub fn sum_device_column<R: Runtime>(
    client: &ComputeClient<R>,
    column: &super::DeviceGpairs,
) -> GradientPairInt64 {
    let (block, cubes, chunk) = reduce_geometry(client, column.n);
    let partials = client.empty(cubes as usize * 2 * size_of::<i64>());
    // SAFETY: as for `gpair_bounds_kernel`.
    unsafe {
        gpair_sum_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(block),
            ArrayArg::from_raw_parts(column.handle.clone(), column.n * 2),
            ArrayArg::from_raw_parts(partials.clone(), cubes as usize * 2),
            column.n as u32,
            chunk,
            block as usize,
        );
    }
    let bytes = client.read_one_unchecked(partials);
    let words: &[i64] = bytemuck::cast_slice(&bytes);
    words.chunks_exact(2).fold(GradientPairInt64::default(), |a, w| {
        GradientPairInt64 { grad: a.grad + w[0], hess: a.hess + w[1] }
    })
}
