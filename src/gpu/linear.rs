//! `gblinear`'s coordinate descent on the device, ported from
//! `xgboost/src/linear/updater_gpu_coordinate.cu` (`GPUCoordinateUpdater`).
//!
//! Coordinate descent is a sequence of two O(nnz) passes over one column:
//! total `(g·x, h·x²)` for the column, then — once the step is known — correct
//! every one of that column's residual gradients by it. Both are what this
//! module runs on the device; the step itself is three lines of arithmetic on
//! two scalars and stays on the host, exactly as upstream keeps
//! `CoordinateDelta` on the host.
//!
//! # Why the answer is bit-identical to the CPU's
//!
//! A parallel sum has to fix its association order or the answer depends on
//! the hardware. [`crate::linear::coordinate::column_gradient`] already does:
//! it folds the column in fixed [`LIN_BLOCK`]-entry blocks and then adds the
//! block totals in block order. This module gives each *thread* one such
//! block, to sum serially in the same order, and folds the block totals on the
//! host in the same order — so the device produces the identical `f64`, not
//! merely a close one. The residual update has no such freedom: a column holds
//! each row at most once, so every thread writes a different gradient.
//!
//! That is what lets `device=cuda` and `device=cpu` be held to the same fit
//! rather than to a tolerance, and it is why this file exists at all instead of
//! a `reduce`.

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::data::csc::{CscPage, CscPages};
use crate::objective::GradientPair;

/// Entries folded per thread.
///
/// Must stay equal to `linear::coordinate::BLOCK`: the two split the same
/// column the same way, which is the whole reason the sums agree bit for bit.
pub const LIN_BLOCK: u32 = 4096;

/// Threads per workgroup for the elementwise kernels.
const DIM: u32 = 256;

/// Sum `(g·x, h·x²)` over one block of one column.
///
/// One thread per block of [`LIN_BLOCK`] entries; `out` is interleaved
/// `[grad, hess]` per block, to be folded in block order by the host.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn column_sums_kernel(
    row_idx: &Array<u32>,
    values: &Array<f32>,
    gpair: &Array<f32>,
    out: &mut Array<f64>,
    begin: u32,
    len: u32,
    group: u32,
    n_groups: u32,
    block: u32,
) {
    let b = ABSOLUTE_POS as u32;
    let lo = b * block;
    if lo < len {
        let hi = if lo + block < len { lo + block } else { len };
        let sum_grad = RuntimeCell::<f64>::new(0.0f64);
        let sum_hess = RuntimeCell::<f64>::new(0.0f64);
        let k = RuntimeCell::<u32>::new(lo);
        while k.read() < hi {
            let i = (begin + k.read()) as usize;
            let slot = (2u32 * (row_idx[i] * n_groups + group)) as usize;
            let h = gpair[slot + 1usize];
            // A negative hessian is the objective's marker for a row it has
            // excluded, and is skipped rather than summed.
            if h >= 0.0f32 {
                let v = f64::cast_from(values[i]);
                sum_grad.store(sum_grad.read() + f64::cast_from(gpair[slot]) * v);
                sum_hess.store(sum_hess.read() + f64::cast_from(h) * v * v);
            }
            k.store(k.read() + 1u32);
        }
        out[(2u32 * b) as usize] = sum_grad.read();
        out[(2u32 * b + 1u32) as usize] = sum_hess.read();
    }
}

/// The same fold over every row, which is the intercept's "column".
#[cube(launch)]
pub fn bias_sums_kernel(
    gpair: &Array<f32>,
    out: &mut Array<f64>,
    n_rows: u32,
    group: u32,
    n_groups: u32,
    block: u32,
) {
    let b = ABSOLUTE_POS as u32;
    let lo = b * block;
    if lo < n_rows {
        let hi = if lo + block < n_rows { lo + block } else { n_rows };
        let sum_grad = RuntimeCell::<f64>::new(0.0f64);
        let sum_hess = RuntimeCell::<f64>::new(0.0f64);
        let r = RuntimeCell::<u32>::new(lo);
        while r.read() < hi {
            let slot = (2u32 * (r.read() * n_groups + group)) as usize;
            let h = gpair[slot + 1usize];
            if h >= 0.0f32 {
                sum_grad.store(sum_grad.read() + f64::cast_from(gpair[slot]));
                sum_hess.store(sum_hess.read() + f64::cast_from(h));
            }
            r.store(r.read() + 1u32);
        }
        out[(2u32 * b) as usize] = sum_grad.read();
        out[(2u32 * b + 1u32) as usize] = sum_hess.read();
    }
}

/// `g += h · x · dw` for every entry of one column.
///
/// One thread per entry. A column holds each row at most once, so no two
/// threads touch the same gradient and the result does not depend on the order
/// they run in.
///
/// # Why the intermediate goes through `scratch`
///
/// The backend evaluates an `f32` expression at wider precision and narrows
/// once at the end, so `g + h * v * dw` written directly is *not* the CPU's
/// three separately-rounded `f32` steps — it lands a ulp away, and a ulp in a
/// residual is a ulp in the next column's sum, which is how it reaches the
/// model. Neither an explicit `f32::cast_from(f64)` nor a `RuntimeCell` round
/// trip forces the narrowing; a store to an `f32` buffer does, because the
/// element type leaves the backend nowhere to keep the extra bits. That is
/// what `scratch` is: one slot per thread, written and read back between each
/// operation. `probe_f32_rounding` in `tests/gpu_linear.rs` is the experiment
/// this conclusion comes from, and is kept so it can be re-run against a new
/// CubeCL or a new backend.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn update_residual_kernel(
    row_idx: &Array<u32>,
    values: &Array<f32>,
    gpair: &mut Array<f32>,
    scratch: &mut Array<f32>,
    begin: u32,
    len: u32,
    group: u32,
    n_groups: u32,
    dw: f32,
) {
    let k = ABSOLUTE_POS as u32;
    if k < len {
        let i = (begin + k) as usize;
        let slot = (2u32 * (row_idx[i] * n_groups + group)) as usize;
        let h = gpair[slot + 1usize];
        if h >= 0.0f32 {
            let t = k as usize;
            scratch[t] = h * values[i];
            scratch[t] = scratch[t] * dw;
            gpair[slot] = gpair[slot] + scratch[t];
        }
    }
}

/// The same correction for the intercept, which every row sees.
#[cube(launch)]
pub fn update_bias_residual_kernel(
    gpair: &mut Array<f32>,
    scratch: &mut Array<f32>,
    n_rows: u32,
    group: u32,
    n_groups: u32,
    dbias: f32,
) {
    let r = ABSOLUTE_POS as u32;
    if r < n_rows {
        let slot = (2u32 * (r * n_groups + group)) as usize;
        let h = gpair[slot + 1usize];
        if h >= 0.0f32 {
            let t = r as usize;
            scratch[t] = h * dbias;
            gpair[slot] = gpair[slot] + scratch[t];
        }
    }
}

/// Six spellings of one `f32` expression, so a test can say which of them the
/// backend rounds where the CPU does.
///
/// Kept in the shipped code rather than deleted with the investigation: the
/// answer belongs to a CubeCL version and a backend, not to this crate, and
/// `probe_f32_rounding` re-runs it in one second against whatever is installed.
#[cube(launch)]
pub fn probe_kernel(
    out: &mut Array<f32>,
    scratch: &mut Array<f32>,
    h: f32,
    v: f32,
    dw: f32,
    g: f32,
) {
    if ABSOLUTE_POS == 0 {
        // 0: written straight out.
        out[0] = g + h * v * dw;
        // 1: through f64 with an explicit narrow after each operation.
        let hv = f32::cast_from(f64::cast_from(h) * f64::cast_from(v));
        let step = f32::cast_from(f64::cast_from(hv) * f64::cast_from(dw));
        out[1] = f32::cast_from(f64::cast_from(g) + f64::cast_from(step));
        // 2: through two slots of a global f32 array.
        scratch[0] = h * v;
        scratch[1] = scratch[0] * dw;
        out[2] = g + scratch[1];
        // 3: everything in f64, narrowed once — what a wide backend gives.
        out[3] = f32::cast_from(
            f64::cast_from(g) + f64::cast_from(h) * f64::cast_from(v) * f64::cast_from(dw),
        );
        // 4: through a RuntimeCell.
        let cell = RuntimeCell::<f32>::new(h * v);
        cell.store(cell.read() * dw);
        out[4] = g + cell.read();
        // 5: through one slot of a global f32 array, read back each time —
        //    what the residual kernels use.
        scratch[0] = h * v;
        scratch[0] = scratch[0] * dw;
        out[5] = g + scratch[0];
    }
}

// ------------------------------------------------------------- host API ----

/// One uploaded [`CscPage`].
struct DevicePage {
    row_idx: Handle,
    values: Handle,
    /// Kept on the host: every launch needs a column's `(begin, len)`, and one
    /// `usize` pair per feature is not worth a device read to fetch.
    col_ptr: Vec<usize>,
    nnz: usize,
}

/// The transposed matrix and the round's residual gradients, both resident on
/// the device for the length of a fit.
pub struct GpuLinear<R: Runtime> {
    client: ComputeClient<R>,
    pages: Vec<DevicePage>,
    /// Interleaved `[grad, hess]` per `(row, group)`, mirroring the host
    /// buffer's layout so a readback is a straight reinterpret.
    gpair: Handle,
    /// One `f32` slot per thread of the widest residual update, which is what
    /// makes that update round exactly where the CPU does — see
    /// [`update_residual_kernel`].
    scratch: Handle,
    n_pairs: usize,
    n_rows: usize,
    n_groups: usize,
}

impl<R: Runtime> GpuLinear<R> {
    /// Upload the transpose. Done once per fit: the matrix does not change.
    pub fn new(client: ComputeClient<R>, pages: &CscPages, n_rows: usize, n_groups: usize) -> Self {
        let upload = |page: &CscPage| -> DevicePage {
            let (col_ptr, row_idx, values) = page.raw();
            // An empty column set still has to bind an array.
            let (row_idx, values): (&[u32], &[f32]) = if row_idx.is_empty() {
                (&[0u32], &[0.0f32])
            } else {
                (row_idx, values)
            };
            DevicePage {
                row_idx: client.create_from_slice(bytemuck::cast_slice(row_idx)),
                values: client.create_from_slice(bytemuck::cast_slice(values)),
                col_ptr: col_ptr.to_vec(),
                nnz: row_idx.len(),
            }
        };
        let pages: Vec<DevicePage> = pages.iter().map(upload).collect();
        let n_pairs = n_rows * n_groups;
        let gpair = client.empty((2 * n_pairs).max(1) * size_of::<f32>());

        // One slot per thread of the widest launch: the longest column of any
        // page, or every row for the intercept.
        let widest = pages
            .iter()
            .flat_map(|p| p.col_ptr.windows(2).map(|w| w[1] - w[0]))
            .chain(std::iter::once(n_rows))
            .max()
            .unwrap_or(1);
        let scratch = client.empty(widest.max(1) * size_of::<f32>());

        Self { client, pages, gpair, scratch, n_pairs, n_rows, n_groups }
    }

    /// Replace the resident gradients with this round's.
    pub fn upload_gpair(&mut self, gpair: &[GradientPair]) {
        let flat: Vec<f32> = gpair.iter().flat_map(|g| [g.grad, g.hess]).collect();
        self.gpair = self.client.create_from_slice(bytemuck::cast_slice(&flat));
        self.n_pairs = gpair.len();
    }

    /// Read the residual gradients back, for the two feature selectors that
    /// score features by them.
    pub fn download_gpair(&self, out: &mut [GradientPair]) {
        let bytes = self.client.read_one_unchecked(self.gpair.clone());
        let flat: &[f32] = bytemuck::cast_slice(&bytes);
        for (i, p) in out.iter_mut().enumerate() {
            p.grad = flat[2 * i];
            p.hess = flat[2 * i + 1];
        }
    }

    /// How many pages the matrix was cut into.
    pub fn num_pages(&self) -> usize {
        self.pages.len()
    }

    /// `(g, h)` for one column of one page, folded exactly as the CPU folds it.
    pub fn column_gradient(&self, page: usize, fidx: usize, group: usize) -> (f64, f64) {
        let p = &self.pages[page];
        let (begin, end) = (p.col_ptr[fidx], p.col_ptr[fidx + 1]);
        let len = end - begin;
        if len == 0 {
            return (0.0, 0.0);
        }
        let n_blocks = (len as u32).div_ceil(LIN_BLOCK) as usize;
        let out = self.client.empty(2 * n_blocks * size_of::<f64>());

        column_sums_kernel::launch::<R>(
            &self.client,
            CubeCount::Static((n_blocks as u32).div_ceil(DIM).max(1), 1, 1),
            CubeDim::new_1d(DIM),
            unsafe { ArrayArg::from_raw_parts(p.row_idx.clone(), p.nnz) },
            unsafe { ArrayArg::from_raw_parts(p.values.clone(), p.nnz) },
            unsafe { ArrayArg::from_raw_parts(self.gpair.clone(), 2 * self.n_pairs) },
            unsafe { ArrayArg::from_raw_parts(out.clone(), 2 * n_blocks) },
            begin as u32,
            len as u32,
            group as u32,
            self.n_groups as u32,
            LIN_BLOCK,
        );
        fold_blocks(&self.client, out, n_blocks)
    }

    /// `(g, h)` over every row, for the intercept.
    pub fn bias_gradient(&self, group: usize) -> (f64, f64) {
        if self.n_rows == 0 {
            return (0.0, 0.0);
        }
        let n_blocks = (self.n_rows as u32).div_ceil(LIN_BLOCK) as usize;
        let out = self.client.empty(2 * n_blocks * size_of::<f64>());

        bias_sums_kernel::launch::<R>(
            &self.client,
            CubeCount::Static((n_blocks as u32).div_ceil(DIM).max(1), 1, 1),
            CubeDim::new_1d(DIM),
            unsafe { ArrayArg::from_raw_parts(self.gpair.clone(), 2 * self.n_pairs) },
            unsafe { ArrayArg::from_raw_parts(out.clone(), 2 * n_blocks) },
            self.n_rows as u32,
            group as u32,
            self.n_groups as u32,
            LIN_BLOCK,
        );
        fold_blocks(&self.client, out, n_blocks)
    }

    /// Correct one column's residual gradients by a weight step.
    pub fn update_residual(&mut self, page: usize, fidx: usize, group: usize, dw: f32) {
        if dw == 0.0 {
            return;
        }
        let p = &self.pages[page];
        let (begin, end) = (p.col_ptr[fidx], p.col_ptr[fidx + 1]);
        let len = end - begin;
        if len == 0 {
            return;
        }
        update_residual_kernel::launch::<R>(
            &self.client,
            CubeCount::Static((len as u32).div_ceil(DIM).max(1), 1, 1),
            CubeDim::new_1d(DIM),
            unsafe { ArrayArg::from_raw_parts(p.row_idx.clone(), p.nnz) },
            unsafe { ArrayArg::from_raw_parts(p.values.clone(), p.nnz) },
            unsafe { ArrayArg::from_raw_parts(self.gpair.clone(), 2 * self.n_pairs) },
            unsafe { ArrayArg::from_raw_parts(self.scratch.clone(), len) },
            begin as u32,
            len as u32,
            group as u32,
            self.n_groups as u32,
            dw,
        );
    }

    /// The same correction for the intercept.
    pub fn update_bias_residual(&mut self, group: usize, dbias: f32) {
        if dbias == 0.0 || self.n_rows == 0 {
            return;
        }
        update_bias_residual_kernel::launch::<R>(
            &self.client,
            CubeCount::Static((self.n_rows as u32).div_ceil(DIM).max(1), 1, 1),
            CubeDim::new_1d(DIM),
            unsafe { ArrayArg::from_raw_parts(self.gpair.clone(), 2 * self.n_pairs) },
            unsafe { ArrayArg::from_raw_parts(self.scratch.clone(), self.n_rows) },
            self.n_rows as u32,
            group as u32,
            self.n_groups as u32,
            dbias,
        );
    }
}

/// Add the per-block totals in block order, which is what makes this equal to
/// the CPU's `reduce_blocks` rather than merely close to it.
fn fold_blocks<R: Runtime>(
    client: &ComputeClient<R>,
    out: Handle,
    n_blocks: usize,
) -> (f64, f64) {
    let bytes = client.read_one_unchecked(out);
    let partials: &[f64] = bytemuck::cast_slice(&bytes);
    let mut sum = (0.0f64, 0.0f64);
    for b in 0..n_blocks {
        sum.0 += partials[2 * b];
        sum.1 += partials[2 * b + 1];
    }
    sum
}
