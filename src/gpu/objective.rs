//! The round's host-side passes, on the device: the objective's gradients,
//! the prediction update a grown tree makes, and the training metric.
//!
//! With the tree grown on the device, what remained on the host each round
//! was the elementwise work around it — the gradients from the predictions,
//! the leaf values added back into the predictions, the metric over them —
//! plus a readback of the row index to do the second. At 500 000 rows on a
//! four-core cloud VM that was 4.7 ms of a 16 ms round, and the readback
//! another 0.5. Upstream keeps all of it on the device, and so does this,
//! for the objectives it covers: the predictions live in a device buffer for
//! as long as every round can be done there, and come back to the host only
//! when something host-side asks for them (`GBTree::sync_preds`).
//!
//! Every kernel here reproduces the host arithmetic operation for operation
//! — the same `f32` products, the metric's `f64` squares summed in the same
//! 4 096-row blocks and the blocks added in the same order — so a device
//! round gives the host round's numbers, not an approximation of them.

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::launch;
use super::quantiser::DeviceGpairsF32;
use super::tables::{TableBuilder, upload_slice};
use crate::objective::DeviceObjective;

/// Halving steps of the update kernel's segment search: enough for `2^24`
/// leaf segments.
const SEARCH_STEPS: u32 = 24;

/// Rows per block of the metric reduction: `metric::REDUCE_ROWS`, so the
/// device sums the same blocks in the same order as the host.
const REDUCE_ROWS: u32 = crate::metric::REDUCE_ROWS as u32;
/// Lanes per block: `metric::REDUCE_LANES`.
const REDUCE_LANES: u32 = crate::metric::REDUCE_LANES as u32;
/// Rows a lane walks.
const ROWS_PER_LANE: u32 = REDUCE_ROWS / REDUCE_LANES;

/// `RegLossObj::get_gradient` for `reg:squarederror`, per row: `grad =
/// (pred - label) * w`, `hess = 1 * w`, with `w` the row weight scaled by
/// `scale_pos_weight` where the label is exactly one.
#[cube(launch_unchecked)]
pub fn squared_error_gradient_kernel(
    preds: &Array<f32>,
    labels: &Array<f32>,
    weights: &Array<f32>,
    out: &mut Array<f32>,
    n: u32,
    scale_pos_weight: f32,
    #[comptime] weighted: bool,
) {
    let i = ABSOLUTE_POS as u32;
    if i < n {
        let p = preds[i as usize];
        let label = labels[i as usize];
        let base = if weighted { weights[i as usize] } else { 1.0f32.into() };
        // `if label == 1 { base * spw } else { base }`, as arithmetic the
        // CPU runtime's pipeline lowers: exact, since `x * 1 + y * 0 == x`.
        let one = f32::cast_from(label == 1.0f32);
        let w = base * scale_pos_weight * one + base * (1.0f32 - one);
        out[(2u32 * i) as usize] = (p - label) * w;
        out[(2u32 * i + 1u32) as usize] = 1.0f32 * w;
    }
}

/// `GrownTree::update_predictions`: every row of every leaf segment gets its
/// leaf's value times the tree weight. A unit is one position of the row
/// index; its segment is the one whose range holds the position, found by
/// binary search over the sorted segment starts, and the row it names is
/// read from whichever of the partitioner's two buffers holds that segment.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn update_predictions_kernel(
    ridx0: &Array<u32>,
    ridx1: &Array<u32>,
    seg_begin: &Array<u32>,
    seg_side: &Array<u32>,
    seg_value: &Array<f32>,
    preds: &mut Array<f32>,
    n_rows: u32,
    n_seg: u32,
    weight: f32,
) {
    // Clamped rather than guarded: a search loop inside a guarded body is
    // the shape the CPU runtime's MLIR pipeline cannot lower, so every unit
    // searches (a dead unit for the last position) and only the write is
    // conditional. The step is arithmetic for the same reason.
    let pos = ABSOLUTE_POS as u32;
    // Arithmetic selects throughout: an `if` expression before a loop is a
    // shape the CPU runtime's pipeline refuses as well.
    let live = u32::cast_from(pos < n_rows);
    let i = pos * live + (n_rows - 1u32) * (1u32 - live);
    // The last segment whose start is at or below `i`.
    // A fixed number of halving steps rather than a `while`: the CPU
    // runtime's pipeline would not lower this loop in any form tried (a
    // guarded body, a bare body, arithmetic in place of every branch), and
    // `2^24` segments is more leaves than any tree has. Once the range is one
    // segment wide the step is a no-op.
    let lo = RuntimeCell::<u32>::new(0u32);
    let n = RuntimeCell::<u32>::new(n_seg);
    #[unroll]
    for _ in 0..SEARCH_STEPS {
        let half = n.read() / 2u32;
        let mid = lo.read() + half;
        let take = u32::cast_from(seg_begin[mid as usize] <= i) * u32::cast_from(half > 0u32);
        lo.store(lo.read() + take * half);
        n.store(n.read() - half);
    }
    let s = lo.read();
    let side = seg_side[s as usize];
    // A dead unit lands on the buffer's spare slot past the last row: the
    // store is unconditional, which is the other shape the pipeline wants.
    let row = (ridx0[i as usize] * (1u32 - side) + ridx1[i as usize] * side) * live
        + n_rows * (1u32 - live);
    preds[row as usize] += seg_value[s as usize] * weight;
}

/// `elementwise_reduce` for `rmse`, first half: a unit is one lane of one
/// block of [`REDUCE_ROWS`] rows — the rows congruent to it modulo
/// [`REDUCE_LANES`], in row order — and writes its weighted sum of
/// `(label - pred)^2` in `f64` and its weight sum. The host walks the same
/// lanes in the same order, so the block sums are the host's.
#[cube(launch_unchecked)]
pub fn rmse_lanes_kernel(
    preds: &Array<f32>,
    labels: &Array<f32>,
    weights: &Array<f32>,
    lanes: &mut Array<f64>,
    n: u32,
    n_lanes: u32,
    #[comptime] weighted: bool,
) {
    // Arithmetic guards and a fixed trip count, the shapes every runtime's
    // pipeline lowers (see `update_predictions_kernel`): a dead unit, or a
    // lane past its block's end, reads row `n - 1` and adds nothing.
    let l = ABSOLUTE_POS as u32;
    let live = u32::cast_from(l < n_lanes);
    let lane = l * live;
    let first = (lane / REDUCE_LANES) * REDUCE_ROWS + lane % REDUCE_LANES;
    let esum = RuntimeCell::<f64>::new(0.0f64);
    let wsum = RuntimeCell::<f64>::new(0.0f64);
    for k in 0..ROWS_PER_LANE {
        let i = first + k * REDUCE_LANES;
        let ok = u32::cast_from(i < n) * live;
        let idx = i * ok + (n - 1u32) * (1u32 - ok);
        let w = if weighted { f64::cast_from(weights[idx as usize]) } else { 1.0f64.into() };
        let d = f64::cast_from(labels[idx as usize] - preds[idx as usize]);
        let m = f64::cast_from(ok);
        esum.store(esum.read() + d * d * w * m);
        wsum.store(wsum.read() + w * m);
    }
    // A dead unit lands on the spare slot past the last lane.
    let out = lane * live + n_lanes * (1u32 - live);
    lanes[(2u32 * out) as usize] = esum.read();
    lanes[(2u32 * out + 1u32) as usize] = wsum.read();
}

/// Second half: a unit adds one block's [`REDUCE_LANES`] lane sums in lane
/// order.
#[cube(launch_unchecked)]
pub fn rmse_fold_kernel(lanes: &Array<f64>, out: &mut Array<f64>, n_blocks: u32) {
    let b = ABSOLUTE_POS as u32;
    let live = u32::cast_from(b < n_blocks);
    let block = b * live;
    let esum = RuntimeCell::<f64>::new(0.0f64);
    let wsum = RuntimeCell::<f64>::new(0.0f64);
    for lane in 0..REDUCE_LANES {
        let at = block * REDUCE_LANES + lane;
        esum.store(esum.read() + lanes[(2u32 * at) as usize]);
        wsum.store(wsum.read() + lanes[(2u32 * at + 1u32) as usize]);
    }
    let dst = block * live + n_blocks * (1u32 - live);
    out[(2u32 * dst) as usize] = esum.read();
    out[(2u32 * dst + 1u32) as usize] = wsum.read();
}

/// The round's device-resident state: the predictions, and the labels and
/// weights they are scored against.
pub struct DeviceRound {
    pub(crate) preds: Handle,
    labels: Handle,
    /// `None` when every row weighs one.
    weights: Option<Handle>,
    n_rows: usize,
    /// Whether the host copy of the predictions is behind this one.
    pub(crate) host_stale: bool,
}

impl DeviceRound {
    /// Upload the labels and weights once, and the predictions as they are.
    pub fn new<R: Runtime>(
        client: &ComputeClient<R>,
        info: &crate::data::MetaInfo,
        preds: &[f32],
    ) -> Self {
        Self {
            preds: Self::padded(client, preds),
            labels: upload_slice(client, &info.labels),
            weights: info.weights.as_ref().map(|w| upload_slice(client, w)),
            n_rows: preds.len(),
            host_stale: false,
        }
    }

    /// Replace the predictions with the host's.
    pub fn upload_preds<R: Runtime>(&mut self, client: &ComputeClient<R>, preds: &[f32]) {
        self.preds = Self::padded(client, preds);
        self.host_stale = false;
    }

    /// The predictions with one spare slot after the last row, which the
    /// update kernel's dead units write to.
    fn padded<R: Runtime>(client: &ComputeClient<R>, preds: &[f32]) -> Handle {
        let mut padded = Vec::with_capacity(preds.len() + 1);
        padded.extend_from_slice(preds);
        padded.push(0.0);
        super::tables::upload_vec(client, padded)
    }

    /// The gradients of `kind` at the current predictions, as the histogram
    /// path takes them.
    pub fn gradients<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        kind: DeviceObjective,
    ) -> DeviceGpairsF32 {
        let n = self.n_rows;
        let out = client.empty(n.max(1) * 2 * size_of::<f32>());
        let (weights, weighted) = match &self.weights {
            Some(w) => (w.clone(), true),
            None => (self.labels.clone(), false),
        };
        let (count, dim) = launch::elementwise(client, n);
        match kind {
            DeviceObjective::SquaredError { scale_pos_weight } => {
                // SAFETY: every index is guarded by `n`, the arrays' length.
                unsafe {
                    squared_error_gradient_kernel::launch_unchecked::<R>(
                        client,
                        count,
                        dim,
                        ArrayArg::from_raw_parts(self.preds.clone(), n),
                        ArrayArg::from_raw_parts(self.labels.clone(), n),
                        ArrayArg::from_raw_parts(weights, n),
                        ArrayArg::from_raw_parts(out.clone(), 2 * n),
                        n as u32,
                        scale_pos_weight,
                        weighted,
                    );
                }
            }
        }
        DeviceGpairsF32 { handle: out, n }
    }

    /// Add the grown tree's leaf values into the predictions.
    ///
    /// `segments` are `(row buffer, begin, len, value)` in any order; every
    /// position of the row index belongs to exactly one.
    pub fn update_predictions<R: Runtime>(
        &mut self,
        client: &ComputeClient<R>,
        bufs: [&Handle; 2],
        segments: &[(u32, u32, u32, f32)],
        weight: f32,
    ) {
        let n = self.n_rows;
        if n == 0 || segments.is_empty() {
            return;
        }
        let mut sorted: Vec<(u32, u32, u32, f32)> = segments.to_vec();
        sorted.sort_by_key(|s| s.1);
        let begin: Vec<u32> = sorted.iter().map(|s| s.1).collect();
        let side: Vec<u32> = sorted.iter().map(|s| s.0).collect();
        let value: Vec<f32> = sorted.iter().map(|s| s.3).collect();
        debug_assert_eq!(begin[0], 0);
        debug_assert_eq!(
            sorted.iter().map(|s| s.2 as usize).sum::<usize>(),
            n,
            "the leaf segments must cover every row exactly once"
        );
        let mut tb = TableBuilder::new();
        let t_begin = tb.push(&begin);
        let t_side = tb.push(&side);
        let t_value = tb.push(&value);
        let tables = tb.upload(client);
        let (count, dim) = launch::elementwise(client, n);
        // SAFETY: positions are guarded by `n_rows`, segments by the search
        // over `n_seg` starts, and the row read is a row of the matrix.
        unsafe {
            update_predictions_kernel::launch_unchecked::<R>(
                client,
                count,
                dim,
                ArrayArg::from_raw_parts(bufs[0].clone(), n),
                ArrayArg::from_raw_parts(bufs[1].clone(), n),
                tables.arg(t_begin, begin.len()),
                tables.arg(t_side, side.len()),
                tables.arg(t_value, value.len()),
                ArrayArg::from_raw_parts(self.preds.clone(), n + 1),
                n as u32,
                begin.len() as u32,
                weight,
            );
        }
        self.host_stale = true;
    }

    /// `rmse` over the current predictions — the host metric's number.
    pub fn rmse<R: Runtime>(&self, client: &ComputeClient<R>) -> f64 {
        let n = self.n_rows;
        if n == 0 {
            return 0.0;
        }
        let n_blocks = (n as u32).div_ceil(REDUCE_ROWS);
        let n_lanes = n_blocks * REDUCE_LANES;
        // One spare slot each for the dead units' writes.
        let lanes = client.empty((n_lanes as usize + 1) * 2 * size_of::<f64>());
        let out = client.empty((n_blocks as usize + 1) * 2 * size_of::<f64>());
        let (weights, weighted) = match &self.weights {
            Some(w) => (w.clone(), true),
            None => (self.labels.clone(), false),
        };
        let (count, dim) = launch::elementwise(client, n_lanes as usize);
        // SAFETY: every index is clamped or lands on the spare slot.
        unsafe {
            rmse_lanes_kernel::launch_unchecked::<R>(
                client,
                count,
                dim,
                ArrayArg::from_raw_parts(self.preds.clone(), n),
                ArrayArg::from_raw_parts(self.labels.clone(), n),
                ArrayArg::from_raw_parts(weights, n),
                ArrayArg::from_raw_parts(lanes.clone(), (n_lanes as usize + 1) * 2),
                n as u32,
                n_lanes,
                weighted,
            );
        }
        let (count, dim) = launch::elementwise(client, n_blocks as usize);
        // SAFETY: as above.
        unsafe {
            rmse_fold_kernel::launch_unchecked::<R>(
                client,
                count,
                dim,
                ArrayArg::from_raw_parts(lanes, (n_lanes as usize + 1) * 2),
                ArrayArg::from_raw_parts(out.clone(), (n_blocks as usize + 1) * 2),
                n_blocks,
            );
        }
        let bytes = client.read_one_unchecked(out);
        let words: &[f64] = bytemuck::cast_slice(&bytes);
        let (esum, wsum) = words
            .chunks_exact(2)
            .take(n_blocks as usize)
            .fold((0.0f64, 0.0f64), |(e, w), b| (e + b[0], w + b[1]));
        if wsum == 0.0 { esum.sqrt() } else { (esum / wsum).sqrt() }
    }

    /// Read the predictions back into `host`.
    pub fn download<R: Runtime>(&mut self, client: &ComputeClient<R>, host: &mut [f32]) {
        let bytes = client.read_one_unchecked(self.preds.clone());
        let words: &[f32] = bytemuck::cast_slice(&bytes);
        host.copy_from_slice(&words[..host.len()]);
        self.host_stale = false;
    }
}
