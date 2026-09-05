//! On-device row partitioning, ported from
//! `xgboost/src/tree/gpu_hist/row_partitioner.{cu,cuh}`
//! (`RowPartitioner::UpdatePositionBatch` / `SortPositionCopyKernel`).
//!
//! # Layout
//!
//! Row indices live in one `ridx` buffer, and each live tree node owns a
//! contiguous segment of it. Splitting a node rewrites its segment in place:
//! the rows going left come first, then the rows going right, so the two
//! children are themselves contiguous segments and the whole level can be
//! re-partitioned in one pass.
//!
//! # Why two passes
//!
//! A segment can hold millions of rows, so a workgroup-per-segment scan would
//! leave the machine idle at the top of the tree, where there is only one
//! segment. Instead the level's rows are cut into fixed-size tiles, and every
//! tile of every segment is processed in parallel:
//!
//! 1. [`count_tile_kernel`] — one tile per work item, counting the rows that
//!    go left.
//! 2. [`scatter_tile_kernel`] — one tile per work item again, re-deriving the
//!    predicate and writing each row to its final slot. Where that is needs
//!    the segment's left-count prefix over its tiles, and the tile sums it
//!    from the first pass's counts itself: a segment has one count per tile,
//!    so that is a few dozen loads at the root and one or two at depth, and it
//!    is cheaper than the launch a separate scan pass would cost on a runtime
//!    whose launch is a thread hand-off.
//!
//! The scatter writes into a second buffer, and rather than copy the result
//! back the partitioner *swaps*: the buffer just written becomes `ridx`, and a
//! per-row table on the host remembers which of the two holds each row's
//! current value. Every segment a level reads was written by the previous
//! level, so on the depth-wise path both buffers are never mixed in one launch
//! and the copy pass ([`commit_tile_kernel`]) is never launched; it remains
//! for the loss-guided queue, which can come back to a node whose segment was
//! written several partitions ago, and for [`RowPartitioner::read`], which
//! assembles the final index from both buffers. On the CPU runtime the copy
//! was a launch per level at the runtime's ~130 µs floor, about 5% of a
//! depth-6 round, for 800 KB of memory traffic. The partition is stable on
//! both sides, which is what [`crate::tree::hist`]'s `partition_block`
//! produces.
//!
//! # Two shapes of a tile
//!
//! What a "work item" is depends on the runtime, through the comptime `coop`
//! flag of the two tile kernels — the cooperation width of
//! [`launch::cooperative`]:
//!
//! * **Cooperative** (a runtime with planes): a tile is [`PART_BLOCK`] rows and
//!   a whole cube owns it, one row per unit. The rows' ranks come from a
//!   shared-memory scan over the units' flags, which is what a GPU wants.
//! * **Serial** (a plane-less runtime): a tile is [`SERIAL_TILE_ROWS`] rows and
//!   one *unit* owns it, walking the rows with a running counter. No shared
//!   memory and no barrier, so the launch can use every core — the shape the
//!   CPU grower's `partition_blocked` has, with its 4096-row blocks.
//!
//! Both leave every row in the same slot: a row's rank among the left-going
//! rows of its tile is the same number whether it was scanned or counted.

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::ellpack::{DeviceEllpack, load_bin};
use super::launch;
use crate::error::{Error, Result};

/// Threads per partitioning workgroup, and rows per tile, in the cooperative
/// shape.
pub const PART_BLOCK: u32 = 256;

/// Rows per tile in the serial shape, where one unit walks a whole tile.
///
/// The CPU grower's `BLOCK_ROWS`: enough rows that a unit is worth its thread
/// dispatch, few enough that a level of small segments still yields a tile per
/// unit.
pub const SERIAL_TILE_ROWS: u32 = 4096;

/// Whether this runtime can run the partitioner against a
/// [`EllpackLayout::Sparse`] matrix.
///
/// The sparse layout is the only one whose [`goes_left`] has to *search* a row
/// for the feature's entry, because its bins are global rather than
/// feature-local. That search loop, inlined into the guarded body of
/// [`count_tile_kernel`], is a shape the CubeCL CPU runtime's MLIR pipeline
/// cannot lower: it fails the `scf`-to-`cf` conversion with "operation with
/// block successors must terminate its parent block". The failure is in a
/// worker thread at kernel-compile time and takes that worker down with it, so
/// it has to be refused before the launch rather than allowed to happen — a
/// dead worker leaves every later launch in the process waiting forever.
///
/// Plane-lessness is the proxy for "is that runtime", which is exact today:
/// the CPU runtime is the only plane-less backend CubeCL has. Nothing else in
/// the crate is affected, because [`build_ellpack`] only ever produces `Dense`
/// or `DenseCompressed` — `Sparse` reaches here only from a hand-built
/// [`EllpackMatrix`], which is to say from the kernel tests.
///
/// [`build_ellpack`]: super::ellpack::build_ellpack
/// [`EllpackMatrix`]: super::ellpack::EllpackMatrix
/// [`EllpackLayout::Sparse`]: super::ellpack::EllpackLayout::Sparse
pub fn supports_sparse_layout<R: Runtime>(client: &ComputeClient<R>) -> bool {
    launch::has_planes(client)
}

/// Which side of a split one row falls on.
///
/// Port of `HistGrower::goes_left` (`src/tree/hist.rs`) against the ELLPACK
/// layouts: a row with no value for the feature follows `default_left`, a
/// numeric split compares the *feature-local* bin against `cond`, and a
/// categorical split tests membership of the right-hand category set.
///
/// `fbegin..fend` is the feature's global bin range, `cut_ptrs[fidx]..cut_ptrs[fidx + 1]`,
/// looked up once per tile by the caller rather than once per row here: the
/// CPU runtime's JIT does not hoist loop-invariant loads out of the row loop.
///
/// The feature-local layouts read the feature-major `gidx_t` (see
/// [`DeviceEllpack`]): rows of a segment ascend, so the reads of one tile walk
/// one column of it. The sparse layout has to search the row-major `gidx`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn goes_left(
    gidx: &Array<u32>,
    gidx_t: &Array<u32>,
    cat_bits: &Array<u32>,
    row: u32,
    fidx: u32,
    fbegin: u32,
    fend: u32,
    cond: i64,
    default_left: bool,
    is_cat: bool,
    cat_base: u32,
    row_stride: u32,
    n_rows: u32,
    base_rowid: u32,
    null_value: u32,
    #[comptime] dense: bool,
    #[comptime] compressed: bool,
    #[comptime] bits: u32,
) -> bool {
    let local_row = row - base_rowid;

    // `EllpackDeviceAccessor::GetBinIndex`. The dense layouts index the
    // feature directly, in the feature-major copy; the sparse layout stores a
    // row's entries in ascending global-bin order, so the feature's entry is
    // found by searching its bin range.
    let raw = if dense || compressed {
        load_bin(gidx_t, fidx * n_rows + local_row, bits)
    } else {
        let row_begin = local_row * row_stride;
        let found = RuntimeCell::<u32>::new(null_value);
        let k = RuntimeCell::<u32>::new(0u32);
        while k.read() < row_stride {
            let v = load_bin(gidx, row_begin + k.read(), bits);
            if v != null_value && v >= fbegin && v < fend {
                found.store(v);
            }
            k.store(k.read() + 1u32);
        }
        found.read()
    };

    let missing = !dense && raw == null_value;
    if missing {
        default_left
    } else {
        // Feature-local bin: the dense layouts already store it that way, the
        // sparse layout stores the global bin.
        let local = if compressed { raw } else { raw - fbegin };
        if is_cat {
            // `!cat::check_bit`: the bit set names the categories going right.
            let word = cat_base + local / 32u32;
            let bit = (cat_bits[word as usize] >> (local % 32u32)) & 1u32;
            bit == 0u32
        } else {
            i64::cast_from(local) <= cond
        }
    }
}

/// Inclusive block scan of `s_flag`, leaving the tile total in the last slot.
#[cube]
fn scan_flags(s_flag: &mut SharedMemory<u32>, #[comptime] block: usize) {
    let t = UNIT_POS_X as usize;
    let off = RuntimeCell::<u32>::new(1u32);
    while off.read() < block as u32 {
        let d = off.read();
        let add = if UNIT_POS_X >= d { s_flag[t - d as usize] } else { 0u32.into() };
        sync_cube();
        if UNIT_POS_X >= d {
            s_flag[t] += add;
        }
        sync_cube();
        off.store(d * 2u32);
    }
}

/// The tile this work item owns: the flattened cube index when a cube owns a
/// tile, the flattened unit index when a unit does.
///
/// Both grids can be overprovisioned (`launch::cubes_1d` rounds the tile
/// count up to a rectangle, `elementwise` to whole cubes), so the caller has
/// to bounds-check the result against the tile count.
#[cube]
fn tile_index(#[comptime] coop: bool) -> u32 {
    if coop {
        CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_Z * CUBE_COUNT_X * CUBE_COUNT_Y
    } else {
        ABSOLUTE_POS as u32
    }
}

/// Count the rows of each tile that go left.
///
/// `tile_seg[t]` names the segment tile `t` belongs to and `tile_off[t]` its
/// first row within that segment. `block` is the tile's row count, and in the
/// cooperative shape also the cube width; see the module docs for the two
/// shapes `coop` selects between.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn count_tile_kernel(
    ridx: &Array<u32>,
    gidx: &Array<u32>,
    gidx_t: &Array<u32>,
    cut_ptrs: &Array<u32>,
    cat_bits: &Array<u32>,
    tile_seg: &Array<u32>,
    tile_off: &Array<u32>,
    seg_begin: &Array<u32>,
    seg_len: &Array<u32>,
    seg_fidx: &Array<u32>,
    seg_cond: &Array<i64>,
    seg_flags: &Array<u32>,
    seg_cat_base: &Array<u32>,
    tile_left: &mut Array<u32>,
    row_stride: u32,
    n_rows: u32,
    base_rowid: u32,
    null_value: u32,
    #[comptime] dense: bool,
    #[comptime] compressed: bool,
    #[comptime] bits: u32,
    #[comptime] coop: bool,
    #[comptime] block: usize,
) {
    // In the cooperative shape every unit of a cube computes the same `tile`,
    // so an overprovisioned cube skips the body as a whole and the `sync_cube`
    // counts stay matched across units.
    let tile = tile_index(coop);
    if tile < tile_seg.len() as u32 {
        let s = tile_seg[tile as usize];
        let toff = tile_off[tile as usize];
        let len = seg_len[s as usize];
        let begin = seg_begin[s as usize];
        let fidx = seg_fidx[s as usize];
        let fbegin = cut_ptrs[fidx as usize];
        let fend = cut_ptrs[(fidx + 1u32) as usize];
        let cond = seg_cond[s as usize];
        let flags = seg_flags[s as usize];
        let cat_base = seg_cat_base[s as usize];

        if coop {
            let off = toff + UNIT_POS_X;
            let mut s_flag = SharedMemory::<u32>::new(block);
            let t = UNIT_POS_X as usize;

            let live = off < len;
            let left = if live {
                goes_left(
                    gidx,
                    gidx_t,
                    cat_bits,
                    ridx[(begin + off) as usize],
                    fidx,
                    fbegin,
                    fend,
                    cond,
                    (flags & 1u32) == 1u32,
                    (flags & 2u32) == 2u32,
                    cat_base,
                    row_stride,
                    n_rows,
                    base_rowid,
                    null_value,
                    dense,
                    compressed,
                    bits,
                )
            } else {
                false.into()
            };
            s_flag[t] = u32::cast_from(left);
            sync_cube();

            scan_flags(&mut s_flag, block);

            if UNIT_POS_X == block as u32 - 1u32 {
                tile_left[tile as usize] = s_flag[t];
            }
        } else {
            // One unit walks the whole tile: a running count needs no scan.
            // Both arms runtime values: a comptime constant cannot be one arm
            // of a runtime `if`.
            let cap = toff + block as u32;
            let n = if cap > len { len - toff } else { cap - toff };
            let count = RuntimeCell::<u32>::new(0u32);
            let k = RuntimeCell::<u32>::new(0u32);
            while k.read() < n {
                let off = toff + k.read();
                let left = goes_left(
                    gidx,
                    gidx_t,
                    cat_bits,
                    ridx[(begin + off) as usize],
                    fidx,
                    fbegin,
                    fend,
                    cond,
                    (flags & 1u32) == 1u32,
                    (flags & 2u32) == 2u32,
                    cat_base,
                    row_stride,
                    n_rows,
                    base_rowid,
                    null_value,
                    dense,
                    compressed,
                    bits,
                );
                count.store(count.read() + u32::cast_from(left));
                k.store(k.read() + 1u32);
            }
            tile_left[tile as usize] = count.read();
        }
    }

    // Outside the `tile` guard, so every cube — including an overprovisioned
    // one that skipped the body — arrives, which is what keeps the barrier
    // counts matching across units. A runtime that runs cubes sequentially
    // shares one `s_flag` across all of them (see the `gpu` module docs), so a
    // unit that has written its tile total must not start the next tile's
    // flags before the last unit has read this one's scan. The serial shape
    // shares nothing between units and needs no barrier at all.
    if coop {
        sync_cube();
    }
}

/// The slot a row lands in.
///
/// Left rows land at `seg_begin + tile_base[t] + rank`, right rows after all
/// the left ones — `seg_begin + seg_left[s] + (off - tile_base[t]) - rank` —
/// which keeps both sides in their original order. `rank` is the row's rank
/// among the left-going rows of its tile, exclusive of itself; `off - base`
/// counts the right-going rows before it, in this tile and the tiles before.
#[cube]
fn scatter_slot(left: bool, begin: u32, base: u32, seg_left: u32, off: u32, rank: u32) -> u32 {
    if left { begin + base + rank } else { begin + seg_left + (off - base) - rank }
}

/// Write every row to its final slot; see [`scatter_slot`] for where that is.
///
/// `tile_left` is [`count_tile_kernel`]'s output and `seg_tile_begin[s]..seg_tile_begin[s + 1]`
/// the segment's run of tiles; the tile derives its own base and the
/// segment's total from them, and the segment's first tile records the total
/// in `seg_left` for the host.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn scatter_tile_kernel(
    ridx: &Array<u32>,
    out: &mut Array<u32>,
    gidx: &Array<u32>,
    gidx_t: &Array<u32>,
    cut_ptrs: &Array<u32>,
    cat_bits: &Array<u32>,
    tile_seg: &Array<u32>,
    tile_off: &Array<u32>,
    tile_left: &Array<u32>,
    seg_begin: &Array<u32>,
    seg_len: &Array<u32>,
    seg_fidx: &Array<u32>,
    seg_cond: &Array<i64>,
    seg_flags: &Array<u32>,
    seg_cat_base: &Array<u32>,
    seg_tile_begin: &Array<u32>,
    seg_left: &mut Array<u32>,
    row_stride: u32,
    n_rows: u32,
    base_rowid: u32,
    null_value: u32,
    #[comptime] dense: bool,
    #[comptime] compressed: bool,
    #[comptime] bits: u32,
    #[comptime] coop: bool,
    #[comptime] block: usize,
) {
    // See `count_tile_kernel` for why an overprovisioned grid needs this guard.
    let tile = tile_index(coop);
    if tile < tile_seg.len() as u32 {
        let s = tile_seg[tile as usize];
        let toff = tile_off[tile as usize];
        let len = seg_len[s as usize];
        let begin = seg_begin[s as usize];
        let fidx = seg_fidx[s as usize];
        let fbegin = cut_ptrs[fidx as usize];
        let fend = cut_ptrs[(fidx + 1u32) as usize];
        let cond = seg_cond[s as usize];
        let flags = seg_flags[s as usize];
        let cat_base = seg_cat_base[s as usize];

        // The tile's base — left rows in the segment's earlier tiles — and the
        // segment's total, summed from the counts here rather than by a scan
        // launch (see the module docs). Uniform across a cube's units.
        let first = seg_tile_begin[s as usize];
        let end = seg_tile_begin[(s + 1u32) as usize];
        let base_acc = RuntimeCell::<u32>::new(0u32);
        let total_acc = RuntimeCell::<u32>::new(0u32);
        let k = RuntimeCell::<u32>::new(first);
        while k.read() < end {
            let cnt = tile_left[k.read() as usize];
            if k.read() < tile {
                base_acc.store(base_acc.read() + cnt);
            }
            total_acc.store(total_acc.read() + cnt);
            k.store(k.read() + 1u32);
        }
        let base = base_acc.read();
        let n_left = total_acc.read();
        // One writer per segment: its first tile, and in the cooperative
        // shape that tile's first unit.
        let reports = if coop { tile == first && UNIT_POS_X == 0u32 } else { tile == first };
        if reports {
            seg_left[s as usize] = n_left;
        }

        if coop {
            let off = toff + UNIT_POS_X;
            let mut s_flag = SharedMemory::<u32>::new(block);
            let t = UNIT_POS_X as usize;

            let live = off < len;
            let row = if live { ridx[(begin + off) as usize] } else { 0u32.into() };
            let left = if live {
                goes_left(
                    gidx,
                    gidx_t,
                    cat_bits,
                    row,
                    fidx,
                    fbegin,
                    fend,
                    cond,
                    (flags & 1u32) == 1u32,
                    (flags & 2u32) == 2u32,
                    cat_base,
                    row_stride,
                    n_rows,
                    base_rowid,
                    null_value,
                    dense,
                    compressed,
                    bits,
                )
            } else {
                false.into()
            };
            s_flag[t] = u32::cast_from(left);
            sync_cube();

            scan_flags(&mut s_flag, block);

            if live {
                // Inclusive scan minus this row's own flag gives its rank.
                let rank_left = s_flag[t] - u32::cast_from(left);
                out[scatter_slot(left, begin, base, n_left, off, rank_left) as usize] = row;
            }
        } else {
            // One unit walks the tile in row order, so the running count of
            // left-going rows *is* each row's rank.
            // Both arms runtime values: a comptime constant cannot be one arm
            // of a runtime `if`.
            let cap = toff + block as u32;
            let n = if cap > len { len - toff } else { cap - toff };
            let rank = RuntimeCell::<u32>::new(0u32);
            let k = RuntimeCell::<u32>::new(0u32);
            while k.read() < n {
                let off = toff + k.read();
                let row = ridx[(begin + off) as usize];
                let left = goes_left(
                    gidx,
                    gidx_t,
                    cat_bits,
                    row,
                    fidx,
                    fbegin,
                    fend,
                    cond,
                    (flags & 1u32) == 1u32,
                    (flags & 2u32) == 2u32,
                    cat_base,
                    row_stride,
                    n_rows,
                    base_rowid,
                    null_value,
                    dense,
                    compressed,
                    bits,
                );
                let r = rank.read();
                out[scatter_slot(left, begin, base, n_left, off, r) as usize] = row;
                rank.store(r + u32::cast_from(left));
                k.store(k.read() + 1u32);
            }
        }
    }

    // As in `count_tile_kernel`, and for the same reason.
    if coop {
        sync_cube();
    }
}

/// Copy segments from one row buffer into the other.
///
/// Used to bring a segment that was written several partitions ago into the
/// buffer the current level reads (the loss-guided queue can revisit such a
/// node); the depth-wise path never needs it, since it swaps buffers instead
/// of copying — see the module docs.
///
/// One cube per tile in both shapes, the units striding through the tile's
/// rows: a cooperative cube is as wide as its tile and takes one row per unit,
/// a serial cube is `cores` wide and takes several. Disjoint writes, so no
/// barrier either way.
#[cube(launch_unchecked)]
pub fn commit_tile_kernel(
    scratch: &Array<u32>,
    ridx: &mut Array<u32>,
    tile_seg: &Array<u32>,
    tile_off: &Array<u32>,
    seg_begin: &Array<u32>,
    seg_len: &Array<u32>,
    #[comptime] block: usize,
) {
    // See `count_tile_kernel` for why an overprovisioned grid needs this guard.
    let tile = tile_index(true);
    if tile < tile_seg.len() as u32 {
        let s = tile_seg[tile as usize];
        let toff = tile_off[tile as usize];
        let len = seg_len[s as usize];
        let begin = seg_begin[s as usize];
        let k = RuntimeCell::<u32>::new(UNIT_POS_X);
        while k.read() < block as u32 {
            let off = toff + k.read();
            if off < len {
                let i = (begin + off) as usize;
                ridx[i] = scratch[i];
            }
            k.store(k.read() + CUBE_DIM_X);
        }
    }
}

// ------------------------------------------------------------- host API ----

/// One node's split, as the partitioner needs it.
#[derive(Clone, Debug)]
pub struct SegmentSplit {
    /// The segment being rewritten, as `(begin, len)` in the `ridx` buffer.
    pub begin: u32,
    pub len: u32,
    pub fidx: u32,
    /// Feature-local bin threshold; rows at or below it go left. Ignored by a
    /// categorical split.
    pub cond: i64,
    pub default_left: bool,
    /// Categories that go *right*, as a bit set over feature-local bins.
    /// Empty for a numeric split.
    pub cat_bits: Vec<u32>,
}

/// Device-resident row index, partitioned by node.
pub struct RowPartitioner<R: Runtime> {
    client: ComputeClient<R>,
    /// The two row buffers a partition ping-pongs between.
    bufs: [Handle; 2],
    /// Which of `bufs` the last partition wrote — the one [`Self::ridx`]
    /// hands out.
    cur: usize,
    /// Per row, which of `bufs` holds its current value. A segment is always
    /// written whole, so one row of a segment speaks for all of it.
    side: Vec<u8>,
    n_rows: usize,
}

impl<R: Runtime> RowPartitioner<R> {
    /// Start with `rows` in the single root segment.
    pub fn new(client: ComputeClient<R>, rows: &[u32]) -> Self {
        let ridx = client.create_from_slice(bytemuck::cast_slice(rows));
        let scratch = client.empty(rows.len().max(1) * size_of::<u32>());
        let side = vec![0; rows.len()];
        Self { client, bufs: [ridx, scratch], cur: 0, side, n_rows: rows.len() }
    }

    /// All rows, in the root segment.
    pub fn all_rows(client: ComputeClient<R>, n_rows: usize) -> Self {
        let rows: Vec<u32> = (0..n_rows as u32).collect();
        Self::new(client, &rows)
    }

    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// The row-index buffer the last partition wrote. Segments index into it,
    /// and every segment that partition produced is current in it; a segment
    /// from an earlier partition may not be (see the module docs), which is
    /// why the grower only ever builds histograms for children of the split
    /// it just applied. The handle changes from one partition to the next.
    pub fn ridx(&self) -> &Handle {
        &self.bufs[self.cur]
    }

    /// Read the whole row index back, for tests and for the final leaf pass.
    ///
    /// Assembled from both buffers: a segment is current in whichever one the
    /// partition that produced it wrote, and a leaf finished early is left
    /// where it was.
    pub fn read(&self) -> Vec<u32> {
        let bytes = self.client.read_one_unchecked(self.bufs[self.cur].clone());
        let mut rows: Vec<u32> = bytemuck::cast_slice(&bytes).to_vec();
        let other = 1 - self.cur;
        if self.side.iter().any(|&b| b as usize == other) {
            let bytes = self.client.read_one_unchecked(self.bufs[other].clone());
            let stale: &[u32] = bytemuck::cast_slice(&bytes);
            for (i, &b) in self.side.iter().enumerate() {
                if b as usize == other {
                    rows[i] = stale[i];
                }
            }
        }
        rows
    }

    /// Partition every segment in `splits` in one pass.
    ///
    /// Returns each split's left-child length, so the caller can turn one
    /// segment into the children's two.
    pub fn partition(&mut self, ell: &DeviceEllpack, splits: &[SegmentSplit]) -> Result<Vec<u32>> {
        if splits.is_empty() {
            return Ok(Vec::new());
        }
        let c = self.client.clone();
        let c = &c;
        // Sparse is exactly the layout that is neither dense nor compressed.
        if !ell.compressed && !ell.dense && !supports_sparse_layout(c) {
            return Err(Error::SparseEllpackUnsupported);
        }
        // Cooperation width: a cube per tile where a barrier is cheap, a unit
        // per tile where it is not. See the module docs and `launch::cooperative`.
        let coop = launch::cooperative(c);
        let block = if coop { launch::scan_block_1d(c, PART_BLOCK) } else { SERIAL_TILE_ROWS };

        // Tile table: every segment is cut into `ceil(len / block)` tiles, and
        // tiles of every segment are launched together so the top of the tree
        // is as parallel as the bottom.
        let mut tile_seg = Vec::new();
        let mut tile_off = Vec::new();
        let mut seg_tile_begin = Vec::with_capacity(splits.len() + 1);
        for (s, sp) in splits.iter().enumerate() {
            seg_tile_begin.push(tile_seg.len() as u32);
            let n_tiles = sp.len.div_ceil(block).max(1);
            for k in 0..n_tiles {
                tile_seg.push(s as u32);
                tile_off.push(k * block);
            }
        }
        seg_tile_begin.push(tile_seg.len() as u32);
        let n_tiles = tile_seg.len();
        let n_seg = splits.len();

        // The tile kernels' geometry. `block` is a comptime argument, and it
        // is a constant in both shapes, so each kernel compiles once per shape
        // however the tile count moves from level to level.
        let (tile_count, tile_dim) = if coop {
            (launch::cubes_1d(c, n_tiles as u32), CubeDim::new_1d(block))
        } else {
            // A unit is worth a whole tile of predicate evaluations.
            launch::elementwise_with_work(c, n_tiles, block as usize)
        };
        // Category bit sets, concatenated; `seg_cat_base` gives each its start.
        let mut cat_bits: Vec<u32> = Vec::new();
        let mut seg_cat_base = Vec::with_capacity(n_seg);
        for sp in splits {
            seg_cat_base.push(cat_bits.len() as u32);
            cat_bits.extend_from_slice(&sp.cat_bits);
        }
        if cat_bits.is_empty() {
            cat_bits.push(0); // kernels still bind the array
        }

        let seg_begin: Vec<u32> = splits.iter().map(|s| s.begin).collect();
        let seg_len: Vec<u32> = splits.iter().map(|s| s.len).collect();
        let seg_fidx: Vec<u32> = splits.iter().map(|s| s.fidx).collect();
        let seg_cond: Vec<i64> = splits.iter().map(|s| s.cond).collect();
        // bit 0 = default_left, bit 1 = is_cat.
        let seg_flags: Vec<u32> = splits
            .iter()
            .map(|s| u32::from(s.default_left) | (u32::from(!s.cat_bits.is_empty()) << 1))
            .collect();

        let d_tile_seg = c.create_from_slice(bytemuck::cast_slice(&tile_seg));
        let d_tile_off = c.create_from_slice(bytemuck::cast_slice(&tile_off));
        let d_seg_tile_begin = c.create_from_slice(bytemuck::cast_slice(&seg_tile_begin));
        let d_seg_begin = c.create_from_slice(bytemuck::cast_slice(&seg_begin));
        let d_seg_len = c.create_from_slice(bytemuck::cast_slice(&seg_len));
        let d_seg_fidx = c.create_from_slice(bytemuck::cast_slice(&seg_fidx));
        let d_seg_cond = c.create_from_slice(bytemuck::cast_slice(&seg_cond));
        let d_seg_flags = c.create_from_slice(bytemuck::cast_slice(&seg_flags));
        let d_seg_cat_base = c.create_from_slice(bytemuck::cast_slice(&seg_cat_base));
        let d_cat_bits = c.create_from_slice(bytemuck::cast_slice(&cat_bits));

        let d_tile_left = c.empty(n_tiles * size_of::<u32>());
        let d_seg_left = c.empty(n_seg * size_of::<u32>());

        let cur = self.cur;
        let other = 1 - cur;
        self.catch_up(splits, block, tile_dim);
        let src = self.bufs[cur].clone();
        let dst = self.bufs[other].clone();

        // SAFETY: the kernel guards every index against the lengths it is
        // given; see the `gpu` module docs on unchecked launches.
        unsafe {
            count_tile_kernel::launch_unchecked::<R>(
                c,
                tile_count.clone(),
                tile_dim,
                ArrayArg::from_raw_parts(src.clone(), self.n_rows),
                ArrayArg::from_raw_parts(ell.gidx.clone(), ell.gidx_len),
                ArrayArg::from_raw_parts(ell.gidx_t.clone(), ell.gidx_t_len),
                ArrayArg::from_raw_parts(ell.cut_ptrs.clone(), ell.n_cuts),
                ArrayArg::from_raw_parts(d_cat_bits.clone(), cat_bits.len()),
                ArrayArg::from_raw_parts(d_tile_seg.clone(), n_tiles),
                ArrayArg::from_raw_parts(d_tile_off.clone(), n_tiles),
                ArrayArg::from_raw_parts(d_seg_begin.clone(), n_seg),
                ArrayArg::from_raw_parts(d_seg_len.clone(), n_seg),
                ArrayArg::from_raw_parts(d_seg_fidx.clone(), n_seg),
                ArrayArg::from_raw_parts(d_seg_cond.clone(), n_seg),
                ArrayArg::from_raw_parts(d_seg_flags.clone(), n_seg),
                ArrayArg::from_raw_parts(d_seg_cat_base.clone(), n_seg),
                ArrayArg::from_raw_parts(d_tile_left.clone(), n_tiles),
                ell.row_stride,
                ell.n_rows as u32,
                ell.base_rowid,
                ell.null_value,
                ell.dense,
                ell.compressed,
                ell.bits,
                coop,
                block as usize,
            );
        }

        // SAFETY: the kernel guards every index against the lengths it is
        // given; see the `gpu` module docs on unchecked launches.
        unsafe {
            scatter_tile_kernel::launch_unchecked::<R>(
                c,
                tile_count,
                tile_dim,
                ArrayArg::from_raw_parts(src, self.n_rows),
                ArrayArg::from_raw_parts(dst, self.n_rows),
                ArrayArg::from_raw_parts(ell.gidx.clone(), ell.gidx_len),
                ArrayArg::from_raw_parts(ell.gidx_t.clone(), ell.gidx_t_len),
                ArrayArg::from_raw_parts(ell.cut_ptrs.clone(), ell.n_cuts),
                ArrayArg::from_raw_parts(d_cat_bits, cat_bits.len()),
                ArrayArg::from_raw_parts(d_tile_seg.clone(), n_tiles),
                ArrayArg::from_raw_parts(d_tile_off.clone(), n_tiles),
                ArrayArg::from_raw_parts(d_tile_left, n_tiles),
                ArrayArg::from_raw_parts(d_seg_begin.clone(), n_seg),
                ArrayArg::from_raw_parts(d_seg_len.clone(), n_seg),
                ArrayArg::from_raw_parts(d_seg_fidx, n_seg),
                ArrayArg::from_raw_parts(d_seg_cond, n_seg),
                ArrayArg::from_raw_parts(d_seg_flags, n_seg),
                ArrayArg::from_raw_parts(d_seg_cat_base, n_seg),
                ArrayArg::from_raw_parts(d_seg_tile_begin, n_seg + 1),
                ArrayArg::from_raw_parts(d_seg_left.clone(), n_seg),
                ell.row_stride,
                ell.n_rows as u32,
                ell.base_rowid,
                ell.null_value,
                ell.dense,
                ell.compressed,
                ell.bits,
                coop,
                block as usize,
            );
        }

        // The rewritten segments are now current in the other buffer; make it
        // the front buffer rather than copy them back. Segments not named in
        // `splits` keep their rows where they were, and `side` says where.
        for sp in splits {
            let (b, e) = (sp.begin as usize, (sp.begin + sp.len) as usize);
            self.side[b..e].fill(other as u8);
        }
        self.cur = other;

        // One small read per level, of one count per split node. Upstream's
        // `RowPartitioner` keeps its segment table on the host the same way.
        let bytes = self.client.read_one_unchecked(d_seg_left);
        Ok(bytemuck::cast_slice::<u8, u32>(&bytes).to_vec())
    }

    /// Bring every segment of `splits` that is current in the back buffer
    /// into the front one, so the tile kernels can read them all from one
    /// binding. A no-op on the depth-wise path, where a level only ever splits
    /// the segments the previous partition wrote.
    fn catch_up(&mut self, splits: &[SegmentSplit], block: u32, tile_dim: CubeDim) {
        let cur = self.cur;
        let stale: Vec<&SegmentSplit> = splits
            .iter()
            .filter(|sp| sp.len > 0 && self.side[sp.begin as usize] as usize != cur)
            .collect();
        if stale.is_empty() {
            return;
        }
        let c = &self.client;
        let mut tile_seg = Vec::new();
        let mut tile_off = Vec::new();
        for (s, sp) in stale.iter().enumerate() {
            for k in 0..sp.len.div_ceil(block) {
                tile_seg.push(s as u32);
                tile_off.push(k * block);
            }
        }
        let seg_begin: Vec<u32> = stale.iter().map(|s| s.begin).collect();
        let seg_len: Vec<u32> = stale.iter().map(|s| s.len).collect();
        let n_tiles = tile_seg.len();
        let n_seg = stale.len();
        let d_tile_seg = c.create_from_slice(bytemuck::cast_slice(&tile_seg));
        let d_tile_off = c.create_from_slice(bytemuck::cast_slice(&tile_off));
        let d_seg_begin = c.create_from_slice(bytemuck::cast_slice(&seg_begin));
        let d_seg_len = c.create_from_slice(bytemuck::cast_slice(&seg_len));
        // SAFETY: the kernel guards every index against the lengths it is
        // given; see the `gpu` module docs on unchecked launches.
        unsafe {
            commit_tile_kernel::launch_unchecked::<R>(
                c,
                launch::cubes_1d(c, n_tiles as u32),
                tile_dim,
                ArrayArg::from_raw_parts(self.bufs[1 - cur].clone(), self.n_rows),
                ArrayArg::from_raw_parts(self.bufs[cur].clone(), self.n_rows),
                ArrayArg::from_raw_parts(d_tile_seg, n_tiles),
                ArrayArg::from_raw_parts(d_tile_off, n_tiles),
                ArrayArg::from_raw_parts(d_seg_begin, n_seg),
                ArrayArg::from_raw_parts(d_seg_len, n_seg),
                block as usize,
            );
        }
        for sp in stale {
            let (b, e) = (sp.begin as usize, (sp.begin + sp.len) as usize);
            self.side[b..e].fill(cur as u8);
        }
    }
}
