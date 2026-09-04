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
//! # Why three kernels
//!
//! A segment can hold millions of rows, so a workgroup-per-segment scan would
//! leave the machine idle at the top of the tree, where there is only one
//! segment. Instead the level's rows are cut into fixed-size tiles, and every
//! tile of every segment is processed in parallel:
//!
//! 1. [`count_tile_kernel`] — one workgroup per tile, counting the rows that go
//!    left.
//! 2. [`scan_tiles_kernel`] — one workgroup per *segment*, exclusive-scanning
//!    that segment's tile counts. This array has one entry per tile, not per
//!    row, so a single workgroup is ample and no host round trip is needed.
//! 3. [`scatter_tile_kernel`] — one workgroup per tile again, re-deriving the
//!    predicate and writing each row to its final slot.
//!
//! The partition is stable on both sides, which is what
//! [`crate::tree::hist`]'s `partition_block` produces.

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::ellpack::DeviceEllpack;
use super::launch;
use crate::error::Result;

/// Threads per partitioning workgroup, and rows per tile.
pub const PART_BLOCK: u32 = 256;

/// Which side of a split one row falls on.
///
/// Port of `HistGrower::goes_left` (`src/tree/hist.rs`) against the ELLPACK
/// layouts: a row with no value for the feature follows `default_left`, a
/// numeric split compares the *feature-local* bin against `cond`, and a
/// categorical split tests membership of the right-hand category set.
#[cube]
#[allow(clippy::too_many_arguments)]
fn goes_left(
    gidx: &Array<u32>,
    cut_ptrs: &Array<u32>,
    cat_bits: &Array<u32>,
    row: u32,
    fidx: u32,
    cond: i64,
    default_left: bool,
    is_cat: bool,
    cat_base: u32,
    row_stride: u32,
    base_rowid: u32,
    null_value: u32,
    #[comptime] dense: bool,
    #[comptime] compressed: bool,
) -> bool {
    let row_begin = (row - base_rowid) * row_stride;
    let fbegin = cut_ptrs[fidx as usize];
    let fend = cut_ptrs[(fidx + 1u32) as usize];

    // `EllpackDeviceAccessor::GetBinIndex`. The dense layouts index the
    // feature directly; the sparse layout stores a row's entries in ascending
    // global-bin order, so the feature's entry is found by searching its bin
    // range.
    let raw = if dense || compressed {
        gidx[(row_begin + fidx) as usize]
    } else {
        let found = RuntimeCell::<u32>::new(null_value);
        let k = RuntimeCell::<u32>::new(0u32);
        while k.read() < row_stride {
            let v = gidx[(row_begin + k.read()) as usize];
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

/// Count the rows of each tile that go left.
///
/// `tile_seg[t]` names the segment tile `t` belongs to and `tile_off[t]` its
/// first row within that segment.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn count_tile_kernel(
    ridx: &Array<u32>,
    gidx: &Array<u32>,
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
    base_rowid: u32,
    null_value: u32,
    #[comptime] dense: bool,
    #[comptime] compressed: bool,
    #[comptime] block: usize,
) {
    // A whole cube, not a unit, maps to one tile, so an overprovisioned grid
    // (`launch::cubes_1d` rounds the tile count up to a rectangle) can hand
    // out cubes past the real tile count; skip them uniformly (every unit in
    // the cube computes the same `tile`, so this branches the whole cube the
    // same way and `sync_cube` below stays safe).
    let tile = CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_Z * CUBE_COUNT_X * CUBE_COUNT_Y;
    if tile < tile_seg.len() as u32 {
        let s = tile_seg[tile as usize];
        let off = tile_off[tile as usize] + UNIT_POS_X;
        let len = seg_len[s as usize];

        let mut s_flag = SharedMemory::<u32>::new(block);
        let t = UNIT_POS_X as usize;

        let live = off < len;
        let flags = seg_flags[s as usize];
        let left = if live {
            goes_left(
                gidx,
                cut_ptrs,
                cat_bits,
                ridx[(seg_begin[s as usize] + off) as usize],
                seg_fidx[s as usize],
                seg_cond[s as usize],
                (flags & 1u32) == 1u32,
                (flags & 2u32) == 2u32,
                seg_cat_base[s as usize],
                row_stride,
                base_rowid,
                null_value,
                dense,
                compressed,
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
    }
}

/// Exclusive-scan each segment's tile counts, and record the segment total.
///
/// One workgroup per segment; `seg_tile_begin[s]..seg_tile_begin[s+1]` is the
/// segment's run of tiles.
#[cube(launch)]
pub fn scan_tiles_kernel(
    tile_left: &Array<u32>,
    seg_tile_begin: &Array<u32>,
    tile_base: &mut Array<u32>,
    seg_left: &mut Array<u32>,
    #[comptime] block: usize,
) {
    // A whole cube maps to one segment; see `count_tile_kernel` for why an
    // overprovisioned grid needs this guard.
    let s = CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_Z * CUBE_COUNT_X * CUBE_COUNT_Y;
    if s < seg_left.len() as u32 {
        let begin = seg_tile_begin[s as usize];
        let end = seg_tile_begin[(s + 1u32) as usize];

        // Serial over the segment's tiles: there is one entry per tile, not
        // per row, so this is a short loop even for a very wide segment.
        if UNIT_POS_X == 0u32 {
            let acc = RuntimeCell::<u32>::new(0u32);
            let i = RuntimeCell::<u32>::new(begin);
            while i.read() < end {
                let k = i.read();
                tile_base[k as usize] = acc.read();
                acc.store(acc.read() + tile_left[k as usize]);
                i.store(k + 1u32);
            }
            seg_left[s as usize] = acc.read();
        }
    }
    comptime![let _ = block;];
}

/// Write every row to its final slot.
///
/// Left rows land at `seg_begin + tile_base[t] + rank`, right rows after all
/// the left ones — `seg_begin + seg_left[s] + (tile_off - tile_base[t]) +
/// rank_right` — which keeps both sides in their original order.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn scatter_tile_kernel(
    ridx: &Array<u32>,
    out: &mut Array<u32>,
    gidx: &Array<u32>,
    cut_ptrs: &Array<u32>,
    cat_bits: &Array<u32>,
    tile_seg: &Array<u32>,
    tile_off: &Array<u32>,
    tile_base: &Array<u32>,
    seg_begin: &Array<u32>,
    seg_len: &Array<u32>,
    seg_fidx: &Array<u32>,
    seg_cond: &Array<i64>,
    seg_flags: &Array<u32>,
    seg_cat_base: &Array<u32>,
    seg_left: &Array<u32>,
    row_stride: u32,
    base_rowid: u32,
    null_value: u32,
    #[comptime] dense: bool,
    #[comptime] compressed: bool,
    #[comptime] block: usize,
) {
    // A whole cube maps to one tile; see `count_tile_kernel` for why an
    // overprovisioned grid needs this guard.
    let tile = CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_Z * CUBE_COUNT_X * CUBE_COUNT_Y;
    if tile < tile_seg.len() as u32 {
        let s = tile_seg[tile as usize];
        let toff = tile_off[tile as usize];
        let off = toff + UNIT_POS_X;
        let len = seg_len[s as usize];
        let begin = seg_begin[s as usize];

        let mut s_flag = SharedMemory::<u32>::new(block);
        let t = UNIT_POS_X as usize;

        let live = off < len;
        let row = if live { ridx[(begin + off) as usize] } else { 0u32.into() };
        let flags = seg_flags[s as usize];
        let left = if live {
            goes_left(
                gidx,
                cut_ptrs,
                cat_bits,
                row,
                seg_fidx[s as usize],
                seg_cond[s as usize],
                (flags & 1u32) == 1u32,
                (flags & 2u32) == 2u32,
                seg_cat_base[s as usize],
                row_stride,
                base_rowid,
                null_value,
                dense,
                compressed,
            )
        } else {
            false.into()
        };
        s_flag[t] = u32::cast_from(left);
        sync_cube();

        scan_flags(&mut s_flag, block);

        if live {
            // Inclusive scan minus this row's own flag gives its rank among
            // the left-going rows of the tile.
            let rank_left = s_flag[t] - u32::cast_from(left);
            let base = tile_base[tile as usize];
            let dst = if left {
                begin + base + rank_left
            } else {
                // Rows before this one in the tile that went right, plus the
                // tiles before it: `off - base` counts both.
                begin + seg_left[s as usize] + (off - base) - rank_left
            };
            out[dst as usize] = row;
        }
    }
}

/// Copy the rewritten segments from the scratch buffer back into `ridx`.
#[cube(launch)]
pub fn commit_tile_kernel(
    scratch: &Array<u32>,
    ridx: &mut Array<u32>,
    tile_seg: &Array<u32>,
    tile_off: &Array<u32>,
    seg_begin: &Array<u32>,
    seg_len: &Array<u32>,
) {
    // A whole cube maps to one tile; see `count_tile_kernel` for why an
    // overprovisioned grid needs this guard.
    let tile = CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_Z * CUBE_COUNT_X * CUBE_COUNT_Y;
    if tile < tile_seg.len() as u32 {
        let s = tile_seg[tile as usize];
        let off = tile_off[tile as usize] + UNIT_POS_X;
        if off < seg_len[s as usize] {
            let i = (seg_begin[s as usize] + off) as usize;
            ridx[i] = scratch[i];
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
    ridx: Handle,
    scratch: Handle,
    n_rows: usize,
}

impl<R: Runtime> RowPartitioner<R> {
    /// Start with `rows` in the single root segment.
    pub fn new(client: ComputeClient<R>, rows: &[u32]) -> Self {
        let ridx = client.create_from_slice(bytemuck::cast_slice(rows));
        let scratch = client.empty(rows.len().max(1) * size_of::<u32>());
        Self { client, ridx, scratch, n_rows: rows.len() }
    }

    /// All rows, in the root segment.
    pub fn all_rows(client: ComputeClient<R>, n_rows: usize) -> Self {
        let rows: Vec<u32> = (0..n_rows as u32).collect();
        Self::new(client, &rows)
    }

    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// The row-index buffer. Segments index into it.
    pub fn ridx(&self) -> &Handle {
        &self.ridx
    }

    /// Read the whole row index back, for tests and for the final leaf pass.
    pub fn read(&self) -> Vec<u32> {
        let bytes = self.client.read_one_unchecked(self.ridx.clone());
        bytemuck::cast_slice(&bytes).to_vec()
    }

    /// Partition every segment in `splits` in one pass.
    ///
    /// Returns each split's left-child length, so the caller can turn one
    /// segment into the children's two.
    pub fn partition(&mut self, ell: &DeviceEllpack, splits: &[SegmentSplit]) -> Result<Vec<u32>> {
        if splits.is_empty() {
            return Ok(Vec::new());
        }
        let c = &self.client;
        let block = launch::block_1d(c, PART_BLOCK);

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
        let d_tile_base = c.empty(n_tiles * size_of::<u32>());
        let d_seg_left = c.empty(n_seg * size_of::<u32>());

        count_tile_kernel::launch::<R>(
            c,
            launch::cubes_1d(c, n_tiles as u32),
            CubeDim::new_1d(block),
            unsafe { ArrayArg::from_raw_parts(self.ridx.clone(), self.n_rows) },
            unsafe { ArrayArg::from_raw_parts(ell.gidx.clone(), ell.gidx_len) },
            unsafe { ArrayArg::from_raw_parts(ell.cut_ptrs.clone(), ell.n_cuts) },
            unsafe { ArrayArg::from_raw_parts(d_cat_bits.clone(), cat_bits.len()) },
            unsafe { ArrayArg::from_raw_parts(d_tile_seg.clone(), n_tiles) },
            unsafe { ArrayArg::from_raw_parts(d_tile_off.clone(), n_tiles) },
            unsafe { ArrayArg::from_raw_parts(d_seg_begin.clone(), n_seg) },
            unsafe { ArrayArg::from_raw_parts(d_seg_len.clone(), n_seg) },
            unsafe { ArrayArg::from_raw_parts(d_seg_fidx.clone(), n_seg) },
            unsafe { ArrayArg::from_raw_parts(d_seg_cond.clone(), n_seg) },
            unsafe { ArrayArg::from_raw_parts(d_seg_flags.clone(), n_seg) },
            unsafe { ArrayArg::from_raw_parts(d_seg_cat_base.clone(), n_seg) },
            unsafe { ArrayArg::from_raw_parts(d_tile_left.clone(), n_tiles) },
            ell.row_stride,
            ell.base_rowid,
            ell.null_value,
            ell.dense,
            ell.compressed,
            block as usize,
        );

        scan_tiles_kernel::launch::<R>(
            c,
            launch::cubes_1d(c, n_seg as u32),
            CubeDim::new_1d(block),
            unsafe { ArrayArg::from_raw_parts(d_tile_left, n_tiles) },
            unsafe { ArrayArg::from_raw_parts(d_seg_tile_begin, n_seg + 1) },
            unsafe { ArrayArg::from_raw_parts(d_tile_base.clone(), n_tiles) },
            unsafe { ArrayArg::from_raw_parts(d_seg_left.clone(), n_seg) },
            block as usize,
        );

        scatter_tile_kernel::launch::<R>(
            c,
            launch::cubes_1d(c, n_tiles as u32),
            CubeDim::new_1d(block),
            unsafe { ArrayArg::from_raw_parts(self.ridx.clone(), self.n_rows) },
            unsafe { ArrayArg::from_raw_parts(self.scratch.clone(), self.n_rows) },
            unsafe { ArrayArg::from_raw_parts(ell.gidx.clone(), ell.gidx_len) },
            unsafe { ArrayArg::from_raw_parts(ell.cut_ptrs.clone(), ell.n_cuts) },
            unsafe { ArrayArg::from_raw_parts(d_cat_bits, cat_bits.len()) },
            unsafe { ArrayArg::from_raw_parts(d_tile_seg.clone(), n_tiles) },
            unsafe { ArrayArg::from_raw_parts(d_tile_off.clone(), n_tiles) },
            unsafe { ArrayArg::from_raw_parts(d_tile_base, n_tiles) },
            unsafe { ArrayArg::from_raw_parts(d_seg_begin.clone(), n_seg) },
            unsafe { ArrayArg::from_raw_parts(d_seg_len.clone(), n_seg) },
            unsafe { ArrayArg::from_raw_parts(d_seg_fidx, n_seg) },
            unsafe { ArrayArg::from_raw_parts(d_seg_cond, n_seg) },
            unsafe { ArrayArg::from_raw_parts(d_seg_flags, n_seg) },
            unsafe { ArrayArg::from_raw_parts(d_seg_cat_base, n_seg) },
            unsafe { ArrayArg::from_raw_parts(d_seg_left.clone(), n_seg) },
            ell.row_stride,
            ell.base_rowid,
            ell.null_value,
            ell.dense,
            ell.compressed,
            block as usize,
        );

        // Only the rewritten segments changed, so copy just those ranges back
        // — the segments not named in `splits` must keep their rows. Done on
        // device: the point of the three-kernel scan is that a level costs no
        // host round trip beyond the small per-segment counts below.
        commit_tile_kernel::launch::<R>(
            c,
            launch::cubes_1d(c, n_tiles as u32),
            CubeDim::new_1d(block),
            unsafe { ArrayArg::from_raw_parts(self.scratch.clone(), self.n_rows) },
            unsafe { ArrayArg::from_raw_parts(self.ridx.clone(), self.n_rows) },
            unsafe { ArrayArg::from_raw_parts(d_tile_seg, n_tiles) },
            unsafe { ArrayArg::from_raw_parts(d_tile_off, n_tiles) },
            unsafe { ArrayArg::from_raw_parts(d_seg_begin, n_seg) },
            unsafe { ArrayArg::from_raw_parts(d_seg_len, n_seg) },
        );

        // One small read per level, of one count per split node. Upstream's
        // `RowPartitioner` keeps its segment table on the host the same way.
        let bytes = self.client.read_one_unchecked(d_seg_left);
        Ok(bytemuck::cast_slice::<u8, u32>(&bytes).to_vec())
    }
}
