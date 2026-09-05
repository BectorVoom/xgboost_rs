//! Gradient histogram construction, ported from
//! `xgboost/src/tree/gpu_hist/histogram.{cu,cuh}` (single-target path:
//! `StHistKernel` / `HistKernelOneNodeTarget` / `AtomicAddGpairShared` /
//! `AtomicAddGpairGlobal` / `AtomicAdd64As32`) and the subtraction trick from
//! `DeviceHistogramBuilder::SubtractionTrick`.
//!
//! # Host API
//!
//! [`HistogramBuilder`] configures the engine and uploads the ELLPACK matrix
//! once; the resulting [`HistogramEngine`] then builds any number of node
//! histograms against device-resident data — mirroring how XGBoost's
//! `DeviceHistogramBuilder` is `Reset` once per iteration and dispatched per
//! node.
//!
//! # Accumulator representation
//!
//! A histogram bin is a `GradientPairInt64` (two `i64`s). On device it is
//! stored as four `u32` words `[grad_lo, grad_hi, hess_lo, hess_hi]` and
//! updated with two 32-bit atomics plus manual carry propagation — the exact
//! scheme of `AtomicAdd64As32` in histogram.cuh. XGBoost uses it for shared
//! memory because 64-bit shared atomics are slow on NVIDIA hardware; here it
//! is also used for global memory so the kernel runs on runtimes without
//! 64-bit atomics (WebGPU, older Vulkan devices). The end result is bitwise
//! identical either way: two's-complement addition is associative and
//! commutative, so the quantised sums are deterministic.

use cubecl::features::AtomicUsage;
use cubecl::ir::{StorageType, Type};
use cubecl::bytes::Bytes;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::ellpack::{EllpackMatrix, device_bits, load_bin, pack_bins};
use super::launch;
use super::{DeviceGpairs, DeviceHistogram, DeviceRows, GradientPairInt64};
use crate::error::{Error, Result};

/// `kItemsPerThread` in histogram.cu.
pub const ITEMS_PER_THREAD: u32 = 8;
/// Block size. The CUDA code tunes this per SM architecture (768–1024);
/// 256 is a portable choice within every backend's workgroup limits.
pub const BLOCK_THREADS: u32 = 256;
/// Default shared-memory budget per workgroup (the CUDA code queries
/// `MaxSharedMemoryOptin`; 48 KiB is the portable baseline).
pub const DEFAULT_SHMEM_BYTES: usize = 48 * 1024;
/// Default upper bound on the grid size, standing in for the occupancy-based
/// `n_blocks_per_mp * n_mps` limit computed by `HistKernel::SetCfg`.
pub const DEFAULT_MAX_BLOCKS_PER_GROUP: u32 = 1024;
/// Rows per work item on the atomic-free path: the CPU grower's `BLOCK_ROWS`.
///
/// Enough `(row, feature)` visits that a unit is worth its thread dispatch,
/// few enough that a node of a few tens of thousands of rows still yields an
/// item per core.
pub const CHUNK_ROWS: u32 = 4096;
/// Floor on the per-node chunk cap of the atomic-free path: the CPU grower's
/// `MAX_LANES`. The cap itself is raised to twice the serial cube width, so a
/// wide machine still gets an item per unit from a single node.
const MAX_CHUNKS_PER_NODE: u32 = 16;
/// Rows per block of the serial histogram's two-phase row loop on a
/// plane-less runtime; `0` is the plain one-row-at-a-time loop.
///
/// At depth a node's rows are a sparse, sorted subset of the matrix — at
/// depth 9 of a 200 000-row fit, one row in 512 — so every row is a cache
/// miss the hardware prefetcher cannot see coming, and a loop that finishes
/// one row before starting the next overlaps only as many misses as its
/// reorder window holds: about two rows' worth. The kernel therefore visits
/// the first bin of each cache line of a block of rows first (every miss of
/// the block in flight together), then the remaining bins of the same rows,
/// now from cache. Measured on 256 nodes × 390 random rows, 32 features:
/// 3.7 ms → 2.5 ms per build; 16 nodes × 6 250 rows: 3.3 → 2.0. Blocks of 8,
/// 16 and 32 rows were within noise of each other; the pattern is exact
/// because the order of `i64` adds into a private slot cannot change a sum.
/// One node over near-sequential rows (level 0) is unchanged at 0.4 ns per
/// visit, as the prefetcher was already ahead there.
const TOUCH_ROWS: u32 = 16;

/// Port of `AtomicAdd64As32` (histogram.cuh): add a signed 64-bit value into
/// two consecutive `u32` words `[lo, hi]` using 32-bit atomics with carry.
#[cube]
fn atomic_add_i64_as_u32_shared(dst: &SharedMemory<Atomic<u32>>, word: u32, val: i64) {
    let lo = u32::cast_from(val & 0xFFFF_FFFFi64);
    let hi = u32::cast_from((val >> 32) & 0xFFFF_FFFFi64);
    let old = dst[word as usize].fetch_add(lo);
    let carry = u32::cast_from(old > 0xFFFF_FFFFu32 - lo);
    dst[(word + 1) as usize].fetch_add(hi + carry);
}

/// Same carry scheme against a global-memory histogram, used when the runtime
/// has no native 64-bit atomics; yields bit-identical results.
#[cube]
fn atomic_add_i64_as_u32_global(dst: &Array<Atomic<u32>>, word: u32, val: i64) {
    let lo = u32::cast_from(val & 0xFFFF_FFFFi64);
    let hi = u32::cast_from((val >> 32) & 0xFFFF_FFFFi64);
    let old = dst[word as usize].fetch_add(lo);
    let carry = u32::cast_from(old > 0xFFFF_FFFFu32 - lo);
    dst[(word + 1) as usize].fetch_add(hi + carry);
}

/// Global-memory gradient-pair add: the native branch is the port of
/// `AtomicAddGpairGlobal` (histogram.cu), which relies on 64-bit `atomicAdd`
/// ("global 64 bit integer atomics at the time of writing do not benefit from
/// being separated into two 32 bit atomics"); the fallback is the u32-carry
/// scheme. Exactly one of `hist32` / `hist64` is real — the other is a 1-word
/// dummy, and the comptime branch removes every access to it.
#[cube]
fn add_gpair_global(
    hist32: &Array<Atomic<u32>>,
    hist64: &Array<Atomic<i64>>,
    bin: u32,
    grad: i64,
    hess: i64,
    #[comptime] native_i64: bool,
) {
    if native_i64 {
        hist64[(bin * 2) as usize].fetch_add(grad);
        hist64[(bin * 2 + 1) as usize].fetch_add(hess);
    } else {
        atomic_add_i64_as_u32_global(hist32, bin * 4, grad);
        atomic_add_i64_as_u32_global(hist32, bin * 4 + 2, hess);
    }
}

/// Port of `StHistKernel` + `HistKernelOneNodeTarget` (histogram.cu): build
/// the gradient histogram for one node, one target.
///
/// Grid: `CUBE_POS_X` indexes the grid-strided tile loop, `CUBE_POS_Y` the
/// feature group (as in the CUDA `dim3 conf(n_blocks, n_groups)` launch).
///
/// * `gidx` — ELLPACK bin matrix, `n_rows * row_stride` entries packed at
///   `bits` per entry (see [`super::ellpack::load_bin`]).
/// * `cut_ptrs` — per-feature bin offsets (`feature_segments`).
/// * `groups` — feature groups, 4 words each:
///   `[start_feature, num_features, start_bin, num_bins]`.
/// * `ridx` — row indices belonging to the node.
/// * `gpair` — quantised gradients, interleaved `[grad, hess]` per row.
/// * `hist` — global histogram as 4 `u32` words per bin (used when
///   `native_i64` is false; 1-word dummy otherwise).
/// * `hist_i64` — global histogram as 2 `i64` words per bin (used when
///   `native_i64` is true; 1-word dummy otherwise). Same byte layout.
/// * `node_n_ridx` / `node_ridx_base` / `node_hist_offset` — per node, indexed
///   by `CUBE_POS_Z`: how many rows it has, where its slice of the shared
///   `ridx` starts, and the bin offset of its slot in `hist`. A level is one
///   launch rather than one launch per node, which is where a deep tree's time
///   used to go.
/// * `dense` / `compressed` — comptime layout flags (`kDense`/`kCompressed`).
/// * `use_shared` — comptime `kSharedMem`: privatise the group's bins in
///   shared memory, then flush to global.
/// * `native_i64` — comptime: use native 64-bit atomics for global-memory
///   accumulation (`AtomicAddGpairGlobal`) instead of the u32-carry scheme.
/// * `smem_words` — comptime shared buffer size; must be at least
///   `4 * max(group num_bins)` when `use_shared` (4 = `SMEM_MIN_WORDS` filler
///   otherwise, since the declaration cannot be elided).
///
/// A runtime with no atomics at all runs
/// [`hist_atomic_free_kernel`](fn@hist_atomic_free_kernel) instead.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn hist_kernel(
    gidx: &Array<u32>,
    cut_ptrs: &Array<u32>,
    groups: &Array<u32>,
    ridx: &Array<u32>,
    gpair: &Array<i64>,
    hist: &mut Array<Atomic<u32>>,
    hist_i64: &mut Array<Atomic<i64>>,
    node_n_ridx: &Array<u32>,
    node_ridx_base: &Array<u32>,
    node_hist_offset: &Array<u32>,
    row_stride: u32,
    base_rowid: u32,
    null_value: u32,
    #[comptime] dense: bool,
    #[comptime] compressed: bool,
    #[comptime] use_shared: bool,
    #[comptime] native_i64: bool,
    #[comptime] smem_words: usize,
    #[comptime] bits: u32,
) {
    // `CUBE_POS_Z` selects the node, so a whole level of them is one launch —
    // upstream's `BuildHistBatch`. A node with fewer rows simply leaves its
    // grid-strided loop early.
    let node = CUBE_POS_Z;
    let n_ridx = node_n_ridx[node as usize];
    let ridx_base = node_ridx_base[node as usize];
    let hist_offset = node_hist_offset[node as usize];

    let group_base = (CUBE_POS_Y * 4) as usize;
    let start_feature = groups[group_base];
    let num_features = groups[group_base + 1];
    let start_bin = groups[group_base + 2];
    let num_bins = groups[group_base + 3];

    let smem = SharedMemory::<Atomic<u32>>::new(smem_words);

    if use_shared {
        // dh::BlockFill of the privatised histogram.
        let mut i = UNIT_POS_X;
        while i < num_bins * 4 {
            smem[i as usize].store(0u32);
            i += CUBE_DIM_X;
        }
        sync_cube();
    }

    // `feature_stride` in HistKernelOneNodeTarget: for compressed layouts each
    // block only walks its group's features, otherwise the full row stride.
    let feature_stride = if compressed { num_features } else { row_stride };
    let n_elements = n_ridx * feature_stride;

    let tile_size = CUBE_DIM_X * ITEMS_PER_THREAD;
    // Grid-strided tile loop over (row, feature) pairs.
    let mut offset = CUBE_POS_X * tile_size;
    let stride = tile_size * CUBE_COUNT_X;

    while offset < n_elements {
        #[unroll]
        for j in 0..ITEMS_PER_THREAD {
            let idx = offset + j * CUBE_DIM_X + UNIT_POS_X;
            if idx < n_elements {
                // process_valid_tile: unravel idx -> (row-in-set, feature-in-group).
                let ridx_in_set = idx / feature_stride;
                let fidx_in_set = idx - ridx_in_set * feature_stride;

                let row = ridx[(ridx_base + ridx_in_set) as usize];
                let fidx = fidx_in_set + start_feature;

                // IterIdx: entry for (row, fidx) in the ELLPACK matrix.
                let entry = (row - base_rowid) * row_stride + fidx;
                let bin = load_bin(gidx, entry, bits);

                if dense || bin != null_value {
                    let grad = gpair[(2 * row) as usize];
                    let hess = gpair[(2 * row + 1) as usize];

                    let mut global_bin = bin;
                    if compressed {
                        global_bin += cut_ptrs[fidx as usize];
                    }

                    if use_shared {
                        let local = global_bin - start_bin;
                        atomic_add_i64_as_u32_shared(&smem, local * 4, grad);
                        atomic_add_i64_as_u32_shared(&smem, local * 4 + 2, hess);
                    } else {
                        add_gpair_global(
                            hist,
                            hist_i64,
                            hist_offset + global_bin,
                            grad,
                            hess,
                            native_i64,
                        );
                    }
                }
            }
        }
        offset += stride;
    }

    if use_shared {
        // Flush the privatised histogram back to global memory.
        sync_cube();
        let mut bin = UNIT_POS_X;
        while bin < num_bins {
            let src = (bin * 4) as usize;
            // The shared accumulator words already hold a finished i64 pair;
            // re-assemble and push it through the global add.
            let grad = i64::cast_from(smem[src].load())
                | (i64::cast_from(smem[src + 1].load()) << 32);
            let hess = i64::cast_from(smem[src + 2].load())
                | (i64::cast_from(smem[src + 3].load()) << 32);
            add_gpair_global(hist, hist_i64, hist_offset + start_bin + bin, grad, hess, native_i64);
            bin += CUBE_DIM_X;
        }
        // Pin every unit to this cube before any of them loops on to the next.
        // A runtime that executes cubes sequentially allocates `smem` once for
        // the whole launch (see the module docs), so without this a unit that
        // has finished flushing would race ahead and re-zero the buffer under a
        // unit still reading it. Redundant where cubes are concurrent.
        sync_cube();
    }
}

/// [`hist_kernel`](fn@hist_kernel) for a runtime that has **no atomics at all**:
/// the serial shape of the histogram build.
///
/// # Why it is a second kernel rather than a comptime branch
///
/// Every other dispatch in this module is a `#[comptime]` flag inside one
/// kernel, with the unused buffer bound as a one-word dummy. That does not work
/// here. `SharedMemory::<Atomic<u32>>::new(..)` and an `Array<Atomic<u32>>`
/// parameter are *type* instantiations, and a backend without atomics rejects
/// them where they are written, not where they are used — the CubeCL CPU
/// runtime answers `not yet implemented: atomic<u32>` at compile time even when
/// the comptime branch that would touch them is dead. An atomic-free kernel
/// therefore has to be a kernel in which the word "atomic" never appears, which
/// is what this is.
///
/// # How it accumulates
///
/// The only runtime without atomics is one whose units are OS threads, where a
/// barrier costs a scheduler round (see [`super::launch::cooperative`]), so
/// this kernel has no barrier and no shared memory at all. A work item is one
/// **row chunk of one node**, and one unit owns it end to end: it zeroes its
/// own `n_bins` slice of `dst`, accumulates every `(row, feature)` of the
/// chunk into that slice, and stops. Nothing it writes is addressable by any
/// other unit, which is what makes the add a plain read-modify-write.
///
/// Where the slice is depends on how many chunks the node has, and the host
/// (`HistogramEngine::build_private_batch`) decides per node:
///
/// * a node of **one** chunk accumulates straight into its slot of the
///   frontier — `dst` is the histogram and `item_dst_base` its slot — so the
///   slot never has to be zeroed by anyone else;
/// * a node of several chunks accumulates into a private `partials` buffer,
///   and [`merge_partials_kernel`](fn@merge_partials_kernel) then sums its
///   chunks bin by bin into the slot, a run of `(node, bin)` lanes per unit.
///
/// # How it walks the rows
///
/// `touch` selects the row loop. `0` is the plain loop: a row, then all its
/// features. `n` visits rows in blocks of `n`: first the bin at the head of
/// every cache line ([`LINE_BINS`] apart) of every row in the block, then the
/// rest of each line — so that at depth, where a node's rows are scattered
/// across the matrix, a block's cache misses are issued together instead of
/// one row at a time. The host passes [`TOUCH_ROWS`] on a plane-less runtime
/// and `0` elsewhere; the arithmetic per visit is [`hist_add`] in both.
///
/// It is the CPU grower's own scheme (`HistGrower::build_hists_target`: single
/// lanes accumulate in place, multi-lane nodes reduce afterwards), with the
/// lane count sized to the machine by the host. The sums are exact `i64` on
/// every path, so the merge order is irrelevant and the result is bit-identical
/// to `hist_kernel`'s — the property the quantiser exists to give.
///
/// Feature groups do not apply: with no shared budget to fit, the whole
/// histogram is one group, and the chunk walks the full `row_stride` of every
/// row.
///
/// * `dst` — `[grad, hess]` pairs, each one `Vector<i64, 2>`; a work item owns
///   the `n_bins` pairs from `item_dst_base[item]` (an offset in pairs).
/// * `item_ridx_begin` / `item_n_ridx` — per work item, its slice of `ridx`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn hist_atomic_free_kernel<N: Size>(
    gidx: &Array<u32>,
    cut_ptrs: &Array<u32>,
    ridx: &Array<u32>,
    gpair: &Array<Vector<i64, N>>,
    dst: &mut Array<Vector<i64, N>>,
    item_ridx_begin: &Array<u32>,
    item_n_ridx: &Array<u32>,
    item_dst_base: &Array<u32>,
    row_stride: u32,
    base_rowid: u32,
    null_value: u32,
    n_items: u32,
    n_bins: u32,
    #[comptime] dense: bool,
    #[comptime] compressed: bool,
    #[comptime] bits: u32,
    #[comptime] touch: u32,
) {
    let item = ABSOLUTE_POS as u32;
    // The elementwise grid overprovisions, so a unit past the table exits.
    if item < n_items {
        let pbase = item_dst_base[item as usize];

        let w = RuntimeCell::<u32>::new(0u32);
        while w.read() < n_bins {
            dst[(pbase + w.read()) as usize] = Vector::<i64, N>::new(0i64);
            w.store(w.read() + 1u32);
        }

        let ridx_begin = item_ridx_begin[item as usize];
        let n_ridx = item_n_ridx[item as usize];
        if touch == 0u32 {
            let i = RuntimeCell::<u32>::new(0u32);
            while i.read() < n_ridx {
                let row = ridx[(ridx_begin + i.read()) as usize];
                let gh = gpair[row as usize];
                let row_begin = (row - base_rowid) * row_stride;

                // One entry per trip. Unrolling by four was measured at 10.0 ms
                // against 10.4 ms per build — noise — so the loop stays plain.
                let f = RuntimeCell::<u32>::new(0u32);
                while f.read() < row_stride {
                    let fidx = f.read();
                    hist_add(
                        gidx, cut_ptrs, dst, pbase, row_begin, fidx, gh, null_value, dense,
                        compressed, bits,
                    );
                    f.store(fidx + 1u32);
                }
                i.store(i.read() + 1u32);
            }
        } else {
            let i = RuntimeCell::<u32>::new(0u32);
            while i.read() < n_ridx {
                let mut block_end = i.read() + touch;
                if block_end > n_ridx {
                    block_end = n_ridx;
                }
                // Phase 1: the first bin of every cache line of every row in
                // the block, so the block's misses are all in flight together.
                let r = RuntimeCell::<u32>::new(i.read());
                while r.read() < block_end {
                    let row = ridx[(ridx_begin + r.read()) as usize];
                    let gh = gpair[row as usize];
                    let row_begin = (row - base_rowid) * row_stride;
                    let f = RuntimeCell::<u32>::new(0u32);
                    while f.read() < row_stride {
                        let fidx = f.read();
                        hist_add(
                            gidx, cut_ptrs, dst, pbase, row_begin, fidx, gh, null_value,
                            dense, compressed, bits,
                        );
                        f.store(fidx + LINE_BINS);
                    }
                    r.store(r.read() + 1u32);
                }
                // Phase 2: the rest of each line.
                let r = RuntimeCell::<u32>::new(i.read());
                while r.read() < block_end {
                    let row = ridx[(ridx_begin + r.read()) as usize];
                    let gh = gpair[row as usize];
                    let row_begin = (row - base_rowid) * row_stride;
                    let l = RuntimeCell::<u32>::new(0u32);
                    while l.read() < row_stride {
                        let mut line_end = l.read() + LINE_BINS;
                        if line_end > row_stride {
                            line_end = row_stride;
                        }
                        let f = RuntimeCell::<u32>::new(l.read() + 1u32);
                        while f.read() < line_end {
                            let fidx = f.read();
                            hist_add(
                                gidx, cut_ptrs, dst, pbase, row_begin, fidx, gh, null_value,
                                dense, compressed, bits,
                            );
                            f.store(fidx + 1u32);
                        }
                        l.store(l.read() + LINE_BINS);
                    }
                    r.store(r.read() + 1u32);
                }
                i.store(block_end);
            }
        }
    }
}

/// Bins per 64-byte cache line of a 32-bit bin matrix.
const LINE_BINS: u32 = 16;

/// One `(row, feature)` visit of the serial histogram: the same arithmetic as
/// `hist_kernel`'s, into the unit's private slot.
///
/// A bin is one `Vector<i64, 2>` — `[grad, hess]` — so the visit is one load,
/// one add and one store rather than two of each, and the slot index is the
/// bin itself. On the CPU runtime's `-O0` code every operation is paid for
/// in full, so the vector form is not a bandwidth trick but an instruction
/// count: measured 2.1 → 1.6 ms on the 200 000-row root build and
/// proportionally on every deeper shape.
#[cube]
#[allow(clippy::too_many_arguments)]
fn hist_add<N: Size>(
    gidx: &Array<u32>,
    cut_ptrs: &Array<u32>,
    dst: &mut Array<Vector<i64, N>>,
    pbase: u32,
    row_begin: u32,
    fidx: u32,
    gh: Vector<i64, N>,
    null_value: u32,
    #[comptime] dense: bool,
    #[comptime] compressed: bool,
    #[comptime] bits: u32,
) {
    let bin = load_bin(gidx, row_begin + fidx, bits);
    if dense || bin != null_value {
        let mut global_bin = bin;
        if compressed {
            global_bin += cut_ptrs[fidx as usize];
        }
        dst[(pbase + global_bin) as usize] += gh;
    }
}

/// Sum each node's private partial histograms into its slot of `hist`.
///
/// A lane is a `(node, bin)`: it adds the bin across the node's work items,
/// `node_item_begin[node]..node_item_end[node]`, and *writes* the total to
/// `hist` at `node_hist_offset[node] + bin` — writes rather than adds, so the
/// slot needs no zeroing first, which is what lets the atomic-free frontier be
/// allocated without a clearing pass. Lanes own disjoint cells, so — like
/// [`hist_atomic_free_kernel`](fn@hist_atomic_free_kernel) — this needs no
/// atomic and no barrier, and the exact `i64` sum does not depend on the item
/// order.
///
/// A unit takes `run` consecutive lanes from `ABSOLUTE_POS * run`: one on a
/// GPU, a static share of the range on the CPU runtime, where the per-cube
/// loop the runtime wraps around the body would otherwise be paid per bin
/// (`launch::elementwise_runs`).
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn merge_partials_kernel(
    partials: &Array<i64>,
    hist: &mut Array<i64>,
    node_item_begin: &Array<u32>,
    node_item_end: &Array<u32>,
    node_hist_offset: &Array<u32>,
    n_bins: u32,
    n_lanes: u32,
    run: u32,
) {
    let first = ABSOLUTE_POS as u32 * run;
    let mut end = first + run;
    if end > n_lanes {
        end = n_lanes;
    }
    let lane = RuntimeCell::<u32>::new(first);
    while lane.read() < end {
        let node = lane.read() / n_bins;
        let bin = lane.read() - node * n_bins;

        let g = RuntimeCell::<i64>::new(0i64);
        let h = RuntimeCell::<i64>::new(0i64);
        let item_end = node_item_end[node as usize];
        let it = RuntimeCell::<u32>::new(node_item_begin[node as usize]);
        while it.read() < item_end {
            let slot = ((it.read() * n_bins + bin) * 2u32) as usize;
            g.store(g.read() + partials[slot]);
            h.store(h.read() + partials[slot + 1]);
            it.store(it.read() + 1u32);
        }

        let cell = (node_hist_offset[node as usize] + bin) as usize;
        hist[cell * 2usize] = g.read();
        hist[cell * 2usize + 1] = h.read();
        lane.store(lane.read() + 1u32);
    }
}

/// Zero a buffer on device.
///
/// A frontier is allocated per batch and must start at zero because the
/// histogram kernel only adds. Filling it host-side and uploading costs a
/// transfer of the whole frontier per level — ~105 MB at depth 10 — for a
/// buffer whose contents are known.
///
/// Vectorised: this is a pure store-bandwidth kernel over the largest buffer
/// the fit allocates, so it writes `Vector`s rather than words — one 128-bit
/// store per unit where the device supports it instead of four 32-bit ones.
/// `n_lines` counts vectors, not words; the caller picks a width that divides
/// the buffer exactly, falling back to 1 when it does not.
#[cube(launch_unchecked)]
pub fn zero_u32_kernel<N: Size>(buf: &mut Array<Vector<u32, N>>, n_lines: u32) {
    let i = ABSOLUTE_POS as u32;
    if i < n_lines {
        buf[i as usize] = Vector::<u32, N>::new(0u32);
    }
}

/// [`subtract_within_kernel`] for a whole batch: `CUBE_POS_Y` selects the node.
///
/// Every parent of a batch shares one buffer — a depth-wise batch is a whole
/// level, produced by a single `build_children` call — so only the slots
/// differ per node.
///
/// Vectorised: a lane is one `Vector` of `N` words, so the index arithmetic
/// is paid once per `N` elements. Offsets and `n_lines` are in vectors; the
/// caller picks `N` so every offset divides.
///
/// A unit takes `run` consecutive lines from its flattened X/Z index times
/// `run`: one on a GPU, a static share of the node's lines on the CPU
/// runtime. Measured scalar at depth 10 this kernel ran at 17 GB/s over a
/// 100 MB frontier and vectorising it changed nothing, because the cost was
/// never the memory: the CPU runtime loops every unit over every cube of the
/// grid, and with one vector per unit that loop's bookkeeping was the kernel.
/// With a run per unit it is one loop over a contiguous range, and the three
/// offset loads happen once per unit rather than once per vector.
///
/// The built child's histogram may still be in pieces: a node the builder cut
/// into several row chunks has one private partial per chunk, `item_begin[
/// node]..item_end[node]` of `partials` (`item_lines` vectors each), and its
/// slot of `frontier` unwritten. Such a node's lanes sum the partials, *write*
/// the built slot and subtract in one pass, so the merge that was a launch of
/// its own ([`merge_partials_kernel`](fn@merge_partials_kernel)) rides along
/// with the subtraction the level makes anyway. A node with one chunk has an
/// empty item range and reads its built slot as before.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn subtract_batch_kernel<N: Size>(
    parent: &Array<Vector<i64, N>>,
    frontier: &mut Array<Vector<i64, N>>,
    partials: &Array<Vector<i64, N>>,
    parent_off: &Array<u32>,
    built_off: &Array<u32>,
    out_off: &Array<u32>,
    item_begin: &Array<u32>,
    item_end: &Array<u32>,
    n_lines: u32,
    item_lines: u32,
    run: u32,
) {
    // Not `ABSOLUTE_POS`: that is linear over the *whole* grid, and this
    // grid's Y axis is the node index, so it would fold the node into the
    // word index. Z, instead, is the overflow the X axis couldn't hold.
    let unit = (CUBE_POS_Z * CUBE_COUNT_X + CUBE_POS_X) * CUBE_DIM_X + UNIT_POS_X;
    let first = unit * run;
    if first < n_lines {
        let mut end = first + run;
        if end > n_lines {
            end = n_lines;
        }
        let node = CUBE_POS_Y as usize;
        // Read the offsets into locals: an index expression on the left of an
        // assignment is taken as a mutable borrow of that array.
        let dst = out_off[node];
        let src_parent = parent_off[node];
        let src_built = built_off[node];
        let it_begin = item_begin[node];
        let it_end = item_end[node];
        let i = RuntimeCell::<u32>::new(first);
        if it_end > it_begin {
            while i.read() < end {
                let k = i.read();
                let mut built = partials[(it_begin * item_lines + k) as usize];
                let it = RuntimeCell::<u32>::new(it_begin + 1u32);
                while it.read() < it_end {
                    built = built + partials[(it.read() * item_lines + k) as usize];
                    it.store(it.read() + 1u32);
                }
                frontier[(src_built + k) as usize] = built;
                frontier[(dst + k) as usize] = parent[(src_parent + k) as usize] - built;
                i.store(k + 1u32);
            }
        } else {
            while i.read() < end {
                let k = i.read();
                frontier[(dst + k) as usize] =
                    parent[(src_parent + k) as usize] - frontier[(src_built + k) as usize];
                i.store(k + 1u32);
            }
        }
    }
}

/// [`subtract_hist_kernel`] for the case the grower actually hits: the built
/// child and the sibling being written are two slots of the *same* frontier
/// buffer.
///
/// It needs its own kernel because binding one buffer twice — once read-only
/// as `built`, once read-write as `out` — is not a thing a bind group can
/// express. Here the frontier is one read-write binding read at `built_off` and
/// written at `out_off`.
#[cube(launch_unchecked)]
pub fn subtract_within_kernel(
    parent: &Array<i64>,
    frontier: &mut Array<i64>,
    parent_off: u32,
    built_off: u32,
    out_off: u32,
    n_words: u32,
) {
    let i = ABSOLUTE_POS as u32;
    if i < n_words {
        frontier[(out_off + i) as usize] =
            parent[(parent_off + i) as usize] - frontier[(built_off + i) as usize];
    }
}

/// Port of the `SubtractionTrick` device lambda (histogram.cuh):
/// `sibling = parent - built`, elementwise over interleaved i64 words.
///
/// The three word offsets let parent, built and sibling each live in a
/// different frontier buffer at a different slot, which is what happens once a
/// level's histograms are allocated together.
#[cube(launch_unchecked)]
pub fn subtract_hist_kernel(
    parent: &Array<i64>,
    built: &Array<i64>,
    out: &mut Array<i64>,
    parent_off: u32,
    built_off: u32,
    out_off: u32,
    n_words: u32,
) {
    let i = ABSOLUTE_POS as u32;
    if i < n_words {
        out[(out_off + i) as usize] =
            parent[(parent_off + i) as usize] - built[(built_off + i) as usize];
    }
}

/// The work items of one [`hist_atomic_free_kernel`](fn@hist_atomic_free_kernel)
/// launch: parallel columns, one entry per item.
#[derive(Default)]
struct ItemTable {
    ridx_begin: Vec<u32>,
    n_ridx: Vec<u32>,
    /// Offset of the item's `n_bins` pairs in its destination, in pairs.
    dst_base: Vec<u32>,
}

impl ItemTable {
    fn push(&mut self, ridx_begin: u32, n_ridx: u32, dst_base: u32) {
        self.ridx_begin.push(ridx_begin);
        self.n_ridx.push(n_ridx);
        self.dst_base.push(dst_base);
    }

    fn len(&self) -> usize {
        self.ridx_begin.len()
    }

    fn is_empty(&self) -> bool {
        self.ridx_begin.is_empty()
    }
}

/// Partial histograms a batched build left unmerged, for the subtraction to
/// fold in (`HistogramEngine::build_into_batch_deferred`).
///
/// A node the builder cut into several row chunks has one private partial per
/// chunk in `partials`; `job_item_begin[j]..job_item_end[j]` are job `j`'s
/// chunks (empty for a node of one chunk, whose slot is already written) and
/// `chunked` the subset of jobs that have any. Dropping it without merging
/// leaves those nodes' slots unwritten.
#[derive(Clone, Debug)]
pub struct PendingMerge {
    partials: Handle,
    partial_words: usize,
    job_item_begin: Vec<u32>,
    job_item_end: Vec<u32>,
    /// `(job index, slot in bins)` of every job with more than one chunk.
    chunked: Vec<(usize, u32)>,
}

/// One node's slice of the work in a batched histogram build.
#[derive(Clone, Copy, Debug)]
pub struct NodeHistJob {
    /// First index of this node's rows in the shared `ridx` buffer.
    pub ridx_base: u32,
    pub n_ridx: u32,
    /// Bin offset of this node's slot in the destination buffer.
    pub slot: u32,
}

/// A contiguous run of features sharing one shared-memory histogram, the
/// host-side analogue of `FeatureGroup` (feature_groups.cuh).
#[derive(Clone, Copy, Debug)]
pub struct FeatureGroup {
    pub start_feature: u32,
    pub num_features: u32,
    pub start_bin: u32,
    pub num_bins: u32,
}

/// Port of the grouping logic in `FeatureGroups` (feature_groups.cu): pack
/// consecutive features into groups whose bins fit in `shmem_bytes` of shared
/// memory. Returns `None` if some single feature does not fit, in which case
/// the caller must fall back to one all-features group in global memory.
///
/// Only valid for **compressed** (feature-local bin) matrices: the multi-group
/// shared path relies on each block walking exactly its group's features. For
/// sparse (global-bin) matrices the caller must not split into multiple groups
/// (see `HistogramBuilder::build`).
pub fn build_feature_groups(cut_ptrs: &[u32], shmem_bytes: usize) -> Option<Vec<FeatureGroup>> {
    // 4 u32 accumulator words per bin.
    let max_bins = (shmem_bytes / (4 * core::mem::size_of::<u32>())) as u32;
    let n_features = cut_ptrs.len() - 1;

    let mut groups = Vec::new();
    let mut start = 0usize;
    while start < n_features {
        let mut end = start;
        while end < n_features && cut_ptrs[end + 1] - cut_ptrs[start] <= max_bins {
            end += 1;
        }
        if end == start {
            return None; // A single feature exceeds the shared-memory budget.
        }
        groups.push(FeatureGroup {
            start_feature: start as u32,
            num_features: (end - start) as u32,
            start_bin: cut_ptrs[start],
            num_bins: cut_ptrs[end] - cut_ptrs[start],
        });
        start = end;
    }
    Some(groups)
}

/// Filler size for the (mandatory) shared-memory declaration on the
/// global-memory path.
const SMEM_MIN_WORDS: usize = 4;

/// Whether the runtime supports native `i64` atomic adds (e.g. CUDA, or
/// Vulkan devices with `VK_KHR_shader_atomic_int64`).
pub fn supports_native_i64_atomics<R: Runtime>(client: &ComputeClient<R>) -> bool {
    let ty = StorageType::Atomic(i64::as_type_native_unchecked().elem_type());
    client.properties().atomic_type_usage(Type::new(ty)).contains(AtomicUsage::Add)
}

/// Whether the runtime supports `u32` atomic adds — the narrower of the two
/// accumulation schemes, and the one the u32-carry fallback rests on.
///
/// False on the CubeCL CPU runtime, which registers no atomic types at all
/// (`register_supported_types`, cubecl-cpu). That is what
/// [`hist_atomic_free_kernel`](fn@hist_atomic_free_kernel) exists for: a
/// runtime answering `false` here gets that kernel instead, and gets
/// bit-identical histograms from it.
pub fn supports_atomic_add_u32<R: Runtime>(client: &ComputeClient<R>) -> bool {
    let ty = StorageType::Atomic(u32::as_type_native_unchecked().elem_type());
    client.properties().atomic_type_usage(Type::new(ty)).contains(AtomicUsage::Add)
}

/// Configures a [`HistogramEngine`]; the analogue of
/// `DeviceHistogramBuilder::Reset` + `HistKernel`'s constructor.
///
/// ```no_run
/// # use xgboost_rs::gpu::histogram::HistogramBuilder;
/// // Whichever backend this build resolved to; see `gpu::DefaultRuntime`.
/// let client = xgboost_rs::gpu::default_client(0);
/// # let matrix: xgboost_rs::gpu::ellpack::EllpackMatrix = unimplemented!();
/// let engine = HistogramBuilder::new(&client)
///     .shmem_bytes(48 * 1024)
///     .force_global(false)
///     .build(&matrix)?;
/// # Ok::<(), xgboost_rs::Error>(())
/// ```
pub struct HistogramBuilder<'a, R: Runtime> {
    client: &'a ComputeClient<R>,
    /// `None` until the caller names one, so a runtime with no global-memory
    /// fallback can pick its own budget without overriding an explicit ask.
    shmem_bytes: Option<usize>,
    force_global: bool,
    max_blocks_per_group: u32,
    native_i64_atomics: Option<bool>,
}

impl<'a, R: Runtime> HistogramBuilder<'a, R> {
    pub fn new(client: &'a ComputeClient<R>) -> Self {
        Self {
            client,
            shmem_bytes: None,
            force_global: false,
            max_blocks_per_group: DEFAULT_MAX_BLOCKS_PER_GROUP,
            native_i64_atomics: None,
        }
    }

    /// Per-workgroup shared-memory budget for the privatised histogram path.
    ///
    /// Defaults to [`DEFAULT_SHMEM_BYTES`]. Ignored on a runtime with no
    /// atomics, whose privatised path keeps its partials in global memory and
    /// has no shared budget to fit.
    pub fn shmem_bytes(mut self, bytes: usize) -> Self {
        self.shmem_bytes = Some(bytes);
        self
    }

    /// Force the global-memory path (the `force_global_memory` test hook of
    /// `DeviceHistogramBuilder::Reset`).
    ///
    /// [`build`](Self::build) fails with [`Error::NoGlobalHistogramPath`] on a
    /// runtime whose atomics this path is built on do not exist.
    pub fn force_global(mut self, force: bool) -> Self {
        self.force_global = force;
        self
    }

    /// Cap on the grid size per feature group.
    pub fn max_blocks_per_group(mut self, n: u32) -> Self {
        self.max_blocks_per_group = n.max(1);
        self
    }

    /// Force native 64-bit atomics on (`true`) or off (`false`) for
    /// global-memory accumulation. Default: auto-detect from the runtime
    /// ([`supports_native_i64_atomics`]). Forcing `true` on a runtime without
    /// support fails at kernel compilation.
    pub fn native_i64_atomics(mut self, enable: bool) -> Self {
        self.native_i64_atomics = Some(enable);
        self
    }

    /// Validate the matrix, decide the shared/global dispatch, and upload the
    /// matrix to the device.
    pub fn build(self, matrix: &EllpackMatrix) -> Result<HistogramEngine<R>> {
        self.build_impl(matrix, None)
    }

    /// [`build`](Self::build) against a matrix already on device.
    ///
    /// `gidx` is the largest buffer a fit holds, and the row partitioner reads
    /// the same one, so a grower uploads it once and hands it to both.
    pub fn build_shared(
        self,
        matrix: &EllpackMatrix,
        shared: &super::ellpack::DeviceEllpack,
    ) -> Result<HistogramEngine<R>> {
        self.build_impl(matrix, Some(shared))
    }

    fn build_impl(
        self,
        matrix: &EllpackMatrix,
        shared: Option<&super::ellpack::DeviceEllpack>,
    ) -> Result<HistogramEngine<R>> {
        if matrix.cut_ptrs.len() < 2 {
            return Err(Error::InvalidCuts { got: matrix.cut_ptrs.len() });
        }
        let expected = matrix.n_rows * matrix.row_stride;
        if matrix.gidx.len() != expected {
            return Err(Error::MatrixShape { expected, got: matrix.gidx.len() });
        }

        // A runtime with no atomics has no global-memory accumulation path at
        // all, so `force_global` cannot be honoured. It has no shared-memory
        // path either: `hist_atomic_free_kernel` privatises the whole
        // histogram per work item in global memory, so there is no budget to
        // fit and the matrix is always one all-features group.
        let atomic_free = !supports_atomic_add_u32(self.client);
        if atomic_free && self.force_global {
            return Err(Error::NoGlobalHistogramPath);
        }
        let shmem_bytes = self.shmem_bytes.unwrap_or(DEFAULT_SHMEM_BYTES);

        let groups = if self.force_global || atomic_free {
            None
        } else if matrix.is_compressed() {
            // Feature-local bins: a block can walk just its group's features
            // (feature_stride = num_features), so multi-group shared is valid.
            build_feature_groups(&matrix.cut_ptrs, shmem_bytes)
        } else {
            // Sparse bins are global; the kernel walks the full row per block
            // (feature_stride = row_stride), so only a single all-features
            // group is valid for shared memory. Use it if every bin fits,
            // otherwise fall back to global.
            let n_bins = matrix.n_bins() as usize;
            if n_bins * 4 * core::mem::size_of::<u32>() <= shmem_bytes {
                Some(vec![FeatureGroup {
                    start_feature: 0,
                    num_features: matrix.n_features() as u32,
                    start_bin: 0,
                    num_bins: matrix.n_bins(),
                }])
            } else {
                None
            }
        };
        // Private partials are the privatised path of an atomic-free runtime.
        let use_shared = groups.is_some() || atomic_free;
        let groups = groups.unwrap_or_else(|| {
            vec![FeatureGroup {
                start_feature: 0,
                num_features: matrix.n_features() as u32,
                start_bin: 0,
                num_bins: matrix.n_bins(),
            }]
        });

        let max_group_bins = groups.iter().map(|g| g.num_bins).max().unwrap() as usize;
        // The shared buffer is declared at its filler size wherever the
        // comptime branch drops every access to it.
        let smem_words = if use_shared && !atomic_free {
            (max_group_bins * 4).max(SMEM_MIN_WORDS)
        } else {
            SMEM_MIN_WORDS
        };
        // Row chunks per node on the atomic-free path: at least the CPU
        // grower's lane cap, and at least two per unit so a root build has
        // something for every core.
        let max_chunks = (2 * launch::serial_width(self.client)).max(MAX_CHUNKS_PER_NODE);

        let groups_flat: Vec<u32> = groups
            .iter()
            .flat_map(|g| [g.start_feature, g.num_features, g.start_bin, g.num_bins])
            .collect();

        let native_i64 =
            self.native_i64_atomics.unwrap_or_else(|| supports_native_i64_atomics(self.client));

        let client = self.client.clone();
        let (gidx, cut_ptrs, gidx_len, bits) = match shared {
            Some(ell) => (ell.gidx.clone(), ell.cut_ptrs.clone(), ell.gidx_len, ell.bits),
            None => {
                let bits = device_bits(&client, matrix);
                let packed = pack_bins(&matrix.gidx, bits);
                (
                    client.create_from_slice(bytemuck::cast_slice(&packed)),
                    client.create_from_slice(bytemuck::cast_slice(&matrix.cut_ptrs)),
                    packed.len(),
                    bits,
                )
            }
        };
        let groups_dev = client.create_from_slice(bytemuck::cast_slice(&groups_flat));
        // 1-word placeholders for whichever histogram view the comptime
        // `native_i64` flag leaves unused.
        let dummy_u32 = client.create_from_slice(bytemuck::cast_slice(&[0u32]));
        let dummy_i64 = client.create_from_slice(bytemuck::cast_slice(&[0i64]));

        Ok(HistogramEngine {
            client,
            gidx,
            cut_ptrs,
            groups_dev,
            dummy_u32,
            dummy_i64,
            native_i64,
            n_groups: groups.len() as u32,
            n_bins: matrix.n_bins() as usize,
            n_rows: matrix.n_rows,
            n_cuts: matrix.cut_ptrs.len(),
            gidx_len,
            bits,
            row_stride: matrix.row_stride as u32,
            base_rowid: matrix.base_rowid,
            null_value: matrix.null_value,
            dense: matrix.is_dense(),
            compressed: matrix.is_compressed(),
            use_shared,
            atomic_free,
            smem_words,
            max_chunks,
            max_blocks_per_group: self.max_blocks_per_group,
        })
    }
}

/// Builds gradient histograms against a device-resident ELLPACK matrix; the
/// analogue of `DeviceHistogramBuilder` + `HistKernel::Dispatch`.
///
/// Created by [`HistogramBuilder::build`].
pub struct HistogramEngine<R: Runtime> {
    client: ComputeClient<R>,
    gidx: Handle,
    cut_ptrs: Handle,
    groups_dev: Handle,
    dummy_u32: Handle,
    dummy_i64: Handle,
    native_i64: bool,
    n_groups: u32,
    n_bins: usize,
    n_rows: usize,
    n_cuts: usize,
    /// Length of the packed `gidx`, in `u32` words, and its bits per entry.
    gidx_len: usize,
    bits: u32,
    row_stride: u32,
    base_rowid: u32,
    null_value: u32,
    dense: bool,
    compressed: bool,
    use_shared: bool,
    /// No atomics anywhere on this runtime; see `hist_atomic_free_kernel`.
    atomic_free: bool,
    smem_words: usize,
    /// Cap on the row chunks one node is cut into on the atomic-free path.
    max_chunks: u32,
    max_blocks_per_group: u32,
}

impl<R: Runtime> HistogramEngine<R> {
    pub fn client(&self) -> &ComputeClient<R> {
        &self.client
    }

    pub fn n_bins(&self) -> usize {
        self.n_bins
    }

    /// Whether the shared-memory (privatised histogram) path is active.
    pub fn uses_shared_memory(&self) -> bool {
        self.use_shared
    }

    /// Whether global-memory accumulation uses native 64-bit atomics
    /// (`AtomicAddGpairGlobal`) instead of the u32-carry scheme.
    pub fn uses_native_i64_atomics(&self) -> bool {
        self.native_i64
    }

    /// Upload quantised gradient pairs (one per matrix row).
    pub fn upload_gpairs(&self, gpairs: &[GradientPairInt64]) -> Result<DeviceGpairs> {
        if gpairs.len() != self.n_rows {
            return Err(Error::GpairCount { expected: self.n_rows, got: gpairs.len() });
        }
        Ok(DeviceGpairs {
            handle: self.client.create_from_slice(bytemuck::cast_slice(gpairs)),
            n: gpairs.len(),
        })
    }

    /// [`upload_gpairs`](Self::upload_gpairs), taking the buffer rather than
    /// copying it.
    ///
    /// `create_from_slice` copies the slice into a fresh vector before the
    /// device sees it; handing over the allocation skips that copy, and the
    /// gradient pairs are the one per-round upload large enough to notice —
    /// 3.2 MB at 200 000 rows, measured 0.80 → 0.33 ms per upload on the CPU
    /// runtime, 3.9 → 1.75 ms at a million rows.
    pub fn upload_gpairs_owned(&self, gpairs: Vec<GradientPairInt64>) -> Result<DeviceGpairs> {
        if gpairs.len() != self.n_rows {
            return Err(Error::GpairCount { expected: self.n_rows, got: gpairs.len() });
        }
        let n = gpairs.len();
        Ok(DeviceGpairs { handle: self.client.create(Bytes::from_elems(gpairs)), n })
    }

    /// Upload the row indices of one tree node.
    pub fn upload_rows(&self, ridx: &[u32]) -> DeviceRows {
        DeviceRows {
            handle: self.client.create_from_slice(bytemuck::cast_slice(ridx)),
            base: 0,
            n: ridx.len(),
        }
    }

    /// Launch the histogram kernel, leaving the result on device.
    ///
    /// This is the hot path used per tree node; it performs no host round
    /// trips beyond the launch itself.
    pub fn build_to_device(&self, gpairs: &DeviceGpairs, rows: &DeviceRows) -> DeviceHistogram {
        // Accumulator: 4 u32 words per bin.
        let hist = self.frontier(self.n_bins * 4);
        self.build_into(gpairs, rows, &hist, self.n_bins, 0);
        DeviceHistogram { handle: hist, n_bins: self.n_bins }
    }

    /// Accumulate one node's histogram into `dst` at bin offset `slot`.
    ///
    /// `dst` comes from [`frontier`](Self::frontier): on the atomic path it
    /// must be zeroed over that slot, because the kernel only adds; on the
    /// atomic-free path the slot is written whole. This is how a whole frontier
    /// shares one allocation, which is what lets the split evaluator read every
    /// node of a level in a single launch.
    pub fn build_into(
        &self,
        gpairs: &DeviceGpairs,
        rows: &DeviceRows,
        dst: &Handle,
        dst_bins: usize,
        slot: u32,
    ) {
        self.build_into_batch(
            gpairs,
            &rows.handle,
            rows.base + rows.n,
            &[NodeHistJob { ridx_base: rows.base as u32, n_ridx: rows.n as u32, slot }],
            dst,
            dst_bins,
        );
    }

    /// Accumulate a whole level of node histograms in one launch.
    ///
    /// The grid is sized for the widest node; narrower ones exit their
    /// grid-strided loop early. That is far cheaper than one launch per node —
    /// a depth-10 tree has ~1000 of them.
    pub fn build_into_batch(
        &self,
        gpairs: &DeviceGpairs,
        ridx: &Handle,
        ridx_len: usize,
        jobs: &[NodeHistJob],
        dst: &Handle,
        dst_bins: usize,
    ) {
        if jobs.is_empty() {
            return;
        }
        if self.atomic_free {
            let pending = self.build_private_batch(gpairs, ridx, ridx_len, jobs, dst, dst_bins);
            if let Some(pending) = pending {
                self.merge_pending(&pending, dst, dst_bins);
            }
            return;
        }

        // Grid sizing, mirroring the `launch` lambda in DispatchHistShmem:
        // enough tiles for (rows x features-per-group) items, occupancy-capped.
        let widest = jobs.iter().map(|j| j.n_ridx).max().unwrap_or(0);
        let columns_per_group = self.row_stride.div_ceil(self.n_groups);
        let items_per_group = widest * columns_per_group;
        // The kernel derives every stride from `CUBE_DIM_X`, so the workgroup
        // can be whatever the runtime wants — but the tile the grid is sized
        // in has to be the same one the kernel walks.
        let block = launch::block_1d(&self.client, BLOCK_THREADS);
        let tile = block * ITEMS_PER_THREAD;
        let n_blocks = items_per_group.div_ceil(tile).clamp(1, self.max_blocks_per_group);

        let n_ridx: Vec<u32> = jobs.iter().map(|j| j.n_ridx).collect();
        let base: Vec<u32> = jobs.iter().map(|j| j.ridx_base).collect();
        let offset: Vec<u32> = jobs.iter().map(|j| j.slot).collect();
        let d_n = self.client.create_from_slice(bytemuck::cast_slice(&n_ridx));
        let d_base = self.client.create_from_slice(bytemuck::cast_slice(&base));
        let d_off = self.client.create_from_slice(bytemuck::cast_slice(&offset));

        let cube_count = CubeCount::Static(n_blocks, self.n_groups, jobs.len() as u32);
        let cube_dim = CubeDim::new_1d(block);

        // Exactly one of the two histogram views is the real buffer; the
        // other is a 1-word dummy never touched by the comptime-elided branch.
        let (hist32, hist32_len, hist64, hist64_len) = if self.native_i64 {
            (self.dummy_u32.clone(), 1, dst.clone(), dst_bins * 2)
        } else {
            (dst.clone(), dst_bins * 4, self.dummy_i64.clone(), 1)
        };

        // SAFETY: the kernel guards every index against the lengths it is
        // given; see the `gpu` module docs on unchecked launches.
        unsafe {
            hist_kernel::launch_unchecked::<R>(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(self.gidx.clone(), self.gidx_len),
                ArrayArg::from_raw_parts(self.cut_ptrs.clone(), self.n_cuts),
                ArrayArg::from_raw_parts(self.groups_dev.clone(), self.n_groups as usize * 4),
                // The kernel indexes `ridx` from each node's base, so the binding
                // spans the whole row index rather than one node's slice.
                ArrayArg::from_raw_parts(ridx.clone(), ridx_len),
                ArrayArg::from_raw_parts(gpairs.handle.clone(), gpairs.n * 2),
                ArrayArg::from_raw_parts(hist32, hist32_len),
                ArrayArg::from_raw_parts(hist64, hist64_len),
                ArrayArg::from_raw_parts(d_n, jobs.len()),
                ArrayArg::from_raw_parts(d_base, jobs.len()),
                ArrayArg::from_raw_parts(d_off, jobs.len()),
                self.row_stride,
                self.base_rowid,
                self.null_value,
                self.dense,
                self.compressed,
                self.use_shared,
                self.native_i64,
                self.smem_words,
                self.bits,
            );
        }
    }

    /// [`build_into_batch`](Self::build_into_batch), leaving multi-chunk
    /// nodes' partials unmerged for [`subtract_batch`](Self::subtract_batch)
    /// to fold into the subtraction the level makes anyway.
    ///
    /// On the CPU runtime a launch costs ~130 µs however little it does, and
    /// the merge was one per level for as long as nodes have more than one
    /// chunk — the top six or so levels at 200 000 rows. `None` when nothing
    /// is pending (every node had one chunk, or the runtime has atomics and
    /// built the batch outright), in which case `dst` is complete. When it is
    /// `Some`, the chunked nodes' slots are *not* written until the pending
    /// merge is passed to `subtract_batch` or to
    /// [`merge_pending`](Self::merge_pending).
    pub fn build_into_batch_deferred(
        &self,
        gpairs: &DeviceGpairs,
        ridx: &Handle,
        ridx_len: usize,
        jobs: &[NodeHistJob],
        dst: &Handle,
        dst_bins: usize,
    ) -> Option<PendingMerge> {
        if jobs.is_empty() {
            return None;
        }
        if self.atomic_free {
            return self.build_private_batch(gpairs, ridx, ridx_len, jobs, dst, dst_bins);
        }
        self.build_into_batch(gpairs, ridx, ridx_len, jobs, dst, dst_bins);
        None
    }

    /// [`build_into_batch`](Self::build_into_batch) on a runtime with no
    /// atomics: the serial shape, barrier-free throughout.
    ///
    /// The job table is the CPU grower's: each node is cut into row chunks of
    /// [`CHUNK_ROWS`], capped at `max_chunks` per node, and every chunk is one
    /// work item. A node of one chunk — every node below the top few levels —
    /// accumulates straight into its slot of `dst`, so neither a partials
    /// buffer nor a merge nor a zeroing pass touches the frontier for it; a
    /// node of several chunks accumulates into a private buffer, and the
    /// returned [`PendingMerge`] says how to sum it into the slot. Either way
    /// every bin of every slot in `jobs` ends up *written*, which is what lets
    /// [`frontier`](Self::frontier) skip the clearing pass on this path. At
    /// depth the frontier is tens of megabytes and those passes ran at memory
    /// bandwidth, so not making them is the whole saving.
    fn build_private_batch(
        &self,
        gpairs: &DeviceGpairs,
        ridx: &Handle,
        ridx_len: usize,
        jobs: &[NodeHistJob],
        dst: &Handle,
        dst_bins: usize,
    ) -> Option<PendingMerge> {
        let bins_per_item = self.n_bins as u32;
        let mut direct = ItemTable::default();
        let mut chunked = ItemTable::default();
        let mut job_item_begin = Vec::with_capacity(jobs.len());
        let mut job_item_end = Vec::with_capacity(jobs.len());
        let mut chunked_jobs = Vec::new();
        for (j, job) in jobs.iter().enumerate() {
            let chunks = job.n_ridx.div_ceil(CHUNK_ROWS).clamp(1, self.max_chunks);
            if chunks == 1 {
                direct.push(job.ridx_base, job.n_ridx, job.slot);
                job_item_begin.push(0);
                job_item_end.push(0);
                continue;
            }
            job_item_begin.push(chunked.len() as u32);
            let chunk_len = job.n_ridx.div_ceil(chunks);
            for c in 0..chunks {
                let begin = c * chunk_len;
                let n = job.n_ridx.saturating_sub(begin).min(chunk_len);
                let base = chunked.len() as u32 * bins_per_item;
                chunked.push(job.ridx_base + begin, n, base);
            }
            job_item_end.push(chunked.len() as u32);
            chunked_jobs.push((j, job.slot));
        }

        if !direct.is_empty() {
            self.launch_private(gpairs, ridx, ridx_len, dst, dst_bins * 2, &direct);
        }
        if chunked.is_empty() {
            return None;
        }

        let partial_words = chunked.len() * self.n_bins * 2;
        let partials = self.client.empty(partial_words * core::mem::size_of::<i64>());
        self.launch_private(gpairs, ridx, ridx_len, &partials, partial_words, &chunked);
        Some(PendingMerge {
            partials,
            partial_words,
            job_item_begin,
            job_item_end,
            chunked: chunked_jobs,
        })
    }

    /// Sum a [`PendingMerge`]'s partials into their nodes' slots of `dst`, as
    /// a launch of its own. [`subtract_batch`](Self::subtract_batch) does the
    /// same work for free where a subtraction follows; this is for the root,
    /// which has no sibling, and for callers that want the batch complete.
    pub fn merge_pending(&self, pending: &PendingMerge, dst: &Handle, dst_bins: usize) {
        let c = &self.client;
        let n_nodes = pending.chunked.len();
        if n_nodes == 0 {
            return;
        }
        let n_items = pending.partial_words / (self.n_bins * 2);
        let node_begin: Vec<u32> =
            pending.chunked.iter().map(|&(j, _)| pending.job_item_begin[j]).collect();
        let node_end: Vec<u32> =
            pending.chunked.iter().map(|&(j, _)| pending.job_item_end[j]).collect();
        let node_offset: Vec<u32> = pending.chunked.iter().map(|&(_, slot)| slot).collect();
        let d_node_begin = c.create_from_slice(bytemuck::cast_slice(&node_begin));
        let d_node_end = c.create_from_slice(bytemuck::cast_slice(&node_end));
        let d_node_offset = c.create_from_slice(bytemuck::cast_slice(&node_offset));
        // A lane is one bin summed over its node's chunks.
        let lanes = n_nodes * self.n_bins;
        let (count, dim, run) = launch::elementwise_runs(c, lanes, n_items.div_ceil(n_nodes));
        // SAFETY: the kernel guards every index against the lengths it is
        // given; see the `gpu` module docs on unchecked launches.
        unsafe {
            merge_partials_kernel::launch_unchecked::<R>(
                c,
                count,
                dim,
                ArrayArg::from_raw_parts(pending.partials.clone(), pending.partial_words),
                ArrayArg::from_raw_parts(dst.clone(), dst_bins * 2),
                ArrayArg::from_raw_parts(d_node_begin, n_nodes),
                ArrayArg::from_raw_parts(d_node_end, n_nodes),
                ArrayArg::from_raw_parts(d_node_offset, n_nodes),
                self.n_bins as u32,
                lanes as u32,
                run,
            );
        }
    }

    /// One launch of [`hist_atomic_free_kernel`](fn@hist_atomic_free_kernel)
    /// over `items`, accumulating into `dst` (`dst_words` `i64` long).
    fn launch_private(
        &self,
        gpairs: &DeviceGpairs,
        ridx: &Handle,
        ridx_len: usize,
        dst: &Handle,
        dst_words: usize,
        items: &ItemTable,
    ) {
        let c = &self.client;
        let n_items = items.len();
        let d_begin = c.create_from_slice(bytemuck::cast_slice(&items.ridx_begin));
        let d_n = c.create_from_slice(bytemuck::cast_slice(&items.n_ridx));
        let d_base = c.create_from_slice(bytemuck::cast_slice(&items.dst_base));
        // A unit is worth a chunk's `(row, feature)` visits.
        let (count, dim) = launch::elementwise_with_work(
            c,
            n_items,
            CHUNK_ROWS as usize * self.row_stride as usize,
        );
        // Blocked row loop where the rows are OS threads' gathers; a GPU that
        // ends up on this path has the memory system to hide them itself.
        let touch = if launch::has_planes(c) { 0 } else { TOUCH_ROWS };
        // SAFETY: the kernel guards every index against the lengths it is
        // given; see the `gpu` module docs on unchecked launches.
        unsafe {
            hist_atomic_free_kernel::launch_unchecked::<R>(
                c,
                count,
                dim,
                // A bin is one `[grad, hess]` vector; counts stay in scalars.
                2,
                ArrayArg::from_raw_parts(self.gidx.clone(), self.gidx_len),
                ArrayArg::from_raw_parts(self.cut_ptrs.clone(), self.n_cuts),
                ArrayArg::from_raw_parts(ridx.clone(), ridx_len),
                ArrayArg::from_raw_parts(gpairs.handle.clone(), gpairs.n * 2),
                ArrayArg::from_raw_parts(dst.clone(), dst_words),
                ArrayArg::from_raw_parts(d_begin, n_items),
                ArrayArg::from_raw_parts(d_n, n_items),
                ArrayArg::from_raw_parts(d_base, n_items),
                self.row_stride,
                self.base_rowid,
                self.null_value,
                n_items as u32,
                self.n_bins as u32,
                self.dense,
                self.compressed,
                self.bits,
                touch,
            );
        }
    }

    /// A frontier of `words` `u32`, ready for [`build_into_batch`](Self::build_into_batch).
    ///
    /// Zeroed on the atomic path, whose kernel only adds. Left as it comes on
    /// the atomic-free path, whose kernels write every bin of every slot they
    /// are given — which is the point: at depth the frontier is tens of
    /// megabytes, and a clearing pass over it runs at memory bandwidth.
    pub fn frontier(&self, words: usize) -> Handle {
        if self.atomic_free {
            self.client.empty(words * core::mem::size_of::<u32>())
        } else {
            self.zeroed(words)
        }
    }

    /// Subtraction trick (`sibling = parent - built`), on device.
    pub fn subtract_to_device(
        &self,
        parent: &DeviceHistogram,
        built: &DeviceHistogram,
    ) -> Result<DeviceHistogram> {
        if parent.n_bins != built.n_bins {
            return Err(Error::HistogramLen { parent: parent.n_bins, built: built.n_bins });
        }
        // Word views: subtraction runs over interleaved i64 pairs.
        let n_words = parent.n_bins * 2;
        let out = self.client.empty(n_words * core::mem::size_of::<i64>());

        let (cube_count, cube_dim) = launch::elementwise(&self.client, n_words);
        // SAFETY: the kernel guards every index against the lengths it is
        // given; see the `gpu` module docs on unchecked launches.
        unsafe {
            subtract_hist_kernel::launch_unchecked::<R>(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(parent.handle.clone(), n_words),
                ArrayArg::from_raw_parts(built.handle.clone(), n_words),
                ArrayArg::from_raw_parts(out.clone(), n_words),
                0,
                0,
                0,
                n_words as u32,
            );
        }

        Ok(DeviceHistogram { handle: out, n_bins: parent.n_bins })
    }

    /// Subtraction trick within a frontier: slot `out_slot` of `frontier`
    /// becomes `parent[parent_slot] - frontier[built_slot]`, in bins.
    ///
    /// `parent` is a different allocation — the previous batch's frontier —
    /// while the built child and the sibling share this batch's, which is why
    /// the frontier is one read-write binding rather than two.
    #[allow(clippy::too_many_arguments)]
    pub fn subtract_into(
        &self,
        parent: &Handle,
        parent_bins: usize,
        parent_slot: u32,
        frontier: &Handle,
        frontier_bins: usize,
        built_slot: u32,
        out_slot: u32,
    ) {
        // Each bin is one interleaved `[grad, hess]` i64 pair.
        let n_words = self.n_bins * 2;
        let (cube_count, cube_dim) = launch::elementwise(&self.client, n_words);
        // SAFETY: the kernel guards every index against the lengths it is
        // given; see the `gpu` module docs on unchecked launches.
        unsafe {
            subtract_within_kernel::launch_unchecked::<R>(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(parent.clone(), parent_bins * 2),
                ArrayArg::from_raw_parts(frontier.clone(), frontier_bins * 2),
                parent_slot * 2,
                built_slot * 2,
                out_slot * 2,
                n_words as u32,
            );
        }
    }

    /// A zeroed buffer of `words` `u32`, filled on device.
    ///
    /// The grid here is the one that overflows: a depth-10 frontier is ~26M
    /// words, and a `div_ceil` into X alone asks for 102,540 cubes against a
    /// 65,535-per-axis limit, which wgpu rejects outright. `elementwise` folds
    /// the excess into Y and Z, which `ABSOLUTE_POS` flattens back.
    pub fn zeroed(&self, words: usize) -> Handle {
        let h = self.client.empty(words * core::mem::size_of::<u32>());
        self.zero_into(&h, words);
        h
    }

    /// Zero the first `words` `u32` of `h` on device.
    pub fn zero_into(&self, h: &Handle, words: usize) {
        let line = launch::line_size_for::<R, u32>(&self.client, &[words]);
        let n_lines = words / line;
        let (cube_count, cube_dim) = launch::elementwise(&self.client, n_lines);
        // SAFETY: the kernel guards every index against the lengths it is
        // given; see the `gpu` module docs on unchecked launches.
        unsafe {
            zero_u32_kernel::launch_unchecked::<R>(
                &self.client,
                cube_count,
                cube_dim,
                line,
                // Scalar count, not line count: the JIT divides by the width.
                ArrayArg::from_raw_parts(h.clone(), words),
                n_lines as u32,
            );
        }
    }

    /// Make a buffer that was used before ready to be built into again, as
    /// [`frontier`](Self::frontier) makes a fresh one: zeroed where the kernel
    /// only adds, untouched where it writes every bin.
    ///
    /// Reuse is worth having. A fresh allocation is first-touched by the
    /// kernel that writes it, and on a host runtime that is a page fault per
    /// page: measured at 3.1 ms to allocate and clear 64 MB against the 1 ms
    /// its bandwidth would take, and 5.6 ms once other large buffers are
    /// live. A depth-10 frontier is that size, and it was being allocated
    /// once per level.
    pub fn prepare_frontier(&self, h: &Handle, words: usize) {
        if !self.atomic_free {
            self.zero_into(h, words);
        }
    }

    /// Subtraction trick for a whole batch, in one launch.
    ///
    /// `slots` gives `(parent_slot, built_slot, out_slot)` per node, in bins.
    /// With a `pending` merge from
    /// [`build_into_batch_deferred`](Self::build_into_batch_deferred) — whose
    /// jobs must be in the same order as `slots` — the built slots still in
    /// pieces are summed and written on the way, so the level pays no merge
    /// launch.
    pub fn subtract_batch(
        &self,
        parent: &Handle,
        parent_bins: usize,
        frontier: &Handle,
        frontier_bins: usize,
        slots: &[(u32, u32, u32)],
        pending: Option<&PendingMerge>,
    ) {
        if slots.is_empty() {
            if let Some(pending) = pending {
                self.merge_pending(pending, frontier, frontier_bins);
            }
            return;
        }
        debug_assert!(pending.is_none_or(|p| p.job_item_begin.len() == slots.len()));
        let n_words = self.n_bins * 2;
        // Offsets are in i64 words; a bin is one interleaved pair.
        let p: Vec<u32> = slots.iter().map(|s| s.0 * 2).collect();
        let b: Vec<u32> = slots.iter().map(|s| s.1 * 2).collect();
        let o: Vec<u32> = slots.iter().map(|s| s.2 * 2).collect();

        // The widest vector that divides the run and every offset, so the
        // kernel can index in whole vectors; the offsets are then given in
        // vectors too.
        let mut extents = vec![n_words, parent_bins * 2, frontier_bins * 2];
        extents.extend(p.iter().chain(&b).chain(&o).map(|&x| x as usize));
        let line = launch::line_size_for::<R, i64>(&self.client, &extents);
        let in_lines = |v: &[u32]| -> Vec<u32> { v.iter().map(|&x| x / line as u32).collect() };
        let dp = self.client.create_from_slice(bytemuck::cast_slice(&in_lines(&p)));
        let db = self.client.create_from_slice(bytemuck::cast_slice(&in_lines(&b)));
        let dobuf = self.client.create_from_slice(bytemuck::cast_slice(&in_lines(&o)));
        let n_lines = n_words / line;

        // The partials, or a one-word dummy and empty item ranges when there
        // are none: one kernel variant either way, the per-node range decides.
        let (partials, partial_words, item_begin, item_end) = match pending {
            Some(p) => {
                (p.partials.clone(), p.partial_words, &p.job_item_begin[..], &p.job_item_end[..])
            }
            None => (self.dummy_i64.clone(), 1, &[][..], &[][..]),
        };
        let zeros = vec![0u32; slots.len()];
        let (item_begin, item_end) =
            if pending.is_some() { (item_begin, item_end) } else { (&zeros[..], &zeros[..]) };
        let d_begin = self.client.create_from_slice(bytemuck::cast_slice(item_begin));
        let d_end = self.client.create_from_slice(bytemuck::cast_slice(item_end));

        // `elementwise` is not usable here: the kernel reads `CUBE_POS_Y` to
        // pick the node, so the cube count's Y axis is spoken for and the X
        // axis cannot spill into it. Plane-align the block, and when the
        // vector count still overflows the X axis's grid limit, spill into Z
        // instead — `subtract_batch_kernel` folds `CUBE_POS_Z` back into the
        // index, the same way `calculate_cube_count_elemwise` folds Y and Z
        // back into `ABSOLUTE_POS`. The kernel has no barrier, so a plane-less
        // runtime gets its full width rather than the one unit a synchronising
        // kernel is held to — and a run of lines per unit, so that one cube
        // in X covers a node and the runtime's cube loop is paid per node,
        // not per vector (`launch::elementwise_runs` for the measurement).
        let block = launch::free_block_1d(&self.client, BLOCK_THREADS);
        let run = launch::run_length(&self.client, n_lines, block);
        let total_cubes_x = (n_lines as u32).div_ceil(block * run).max(1);
        let max_cubes_x = self.client.properties().hardware.max_cube_count.0.max(1);
        let cubes_x = total_cubes_x.min(max_cubes_x);
        let cubes_z = total_cubes_x.div_ceil(cubes_x);
        // SAFETY: the kernel guards every index against the lengths it is
        // given; see the `gpu` module docs on unchecked launches.
        unsafe {
            subtract_batch_kernel::launch_unchecked::<R>(
                &self.client,
                CubeCount::Static(cubes_x, slots.len() as u32, cubes_z),
                CubeDim::new_1d(block),
                line,
                // Scalar counts, not vector counts: the JIT divides by the width.
                ArrayArg::from_raw_parts(parent.clone(), parent_bins * 2),
                ArrayArg::from_raw_parts(frontier.clone(), frontier_bins * 2),
                ArrayArg::from_raw_parts(partials, partial_words),
                ArrayArg::from_raw_parts(dp, slots.len()),
                ArrayArg::from_raw_parts(db, slots.len()),
                ArrayArg::from_raw_parts(dobuf, slots.len()),
                ArrayArg::from_raw_parts(d_begin, slots.len()),
                ArrayArg::from_raw_parts(d_end, slots.len()),
                n_lines as u32,
                (n_words / line) as u32,
                run,
            );
        }
    }

    /// Read a histogram back as `i64` gradient pairs.
    ///
    /// Accumulators (from [`Self::build_to_device`]) hold 4 `u32` words per
    /// bin; subtraction outputs hold plain i64 pairs. Both are the same bytes
    /// in little-endian layout, so one reader serves both.
    pub fn read(&self, hist: &DeviceHistogram) -> Result<Vec<GradientPairInt64>> {
        let bytes = self.client.read_one_unchecked(hist.handle.clone());
        let words: &[u32] = bytemuck::cast_slice(&bytes);
        if words.len() != hist.n_bins * 4 {
            return Err(Error::HistogramBins { expected: hist.n_bins, got: words.len() / 4 });
        }
        Ok(words
            .chunks_exact(4)
            .map(|w| GradientPairInt64 {
                grad: (w[0] as i64) | ((w[1] as i64) << 32),
                hess: (w[2] as i64) | ((w[3] as i64) << 32),
            })
            .collect())
    }

    /// Convenience: upload, build, and read back one node histogram.
    pub fn build(
        &self,
        gpairs: &[GradientPairInt64],
        ridx: &[u32],
    ) -> Result<Vec<GradientPairInt64>> {
        let gpairs = self.upload_gpairs(gpairs)?;
        let rows = self.upload_rows(ridx);
        self.read(&self.build_to_device(&gpairs, &rows))
    }

    /// Convenience: subtraction trick over host-side histograms.
    pub fn subtract(
        &self,
        parent: &[GradientPairInt64],
        built: &[GradientPairInt64],
    ) -> Result<Vec<GradientPairInt64>> {
        if parent.len() != built.len() {
            return Err(Error::HistogramLen { parent: parent.len(), built: built.len() });
        }
        let parent_dev = DeviceHistogram {
            handle: self.client.create_from_slice(bytemuck::cast_slice(parent)),
            n_bins: parent.len(),
        };
        let built_dev = DeviceHistogram {
            handle: self.client.create_from_slice(bytemuck::cast_slice(built)),
            n_bins: built.len(),
        };
        let out = self.subtract_to_device(&parent_dev, &built_dev)?;
        self.read(&out)
    }
}
