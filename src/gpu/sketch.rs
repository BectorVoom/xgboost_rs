//! The quantile cuts, computed on the device.
//!
//! `data::cuts::build_cuts` is a port of upstream's CPU sketch: every value of
//! every column pushed through a weighted-quantile summary, pruned, then read
//! off by rank. On a cloud VM's four cores that was 0.54 s at 500 000 × 50 and
//! 1.1 s at a million rows — more than a whole 20-round `gpu_hist` fit
//! upstream, which sketches on the device (`common::DeviceSketch`). So does
//! this, by a different route than upstream's: the column is **sorted** on the
//! device, and the cuts are read off the sorted column with the same rank
//! queries `query_cut_values` makes against a summary. A sorted column *is* the
//! exact summary — every distinct value with its true rank range — so the
//! result is what the host sketch would give with an unbounded budget, and
//! `data::cuts::exact_cuts_from_sorted` states that reference on the host for
//! the tests to pin the kernels to.
//!
//! What it does not do, and leaves to the host route: row weights (the
//! `hessian` argument of `approx`, and a weighted `DMatrix`), categorical
//! features (their bins are category codes, not quantiles), `max_bin` past
//! [`MAX_DEVICE_BIN`], and the CubeCL CPU runtime, whose units are the host's
//! own cores and which has none of the atomics the sort's histograms use.
//!
//! # The sort
//!
//! An LSD radix sort on the values' order-preserving `u32` keys (`-0.0`
//! folded onto `+0.0`, NaN and padding on the sentinel above every key), four
//! passes of an 8-bit digit, one column per segment of the key buffer so every
//! tile belongs to one column and the passes are one launch each for the whole
//! matrix:
//!
//! 1. [`radix_count_kernel`] — a cube per tile counts its digits in shared
//!    memory and writes them digit-major, so that
//! 2. [`radix_scan_kernel`] — a cube per column — scanning them in that order
//!    gives every `(digit, tile)` its first slot in the column, and
//! 3. [`radix_scatter_kernel`] — a cube per tile — sorts the tile by digit in
//!    shared memory, stably, with eight one-bit splits, and writes each key to
//!    its slot: the tile's slot for the digit plus the key's rank inside the
//!    tile. Stable within a tile, tiles in order, so stable across the pass,
//!    which is what an LSD sort needs of each pass.
//!
//! Then [`cuts_kernel`], a cube per column, reads the cuts off the sorted keys.

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::ellpack::is_nan_bits;
use super::launch;
use super::tables::{upload_slice, upload_vec};
use crate::data::DMatrix;
use crate::data::cuts::HistogramCuts;

/// Keys per tile of the sort: [`SORT_BLOCK`] units taking [`SORT_ITEMS`] each.
pub const SORT_TILE: u32 = 2048;
/// Units per cube of the sort's tile kernels; a power of two, for the scans.
pub const SORT_BLOCK: u32 = 256;
/// Keys a unit owns in a tile: contiguous, so its own order is its rank.
pub const SORT_ITEMS: u32 = 8;
/// Digit width of one pass; `2^RADIX_BITS` counters per tile.
const RADIX_BITS: u32 = 8;
const RADIX: u32 = 1 << RADIX_BITS;
/// The key above every value: what a missing cell and the padding of a column
/// hold, so that both sort to the end of the column.
const SENTINEL: u32 = 0xFFFF_FFFF;
/// Largest `max_bin` the cuts kernel holds candidates for in shared memory;
/// larger is left to the host sketch.
pub const MAX_DEVICE_BIN: usize = 1024;

/// Fill `buf` with `value`.
#[cube(launch_unchecked)]
pub fn fill_u32_kernel(buf: &mut Array<u32>, value: u32, n: u32) {
    let i = ABSOLUTE_POS as u32;
    if i < n {
        buf[i as usize] = value;
    }
}

/// The order-preserving key of a float: sign bit flipped for a positive
/// value, every bit flipped for a negative one, so unsigned order is float
/// order. `-0.0` is folded onto `+0.0`, as `<=` treats them, and NaN — never
/// equal to itself — is the sentinel.
#[cube]
fn float_key(v: f32) -> u32 {
    let mut bits = u32::reinterpret(v);
    // By the bits, not `v != v`, which the CPU runtime's compiler folds away.
    if is_nan_bits(bits) {
        0xFFFF_FFFFu32.into()
    } else {
        if bits == 0x8000_0000u32 {
            bits = 0u32;
        }
        if (bits & 0x8000_0000u32) != 0u32 { !bits } else { bits | 0x8000_0000u32 }
    }
}

/// [`float_key`] inverted, for a key below the sentinel.
#[cube]
fn key_float(k: u32) -> f32 {
    let bits = if (k & 0x8000_0000u32) != 0u32 { k & 0x7FFF_FFFFu32 } else { !k };
    f32::reinterpret(bits)
}

/// Write every present cell's key to column-major `keys`, column `f` at
/// `f * seg`. The buffer is pre-filled with the sentinel, which is what the
/// missing cells and the padding past `n_rows` keep. Same entry walk as
/// `ellpack::bin_csr_kernel`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn float_keys_kernel(
    row_ptr: &Array<u32>,
    index: &Array<u32>,
    value: &Array<f32>,
    keys: &mut Array<u32>,
    n_entries: u32,
    n_rows: u32,
    row_stride: u32,
    seg: u32,
    #[comptime] dense_input: bool,
) {
    let e = ABSOLUTE_POS as u32;
    if e < n_entries {
        let (r, f) = if dense_input {
            let r = e / row_stride;
            (r, e - r * row_stride)
        } else {
            let lo = RuntimeCell::<u32>::new(0u32);
            let n = RuntimeCell::<u32>::new(n_rows);
            while n.read() > 1u32 {
                let half = n.read() / 2u32;
                let mid = lo.read() + half;
                if row_ptr[mid as usize] <= e {
                    lo.store(mid);
                }
                n.store(n.read() - half);
            }
            (lo.read(), index[e as usize])
        };
        keys[(f * seg + r) as usize] = float_key(value[e as usize]);
    }
}

/// Inclusive Hillis–Steele scan of `s` over the cube, in place.
#[cube]
fn scan_inclusive(s: &mut SharedMemory<u32>, #[comptime] block: u32) {
    let t = UNIT_POS_X as usize;
    let off = RuntimeCell::<u32>::new(1u32);
    while off.read() < block {
        let d = off.read();
        let add = if UNIT_POS_X >= d { s[t - d as usize] } else { 0u32.into() };
        sync_cube();
        if UNIT_POS_X >= d {
            s[t] += add;
        }
        sync_cube();
        off.store(d * 2u32);
    }
}

/// Count the digits of one tile. `counts[(f * RADIX + d) * n_tiles + tile]`
/// gets the tile's count of digit `d`: digit-major within the column, so a
/// scan in memory order is the LSD pass's slot table.
#[cube(launch_unchecked)]
pub fn radix_count_kernel(
    keys: &Array<u32>,
    counts: &mut Array<u32>,
    n_tiles: u32,
    seg: u32,
    shift: u32,
    #[comptime] block: u32,
    #[comptime] items: u32,
    #[comptime] radix: u32,
) {
    let tile = CUBE_POS_X;
    let f = CUBE_POS_Y;
    let hist = SharedMemory::<Atomic<u32>>::new(radix as usize);
    let t = UNIT_POS_X;
    // `radix == block` here, one counter per unit.
    hist[t as usize].store(0u32);
    sync_cube();
    let base = f * seg + tile * (block * items);
    let mut k = 0u32;
    while k < items {
        let key = keys[(base + t + k * block) as usize];
        let d = (key >> shift) & (radix - 1u32);
        hist[d as usize].fetch_add(1u32);
        k += 1u32;
    }
    sync_cube();
    counts[((f * radix + t) * n_tiles + tile) as usize] = hist[t as usize].load();
    sync_cube();
}

/// Exclusive scan of one column's `radix * n_tiles` counts into `offsets`,
/// in memory order; a cube per column.
#[cube(launch_unchecked)]
pub fn radix_scan_kernel(
    counts: &Array<u32>,
    offsets: &mut Array<u32>,
    len: u32,
    #[comptime] block: u32,
) {
    let f = CUBE_POS_X;
    let base = f * len;
    let mut s = SharedMemory::<u32>::new(block as usize);
    let t = UNIT_POS_X;
    let carry = RuntimeCell::<u32>::new(0u32);
    let start = RuntimeCell::<u32>::new(0u32);
    while start.read() < len {
        let i = start.read() + t;
        let v = if i < len { counts[(base + i) as usize] } else { 0u32.into() };
        s[t as usize] = v;
        sync_cube();
        scan_inclusive(&mut s, block);
        if i < len {
            offsets[(base + i) as usize] = carry.read() + s[t as usize] - v;
        }
        let total = s[(block - 1u32) as usize];
        // Every unit has read the chunk total before the next chunk lands.
        sync_cube();
        carry.store(carry.read() + total);
        start.store(start.read() + block);
    }
}

/// One stable split of a tile by bit `bit`: `dst` gets `src`'s keys with the
/// bit clear first, then set, each group in `src` order. A unit's keys are
/// its contiguous `items`, so the scan of the units' set-bit counts places
/// every key.
#[cube]
fn split_by_bit(
    src: &SharedMemory<u32>,
    dst: &mut SharedMemory<u32>,
    s_scan: &mut SharedMemory<u32>,
    bit: u32,
    #[comptime] block: u32,
    #[comptime] items: u32,
) {
    let t = UNIT_POS_X;
    let base = t * items;
    let ones = RuntimeCell::<u32>::new(0u32);
    let mut i = 0u32;
    while i < items {
        ones.store(ones.read() + ((src[(base + i) as usize] >> bit) & 1u32));
        i += 1u32;
    }
    s_scan[t as usize] = ones.read();
    sync_cube();
    scan_inclusive(s_scan, block);
    let ones_before = s_scan[t as usize] - ones.read();
    let total_ones = s_scan[(block - 1u32) as usize];
    let zeros_before = base - ones_before;
    let one_base = block * items - total_ones;
    let zi = RuntimeCell::<u32>::new(0u32);
    let oi = RuntimeCell::<u32>::new(0u32);
    let mut i = 0u32;
    while i < items {
        let key = src[(base + i) as usize];
        if ((key >> bit) & 1u32) == 1u32 {
            dst[(one_base + ones_before + oi.read()) as usize] = key;
            oi.store(oi.read() + 1u32);
        } else {
            dst[(zeros_before + zi.read()) as usize] = key;
            zi.store(zi.read() + 1u32);
        }
        i += 1u32;
    }
    // `dst` is the next split's `src`, and `s_scan` is reused.
    sync_cube();
}

/// Sort one tile by its digit in shared memory and write every key to its
/// slot of the pass: the `(digit, tile)` offset plus the key's rank in the
/// tile. `counts` is [`radix_count_kernel`]'s table for this pass and
/// `offsets` its scan.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn radix_scatter_kernel(
    keys_in: &Array<u32>,
    keys_out: &mut Array<u32>,
    counts: &Array<u32>,
    offsets: &Array<u32>,
    n_tiles: u32,
    seg: u32,
    shift: u32,
    #[comptime] block: u32,
    #[comptime] items: u32,
    #[comptime] radix_bits: u32,
) {
    let radix = comptime![1u32 << radix_bits];
    let tile = CUBE_POS_X;
    let f = CUBE_POS_Y;
    let t = UNIT_POS_X;
    let tile_len = block * items;
    let base = f * seg + tile * tile_len;

    let mut a = SharedMemory::<u32>::new((block * items) as usize);
    let mut b = SharedMemory::<u32>::new((block * items) as usize);
    let mut s_scan = SharedMemory::<u32>::new(block as usize);
    let mut s_start = SharedMemory::<u32>::new(radix as usize);

    // Stage the tile, coalesced.
    let mut k = 0u32;
    while k < items {
        let i = t + k * block;
        a[i as usize] = keys_in[(base + i) as usize];
        k += 1u32;
    }
    sync_cube();

    // Eight one-bit splits, `a` -> `b` -> `a` ...: an even count, so the
    // sorted tile ends in `a`.
    let mut bit = 0u32;
    while bit < radix_bits {
        split_by_bit(&a, &mut b, &mut s_scan, shift + bit, block, items);
        split_by_bit(&b, &mut a, &mut s_scan, shift + bit + 1u32, block, items);
        bit += 2u32;
    }

    // Where each digit starts inside the tile: the tile's own counts, scanned.
    // `radix == block`: one digit per unit.
    let cnt = counts[((f * radix + t) * n_tiles + tile) as usize];
    s_start[t as usize] = cnt;
    sync_cube();
    scan_inclusive(&mut s_start, block);
    let excl = s_start[t as usize] - cnt;
    sync_cube();
    s_start[t as usize] = excl;
    sync_cube();

    let mut k = 0u32;
    while k < items {
        let i = t + k * block;
        let key = a[i as usize];
        let d = (key >> shift) & (radix - 1u32);
        let rank = i - s_start[d as usize];
        let slot = offsets[((f * radix + d) * n_tiles + tile) as usize] + rank;
        keys_out[(f * seg + slot) as usize] = key;
        k += 1u32;
    }
    sync_cube();
}

/// Index of the first key of the column at or above `key`, over
/// `sorted[base .. base + len]`.
#[cube]
fn lower_bound(sorted: &Array<u32>, base: u32, len: u32, key: u32) -> u32 {
    let lo = RuntimeCell::<u32>::new(0u32);
    let n = RuntimeCell::<u32>::new(len);
    while n.read() > 0u32 {
        let half = n.read() / 2u32;
        if sorted[(base + lo.read() + half) as usize] < key {
            lo.store(lo.read() + half + 1u32);
            n.store(n.read() - half - 1u32);
        } else {
            n.store(half);
        }
    }
    lo.read()
}

/// `WQSummary::QueryCutValues` against a sorted column, a cube per column:
/// `out[f * (max_bin + 1) ..]` gets `out_count[f]` cut values.
///
/// `n` present keys are the column's exact summary. Under `max_bin` distinct
/// values the cuts are every distinct value but the first; otherwise the
/// `max_bin - 1` rank queries `i * n / max_bin`, each answered by the key at
/// that rank — which is what the summary query resolves to when the summary
/// is exact — forced strictly increasing by taking the next distinct value
/// after a repeat, and ending early when the values run out. The sentinel
/// above the largest value closes the list either way, in the same `f32`
/// arithmetic as the host.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn cuts_kernel(
    sorted: &Array<u32>,
    out: &mut Array<f32>,
    out_count: &mut Array<u32>,
    seg: u32,
    max_bin: u32,
    #[comptime] block: u32,
    #[comptime] max_device_bin: u32,
) {
    let f = CUBE_POS_X;
    let base = f * seg;
    let t = UNIT_POS_X;
    let out_base = f * (max_bin + 1u32);
    let mut s = SharedMemory::<u32>::new(block as usize);
    let mut cand = SharedMemory::<u32>::new(max_device_bin as usize);

    // Present keys: everything below the first sentinel. Uniform.
    let n = lower_bound(sorted, base, seg, 0xFFFF_FFFFu32);

    // Distinct values: run starts, reduced over the cube. Branches, not
    // `||`: CubeCL evaluates both operands, and `sorted[idx - 1]` at `idx
    // == 0` wraps to an address past every buffer — a read CUDA faults on.
    let local = RuntimeCell::<u32>::new(0u32);
    let i = RuntimeCell::<u32>::new(t);
    while i.read() < n {
        let idx = i.read();
        let start = if idx == 0u32 {
            true.into()
        } else {
            sorted[(base + idx) as usize] != sorted[(base + idx - 1u32) as usize]
        };
        local.store(local.read() + u32::cast_from(start));
        i.store(idx + block);
    }
    s[t as usize] = local.read();
    let half = RuntimeCell::<u32>::new(block / 2u32);
    while half.read() > 0u32 {
        let d = half.read();
        sync_cube();
        if t < d {
            s[t as usize] += s[(t + d) as usize];
        }
        half.store(d / 2u32);
    }
    sync_cube();
    let distinct = s[0usize];
    sync_cube();

    if n == 0u32 {
        // `query_cut_values` on an empty summary.
        if t == 0u32 {
            out[out_base as usize] = 1e-5f32;
            out_count[f as usize] = 1u32;
        }
    } else if distinct <= max_bin {
        // Every distinct value but the first, compacted in order.
        let carry = RuntimeCell::<u32>::new(0u32);
        let start = RuntimeCell::<u32>::new(0u32);
        while start.read() < n {
            let idx = start.read() + t;
            // As above: the reads only happen inside the guards.
            let flag = if idx < n {
                if idx > 0u32 {
                    sorted[(base + idx) as usize] != sorted[(base + idx - 1u32) as usize]
                } else {
                    false.into()
                }
            } else {
                false.into()
            };
            let v = u32::cast_from(flag);
            s[t as usize] = v;
            sync_cube();
            scan_inclusive(&mut s, block);
            if flag {
                out[(out_base + carry.read() + s[t as usize] - 1u32) as usize] =
                    key_float(sorted[(base + idx) as usize]);
            }
            let total = s[(block - 1u32) as usize];
            sync_cube();
            carry.store(carry.read() + total);
            start.store(start.read() + block);
        }
        if t == 0u32 {
            let last = key_float(sorted[(base + n - 1u32) as usize]);
            out[(out_base + distinct - 1u32) as usize] = last + (last.abs() + 1e-5f32);
            out_count[f as usize] = distinct;
        }
    } else {
        // The rank queries, one per unit: `floor(i * total / max_bin)` with
        // `total` the count as the host holds it — an `f32`, so an integer
        // that may be rounded above 2^24 — and the answer the key at that
        // rank. In integers: the host's `f64` quotient is never rounded up
        // across a whole number (its error is 2^-19 at most against a gap
        // of at least `1 / max_bin`), so its floor is the integer quotient.
        let total = u64::cast_from(u32::cast_from(f32::cast_from(n)));
        let i = RuntimeCell::<u32>::new(t);
        while i.read() < max_bin {
            let q = i.read();
            if q > 0u32 {
                let rank = u64::cast_from(q) * total / u64::cast_from(max_bin);
                cand[q as usize] = sorted[(base + u32::cast_from(rank)) as usize];
            }
            i.store(q + block);
        }
        sync_cube();
        if t == 0u32 {
            let last = RuntimeCell::<u32>::new(sorted[base as usize]);
            let count = RuntimeCell::<u32>::new(0u32);
            let stop = RuntimeCell::<bool>::new(false);
            let q = RuntimeCell::<u32>::new(1u32);
            while q.read() < max_bin && !stop.read() {
                let mut ck = cand[q.read() as usize];
                if ck <= last.read() {
                    // The first key above the last cut, or nothing left.
                    let nv = lower_bound(sorted, base, n, last.read() + 1u32);
                    if nv == n {
                        stop.store(true);
                    } else {
                        ck = sorted[(base + nv) as usize];
                    }
                }
                if !stop.read() {
                    out[(out_base + count.read()) as usize] = key_float(ck);
                    count.store(count.read() + 1u32);
                    last.store(ck);
                }
                q.store(q.read() + 1u32);
            }
            let cpt = key_float(sorted[(base + n - 1u32) as usize]);
            out[(out_base + count.read()) as usize] = cpt + (cpt.abs() + 1e-5f32);
            out_count[f as usize] = count.read() + 1u32;
        }
    }
    sync_cube();
}

/// Keep the host sketch for every device fit, whatever the runtime.
///
/// The device cuts are exact quantiles where the host's are a pruned
/// summary's, so a fit binned by one is not the fit binned by the other. The
/// parity tests (`tests/gpu_training.rs`, the string-parameter oracle) hold
/// the device fit to the CPU fit bit for bit and set this; `XGB_HOST_SKETCH`
/// in the environment does the same for a measurement.
static FORCE_HOST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);

/// Route every device fit's sketch to the host (`true`) or let
/// [`applies`] decide (`false`).
pub fn force_host_sketch(on: bool) {
    FORCE_HOST.store(u8::from(on), std::sync::atomic::Ordering::Relaxed);
}

fn host_forced() -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    match FORCE_HOST.load(Relaxed) {
        2 => {
            let on = std::env::var_os("XGB_HOST_SKETCH").is_some();
            FORCE_HOST.store(u8::from(on), Relaxed);
            on
        }
        v => v == 1,
    }
}

/// Whether [`device_cuts`] applies to this fit; the host sketch otherwise.
pub fn applies<R: Runtime>(client: &ComputeClient<R>, dmat: &DMatrix, max_bin: u32) -> bool {
    !host_forced()
        && launch::has_planes(client)
        && dmat.info().weights.is_none()
        && !dmat.info().has_categorical()
        && (max_bin as usize) <= MAX_DEVICE_BIN
        && dmat.num_col() > 0
}

/// The matrix's values on the device, as the sketch and the binning read
/// them: the CSR arrays, or just the values when the matrix is full.
pub struct DeviceValues {
    pub row_ptr: Handle,
    pub row_ptr_len: usize,
    pub index: Handle,
    pub index_len: usize,
    pub value: Handle,
    pub n_entries: usize,
    pub n_rows: usize,
    pub n_cols: usize,
    /// Every cell present, in column order: the entry index names the cell.
    pub dense_input: bool,
}

impl DeviceValues {
    /// Whether the device routes can read `dmat`: its rows' entries have to
    /// be in ascending column order, which `DMatrix::from_dense` produces and
    /// the binning kernel's per-row search relies on.
    ///
    /// A full matrix — every row holding every column — can only be in
    /// column order, so it is answered from the row lengths alone rather than
    /// by a pass over every entry (25 million of them at 500 000 × 50, and
    /// ~80 ms on a cloud VM).
    pub fn readable(dmat: &DMatrix) -> bool {
        let n_cols = dmat.num_col();
        if dmat.value.len() == dmat.num_row() * n_cols
            && dmat.row_ptr.windows(2).all(|w| w[1] - w[0] == n_cols)
        {
            return true;
        }
        dmat.row_ptr.windows(2).all(|w| dmat.index[w[0]..w[1]].windows(2).all(|c| c[0] < c[1]))
    }

    /// Upload `dmat`'s entries once, for the sketch and the binning to share.
    ///
    /// Through the pinned pool (`tables::upload_vec`): the client's slice
    /// route moved 100 MB in 290 ms on a cloud VM.
    pub fn upload<R: Runtime>(client: &ComputeClient<R>, dmat: &DMatrix) -> Self {
        let (n_rows, n_cols) = (dmat.num_row(), dmat.num_col());
        let n_entries = dmat.value.len();
        let dense_input =
            n_entries == n_rows * n_cols && dmat.row_ptr.windows(2).all(|w| w[1] - w[0] == n_cols);
        let value = upload_slice(client, &dmat.value);
        let (row_ptr, index, row_ptr_len, index_len) = if dense_input {
            (client.create_from_slice(bytemuck::cast_slice(&[0u32])), value.clone(), 1, 1)
        } else {
            let row_ptr: Vec<u32> = dmat.row_ptr.iter().map(|&p| p as u32).collect();
            let len = row_ptr.len();
            (
                upload_vec(client, row_ptr),
                upload_slice(client, &dmat.index),
                len,
                n_entries.max(1),
            )
        };
        Self { row_ptr, row_ptr_len, index, index_len, value, n_entries, n_rows, n_cols, dense_input }
    }
}

/// `build_cuts` on the device. The caller checks [`applies`] first.
pub fn device_cuts<R: Runtime>(
    client: &ComputeClient<R>,
    values: &DeviceValues,
    max_bin: u32,
) -> HistogramCuts {
    let (n_rows, n_cols) = (values.n_rows, values.n_cols);
    let n_tiles = (n_rows as u32).div_ceil(SORT_TILE).max(1);
    let seg = n_tiles * SORT_TILE;
    let total = n_cols * seg as usize;
    let mut phases = super::PhaseLog::new(client.clone());

    // Keys, column-major, sentinel-filled so the padding sorts last.
    let mut keys = client.empty(total * size_of::<u32>());
    let mut scratch = client.empty(total * size_of::<u32>());
    {
        let (count, dim) = launch::elementwise(client, total);
        // SAFETY: the kernel guards its index against `n`.
        unsafe {
            fill_u32_kernel::launch_unchecked::<R>(
                client,
                count,
                dim,
                ArrayArg::from_raw_parts(keys.clone(), total),
                SENTINEL,
                total as u32,
            );
        }
    }
    if values.n_entries > 0 {
        let (count, dim) = launch::elementwise(client, values.n_entries);
        // SAFETY: the entry index is guarded, and every cell it names is
        // inside the `n_cols * seg` buffer.
        unsafe {
            float_keys_kernel::launch_unchecked::<R>(
                client,
                count,
                dim,
                ArrayArg::from_raw_parts(values.row_ptr.clone(), values.row_ptr_len),
                ArrayArg::from_raw_parts(values.index.clone(), values.index_len),
                ArrayArg::from_raw_parts(values.value.clone(), values.n_entries),
                ArrayArg::from_raw_parts(keys.clone(), total),
                values.n_entries as u32,
                n_rows as u32,
                n_cols as u32,
                seg,
                values.dense_input,
            );
        }
    }
    phases.mark("keys");

    // Four LSD passes.
    let table = n_cols * (RADIX * n_tiles) as usize;
    let counts = client.empty(table * size_of::<u32>());
    let offsets = client.empty(table * size_of::<u32>());
    let tile_count = CubeCount::Static(n_tiles, n_cols as u32, 1);
    let tile_dim = CubeDim::new_1d(SORT_BLOCK);
    for pass in 0..(32 / RADIX_BITS) {
        let shift = pass * RADIX_BITS;
        // SAFETY: every index is derived from the grid and the table sizes
        // the buffers were allocated with.
        unsafe {
            radix_count_kernel::launch_unchecked::<R>(
                client,
                tile_count.clone(),
                tile_dim,
                ArrayArg::from_raw_parts(keys.clone(), total),
                ArrayArg::from_raw_parts(counts.clone(), table),
                n_tiles,
                seg,
                shift,
                SORT_BLOCK,
                SORT_ITEMS,
                RADIX,
            );
            radix_scan_kernel::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_cols as u32, 1, 1),
                tile_dim,
                ArrayArg::from_raw_parts(counts.clone(), table),
                ArrayArg::from_raw_parts(offsets.clone(), table),
                RADIX * n_tiles,
                SORT_BLOCK,
            );
            radix_scatter_kernel::launch_unchecked::<R>(
                client,
                tile_count.clone(),
                tile_dim,
                ArrayArg::from_raw_parts(keys.clone(), total),
                ArrayArg::from_raw_parts(scratch.clone(), total),
                ArrayArg::from_raw_parts(counts.clone(), table),
                ArrayArg::from_raw_parts(offsets.clone(), table),
                n_tiles,
                seg,
                shift,
                SORT_BLOCK,
                SORT_ITEMS,
                RADIX_BITS,
            );
        }
        std::mem::swap(&mut keys, &mut scratch);
    }
    phases.mark("sort");

    let width = max_bin as usize + 1;
    let out = client.empty(n_cols * width * size_of::<f32>());
    let out_count = client.empty(n_cols * size_of::<u32>());
    // SAFETY: a cube per column, every index inside its column and its slice
    // of `out`, which holds `max_bin + 1` per column.
    unsafe {
        cuts_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(n_cols as u32, 1, 1),
            tile_dim,
            ArrayArg::from_raw_parts(keys.clone(), total),
            ArrayArg::from_raw_parts(out.clone(), n_cols * width),
            ArrayArg::from_raw_parts(out_count.clone(), n_cols),
            seg,
            max_bin,
            SORT_BLOCK,
            MAX_DEVICE_BIN as u32,
        );
    }
    let cut_bytes = client.read_one_unchecked(out);
    let count_bytes = client.read_one_unchecked(out_count);
    let all: &[f32] = bytemuck::cast_slice(&cut_bytes);
    let counts_host: &[u32] = bytemuck::cast_slice(&count_bytes);
    phases.mark("cuts");
    phases.report_as("device sketch");

    let mut cuts = HistogramCuts {
        cut_ptrs: vec![0u32],
        cut_values: Vec::new(),
        // 3.4.0's lower bound of every numeric feature's first bin.
        min_values: vec![f32::NEG_INFINITY; n_cols],
        is_categorical: Vec::new(),
    };
    for f in 0..n_cols {
        let n = counts_host[f] as usize;
        cuts.cut_values.extend_from_slice(&all[f * width..f * width + n]);
        cuts.cut_ptrs.push(cuts.cut_values.len() as u32);
    }
    cuts
}
