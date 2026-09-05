//! Host-side ELLPACK quantile matrix, the minimal analogue of
//! `xgboost::EllpackDeviceAccessor` needed by the histogram kernel.
//!
//! Rows are stored with a fixed `row_stride`; each entry is a quantised bin
//! index (`gidx`). Three layouts exist, mirroring the `kDense`/`kCompressed`
//! dispatch in `histogram.cu`:
//!
//! * [`EllpackLayout::Dense`] — every entry valid, bins are local to their
//!   feature (`kDense && kCompressed`).
//! * [`EllpackLayout::DenseCompressed`] — bins local to their feature, missing
//!   entries hold `null_value` (`!kDense && kCompressed`).
//! * [`EllpackLayout::Sparse`] — bins are global (cut pointers already added),
//!   missing entries hold `null_value` (`!kDense && !kCompressed`).
//!
//! # Packing on device
//!
//! The host matrix is one `u32` per entry, which is what the oracle tests
//! read. On device it is **bit-packed** at the narrowest width that holds its
//! largest value ([`bits_for`]): 8 bits for 256-bin data with no missing
//! entry, 9 once the missing sentinel is stored alongside, and so on — the
//! same idea as `xgboost`'s `CompressedBufferWriter`. The matrix is the
//! largest buffer a fit holds and the histogram kernel streams all of it per
//! level, so this is a 3.5–4× cut in that kernel's index traffic. Kernels read
//! entries through [`load_bin`], specialised at comptime on the width: a
//! shift and a mask when the width divides a word, a two-word read otherwise.

use cubecl::prelude::*;
use rayon::prelude::*;

use crate::data::cuts::HistogramCuts;

/// Bits per packed entry for values up to `max_value` inclusive: the narrowest
/// width that holds it, and never fewer than one.
pub fn bits_for(max_value: u32) -> u32 {
    (u32::BITS - max_value.leading_zeros()).max(1)
}

/// Pack `values` at `bits` per entry, least-significant bits first within each
/// `u32` word and entries running across word boundaries.
///
/// Thirty-two entries occupy exactly `bits` words, which is what lets the
/// packing run in parallel over aligned chunks. One padding word follows the
/// data so [`load_bin`]'s two-word read of the last entry stays in bounds.
pub fn pack_bins(values: &[u32], bits: u32) -> Vec<u32> {
    debug_assert!((1..=32).contains(&bits));
    if bits == 32 {
        // Whole words: the identity, plus the padding word.
        let mut words = Vec::with_capacity(values.len() + 1);
        words.extend_from_slice(values);
        words.push(0);
        return words;
    }
    let chunk_words = bits as usize;
    let mut words = vec![0u32; values.len().div_ceil(32) * chunk_words + 1];
    words[..values.len().div_ceil(32) * chunk_words]
        .par_chunks_mut(chunk_words)
        .zip(values.par_chunks(32))
        .for_each(|(out, chunk)| {
            for (i, &v) in chunk.iter().enumerate() {
                debug_assert!(bits == 32 || v < (1u32 << bits));
                let bit = i as u64 * bits as u64;
                let (w, off) = ((bit >> 5) as usize, (bit & 31) as u32);
                let wide = (v as u64) << off;
                out[w] |= wide as u32;
                if off + bits > 32 {
                    out[w + 1] |= (wide >> 32) as u32;
                }
            }
        });
    words
}

/// The narrowest width that holds every entry of `matrix`.
///
/// The sentinel counts as a value wherever the layout can store it, so a
/// kernel's `bin != null_value` test still means what it did.
pub fn natural_bits(matrix: &EllpackMatrix) -> u32 {
    let data_max = matrix.gidx.iter().copied().max().unwrap_or(0);
    let max_value = if matrix.is_dense() { data_max } else { data_max.max(matrix.null_value) };
    bits_for(max_value)
}

/// The width the matrix is packed at on `client`'s runtime.
///
/// Packing is a bandwidth trade: fewer bytes per entry, a shift and a mask
/// per read. A GPU streaming the matrix from device memory is on the right
/// side of it. The CPU runtime is not — its histogram kernel is
/// instruction-bound, and packing it was measured at 16.3 ms against 10.0 ms
/// per build at 8 bits, and 53 ms against 25 ms on the two-word path — so a
/// plane-less runtime keeps whole words, which [`load_bin`] reads with no
/// arithmetic at all.
pub fn device_bits<R: Runtime>(client: &ComputeClient<R>, matrix: &EllpackMatrix) -> u32 {
    if super::launch::has_planes(client) { natural_bits(matrix) } else { 32 }
}

/// Read packed entry `entry` of a [`pack_bins`] buffer at `bits` per entry.
///
/// Comptime on the width, so a build carries one shape per width it meets: a
/// word index, a shift and a mask where the width divides a word (8, 16, 32);
/// a 64-bit bit position and a two-word read otherwise (9 bits is the common
/// case — 256-bin data with a missing sentinel). The buffer's padding word is
/// what makes the second read of the last entry sound.
#[cube]
pub fn load_bin(words: &Array<u32>, entry: u32, #[comptime] bits: u32) -> u32 {
    // Comptime values become runtime constants where they meet runtime
    // operands.
    let mask = comptime![if bits == 32 { u32::MAX } else { (1u32 << bits) - 1 }].runtime();
    if comptime![bits == 32] {
        // Unpacked: the plain load, which is what a plane-less runtime wants.
        words[entry as usize]
    } else if comptime![32 % bits == 0] {
        let per_word = comptime![32 / bits].runtime();
        let width = comptime![bits].runtime();
        let word = entry / per_word;
        let off = (entry - word * per_word) * width;
        (words[word as usize] >> off) & mask
    } else {
        let width = comptime![bits as u64].runtime();
        let bit = u64::cast_from(entry) * width;
        let word = u32::cast_from(bit >> 5u64);
        let off = bit & 31u64;
        let lo = u64::cast_from(words[word as usize]);
        let hi = u64::cast_from(words[(word + 1u32) as usize]);
        u32::cast_from(((hi << 32u64) | lo) >> off) & mask
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EllpackLayout {
    Dense,
    DenseCompressed,
    Sparse,
}

#[derive(Clone, Debug)]
pub struct EllpackMatrix {
    /// Bin indices, `n_rows * row_stride` entries.
    pub gidx: Vec<u32>,
    /// Number of matrix columns per row (== `n_features` for dense data).
    pub row_stride: usize,
    /// First row id of this batch.
    pub base_rowid: u32,
    pub n_rows: usize,
    /// Cut pointers (`feature_segments` on device): bin range of feature `f`
    /// is `cut_ptrs[f]..cut_ptrs[f + 1]`. Length `n_features + 1`.
    pub cut_ptrs: Vec<u32>,
    /// Sentinel for a missing entry (compared against the *stored* value).
    pub null_value: u32,
    pub layout: EllpackLayout,
}

/// Build an ELLPACK from a matrix and its quantile cuts.
///
/// The analogue of `EllpackPageImpl`'s constructor, and the GPU counterpart of
/// [`crate::data::gradient_index::build_gradient_index`]: it bins with the same
/// [`HistogramCuts::bin_of`], so a GPU fit and a CPU fit see the same bins.
///
/// The result always has `row_stride == n_features`, which is what lets the
/// partitioner index a row's feature directly instead of searching. A value
/// that has no bin — an unseen category — is stored as missing, exactly as the
/// CPU index treats it.
pub fn build_ellpack(dmat: &crate::data::DMatrix, cuts: &HistogramCuts) -> EllpackMatrix {
    let n_rows = dmat.num_row();
    let n_features = dmat.num_col();

    // A single sentinel has to be invalid for *every* feature, so it sits
    // above the widest feature's local bin count.
    let null_value = (0..n_features).map(|f| cuts.feature_bins(f)).max().unwrap_or(0) as u32;

    let mut gidx = vec![null_value; n_rows * n_features];
    gidx.par_chunks_mut(n_features).enumerate().for_each(|(r, row_out)| {
        let (indices, values) = dmat.row(r);
        for (&f, &v) in indices.iter().zip(values) {
            let f = f as usize;
            if let Some(bin) = cuts.bin_of(v, f) {
                // Feature-local, which is what the `compressed` layouts store.
                row_out[f] = bin - cuts.cut_ptrs[f];
            }
        }
    });

    // `Dense` is the layout with no missing entry at all, which is a property
    // of the filled matrix — not of how the `DMatrix` chose to store it. A
    // `DMatrix` elides zeros as well as NaNs, and an unseen category has no
    // bin, so both leave a hole here.
    let layout = if gidx.contains(&null_value) {
        EllpackLayout::DenseCompressed
    } else {
        EllpackLayout::Dense
    };

    EllpackMatrix {
        gidx,
        row_stride: n_features,
        base_rowid: 0,
        n_rows,
        cut_ptrs: cuts.cut_ptrs.clone(),
        null_value,
        layout,
    }
}

/// An [`EllpackMatrix`] resident on device, bit-packed (see the module docs).
///
/// The histogram kernel reads `gidx` row by row, and it is uploaded once and
/// shared with everything else that walks rows — it is by far the largest
/// buffer a fit holds (`bits * n_rows * row_stride / 8` bytes).
///
/// The row partitioner reads one *feature* of many rows instead, and for the
/// feature-local layouts it gets `gidx_t`, the same bins feature-major:
/// entry `f * n_rows + r`. A split's rows are in ascending order within their
/// segment (the partition is stable), so successive decisions read successive
/// entries of one column rather than one cache line per row of the row-major
/// matrix — the CPU grower's `ColumnIndex`, and a coalesced read on a GPU. It
/// costs a second copy of the matrix; the sparse layout, which the partitioner
/// has to search a row for anyway, binds a one-word placeholder.
#[derive(Clone, Debug)]
pub struct DeviceEllpack {
    /// Row-major entries, packed at `bits` per entry.
    pub gidx: cubecl::server::Handle,
    /// Feature-major copy of `gidx` for the feature-local layouts, packed the
    /// same way; a one-word placeholder for the sparse one.
    pub gidx_t: cubecl::server::Handle,
    pub cut_ptrs: cubecl::server::Handle,
    /// Lengths of the two packed buffers, in `u32` words.
    pub gidx_len: usize,
    pub gidx_t_len: usize,
    /// Bits per packed entry, a comptime argument of every kernel that reads
    /// either buffer.
    pub bits: u32,
    pub n_cuts: usize,
    pub n_rows: usize,
    pub row_stride: u32,
    pub base_rowid: u32,
    pub null_value: u32,
    pub n_bins: u32,
    pub dense: bool,
    pub compressed: bool,
}

impl DeviceEllpack {
    /// Upload `matrix` to the device, packed at the width [`device_bits`]
    /// chooses for this runtime.
    pub fn upload<R: Runtime>(client: &ComputeClient<R>, matrix: &EllpackMatrix) -> Self {
        Self::upload_with_bits(client, matrix, device_bits(client, matrix))
    }

    /// [`upload`](Self::upload) at an explicit width, which must hold every
    /// entry ([`natural_bits`] or wider). This is how the tests run the
    /// packed read paths on a runtime that would not choose them.
    pub fn upload_with_bits<R: Runtime>(
        client: &ComputeClient<R>,
        matrix: &EllpackMatrix,
        bits: u32,
    ) -> Self {
        debug_assert!(bits >= natural_bits(matrix), "a width of {bits} bits cannot hold the matrix");
        let packed = pack_bins(&matrix.gidx, bits);
        // Feature-major copy, one column per feature; the placeholder keeps
        // the binding non-empty on a layout whose kernel never reads it.
        let transposed: Vec<u32> = if matrix.is_compressed() && !matrix.gidx.is_empty() {
            let (n_rows, stride) = (matrix.n_rows, matrix.row_stride);
            let mut t = vec![0u32; matrix.gidx.len()];
            t.par_chunks_mut(n_rows).enumerate().for_each(|(f, column)| {
                for (r, cell) in column.iter_mut().enumerate() {
                    *cell = matrix.gidx[r * stride + f];
                }
            });
            pack_bins(&t, bits)
        } else {
            vec![0u32]
        };
        Self {
            gidx: client.create_from_slice(bytemuck::cast_slice(&packed)),
            gidx_t: client.create_from_slice(bytemuck::cast_slice(&transposed)),
            cut_ptrs: client.create_from_slice(bytemuck::cast_slice(&matrix.cut_ptrs)),
            gidx_len: packed.len(),
            gidx_t_len: transposed.len(),
            bits,
            n_cuts: matrix.cut_ptrs.len(),
            n_rows: matrix.n_rows,
            row_stride: matrix.row_stride as u32,
            base_rowid: matrix.base_rowid,
            null_value: matrix.null_value,
            n_bins: matrix.n_bins(),
            dense: matrix.is_dense(),
            compressed: matrix.is_compressed(),
        }
    }

    pub fn n_features(&self) -> usize {
        self.n_cuts - 1
    }
}

impl EllpackMatrix {
    pub fn n_features(&self) -> usize {
        self.cut_ptrs.len() - 1
    }

    /// Total number of histogram bins across all features.
    pub fn n_bins(&self) -> u32 {
        *self.cut_ptrs.last().unwrap()
    }

    pub fn is_dense(&self) -> bool {
        self.layout == EllpackLayout::Dense
    }

    /// Whether stored bins are feature-local (need `cut_ptrs[fidx]` added).
    pub fn is_compressed(&self) -> bool {
        self.layout != EllpackLayout::Sparse
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Host-side reference for [`load_bin`]: the same bit arithmetic, written
    /// the slow way.
    fn unpack(words: &[u32], entry: usize, bits: u32) -> u32 {
        let bit = entry as u64 * bits as u64;
        let (w, off) = ((bit >> 5) as usize, (bit & 31) as u32);
        let pair = (words[w + 1] as u64) << 32 | words[w] as u64;
        let mask = if bits == 32 { u32::MAX as u64 } else { (1u64 << bits) - 1 };
        ((pair >> off) & mask) as u32
    }

    #[test]
    fn packing_round_trips_at_every_width() {
        for bits in 1..=32u32 {
            let cap = if bits == 32 { u32::MAX } else { (1u32 << bits) - 1 };
            let values: Vec<u32> =
                (0..1000u64).map(|i| (i.wrapping_mul(2_654_435_761) % (cap as u64 + 1)) as u32).collect();
            let words = pack_bins(&values, bits);
            let expected_words = if bits == 32 {
                values.len() + 1
            } else {
                values.len().div_ceil(32) * bits as usize + 1
            };
            assert_eq!(words.len(), expected_words, "bits {bits}");
            for (i, &v) in values.iter().enumerate() {
                assert_eq!(unpack(&words, i, bits), v, "bits {bits} entry {i}");
            }
        }
    }

    #[test]
    fn width_follows_the_largest_stored_value() {
        assert_eq!(bits_for(0), 1);
        assert_eq!(bits_for(1), 1);
        assert_eq!(bits_for(255), 8);
        assert_eq!(bits_for(256), 9);
        assert_eq!(bits_for(u32::MAX), 32);
    }
}

// ------------------------------------------------------ binning on device ----
//
// `build_ellpack` + `DeviceEllpack::upload` is three passes over every entry
// on the host — a binary search per value, a bit-pack, a transpose and a
// second pack — and two uploads of the packed result. At 500 000 × 50 that
// was 190 ms on a laptop and 950 ms on a cloud VM's slower cores, against a
// whole 20-round `gpu_hist` fit of 470 ms upstream: the setup alone lost the
// race before a tree was grown. Upstream bins on the device
// (`EllpackPageImpl`'s `CompressBinEllpackKernel`), and so does this: the raw
// values go up once (`sketch::DeviceValues`, shared with the device sketch),
// and one kernel per layout bins and packs straight into the packed buffer.
//
// One kernel, not a bin pass into a word matrix and a pack pass over it: at
// 500 000 × 50 the two word matrices were 200 MB of device memory allocated
// and streamed for nothing — 180 ms on a T4, most of it the allocations.

/// The bits of the value of cell `(r, f)`: the cell's own entry on a full
/// matrix, else the entry of row `r` whose column is `f`, found by binary
/// search over the row's entries, which are in ascending column order; the
/// sentinel `0xFFFF_FFFF` — a NaN pattern — when the row has none.
///
/// Bits rather than the float, and [`is_nan_bits`] rather than `v != v`: the
/// CPU runtime's compiler folds `v != v` to false, which turned every missing
/// value into bin zero.
#[cube]
fn cell_bits(
    row_ptr: &Array<u32>,
    index: &Array<u32>,
    value: &Array<f32>,
    r: u32,
    f: u32,
    row_stride: u32,
    #[comptime] dense_input: bool,
) -> u32 {
    if dense_input {
        u32::reinterpret(value[(r * row_stride + f) as usize])
    } else {
        let begin = row_ptr[r as usize];
        let end = row_ptr[(r + 1u32) as usize];
        let lo = RuntimeCell::<u32>::new(begin);
        let n = RuntimeCell::<u32>::new(end - begin);
        while n.read() > 0u32 {
            let half = n.read() / 2u32;
            if index[(lo.read() + half) as usize] < f {
                lo.store(lo.read() + half + 1u32);
                n.store(n.read() - half - 1u32);
            } else {
                n.store(half);
            }
        }
        let i = lo.read();
        // Nested, not `&&`: CubeCL evaluates both operands, and `index[end]`
        // is one past the buffer on the last row.
        if i < end {
            if index[i as usize] == f {
                u32::reinterpret(value[i as usize])
            } else {
                0xFFFF_FFFFu32.into()
            }
        } else {
            0xFFFF_FFFFu32.into()
        }
    }
}

/// Whether `bits` is a NaN pattern: exponent all ones and a non-zero
/// mantissa, sign ignored.
#[cube]
pub fn is_nan_bits(bits: u32) -> bool {
    (bits & 0x7FFF_FFFFu32) > 0x7F80_0000u32
}

/// `HistogramCuts::search_bin`, feature-local: the first cut of `f` strictly
/// above `v`, and the last bin for a value above every cut.
#[cube]
fn local_bin(cut_ptrs: &Array<u32>, cut_values: &Array<f32>, f: u32, v: f32) -> u32 {
    let beg = cut_ptrs[f as usize];
    let end = cut_ptrs[(f + 1u32) as usize];
    let base = RuntimeCell::<u32>::new(0u32);
    let n = RuntimeCell::<u32>::new(end - beg);
    while n.read() > 1u32 {
        let half = n.read() / 2u32;
        if cut_values[(beg + base.read() + half) as usize] <= v {
            base.store(base.read() + half);
        }
        n.store(n.read() - half);
    }
    let mut lo = base.read();
    if cut_values[(beg + lo) as usize] <= v {
        lo += 1u32;
    }
    if lo == end - beg {
        lo -= 1u32;
    }
    lo
}

/// Bin and pack 32 consecutive row-major cells per unit, straight into the
/// packed buffer: [`pack_bins`]'s chunk, so the words are the host's. Cell
/// `c` is `(c / stride, c % stride)`; the feature-major matrix is
/// [`transpose_packed_kernel`] over this one. A missing value — NaN, or a
/// hole in a CSR row — packs as `null_value`; cells past `n` pack as zero.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn bin_pack_kernel(
    row_ptr: &Array<u32>,
    index: &Array<u32>,
    value: &Array<f32>,
    cut_ptrs: &Array<u32>,
    cut_values: &Array<f32>,
    out: &mut Array<u32>,
    n_cells: u32,
    n_chunks: u32,
    row_stride: u32,
    null_value: u32,
    #[comptime] bits: u32,
    #[comptime] dense_input: bool,
) {
    let chunk = ABSOLUTE_POS as u32;
    if chunk < n_chunks {
        let width = comptime![bits as u64].runtime();
        let first = chunk * 32u32;
        let out_base = chunk * comptime![bits].runtime();
        let acc = RuntimeCell::<u64>::new(0u64);
        let w = RuntimeCell::<u32>::new(0u32);
        let i = RuntimeCell::<u32>::new(0u32);
        while i.read() < 32u32 {
            let c = first + i.read();
            let bin = if c < n_cells {
                let r = c / row_stride;
                let f = c - r * row_stride;
                let bits = cell_bits(row_ptr, index, value, r, f, row_stride, dense_input);
                if is_nan_bits(bits) {
                    null_value
                } else {
                    local_bin(cut_ptrs, cut_values, f, f32::reinterpret(bits))
                }
            } else {
                0u32.into()
            };
            let bit = u64::cast_from(i.read()) * width;
            let off = bit - u64::cast_from(w.read()) * 32u64;
            acc.store(acc.read() | (u64::cast_from(bin) << off));
            if off + width >= 32u64 {
                out[(out_base + w.read()) as usize] = u32::cast_from(acc.read() & 0xFFFF_FFFFu64);
                acc.store(acc.read() >> 32u64);
                w.store(w.read() + 1u32);
            }
            i.store(i.read() + 1u32);
        }
    }
}

/// The feature-major packed matrix from the row-major one: a unit packs 32
/// consecutive feature-major cells, each read back out of `gidx` with
/// [`load_bin`], into [`pack_bins`]'s chunk.
///
/// Binning feature-major directly reads a CSR row per cell with the rows a
/// warp apart, and was 48 ms of a 66 ms device binning on a T4 with 30% of
/// the cells missing (15 of 23 ms dense); this reads bins, not values, and
/// its units are numbered so that a warp reads the same rows for consecutive
/// features — `chunks_per_feature` is the row-block count a unit index is
/// decomposed by — which keeps the rows it touches within the cache.
#[cube(launch_unchecked)]
pub fn transpose_packed_kernel(
    gidx: &Array<u32>,
    out: &mut Array<u32>,
    n_cells: u32,
    n_chunks: u32,
    n_rows: u32,
    row_stride: u32,
    chunks_per_feature: u32,
    #[comptime] bits: u32,
) {
    let u = ABSOLUTE_POS as u32;
    let chunk = (u % row_stride) * chunks_per_feature + u / row_stride;
    if chunk < n_chunks {
        let width = comptime![bits as u64].runtime();
        let first = chunk * 32u32;
        let out_base = chunk * comptime![bits].runtime();
        let acc = RuntimeCell::<u64>::new(0u64);
        let w = RuntimeCell::<u32>::new(0u32);
        let i = RuntimeCell::<u32>::new(0u32);
        while i.read() < 32u32 {
            let c = first + i.read();
            let bin = if c < n_cells {
                let f = c / n_rows;
                let r = c - f * n_rows;
                load_bin(gidx, r * row_stride + f, bits)
            } else {
                0u32.into()
            };
            let bit = u64::cast_from(i.read()) * width;
            let off = bit - u64::cast_from(w.read()) * 32u64;
            acc.store(acc.read() | (u64::cast_from(bin) << off));
            if off + width >= 32u64 {
                out[(out_base + w.read()) as usize] = u32::cast_from(acc.read() & 0xFFFF_FFFFu64);
                acc.store(acc.read() >> 32u64);
                w.store(w.read() + 1u32);
            }
            i.store(i.read() + 1u32);
        }
    }
}

/// What the device path knows about the matrix it binned, in place of the
/// host [`EllpackMatrix`] it never builds.
#[derive(Clone, Debug)]
pub struct EllpackShape {
    pub n_rows: usize,
    pub row_stride: usize,
    pub base_rowid: u32,
    pub cut_ptrs: Vec<u32>,
    pub null_value: u32,
    pub layout: EllpackLayout,
}

impl EllpackShape {
    pub fn of(matrix: &EllpackMatrix) -> Self {
        Self {
            n_rows: matrix.n_rows,
            row_stride: matrix.row_stride,
            base_rowid: matrix.base_rowid,
            cut_ptrs: matrix.cut_ptrs.clone(),
            null_value: matrix.null_value,
            layout: matrix.layout,
        }
    }

    pub fn n_features(&self) -> usize {
        self.cut_ptrs.len() - 1
    }

    pub fn n_bins(&self) -> u32 {
        *self.cut_ptrs.last().unwrap()
    }

    pub fn is_dense(&self) -> bool {
        self.layout == EllpackLayout::Dense
    }

    pub fn is_compressed(&self) -> bool {
        self.layout != EllpackLayout::Sparse
    }
}

impl DeviceEllpack {
    /// Bin `values` against `cuts` on the device: [`build_ellpack`] followed
    /// by [`Self::upload`], without the host ever holding the bin matrix.
    ///
    /// The same bins, the same packing: the kernel is `search_bin` and
    /// `pack_bins`, chunk for chunk, so the buffers are word-identical to the
    /// host route's (`tests/kernels.rs` asserts it). What is not the same is
    /// the width when the matrix has no hole: the host measures the largest
    /// bin actually stored, this takes the largest bin the cuts allow — one
    /// bit at most, and the kernels read either.
    ///
    /// Categorical features are left to the host route by the caller: their
    /// bin is a category-code check, not a cut search. So is the CPU runtime,
    /// whose cores are the host's anyway.
    pub fn bin_on_device<R: Runtime>(
        client: &ComputeClient<R>,
        values: &super::sketch::DeviceValues,
        cuts: &HistogramCuts,
    ) -> (Self, EllpackShape) {
        let n_rows = values.n_rows;
        let n_features = values.n_cols;
        let n_cells = n_rows * n_features;
        let dense_input = values.dense_input;
        let null_value = (0..n_features).map(|f| cuts.feature_bins(f)).max().unwrap_or(0) as u32;
        let layout = if dense_input { EllpackLayout::Dense } else { EllpackLayout::DenseCompressed };
        let shape = EllpackShape {
            n_rows,
            row_stride: n_features,
            base_rowid: 0,
            cut_ptrs: cuts.cut_ptrs.clone(),
            null_value,
            layout,
        };
        let mut phases = super::PhaseLog::new(client.clone());
        let d_cut_ptrs = client.create_from_slice(bytemuck::cast_slice(&cuts.cut_ptrs));
        let d_cut_values = client.create_from_slice(bytemuck::cast_slice(&cuts.cut_values));

        // Width: `natural_bits` over what the cuts can produce rather than
        // what was stored (see above).
        let data_max = null_value.saturating_sub(1);
        let bits = if super::launch::has_planes(client) {
            bits_for(if dense_input { data_max } else { null_value })
        } else {
            32
        };
        // Whole 32-entry chunks plus the padding word, at every width: at 32
        // bits the chunk is the identity and the buffer simply rounds up to
        // it, which the kernels reading the first `n_cells` entries never see.
        let n_chunks = n_cells.div_ceil(32);
        let packed_len = n_chunks * bits as usize + 1;
        let packed = {
            let out = client.empty(packed_len * size_of::<u32>());
            let (count, dim) = super::launch::elementwise(client, n_chunks);
            // SAFETY: the chunk index is guarded against `n_chunks`, cells
            // against `n_cells`; a chunk writes exactly `bits` words inside
            // the `n_chunks * bits` the buffer holds.
            unsafe {
                bin_pack_kernel::launch_unchecked::<R>(
                    client,
                    count,
                    dim,
                    ArrayArg::from_raw_parts(values.row_ptr.clone(), values.row_ptr_len),
                    ArrayArg::from_raw_parts(values.index.clone(), values.index_len),
                    ArrayArg::from_raw_parts(values.value.clone(), values.n_entries.max(1)),
                    ArrayArg::from_raw_parts(d_cut_ptrs.clone(), cuts.cut_ptrs.len()),
                    ArrayArg::from_raw_parts(d_cut_values.clone(), cuts.cut_values.len()),
                    ArrayArg::from_raw_parts(out.clone(), packed_len),
                    n_cells as u32,
                    n_chunks as u32,
                                        n_features as u32,
                    null_value,
                    bits,
                    dense_input,
                );
            }
            out
        };
        phases.mark("bin+pack row-major");
        let packed_t = client.empty(packed_len * size_of::<u32>());
        let chunks_per_feature = n_chunks.div_ceil(n_features.max(1));
        {
            let (count, dim) =
                super::launch::elementwise(client, n_features.max(1) * chunks_per_feature);
            // SAFETY: the chunk a unit decodes to is guarded against
            // `n_chunks`, cells against `n_cells`, and every row-major entry
            // read is below `n_cells` (its two-word read has the pad).
            unsafe {
                transpose_packed_kernel::launch_unchecked::<R>(
                    client,
                    count,
                    dim,
                    ArrayArg::from_raw_parts(packed.clone(), packed_len),
                    ArrayArg::from_raw_parts(packed_t.clone(), packed_len),
                    n_cells as u32,
                    n_chunks as u32,
                    n_rows as u32,
                    n_features as u32,
                    chunks_per_feature as u32,
                    bits,
                );
            }
        }
        phases.mark("transpose");
        phases.report_as("device binning");

        let ell = Self {
            gidx: packed,
            gidx_t: packed_t,
            cut_ptrs: d_cut_ptrs,
            gidx_len: packed_len,
            gidx_t_len: packed_len,
            bits,
            n_cuts: cuts.cut_ptrs.len(),
            n_rows,
            row_stride: n_features as u32,
            base_rowid: 0,
            null_value,
            n_bins: shape.n_bins(),
            dense: dense_input,
            compressed: true,
        };
        (ell, shape)
    }
}
