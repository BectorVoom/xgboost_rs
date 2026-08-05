//! Binned feature matrix — the analogue of XGBoost's `GHistIndexMatrix`.
//!
//! Every stored (non-missing) value is replaced by the bin index it falls into,
//! so histogram building is a gather with no float comparisons in the inner
//! loop.
//!
//! # Layout
//!
//! Histogram building is memory bound, so the index is stored in the narrowest
//! integer that fits:
//!
//! * **Dense** matrices store *feature-local* bins — `bin - cut_ptrs[f]` — with
//!   one entry per feature per row. With the default `max_bin = 256` that is a
//!   single byte per value, and the per-feature offsets needed to recover the
//!   global bin are a small array that stays in L1.
//! * **Sparse** matrices store *global* bins, because a row does not visit every
//!   feature and the offset is not implied by position.

use super::{DMatrix, cuts::HistogramCuts};
use crate::Result;
use rayon::prelude::*;

/// Bin indices in the narrowest integer type that fits them.
#[derive(Clone, Debug)]
pub enum BinStorage {
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
}

impl BinStorage {
    /// Build from bin indices that take `n_values` distinct values (`0..n`),
    /// narrowing to the smallest type that holds them.
    fn from_u32(values: Vec<u32>, n_values: usize) -> Self {
        if n_values <= u8::MAX as usize + 1 {
            Self::U8(values.into_iter().map(|v| v as u8).collect())
        } else if n_values <= u16::MAX as usize + 1 {
            Self::U16(values.into_iter().map(|v| v as u16).collect())
        } else {
            Self::U32(values)
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::U8(v) => v.len(),
            Self::U16(v) => v.len(),
            Self::U32(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Bytes per stored entry.
    pub fn width(&self) -> usize {
        match self {
            Self::U8(_) => 1,
            Self::U16(_) => 2,
            Self::U32(_) => 4,
        }
    }

    #[inline]
    pub fn get(&self, i: usize) -> u32 {
        match self {
            Self::U8(v) => v[i] as u32,
            Self::U16(v) => v[i] as u32,
            Self::U32(v) => v[i],
        }
    }
}

/// Run `body` with the concrete slice type of `storage`.
///
/// The inner loops must be monomorphic to vectorise, so every hot path goes
/// through this dispatch rather than [`BinStorage::get`].
macro_rules! dispatch_bins {
    ($storage:expr, |$slice:ident| $body:expr) => {
        match $storage {
            $crate::data::gradient_index::BinStorage::U8($slice) => $body,
            $crate::data::gradient_index::BinStorage::U16($slice) => $body,
            $crate::data::gradient_index::BinStorage::U32($slice) => $body,
        }
    };
}
pub(crate) use dispatch_bins;

/// Column-major bins, used when partitioning rows on a single feature.
///
/// The row-major index is right for histogram building, which walks every
/// feature of a row, but wrong for partitioning, which walks every row of one
/// feature — there it touches a fresh cache line per row. Keeping a transposed
/// copy (as upstream's `ColumnMatrix` does) makes partitioning sequential.
///
/// # Two layouts, chosen per feature
///
/// `sparse_threshold` picks each column's storage, exactly as upstream's
/// `ColumnMatrix::InitStorage` does:
///
/// * a **dense** column holds one entry per row, so a lookup is one index —
///   fastest, but it pays for the rows that have no value;
/// * a **sparse** column holds only the rows that have a value, so a lookup is
///   a binary search over that column's row ids — slower per probe, but a
///   column that is mostly missing costs a fraction of the memory.
///
/// A column is stored sparsely when fewer than `sparse_threshold * n_rows` of
/// its entries are present. The choice changes speed and memory only: both
/// layouts answer every query identically.
#[derive(Clone, Debug)]
pub(crate) struct ColumnIndex {
    /// Concatenated per-feature segments of feature-local bins.
    data: BinStorage,
    /// `data` segment bounds, `n_features + 1` entries.
    data_ptr: Vec<usize>,
    /// Row ids of the stored entries of every sparse column, ascending within
    /// a column. Dense columns contribute an empty segment.
    row_ind: Vec<u32>,
    /// `row_ind` segment bounds, `n_features + 1` entries.
    row_ptr: Vec<usize>,
    /// Whether each column took the sparse layout.
    sparse: Vec<bool>,
    /// Sentinel for an absent value in a dense column; `u32::MAX` when the
    /// matrix has no missing values at all.
    missing: u32,
}

impl ColumnIndex {
    /// Feature-local bin of row `r`, or `None` when the value is missing.
    #[inline]
    pub(crate) fn get(&self, fidx: u32, r: usize) -> Option<u32> {
        let f = fidx as usize;
        let base = self.data_ptr[f];
        if !self.sparse[f] {
            let v = self.data.get(base + r);
            return if v == self.missing { None } else { Some(v) };
        }
        let rows = &self.row_ind[self.row_ptr[f]..self.row_ptr[f + 1]];
        match rows.binary_search(&(r as u32)) {
            Ok(k) => Some(self.data.get(base + k)),
            Err(_) => None,
        }
    }

    /// How many columns took the sparse layout. Speed and memory only, but it
    /// is what `sparse_threshold` decides, so it is worth being able to check.
    pub(crate) fn sparse_columns(&self) -> usize {
        self.sparse.iter().filter(|s| **s).count()
    }

    /// Bytes held by the column index.
    pub(crate) fn size_bytes(&self) -> usize {
        self.data.len() * self.data.width() + self.row_ind.len() * size_of::<u32>()
    }
}

/// Row-major binned index over a [`DMatrix`].
#[derive(Clone, Debug)]
pub struct GHistIndex {
    /// Dense: feature-local bins, `row_stride` per row. Sparse: global bins.
    pub(crate) index: BinStorage,
    /// Row offsets into `index`. Empty for dense matrices, where the offset is
    /// `row * row_stride`.
    pub(crate) row_ptr: Vec<usize>,
    /// Global bin offset of each feature; dense matrices only.
    pub(crate) offsets: Vec<u32>,
    pub cuts: HistogramCuts,
    /// True when every row holds an entry for every feature.
    pub is_dense: bool,
    pub(crate) row_stride: usize,
    /// Transposed view of the same bins, for row partitioning.
    pub(crate) columns: ColumnIndex,
    num_row: usize,
}

impl GHistIndex {
    pub fn num_row(&self) -> usize {
        self.num_row
    }

    pub fn total_bins(&self) -> usize {
        self.cuts.total_bins()
    }

    /// Bytes held by the binned index.
    pub fn size_bytes(&self) -> usize {
        self.index.len() * self.index.width()
    }

    /// Bytes held by the transposed copy the row partitioner reads. This is
    /// what `sparse_threshold` trades against lookup speed.
    pub fn column_size_bytes(&self) -> usize {
        self.columns.size_bytes()
    }

    /// How many columns took the sparse layout.
    pub fn sparse_columns(&self) -> usize {
        self.columns.sparse_columns()
    }

    /// Range of `index` covered by row `r`.
    #[inline]
    pub(crate) fn row_range(&self, r: usize) -> (usize, usize) {
        if self.is_dense {
            (r * self.row_stride, (r + 1) * self.row_stride)
        } else {
            (self.row_ptr[r], self.row_ptr[r + 1])
        }
    }

    /// Global bins of row `r`. Convenience for tests and cold paths.
    pub fn row_global_bins(&self, r: usize) -> Vec<u32> {
        let (b, e) = self.row_range(r);
        (b..e)
            .map(|k| {
                if self.is_dense {
                    self.offsets[k - b] + self.index.get(k)
                } else {
                    self.index.get(k)
                }
            })
            .collect()
    }

}

/// Feature-local bin of one stored value.
///
/// The cuts were built from the very matrix being binned, so every value has a
/// bin: a numerical one always falls in some quantile bucket, and a category
/// code is at most the largest one the cut builder saw. An unseen category
/// could only come from cuts built elsewhere, and it maps to the last bin
/// rather than out of the histogram.
#[inline]
fn local_bin(cuts: &HistogramCuts, value: f32, fidx: usize) -> u32 {
    if cuts.is_cat(fidx) {
        let n_bins = cuts.feature_bins(fidx) as u32;
        debug_assert!(
            cuts.search_cat_bin(value, fidx).is_some(),
            "category {value} of feature {fidx} has no bin in cuts built from this matrix"
        );
        let code = if crate::tree::cat::invalid_cat(value) {
            0
        } else {
            crate::tree::cat::as_cat(value)
        };
        code.min(n_bins.saturating_sub(1))
    } else {
        cuts.search_bin(value, fidx) - cuts.cut_ptrs[fidx]
    }
}

/// XGBoost's `sparse_threshold` default, used by the callers that have no
/// training parameters to hand.
pub const DEFAULT_SPARSE_THRESHOLD: f64 = 0.2;

/// Bin every entry of `dmat` against `cuts`, with the default column layout
/// rule.
pub fn build_gradient_index(dmat: &DMatrix, cuts: &HistogramCuts) -> Result<GHistIndex> {
    build_gradient_index_with(dmat, cuts, DEFAULT_SPARSE_THRESHOLD)
}

/// Bin every entry of `dmat` against `cuts`.
///
/// Rows are independent, so the work is split across threads by row block. The
/// output depends only on the data, never on how it was blocked.
///
/// `sparse_threshold` selects each column's storage layout in the transposed
/// copy; see [`ColumnIndex`]. It changes memory and speed, never the model.
pub fn build_gradient_index_with(
    dmat: &DMatrix,
    cuts: &HistogramCuts,
    sparse_threshold: f64,
) -> Result<GHistIndex> {
    let is_dense = dmat.is_dense();
    let num_row = dmat.num_row();
    let num_col = dmat.num_col();

    // Sparse rows keep the source matrix's row offsets; dense rows are implied
    // by the stride.
    let row_ptr = if is_dense { Vec::new() } else { dmat.row_ptr.clone() };
    let offsets: Vec<u32> = cuts.cut_ptrs[..num_col].to_vec();

    // Bin once, row-major and feature-local. Both the row-major index and its
    // transpose are derived from this, so no value is searched for twice.
    let mut local = vec![0u32; dmat.num_nonzero()];
    /// Rows per binning task.
    const ROW_BLOCK: usize = 8192;
    crate::threading::install(|| {
        if is_dense {
            local
                .par_chunks_mut(ROW_BLOCK * num_col)
                .enumerate()
                .for_each(|(block, out)| {
                    let first_row = block * ROW_BLOCK;
                    for (i, slot) in out.iter_mut().enumerate() {
                        let (r, c) = (first_row + i / num_col, i % num_col);
                        *slot = local_bin(cuts, dmat.value[r * num_col + c], c);
                    }
                });
        } else {
            // Split on row boundaries so each task owns whole rows.
            let n_blocks = num_row.div_ceil(ROW_BLOCK);
            let mut rest: &mut [u32] = &mut local;
            let mut jobs: Vec<(usize, &mut [u32])> = Vec::with_capacity(n_blocks);
            for b in 0..n_blocks {
                let lo = dmat.row_ptr[b * ROW_BLOCK];
                let hi = dmat.row_ptr[((b + 1) * ROW_BLOCK).min(num_row)];
                let (here, tail) = rest.split_at_mut(hi - lo);
                rest = tail;
                jobs.push((b, here));
            }
            jobs.into_par_iter().for_each(|(b, out)| {
                let first_row = b * ROW_BLOCK;
                let base = dmat.row_ptr[first_row];
                for r in first_row..((b + 1) * ROW_BLOCK).min(num_row) {
                    let (idx, val) = dmat.row(r);
                    let offset = dmat.row_ptr[r] - base;
                    for (k, (&c, &v)) in idx.iter().zip(val).enumerate() {
                        out[offset + k] = local_bin(cuts, v, c as usize);
                    }
                }
            });
        }
    });

    let columns = build_column_index(dmat, cuts, &local, is_dense, sparse_threshold);

    let max_feature_bins = (0..num_col).map(|f| cuts.feature_bins(f)).max().unwrap_or(0);
    let (index, offsets, row_stride) = if is_dense {
        (BinStorage::from_u32(local, max_feature_bins), offsets, num_col)
    } else {
        // Sparse rows do not imply their feature from position, so the
        // row-major index holds global bins.
        let mut global = local;
        crate::threading::install(|| {
            global.par_iter_mut().enumerate().for_each(|(k, bin)| {
                *bin += offsets[dmat.index[k] as usize];
            });
        });
        (BinStorage::from_u32(global, cuts.total_bins()), Vec::new(), 0)
    };

    Ok(GHistIndex {
        index,
        row_ptr,
        offsets,
        cuts: cuts.clone(),
        is_dense,
        row_stride,
        columns,
        num_row,
    })
}

/// Transpose the feature-local bins into column-major order for partitioning,
/// giving each column the layout `sparse_threshold` selects.
fn build_column_index(
    dmat: &DMatrix,
    cuts: &HistogramCuts,
    local: &[u32],
    is_dense: bool,
    sparse_threshold: f64,
) -> ColumnIndex {
    let n_rows = dmat.num_row();
    let n_features = dmat.num_col();
    let counts = dmat.column_sizes();
    let max_feature_bins = (0..n_features).map(|f| cuts.feature_bins(f)).max().unwrap_or(0);
    // A dense column needs one extra value to mark an absent entry; a matrix
    // with no missing values needs none.
    let (missing, n_values) = if is_dense {
        (u32::MAX, max_feature_bins)
    } else {
        (max_feature_bins as u32, max_feature_bins + 1)
    };

    // `ColumnMatrix::InitStorage`: a column with fewer stored values than
    // `sparse_threshold * n_rows` is worth the binary search.
    let limit = sparse_threshold * n_rows as f64;
    let sparse: Vec<bool> =
        (0..n_features).map(|f| (counts[f] as f64) < limit).collect();

    let mut data_ptr = Vec::with_capacity(n_features + 1);
    let mut row_ptr = Vec::with_capacity(n_features + 1);
    let (mut d_acc, mut r_acc) = (0usize, 0usize);
    for f in 0..n_features {
        data_ptr.push(d_acc);
        row_ptr.push(r_acc);
        if sparse[f] {
            d_acc += counts[f];
            r_acc += counts[f];
        } else {
            d_acc += n_rows;
        }
    }
    data_ptr.push(d_acc);
    row_ptr.push(r_acc);

    let mut data = vec![missing; d_acc];
    let mut row_ind = vec![0u32; r_acc];

    // Each task owns whole columns, so its output slices are exclusive. Groups
    // are wide enough that the strided reads still use most of a cache line.
    let group = n_features
        .div_ceil(crate::threading::num_threads() * 2)
        .max(16 / size_of::<u32>())
        .min(n_features.max(1));
    let n_groups = n_features.div_ceil(group.max(1));

    // Carve the two output buffers into one disjoint slice per group.
    let mut jobs: Vec<(usize, usize, &mut [u32], &mut [u32])> = Vec::with_capacity(n_groups);
    {
        let mut data_rest: &mut [u32] = &mut data;
        let mut rows_rest: &mut [u32] = &mut row_ind;
        for g in 0..n_groups {
            let first = g * group;
            let last = (first + group).min(n_features);
            let (d_here, d_tail) = data_rest.split_at_mut(data_ptr[last] - data_ptr[first]);
            let (r_here, r_tail) = rows_rest.split_at_mut(row_ptr[last] - row_ptr[first]);
            data_rest = d_tail;
            rows_rest = r_tail;
            jobs.push((first, last, d_here, r_here));
        }
    }

    crate::threading::install(|| {
        jobs.into_par_iter().for_each(|(first, last, d_out, r_out)| {
            // Where each column of this group starts inside the group's slices.
            let d_base = data_ptr[first];
            let r_base = row_ptr[first];
            // Fill position of each sparse column, so its row ids stay
            // ascending.
            let mut fill = vec![0usize; last - first];

            let mut place = |col: usize, r: usize, v: u32, fill: &mut [usize]| {
                let j = col - first;
                if sparse[col] {
                    let k = fill[j];
                    fill[j] = k + 1;
                    d_out[data_ptr[col] - d_base + k] = v;
                    r_out[row_ptr[col] - r_base + k] = r as u32;
                } else {
                    d_out[data_ptr[col] - d_base + r] = v;
                }
            };

            if is_dense {
                for r in 0..n_rows {
                    for col in first..last {
                        place(col, r, local[r * n_features + col], &mut fill);
                    }
                }
            } else {
                for r in 0..n_rows {
                    let (b, e) = (dmat.row_ptr[r], dmat.row_ptr[r + 1]);
                    for k in b..e {
                        let col = dmat.index[k] as usize;
                        if col >= first && col < last {
                            place(col, r, local[k], &mut fill);
                        }
                    }
                }
            }
        });
    });

    ColumnIndex {
        data: BinStorage::from_u32(data, n_values),
        data_ptr,
        row_ind,
        row_ptr,
        sparse,
        missing,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::cuts::build_cuts;

    #[test]
    fn bins_are_monotone_in_value() {
        let vals: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let d = DMatrix::from_dense(&vals, 64, 1, f32::NAN).unwrap();
        let cuts = build_cuts(&d, 16).unwrap();
        let gi = build_gradient_index(&d, &cuts).unwrap();
        assert!(gi.is_dense);
        let bins: Vec<u32> = (0..64).map(|r| gi.row_global_bins(r)[0]).collect();
        assert!(bins.windows(2).all(|w| w[0] <= w[1]), "bins must not decrease with value");
        assert_eq!(*bins.first().unwrap(), 0);
        assert_eq!(*bins.last().unwrap() as usize, cuts.feature_bins(0) - 1);
    }

    #[test]
    fn dense_index_is_narrowed_to_one_byte() {
        let vals: Vec<f32> = (0..600).map(|i| i as f32).collect();
        let d = DMatrix::from_dense(&vals, 200, 3, f32::NAN).unwrap();
        let cuts = build_cuts(&d, 256).unwrap();
        let gi = build_gradient_index(&d, &cuts).unwrap();
        assert_eq!(gi.index.width(), 1, "200 bins per feature fit in a byte");
        assert_eq!(gi.size_bytes(), 600);
    }

    #[test]
    fn sparse_rows_keep_their_entry_count() {
        let d = DMatrix::from_dense(&[1.0, f32::NAN, 3.0, 4.0], 2, 2, f32::NAN).unwrap();
        let cuts = build_cuts(&d, 8).unwrap();
        let gi = build_gradient_index(&d, &cuts).unwrap();
        assert!(!gi.is_dense);
        assert_eq!(gi.row_global_bins(0).len(), 1);
        assert_eq!(gi.row_global_bins(1).len(), 2);
    }

    #[test]
    fn column_index_reports_missing_values() {
        let d = DMatrix::from_dense(&[1.0, f32::NAN, 3.0, 4.0], 2, 2, f32::NAN).unwrap();
        let cuts = build_cuts(&d, 8).unwrap();
        let gi = build_gradient_index(&d, &cuts).unwrap();
        assert!(gi.columns.get(0, 0).is_some());
        assert_eq!(gi.columns.get(1, 0), None, "row 0 has no feature 1");
        assert!(gi.columns.get(1, 1).is_some());
    }

    /// Both column layouts must answer every query the same way; only their
    /// memory and lookup cost differ.
    #[test]
    fn the_two_column_layouts_agree_on_every_lookup() {
        let (rows, cols) = (400usize, 6usize);
        // Columns of very different density, so a middling threshold splits
        // them between the two layouts.
        let x: Vec<f32> = (0..rows * cols)
            .map(|i| {
                let (r, c) = (i / cols, i % cols);
                if (r * (c + 1)) % (c + 2) == 0 { ((i * 37) % 97) as f32 } else { f32::NAN }
            })
            .collect();
        let d = DMatrix::from_dense(&x, rows, cols, f32::NAN).unwrap();
        let cuts = build_cuts(&d, 32).unwrap();

        // 0 forces every column dense, 1 forces every partly-missing column
        // sparse, and the default splits them.
        let all_dense = build_gradient_index_with(&d, &cuts, 0.0).unwrap();
        let all_sparse = build_gradient_index_with(&d, &cuts, 1.0).unwrap();
        let mixed = build_gradient_index_with(&d, &cuts, 0.5).unwrap();

        assert_eq!(all_dense.sparse_columns(), 0, "threshold 0 keeps every column dense");
        assert!(all_sparse.sparse_columns() > 0, "threshold 1 stores sparse columns sparsely");
        assert!(
            mixed.sparse_columns() > 0 && mixed.sparse_columns() < cols,
            "0.5 should split the columns between the layouts, got {}",
            mixed.sparse_columns()
        );

        for f in 0..cols as u32 {
            for r in 0..rows {
                let expected = all_dense.columns.get(f, r);
                assert_eq!(all_sparse.columns.get(f, r), expected, "feature {f} row {r}");
                assert_eq!(mixed.columns.get(f, r), expected, "feature {f} row {r}");
            }
        }
    }

    /// The sparse layout exists to save memory, so it must actually do so.
    #[test]
    fn a_sparse_column_layout_is_smaller() {
        let (rows, cols) = (500usize, 4usize);
        // Every column is mostly missing.
        let x: Vec<f32> = (0..rows * cols)
            .map(|i| if i % 10 == 0 { (i % 50) as f32 } else { f32::NAN })
            .collect();
        let d = DMatrix::from_dense(&x, rows, cols, f32::NAN).unwrap();
        let cuts = build_cuts(&d, 32).unwrap();

        let dense = build_gradient_index_with(&d, &cuts, 0.0).unwrap();
        let sparse = build_gradient_index_with(&d, &cuts, 1.0).unwrap();
        assert!(
            sparse.column_size_bytes() < dense.column_size_bytes(),
            "sparse columns should cost less: {} vs {}",
            sparse.column_size_bytes(),
            dense.column_size_bytes()
        );
    }

    /// A fully populated column is never worth storing sparsely, whatever the
    /// threshold: the rule compares against `sparse_threshold * n_rows`, and a
    /// full column is never below that for a threshold of at most 1.
    #[test]
    fn a_full_column_always_stays_dense() {
        let vals: Vec<f32> = (0..300).map(|i| (i % 17) as f32).collect();
        let d = DMatrix::from_dense(&vals, 100, 3, f32::NAN).unwrap();
        let cuts = build_cuts(&d, 16).unwrap();
        for threshold in [0.0f64, 0.2, 0.5, 1.0] {
            let gi = build_gradient_index_with(&d, &cuts, threshold).unwrap();
            assert_eq!(gi.sparse_columns(), 0, "threshold {threshold}");
        }
    }

    #[test]
    fn column_index_agrees_with_the_row_major_index() {
        let vals: Vec<f32> = (0..300).map(|i| ((i * 17) % 23) as f32).collect();
        let d = DMatrix::from_dense(&vals, 100, 3, f32::NAN).unwrap();
        let cuts = build_cuts(&d, 8).unwrap();
        let gi = build_gradient_index(&d, &cuts).unwrap();
        for r in 0..d.num_row() {
            for (f, global) in gi.row_global_bins(r).into_iter().enumerate() {
                let local = gi.columns.get(f as u32, r).expect("dense: never missing");
                assert_eq!(local + cuts.cut_ptrs[f], global, "row {r} feature {f}");
            }
        }
    }
}
