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
#[derive(Clone, Debug)]
pub(crate) struct ColumnIndex {
    /// `data[f * n_rows + r]` is the feature-local bin, or [`Self::missing`].
    pub(crate) data: BinStorage,
    /// Sentinel for an absent value; `u32::MAX` for dense matrices, which have
    /// none.
    pub(crate) missing: u32,
    pub(crate) n_rows: usize,
}

impl ColumnIndex {
    /// Feature-local bin of row `r`, or `None` when the value is missing.
    #[inline]
    pub(crate) fn get(&self, fidx: u32, r: usize) -> Option<u32> {
        let v = self.data.get(fidx as usize * self.n_rows + r);
        if v == self.missing { None } else { Some(v) }
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

/// Bin every entry of `dmat` against `cuts`.
///
/// Rows are independent, so the work is split across threads by row block. The
/// output depends only on the data, never on how it was blocked.
pub fn build_gradient_index(dmat: &DMatrix, cuts: &HistogramCuts) -> Result<GHistIndex> {
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
                        *slot = cuts.search_bin(dmat.value[r * num_col + c], c) - offsets[c];
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
                        out[offset + k] = cuts.search_bin(v, c as usize) - offsets[c as usize];
                    }
                }
            });
        }
    });

    let columns = build_column_index(dmat, cuts, &local, is_dense);

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

/// Transpose the feature-local bins into column-major order for partitioning.
fn build_column_index(
    dmat: &DMatrix,
    cuts: &HistogramCuts,
    local: &[u32],
    is_dense: bool,
) -> ColumnIndex {
    let n_rows = dmat.num_row();
    let n_features = dmat.num_col();
    let max_feature_bins = (0..n_features).map(|f| cuts.feature_bins(f)).max().unwrap_or(0);
    // Sparse columns need one extra value to mark an absent entry.
    let (missing, n_values) = if is_dense {
        (u32::MAX, max_feature_bins)
    } else {
        (max_feature_bins as u32, max_feature_bins + 1)
    };

    let mut data = vec![if is_dense { 0 } else { missing }; n_rows * n_features];

    // Each task owns whole columns, so its output slice is exclusive. Groups
    // are wide enough that the strided reads still use most of a cache line.
    let group = n_features
        .div_ceil(crate::threading::num_threads() * 2)
        .max(16 / size_of::<u32>())
        .min(n_features.max(1));
    crate::threading::install(|| {
        data.par_chunks_mut(n_rows * group).enumerate().for_each(|(c, out)| {
            let first = c * group;
            let n_here = out.len() / n_rows;
            if is_dense {
                for r in 0..n_rows {
                    let row = &local[r * n_features + first..r * n_features + first + n_here];
                    for (j, &v) in row.iter().enumerate() {
                        out[j * n_rows + r] = v;
                    }
                }
            } else {
                let last = first + n_here;
                for r in 0..n_rows {
                    let (b, e) = (dmat.row_ptr[r], dmat.row_ptr[r + 1]);
                    for k in b..e {
                        let col = dmat.index[k] as usize;
                        if col >= first && col < last {
                            out[(col - first) * n_rows + r] = local[k];
                        }
                    }
                }
            }
        });
    });

    ColumnIndex { data: BinStorage::from_u32(data, n_values), missing, n_rows }
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
