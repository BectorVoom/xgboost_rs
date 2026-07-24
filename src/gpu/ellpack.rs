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
