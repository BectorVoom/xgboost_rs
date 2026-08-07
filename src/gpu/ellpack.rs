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

/// An [`EllpackMatrix`] resident on device.
///
/// Both the histogram kernel and the row partitioner read the same `gidx`, so
/// it is uploaded once and shared rather than copied per consumer — it is by
/// far the largest buffer a fit holds (`4 * n_rows * row_stride` bytes).
#[derive(Clone, Debug)]
pub struct DeviceEllpack {
    pub gidx: cubecl::server::Handle,
    pub cut_ptrs: cubecl::server::Handle,
    pub gidx_len: usize,
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
    /// Upload `matrix` to the device.
    pub fn upload<R: cubecl::prelude::Runtime>(
        client: &cubecl::prelude::ComputeClient<R>,
        matrix: &EllpackMatrix,
    ) -> Self {
        Self {
            gidx: client.create_from_slice(bytemuck::cast_slice(&matrix.gidx)),
            cut_ptrs: client.create_from_slice(bytemuck::cast_slice(&matrix.cut_ptrs)),
            gidx_len: matrix.gidx.len(),
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
