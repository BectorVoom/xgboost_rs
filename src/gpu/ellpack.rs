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

use rayon::prelude::*;

use crate::data::cuts::HistogramCuts;

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
    let layout = if gidx.iter().any(|&v| v == null_value) {
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
