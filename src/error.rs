//! Library error type, following the `thiserror` pattern: the library defines
//! a structured error enum; binaries wrap it with `anyhow` for context.

/// Errors reported by the GPU kernel front-ends.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("ELLPACK shape mismatch: gidx has {got} entries, expected n_rows * row_stride = {expected}")]
    MatrixShape { expected: usize, got: usize },

    #[error("cut_ptrs must contain at least 2 entries (n_features + 1), got {got}")]
    InvalidCuts { got: usize },

    #[error("gradient pair count {got} does not match matrix rows {expected}")]
    GpairCount { expected: usize, got: usize },

    #[error("histogram length mismatch: parent has {parent} bins, built has {built}")]
    HistogramLen { parent: usize, built: usize },

    #[error("histogram buffer holds {got} bins, engine expects {expected}")]
    HistogramBins { expected: usize, got: usize },

    #[error("device synchronisation failed: {0}")]
    Sync(String),
}

pub type Result<T> = core::result::Result<T, Error>;
