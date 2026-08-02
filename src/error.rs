//! Library error type, following the `thiserror` pattern: the library defines
//! a structured error enum; binaries wrap it with `anyhow` for context.

/// Errors reported by the GPU kernel front-ends and the parameter layer.
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

    /// A single parameter is out of range, or a combination of parameters is
    /// rejected by XGBoost. `name` is the XGBoost parameter name so the message
    /// is greppable against upstream docs.
    #[error("invalid parameter `{name}`: {reason}")]
    InvalidParameter { name: &'static str, reason: String },

    /// A string could not be parsed into a typed parameter value.
    #[error("cannot parse `{value}` as {name}: {reason}")]
    ParseParameter { name: &'static str, value: String, reason: String },
}

impl Error {
    /// Build an [`Error::InvalidParameter`].
    pub(crate) fn invalid(name: &'static str, reason: impl Into<String>) -> Self {
        Self::InvalidParameter { name, reason: reason.into() }
    }

    /// Build an [`Error::ParseParameter`].
    pub(crate) fn parse(
        name: &'static str,
        value: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        Self::ParseParameter { name, value: value.into(), reason: reason.into() }
    }
}

pub type Result<T> = core::result::Result<T, Error>;
