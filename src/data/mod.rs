//! Training data: [`DMatrix`] and its [`MetaInfo`].
//!
//! The layout mirrors XGBoost's `SparsePage`: rows are stored CSR-style, with
//! *missing* values simply absent. Dense input is the special case where every
//! row holds an entry for every feature; [`DMatrix::is_dense`] reports it, and
//! the hist path takes a faster route when it holds.

pub mod cuts;
pub mod gradient_index;

use crate::{Error, Result};

/// Labels and per-row metadata attached to a [`DMatrix`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MetaInfo {
    pub num_row: usize,
    pub num_col: usize,
    pub labels: Vec<f32>,
    pub weights: Option<Vec<f32>>,
    pub base_margin: Option<Vec<f32>>,
}

impl MetaInfo {
    /// Weight of row `i`, or `1.0` when the matrix is unweighted.
    #[inline]
    pub fn weight(&self, i: usize) -> f32 {
        match &self.weights {
            Some(w) => w[i],
            None => 1.0,
        }
    }
}

/// A CSR feature matrix plus its [`MetaInfo`].
#[derive(Clone, Debug)]
pub struct DMatrix {
    /// Row offsets into `index`/`value`, length `num_row + 1`.
    pub(crate) row_ptr: Vec<usize>,
    /// Column index of each stored entry.
    pub(crate) index: Vec<u32>,
    /// Value of each stored entry.
    pub(crate) value: Vec<f32>,
    pub(crate) info: MetaInfo,
}

impl DMatrix {
    /// Build from a row-major dense buffer, dropping entries equal to
    /// `missing` (use `f32::NAN` for the XGBoost default).
    pub fn from_dense(data: &[f32], nrow: usize, ncol: usize, missing: f32) -> Result<Self> {
        if data.len() != nrow * ncol {
            return Err(Error::DataShape { expected: nrow * ncol, got: data.len() });
        }
        let mut row_ptr = Vec::with_capacity(nrow + 1);
        let mut index = Vec::with_capacity(data.len());
        let mut value = Vec::with_capacity(data.len());
        row_ptr.push(0);
        for r in 0..nrow {
            for c in 0..ncol {
                let v = data[r * ncol + c];
                if !is_missing(v, missing) {
                    index.push(c as u32);
                    value.push(v);
                }
            }
            row_ptr.push(index.len());
        }
        Ok(Self {
            row_ptr,
            index,
            value,
            info: MetaInfo { num_row: nrow, num_col: ncol, ..Default::default() },
        })
    }

    /// Build from CSR arrays. `row_ptr` has length `nrow + 1`.
    pub fn from_csr(
        row_ptr: &[usize],
        index: &[u32],
        value: &[f32],
        ncol: usize,
        missing: f32,
    ) -> Result<Self> {
        if row_ptr.is_empty() {
            return Err(Error::DataShape { expected: 1, got: 0 });
        }
        if index.len() != value.len() {
            return Err(Error::DataShape { expected: index.len(), got: value.len() });
        }
        let nrow = row_ptr.len() - 1;
        let mut out_ptr = Vec::with_capacity(nrow + 1);
        let mut out_idx = Vec::with_capacity(index.len());
        let mut out_val = Vec::with_capacity(value.len());
        out_ptr.push(0);
        for r in 0..nrow {
            for k in row_ptr[r]..row_ptr[r + 1] {
                let v = value[k];
                if !is_missing(v, missing) {
                    if index[k] as usize >= ncol {
                        return Err(Error::FeatureIndex { index: index[k] as usize, num_col: ncol });
                    }
                    out_idx.push(index[k]);
                    out_val.push(v);
                }
            }
            out_ptr.push(out_idx.len());
        }
        Ok(Self {
            row_ptr: out_ptr,
            index: out_idx,
            value: out_val,
            info: MetaInfo { num_row: nrow, num_col: ncol, ..Default::default() },
        })
    }

    /// Parse a LIBSVM file (the agaricus format): `label idx:value ...`.
    pub fn from_libsvm(path: &std::path::Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| Error::Io { path: path.display().to_string(), source: e })?;
        Self::from_libsvm_str(&text)
    }

    /// Parse LIBSVM text. Column count is inferred as `max_index + 1`.
    pub fn from_libsvm_str(text: &str) -> Result<Self> {
        let mut row_ptr = vec![0usize];
        let mut index = Vec::new();
        let mut value = Vec::new();
        let mut labels = Vec::new();
        let mut ncol = 0usize;

        for (lineno, line) in text.lines().enumerate() {
            let mut parts = line.split_ascii_whitespace();
            let Some(label) = parts.next() else { continue };
            labels.push(label.parse::<f32>().map_err(|_| Error::LibsvmParse {
                line: lineno + 1,
                token: label.to_string(),
            })?);
            for tok in parts {
                let (k, v) = tok.split_once(':').ok_or_else(|| Error::LibsvmParse {
                    line: lineno + 1,
                    token: tok.to_string(),
                })?;
                let k: usize = k.parse().map_err(|_| Error::LibsvmParse {
                    line: lineno + 1,
                    token: tok.to_string(),
                })?;
                let v: f32 = v.parse().map_err(|_| Error::LibsvmParse {
                    line: lineno + 1,
                    token: tok.to_string(),
                })?;
                index.push(k as u32);
                value.push(v);
                ncol = ncol.max(k + 1);
            }
            row_ptr.push(index.len());
        }

        let num_row = labels.len();
        Ok(Self {
            row_ptr,
            index,
            value,
            info: MetaInfo { num_row, num_col: ncol, labels, ..Default::default() },
        })
    }

    pub fn set_labels(&mut self, y: &[f32]) -> Result<()> {
        if y.len() != self.info.num_row {
            return Err(Error::DataShape { expected: self.info.num_row, got: y.len() });
        }
        self.info.labels = y.to_vec();
        Ok(())
    }

    pub fn set_weights(&mut self, w: &[f32]) -> Result<()> {
        if w.len() != self.info.num_row {
            return Err(Error::DataShape { expected: self.info.num_row, got: w.len() });
        }
        self.info.weights = Some(w.to_vec());
        Ok(())
    }

    pub fn set_base_margin(&mut self, m: &[f32]) -> Result<()> {
        if m.len() != self.info.num_row {
            return Err(Error::DataShape { expected: self.info.num_row, got: m.len() });
        }
        self.info.base_margin = Some(m.to_vec());
        Ok(())
    }

    pub fn num_row(&self) -> usize {
        self.info.num_row
    }

    pub fn num_col(&self) -> usize {
        self.info.num_col
    }

    pub fn info(&self) -> &MetaInfo {
        &self.info
    }

    pub fn num_nonzero(&self) -> usize {
        self.index.len()
    }

    /// True when every row stores every feature — no missing values.
    pub fn is_dense(&self) -> bool {
        self.index.len() == self.info.num_row * self.info.num_col
    }

    /// The `(column, value)` entries of row `i`, in stored order.
    #[inline]
    pub fn row(&self, i: usize) -> (&[u32], &[f32]) {
        let (b, e) = (self.row_ptr[i], self.row_ptr[i + 1]);
        (&self.index[b..e], &self.value[b..e])
    }

    /// Number of stored (non-missing) entries per column.
    pub fn column_sizes(&self) -> Vec<usize> {
        let mut sizes = vec![0usize; self.info.num_col];
        for &c in &self.index {
            sizes[c as usize] += 1;
        }
        sizes
    }
}

#[inline]
fn is_missing(v: f32, missing: f32) -> bool {
    // NaN never compares equal, so it needs the explicit check that XGBoost's
    // `IsValid` predicate also performs.
    v.is_nan() || v == missing
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_drops_missing_and_reports_density() {
        let d = DMatrix::from_dense(&[1.0, 2.0, f32::NAN, 4.0], 2, 2, f32::NAN).unwrap();
        assert_eq!(d.num_row(), 2);
        assert_eq!(d.num_col(), 2);
        assert!(!d.is_dense());
        assert_eq!(d.row(0), ([0u32, 1].as_slice(), [1.0f32, 2.0].as_slice()));
        assert_eq!(d.row(1), ([1u32].as_slice(), [4.0f32].as_slice()));
        assert_eq!(d.column_sizes(), vec![1, 2]);
    }

    #[test]
    fn dense_without_missing_is_dense() {
        let d = DMatrix::from_dense(&[1.0, 0.0, 3.0, 4.0], 2, 2, f32::NAN).unwrap();
        assert!(d.is_dense(), "explicit zeros are values, not missing");
    }

    #[test]
    fn libsvm_parses_labels_and_infers_columns() {
        let d = DMatrix::from_libsvm_str("1 0:1 3:2.5\n0 1:7\n").unwrap();
        assert_eq!(d.num_row(), 2);
        assert_eq!(d.num_col(), 4);
        assert_eq!(d.info().labels, vec![1.0, 0.0]);
        assert_eq!(d.row(1), ([1u32].as_slice(), [7.0f32].as_slice()));
    }

    #[test]
    fn shape_mismatch_is_an_error() {
        assert!(DMatrix::from_dense(&[1.0, 2.0], 2, 2, f32::NAN).is_err());
    }
}
