//! Column-major views of a [`DMatrix`] — XGBoost's `CSCPage` and
//! `SortedCSCPage`.
//!
//! The CSR layout a [`DMatrix`] stores is right for anything that walks a row
//! at a time, and wrong for the two algorithms that walk a *column* at a time:
//! the `gblinear` coordinate solvers, which update one feature's weight from
//! that feature's whole column, and the `exact` tree updater, which enumerates
//! split candidates by scanning a column in value order.
//!
//! Both get their column view from here. They differ in one thing — whether a
//! column is ordered by row or by value — which is what [`CscPage::build`] and
//! [`CscPage::build_sorted`] name.
//!
//! # Row batches
//!
//! A [`CscPages`] is a *sequence* of pages, each covering a contiguous range of
//! rows. That is XGBoost's `GetBatches<CSCPage>`, and it is what
//! `max_row_perbatch` controls: the `shotgun` updater sweeps every feature once
//! per page, so the batch size changes the model, not just the memory profile.

use std::ops::Range;

use super::DMatrix;

/// One column-major page covering a contiguous range of rows.
///
/// Row indices are absolute (they index the source matrix), so a caller never
/// has to know which page an entry came from.
#[derive(Clone, Debug)]
pub struct CscPage {
    /// Offsets into `row_idx`/`value`, length `num_col + 1`.
    col_ptr: Vec<usize>,
    row_idx: Vec<u32>,
    value: Vec<f32>,
    num_col: usize,
}

impl CscPage {
    /// Transpose `rows` of `dmat`, leaving each column in ascending row order.
    pub fn build(dmat: &DMatrix, rows: Range<usize>) -> Self {
        let num_col = dmat.num_col();
        let mut col_ptr = vec![0usize; num_col + 1];
        for r in rows.clone() {
            let (idx, _) = dmat.row(r);
            for &c in idx {
                col_ptr[c as usize + 1] += 1;
            }
        }
        for c in 0..num_col {
            col_ptr[c + 1] += col_ptr[c];
        }

        let nnz = col_ptr[num_col];
        let mut row_idx = vec![0u32; nnz];
        let mut value = vec![0.0f32; nnz];
        // `fill` walks the rows in order, so each column comes out ascending in
        // row index without a sort.
        let mut fill = col_ptr.clone();
        for r in rows {
            let (idx, val) = dmat.row(r);
            for (&c, &v) in idx.iter().zip(val) {
                let slot = fill[c as usize];
                row_idx[slot] = r as u32;
                value[slot] = v;
                fill[c as usize] += 1;
            }
        }
        Self { col_ptr, row_idx, value, num_col }
    }

    /// The same transpose, with each column in ascending *value* order.
    ///
    /// Equal values keep their row order, which is what makes the `exact`
    /// updater's split enumeration reproducible: upstream's `std::sort` leaves
    /// ties unordered, and an unstable tie here would move a split point.
    pub fn build_sorted(dmat: &DMatrix, rows: Range<usize>) -> Self {
        let mut page = Self::build(dmat, rows);
        for c in 0..page.num_col {
            let (b, e) = (page.col_ptr[c], page.col_ptr[c + 1]);
            let mut order: Vec<usize> = (b..e).collect();
            // Stable in the index, so equal values stay in ascending row order.
            order.sort_by(|&i, &j| {
                page.value[i].partial_cmp(&page.value[j]).unwrap_or(std::cmp::Ordering::Equal)
            });
            let rows: Vec<u32> = order.iter().map(|&i| page.row_idx[i]).collect();
            let values: Vec<f32> = order.iter().map(|&i| page.value[i]).collect();
            page.row_idx[b..e].copy_from_slice(&rows);
            page.value[b..e].copy_from_slice(&values);
        }
        page
    }

    /// Column `fidx` as `(row indices, values)`.
    #[inline]
    pub fn column(&self, fidx: usize) -> (&[u32], &[f32]) {
        let (b, e) = (self.col_ptr[fidx], self.col_ptr[fidx + 1]);
        (&self.row_idx[b..e], &self.value[b..e])
    }

    pub fn num_col(&self) -> usize {
        self.num_col
    }

    /// Stored entries in column `fidx`.
    #[inline]
    pub fn column_len(&self, fidx: usize) -> usize {
        self.col_ptr[fidx + 1] - self.col_ptr[fidx]
    }
}

/// Every column-major page of a matrix, in row order.
///
/// A single page unless `max_row_perbatch` asks for smaller batches.
#[derive(Clone, Debug)]
pub struct CscPages {
    pages: Vec<CscPage>,
}

impl CscPages {
    /// Transpose `dmat` into pages of at most `rows_per_batch` rows each.
    /// `None` produces one page covering everything.
    pub fn build(dmat: &DMatrix, rows_per_batch: Option<usize>, sorted: bool) -> Self {
        let n = dmat.num_row();
        let batch = rows_per_batch.unwrap_or(usize::MAX).max(1).min(n.max(1));
        let build = if sorted { CscPage::build_sorted } else { CscPage::build };
        let mut pages = Vec::new();
        let mut start = 0usize;
        while start < n {
            let end = (start + batch).min(n);
            pages.push(build(dmat, start..end));
            start = end;
        }
        if pages.is_empty() {
            pages.push(build(dmat, 0..0));
        }
        Self { pages }
    }

    pub fn iter(&self) -> std::slice::Iter<'_, CscPage> {
        self.pages.iter()
    }

    pub fn len(&self) -> usize {
        self.pages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }

    /// The single page, for callers that never batch.
    pub fn single(&self) -> &CscPage {
        debug_assert_eq!(self.pages.len(), 1, "single() on a batched matrix");
        &self.pages[0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix() -> DMatrix {
        // 3 rows, 2 columns, one hole at (1, 0).
        DMatrix::from_dense(&[3.0, 1.0, f32::NAN, 5.0, 2.0, 4.0], 3, 2, f32::NAN).unwrap()
    }

    #[test]
    fn a_column_holds_every_row_that_has_the_feature() {
        let page = CscPage::build(&matrix(), 0..3);
        assert_eq!(page.column(0), ([0u32, 2].as_slice(), [3.0f32, 2.0].as_slice()));
        assert_eq!(page.column(1), ([0u32, 1, 2].as_slice(), [1.0f32, 5.0, 4.0].as_slice()));
        assert_eq!(page.column_len(0), 2);
    }

    #[test]
    fn sorting_orders_by_value_and_keeps_rows_absolute() {
        let page = CscPage::build_sorted(&matrix(), 0..3);
        assert_eq!(page.column(0), ([2u32, 0].as_slice(), [2.0f32, 3.0].as_slice()));
        assert_eq!(page.column(1), ([0u32, 2, 1].as_slice(), [1.0f32, 4.0, 5.0].as_slice()));
    }

    #[test]
    fn equal_values_keep_their_row_order() {
        let d = DMatrix::from_dense(&[1.0, 1.0, 1.0, 1.0], 4, 1, f32::NAN).unwrap();
        let page = CscPage::build_sorted(&d, 0..4);
        assert_eq!(page.column(0).0, [0u32, 1, 2, 3]);
    }

    #[test]
    fn batching_splits_rows_and_keeps_absolute_indices() {
        let pages = CscPages::build(&matrix(), Some(2), false);
        assert_eq!(pages.len(), 2);
        let mut seen: Vec<u32> = Vec::new();
        for page in pages.iter() {
            seen.extend_from_slice(page.column(1).0);
        }
        assert_eq!(seen, vec![0, 1, 2], "every row appears in exactly one page");
        assert_eq!(pages.iter().next().unwrap().column(0), ([0u32].as_slice(), [3.0f32].as_slice()));
    }

    #[test]
    fn an_unbatched_matrix_is_one_page() {
        assert_eq!(CscPages::build(&matrix(), None, false).len(), 1);
    }

    #[test]
    fn an_empty_matrix_still_yields_a_page() {
        let d = DMatrix::from_dense(&[], 0, 2, f32::NAN).unwrap();
        let pages = CscPages::build(&d, None, false);
        assert_eq!(pages.len(), 1);
        assert_eq!(pages.single().column(0).0.len(), 0);
    }
}
