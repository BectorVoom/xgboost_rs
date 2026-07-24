//! CPU reference ("oracle") implementations and deterministic data generators,
//! shared by the test suite and the benchmark binary.
//!
//! Because gradients are quantised to fixed point before accumulation, GPU
//! results must match these sequential sums *exactly* (`i64` equality) — that
//! determinism is the entire purpose of the quantiser.

use crate::gpu::ellpack::{EllpackLayout, EllpackMatrix};
use crate::gpu::{GradientPair, GradientPairInt64};

/// Deterministic xorshift PRNG so no external crates are needed.
pub struct Rng(pub u64);

impl Rng {
    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 32) as u32
    }

    /// Uniform in [-1, 1).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

pub fn random_gpairs(n: usize, seed: u64) -> Vec<GradientPair> {
    let mut rng = Rng(seed);
    (0..n)
        .map(|_| GradientPair { grad: rng.next_f32(), hess: rng.next_f32().abs() })
        .collect()
}

/// Random ELLPACK matrix; `sparsity` is the fraction of missing entries.
pub fn random_matrix(
    n_rows: usize,
    n_features: usize,
    bins_per_feature: u32,
    sparsity: f32,
    layout: EllpackLayout,
    seed: u64,
) -> EllpackMatrix {
    assert!(sparsity == 0.0 || layout != EllpackLayout::Dense);
    let mut rng = Rng(seed);
    let cut_ptrs: Vec<u32> = (0..=n_features as u32).map(|f| f * bins_per_feature).collect();
    let null_value = match layout {
        EllpackLayout::Dense => u32::MAX,
        EllpackLayout::DenseCompressed => bins_per_feature,
        EllpackLayout::Sparse => n_features as u32 * bins_per_feature,
    };
    let gidx = (0..n_rows * n_features)
        .map(|i| {
            if (rng.next_u32() as f32 / u32::MAX as f32) < sparsity {
                null_value
            } else {
                let local = rng.next_u32() % bins_per_feature;
                match layout {
                    EllpackLayout::Sparse => local + cut_ptrs[i % n_features],
                    _ => local,
                }
            }
        })
        .collect();
    EllpackMatrix {
        gidx,
        row_stride: n_features,
        base_rowid: 0,
        n_rows,
        cut_ptrs,
        null_value,
        layout,
    }
}

/// Sequential CPU oracle of the histogram kernel.
pub fn cpu_histogram(
    matrix: &EllpackMatrix,
    gpair: &[GradientPairInt64],
    ridx: &[u32],
) -> Vec<GradientPairInt64> {
    let mut hist = vec![GradientPairInt64::default(); matrix.n_bins() as usize];
    for &row in ridx {
        for f in 0..matrix.n_features() {
            let entry = (row - matrix.base_rowid) as usize * matrix.row_stride + f;
            let bin = matrix.gidx[entry];
            if matrix.is_dense() || bin != matrix.null_value {
                let global_bin =
                    if matrix.is_compressed() { bin + matrix.cut_ptrs[f] } else { bin };
                hist[global_bin as usize] = hist[global_bin as usize] + gpair[row as usize];
            }
        }
    }
    hist
}
