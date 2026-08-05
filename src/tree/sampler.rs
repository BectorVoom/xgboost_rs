//! Row sampling — `subsample` and `sampling_method`.
//!
//! A port of `xgboost::tree::cpu_impl::Sampler` (`src/tree/hist/sampler.cc`).
//! Rows are not removed from the fit: an unsampled row keeps its place in the
//! row set and has its gradient pair *zeroed*, so it contributes nothing to any
//! histogram, weight or gain but still reaches a leaf and still gets a
//! prediction. Doing it this way is what makes the partitioner, the subtraction
//! trick and the prediction cache all work unchanged.
//!
//! # Determinism
//!
//! Row `i`'s decision comes from the `i`-th value of a linear congruential
//! engine seeded once per tree. Upstream reaches that with
//! `RandomReplace::SimpleSkip`, so a thread that starts at row `b` can jump
//! straight to the right state; here the same closed form gives each row its
//! own engine, so the decision depends on the row index alone — never on how
//! the rows were divided between threads.

use crate::objective::GradientPair;
use crate::parameters::SamplingMethod;
use crate::rng::{Lcg63, bernoulli, canonical_f32};
use crate::threading;
use rayon::prelude::*;

/// `kDefaultMvsLambda`: the hessian weighting inside gradient-based sampling.
const MVS_LAMBDA: f32 = 0.1;
/// `kRtEps`, used to keep the sampling probability away from zero.
const RT_EPS: f32 = 1e-6;
/// Rows per parallel chunk. Fixed, so the work split never changes an answer.
const CHUNK_ROWS: usize = 4096;

/// Draws the rows one tree is grown from.
#[derive(Clone, Copy, Debug)]
pub struct RowSampler {
    method: SamplingMethod,
    subsample: f32,
}

impl RowSampler {
    pub fn new(method: SamplingMethod, subsample: f32) -> Self {
        Self { method, subsample }
    }

    /// Whether this configuration samples at all.
    ///
    /// Upstream compares the *row count* rather than the ratio, so a
    /// `subsample` that rounds up to every row is not sampling and consumes no
    /// randomness.
    pub fn is_sampling(&self, n_rows: usize) -> bool {
        let sample_rows = (n_rows as f32 * self.subsample) as usize;
        n_rows > 0 && sample_rows < n_rows
    }

    /// Zero the gradient pairs of the rows this tree does not see.
    ///
    /// `seed` is one draw from the session engine, taken by the caller so the
    /// engine advances in the same order upstream advances it. `threads` is
    /// the fit's `nthread`; it changes only how fast this runs, never what it
    /// decides.
    pub fn sample(&self, gpair: &mut [GradientPair], seed: u64, threads: usize) {
        if !self.is_sampling(gpair.len()) {
            return;
        }
        let sample_rows = (gpair.len() as f32 * self.subsample) as usize;
        match self.method {
            SamplingMethod::Uniform => self.uniform(gpair, seed, threads),
            SamplingMethod::GradientBased => {
                if sample_rows == 0 {
                    gpair.fill(GradientPair::default());
                    return;
                }
                self.gradient_based(gpair, sample_rows, seed, threads);
            }
        }
    }

    /// Keep each row independently with probability `subsample`.
    fn uniform(&self, gpair: &mut [GradientPair], seed: u64, threads: usize) {
        let p = self.subsample as f64;
        for_each_row(gpair, seed, threads, |pair, draw| {
            if !bernoulli(draw, p) {
                *pair = GradientPair::default();
            }
        });
    }

    /// Minimum variance sampling: keep large gradients for certain, keep the
    /// rest with a probability that rescales them to stay unbiased.
    fn gradient_based(
        &self,
        gpair: &mut [GradientPair],
        sample_rows: usize,
        seed: u64,
        threads: usize,
    ) {
        let reg_abs_grad = reg_abs_grad(gpair);
        let threshold = sampling_threshold(&reg_abs_grad, sample_rows);

        let mut probability: Vec<f32> = Vec::with_capacity(gpair.len());
        probability.extend(reg_abs_grad.iter().map(|&g| sampling_probability(threshold, g)));

        for_each_row_indexed(gpair, seed, threads, |i, pair, draw| {
            let p = probability[i];
            let rnd = canonical_f32(draw);
            if p >= 1.0 {
                // Kept as-is: no rescaling needed.
            } else if p > 0.0 && rnd <= p {
                pair.grad /= p;
                pair.hess /= p;
            } else {
                *pair = GradientPair::default();
            }
        });
    }
}

/// Apply `f` to every row with that row's engine draw.
fn for_each_row(
    gpair: &mut [GradientPair],
    seed: u64,
    threads: usize,
    f: impl Fn(&mut GradientPair, u64) + Sync + Send,
) {
    for_each_row_indexed(gpair, seed, threads, |_, pair, draw| f(pair, draw));
}

/// Apply `f` to every row with its index and its engine draw.
///
/// Row `i` always sees the `(i + 1)`-th value of the engine, whatever the
/// chunking, because each chunk jumps to its own start state.
fn for_each_row_indexed(
    gpair: &mut [GradientPair],
    seed: u64,
    threads: usize,
    f: impl Fn(usize, &mut GradientPair, u64) + Sync + Send,
) {
    threading::install_with(threads, || {
        gpair.par_chunks_mut(CHUNK_ROWS).enumerate().for_each(|(c, chunk)| {
            let begin = c * CHUNK_ROWS;
            let mut engine = Lcg63::new(Lcg63::skip(begin as u64, seed));
            for (k, pair) in chunk.iter_mut().enumerate() {
                f(begin + k, pair, engine.next_u64());
            }
        });
    });
}

/// `sqrt(g^2 + lambda * h^2)` per row, upstream's `CalcRegAbsGrad`.
fn reg_abs_grad(gpair: &[GradientPair]) -> Vec<f32> {
    gpair
        .iter()
        .map(|p| (p.grad * p.grad + MVS_LAMBDA * p.hess * p.hess).sqrt())
        .collect()
}

/// The gradient magnitude above which a row is always kept.
///
/// Mirrors `CalculateThreshold`: a binary search for the `u` that makes the
/// expected number of sampled rows equal `sample_rows`.
fn sampling_threshold(reg_abs_grad: &[f32], sample_rows: usize) -> f32 {
    let n_samples = reg_abs_grad.len();
    if sample_rows == 0 {
        return f32::MAX;
    }

    let mut sorted = reg_abs_grad.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // Sentinel, so the last element still has an upper bound.
    sorted.push(f32::MAX);

    let mut csum = Vec::with_capacity(n_samples);
    let mut running = 0.0f32;
    for &v in &sorted[..n_samples] {
        running += v;
        csum.push(running);
    }

    let (mut low, mut high) = (0i64, n_samples as i64 - 1);
    while low <= high {
        let i = low + (high - low) / 2;
        let (lower, upper) = (sorted[i as usize], sorted[i as usize + 1]);
        let n_above = n_samples as i64 - i - 1;
        let denom = sample_rows as f32 - n_above as f32;
        if denom <= 0.0 {
            low = i + 1;
            continue;
        }
        let u = csum[i as usize] / denom;
        if u > lower && u <= upper {
            return u;
        }
        if u <= lower {
            high = i - 1;
        } else {
            low = i + 1;
        }
    }

    // Every gradient is identical, so no `u` can beat its own lower bound.
    csum[n_samples - 1] / sample_rows as f32
}

/// Probability of keeping a row with this gradient magnitude.
fn sampling_probability(threshold: f32, reg_abs_grad: f32) -> f32 {
    if threshold.is_infinite() {
        // An infinite threshold is an empty sampling budget.
        return 0.0;
    }
    let u = if threshold.abs() < RT_EPS { RT_EPS.copysign(threshold) } else { threshold };
    reg_abs_grad / u
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pairs(n: usize) -> Vec<GradientPair> {
        (0..n)
            .map(|i| GradientPair { grad: (i as f32) - 50.0, hess: 1.0 })
            .collect()
    }

    fn kept(gpair: &[GradientPair]) -> usize {
        gpair.iter().filter(|p| p.hess != 0.0).count()
    }

    #[test]
    fn full_subsample_touches_nothing() {
        let sampler = RowSampler::new(SamplingMethod::Uniform, 1.0);
        assert!(!sampler.is_sampling(100));
        let mut g = pairs(100);
        let before = g.clone();
        sampler.sample(&mut g, 12345, 1);
        assert_eq!(g, before);
    }

    /// Whether a fit samples is decided on the *row count*, not the ratio, so
    /// a ratio that still drops rows samples however close to 1 it is.
    #[test]
    fn sampling_is_decided_on_the_row_count() {
        assert!(RowSampler::new(SamplingMethod::Uniform, 0.999).is_sampling(10));
        assert!(!RowSampler::new(SamplingMethod::Uniform, 1.0).is_sampling(10));
        // Truncation, not rounding: a budget below one row is still sampling,
        // and it keeps nothing.
        let sampler = RowSampler::new(SamplingMethod::Uniform, 0.5);
        assert!(sampler.is_sampling(1));
    }

    #[test]
    fn uniform_sampling_keeps_about_the_requested_fraction() {
        let sampler = RowSampler::new(SamplingMethod::Uniform, 0.5);
        let mut g = pairs(20_000);
        sampler.sample(&mut g, 987_654_321, 1);
        let rate = kept(&g) as f64 / 20_000.0;
        assert!((rate - 0.5).abs() < 0.02, "kept {rate}");
        // Dropped rows are zeroed in both components.
        assert!(g.iter().all(|p| p.hess != 0.0 || p.grad == 0.0));
    }

    #[test]
    fn uniform_sampling_is_reproducible_and_seed_dependent() {
        let sampler = RowSampler::new(SamplingMethod::Uniform, 0.5);
        let (mut a, mut b, mut c) = (pairs(5000), pairs(5000), pairs(5000));
        sampler.sample(&mut a, 1, 1);
        sampler.sample(&mut b, 1, 1);
        sampler.sample(&mut c, 2, 1);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    /// The whole point of the skip-ahead engine: the chunking must not matter.
    #[test]
    fn row_decisions_do_not_depend_on_the_chunk_boundaries() {
        let sampler = RowSampler::new(SamplingMethod::Uniform, 0.4);
        let n = CHUNK_ROWS * 3 + 17;
        let mut whole = pairs(n);
        sampler.sample(&mut whole, 42, 1);

        // Recompute row by row with a fresh engine per row.
        for (i, pair) in whole.iter().enumerate() {
            let draw = Lcg63::skip(i as u64 + 1, 42);
            let expect_kept = bernoulli(draw, 0.4);
            assert_eq!(pair.hess != 0.0, expect_kept, "row {i}");
        }
    }

    #[test]
    fn gradient_based_sampling_keeps_the_largest_gradients() {
        let sampler = RowSampler::new(SamplingMethod::GradientBased, 0.3);
        let mut g = pairs(2000);
        sampler.sample(&mut g, 7, 1);

        // Rows at the extremes have the largest |gradient| and survive; the
        // rows near the middle are the ones dropped.
        assert!(g[0].hess != 0.0, "the largest negative gradient must survive");
        assert!(g[1999].hess != 0.0, "the largest positive gradient must survive");
        let dropped = 2000 - kept(&g);
        assert!(dropped > 0, "some rows must be dropped");
        let mid_dropped = g[900..1100].iter().filter(|p| p.hess == 0.0).count();
        assert!(mid_dropped > 0, "small gradients are the ones sampled away");
    }

    /// Rescaling by `1/p` keeps the gradient sum roughly unbiased, which is the
    /// property gradient-based sampling exists for.
    #[test]
    fn gradient_based_sampling_rescales_to_stay_unbiased() {
        let sampler = RowSampler::new(SamplingMethod::GradientBased, 0.5);
        let mut g = pairs(20_000);
        let full: f64 = g.iter().map(|p| p.grad as f64).sum();
        sampler.sample(&mut g, 11, 1);
        let sampled: f64 = g.iter().map(|p| p.grad as f64).sum();
        let scale = full.abs().max(1.0);
        assert!((sampled - full).abs() / scale < 0.15, "{sampled} vs {full}");
    }

    #[test]
    fn an_empty_budget_drops_every_row() {
        let sampler = RowSampler::new(SamplingMethod::GradientBased, 0.0001);
        let mut g = pairs(100);
        sampler.sample(&mut g, 5, 1);
        assert_eq!(kept(&g), 0);
    }

    #[test]
    fn identical_gradients_still_produce_a_usable_threshold() {
        // The degenerate branch of `CalculateThreshold`: no `u` beats its lower
        // bound, so the fallback has to keep the fit going.
        let flat: Vec<GradientPair> =
            (0..1000).map(|_| GradientPair { grad: 1.0, hess: 1.0 }).collect();
        let sampler = RowSampler::new(SamplingMethod::GradientBased, 0.5);
        let mut g = flat;
        sampler.sample(&mut g, 3, 1);
        let rate = kept(&g) as f64 / 1000.0;
        assert!((rate - 0.5).abs() < 0.1, "kept {rate}");
    }

    #[test]
    fn empty_input_is_not_sampling() {
        let sampler = RowSampler::new(SamplingMethod::Uniform, 0.5);
        assert!(!sampler.is_sampling(0));
        let mut g: Vec<GradientPair> = Vec::new();
        sampler.sample(&mut g, 1, 1);
        assert!(g.is_empty());
    }
}
