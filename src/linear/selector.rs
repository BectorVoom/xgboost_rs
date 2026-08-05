//! The five `feature_selector` strategies.
//!
//! A port of `xgboost::linear::FeatureSelector` and its five subclasses
//! (`coordinate_common.h`). Each one answers the same question — which feature
//! does the solver update next — and they differ only in how much work they do
//! to answer it:
//!
//! | Selector | Cost per sweep | Deterministic |
//! |---|---|---|
//! | `cyclic` | nothing | yes |
//! | `shuffle` | one shuffle per round | yes, given the seed |
//! | `random` | nothing | yes, given the seed |
//! | `greedy` | one full pass over the matrix **per feature** | yes |
//! | `thrifty` | one full pass over the matrix per round | yes |
//!
//! `greedy` is quadratic in the feature count and is the reason `top_k`
//! exists; `thrifty` is its linear approximation, ranking every feature once by
//! the step it *would* take and then cycling that order.

use crate::context::Context;
use crate::data::csc::CscPages;
use crate::objective::GradientPair;
use crate::parameters::FeatureSelector;
use crate::rng::shuffle;

use super::coordinate::{column_gradient, coordinate_delta};

/// The per-round state a selector carries between [`Selector::next_feature`]
/// calls.
#[derive(Clone, Debug)]
pub struct Selector {
    kind: FeatureSelector,
    /// `shuffle`: the permutation this round cycles through.
    order: Vec<u32>,
    /// `greedy` / `thrifty`: features already handed out, per group.
    counter: Vec<usize>,
    /// `greedy` / `thrifty`: the shortlist size, `usize::MAX` for "all".
    top_k: usize,
    /// `thrifty`: features ranked by descending univariate step, per group.
    ranked: Vec<Vec<u32>>,
}

impl Selector {
    pub fn new(kind: FeatureSelector) -> Self {
        Self {
            kind,
            order: Vec::new(),
            counter: Vec::new(),
            top_k: usize::MAX,
            ranked: Vec::new(),
        }
    }

    /// `FeatureSelector::Setup`: called once per boosting round, before any
    /// feature is chosen.
    ///
    /// `top_k` of `0` means "every feature", as upstream's `param <= 0` does.
    #[allow(clippy::too_many_arguments)]
    pub fn setup(
        &mut self,
        ctx: &mut Context,
        weight: &[f32],
        gpair: &[GradientPair],
        pages: &CscPages,
        n_features: usize,
        n_groups: usize,
        alpha: f64,
        lambda: f64,
        top_k: u32,
    ) {
        self.top_k = if top_k == 0 { usize::MAX } else { top_k as usize };
        self.counter = vec![0; n_groups];

        match self.kind {
            FeatureSelector::Cyclic | FeatureSelector::Random => {}
            FeatureSelector::Shuffle => {
                if self.order.len() != n_features {
                    self.order = (0..n_features as u32).collect();
                }
                shuffle(&mut self.order, ctx.rng());
            }
            FeatureSelector::Greedy => {}
            FeatureSelector::Thrifty => {
                self.ranked = (0..n_groups)
                    .map(|gid| {
                        let mut steps: Vec<(u32, f64)> = (0..n_features)
                            .map(|f| {
                                let (g, h) =
                                    sum_over_pages(pages, f, gid, n_groups, gpair);
                                let w = weight[f * n_groups + gid] as f64;
                                // Upstream stores the step as `float` before
                                // ranking, so the comparison sees `f32`
                                // precision.
                                let dw = coordinate_delta(g, h, w, alpha, lambda) as f32;
                                (f as u32, dw.abs() as f64)
                            })
                            .collect();
                        // Descending by magnitude; ties keep ascending feature
                        // order, which upstream's `std::sort` leaves open and
                        // this crate pins down.
                        steps.sort_by(|a, b| {
                            b.1.partial_cmp(&a.1)
                                .unwrap_or(std::cmp::Ordering::Equal)
                                .then(a.0.cmp(&b.0))
                        });
                        steps.into_iter().map(|(f, _)| f).collect()
                    })
                    .collect();
            }
        }
    }

    /// `FeatureSelector::NextFeature`: the index to update, or `None` to stop
    /// sweeping this group.
    #[allow(clippy::too_many_arguments)]
    pub fn next_feature(
        &mut self,
        ctx: &mut Context,
        iteration: usize,
        weight: &[f32],
        gpair: &[GradientPair],
        pages: &CscPages,
        n_features: usize,
        group: usize,
        n_groups: usize,
        alpha: f64,
        lambda: f64,
    ) -> Option<usize> {
        if n_features == 0 {
            return None;
        }
        match self.kind {
            FeatureSelector::Cyclic => Some(iteration % n_features),
            FeatureSelector::Shuffle => Some(self.order[iteration % n_features] as usize),
            FeatureSelector::Random => Some(ctx.rng().next_u32() as usize % n_features),
            FeatureSelector::Greedy => {
                self.take_slot(group, n_features)?;
                // Recompute every feature's univariate step against the current
                // residuals and take the largest. Upstream keeps feature 0 as
                // the fallback when nothing would move.
                let mut best = 0usize;
                let mut best_step = 0.0f32;
                for f in 0..n_features {
                    let (g, h) = sum_over_pages(pages, f, group, n_groups, gpair);
                    let w = weight[f * n_groups + group] as f64;
                    let dw = (coordinate_delta(g, h, w, alpha, lambda) as f32).abs();
                    if dw > best_step {
                        best_step = dw;
                        best = f;
                    }
                }
                Some(best)
            }
            FeatureSelector::Thrifty => {
                let k = self.take_slot(group, n_features)?;
                Some(self.ranked[group][k] as usize)
            }
        }
    }

    /// The shortlist bookkeeping `greedy` and `thrifty` share: hand out the
    /// next slot, or stop once `top_k` — or the feature count — is reached.
    fn take_slot(&mut self, group: usize, n_features: usize) -> Option<usize> {
        let k = self.counter[group];
        self.counter[group] += 1;
        // Upstream stops at `counter == n_features`, one short of a full sweep.
        if k >= self.top_k || self.counter[group] == n_features {
            return None;
        }
        Some(k)
    }
}

/// One feature's gradient sums over every row batch.
fn sum_over_pages(
    pages: &CscPages,
    fidx: usize,
    group: usize,
    n_groups: usize,
    gpair: &[GradientPair],
) -> (f64, f64) {
    let mut sum_grad = 0.0f64;
    let mut sum_hess = 0.0f64;
    for page in pages.iter() {
        let (g, h) = column_gradient(page, fidx, group, n_groups, gpair);
        sum_grad += g;
        sum_hess += h;
    }
    (sum_grad, sum_hess)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DMatrix;

    fn setup_state(kind: FeatureSelector, n_features: usize) -> (Selector, Context, CscPages) {
        let d = DMatrix::from_dense(&vec![1.0f32; n_features], 1, n_features, f32::NAN).unwrap();
        (Selector::new(kind), Context::default(), CscPages::build(&d, None, false))
    }

    #[test]
    fn cyclic_walks_the_features_in_order_and_wraps() {
        let (mut s, mut ctx, pages) = setup_state(FeatureSelector::Cyclic, 3);
        let gpair = vec![GradientPair { grad: 1.0, hess: 1.0 }];
        s.setup(&mut ctx, &[0.0; 3], &gpair, &pages, 3, 1, 0.0, 0.0, 0);
        let picks: Vec<usize> = (0..5)
            .map(|i| {
                s.next_feature(&mut ctx, i, &[0.0; 3], &gpair, &pages, 3, 0, 1, 0.0, 0.0).unwrap()
            })
            .collect();
        assert_eq!(picks, vec![0, 1, 2, 0, 1]);
    }

    #[test]
    fn shuffle_is_a_permutation_and_depends_on_the_seed() {
        let (mut s, mut ctx, pages) = setup_state(FeatureSelector::Shuffle, 8);
        let gpair = vec![GradientPair { grad: 1.0, hess: 1.0 }];
        s.setup(&mut ctx, &[0.0; 8], &gpair, &pages, 8, 1, 0.0, 0.0, 0);
        let mut picks: Vec<usize> = (0..8)
            .map(|i| {
                s.next_feature(&mut ctx, i, &[0.0; 8], &gpair, &pages, 8, 0, 1, 0.0, 0.0).unwrap()
            })
            .collect();
        picks.sort_unstable();
        assert_eq!(picks, (0..8).collect::<Vec<_>>(), "every feature exactly once");

        // A second round reshuffles.
        let before = s.order.clone();
        s.setup(&mut ctx, &[0.0; 8], &gpair, &pages, 8, 1, 0.0, 0.0, 0);
        assert_ne!(s.order, before);
    }

    #[test]
    fn random_stays_in_range() {
        let (mut s, mut ctx, pages) = setup_state(FeatureSelector::Random, 4);
        let gpair = vec![GradientPair { grad: 1.0, hess: 1.0 }];
        s.setup(&mut ctx, &[0.0; 4], &gpair, &pages, 4, 1, 0.0, 0.0, 0);
        for i in 0..50 {
            let f =
                s.next_feature(&mut ctx, i, &[0.0; 4], &gpair, &pages, 4, 0, 1, 0.0, 0.0).unwrap();
            assert!(f < 4);
        }
    }

    #[test]
    fn top_k_bounds_a_shortlist_sweep() {
        let (mut s, mut ctx, pages) = setup_state(FeatureSelector::Thrifty, 6);
        let gpair = vec![GradientPair { grad: 1.0, hess: 1.0 }];
        s.setup(&mut ctx, &[0.0; 6], &gpair, &pages, 6, 1, 0.0, 0.0, 2);
        let mut n = 0;
        while s.next_feature(&mut ctx, n, &[0.0; 6], &gpair, &pages, 6, 0, 1, 0.0, 0.0).is_some() {
            n += 1;
            assert!(n <= 6, "the sweep must terminate");
        }
        assert_eq!(n, 2, "top_k = 2 hands out two features");
    }

    #[test]
    fn an_unbounded_shortlist_stops_one_short_of_a_full_sweep() {
        // Upstream's `counter == num_feature` check ends the sweep on the last
        // feature rather than after it; pinned here so the behaviour is not
        // "fixed" into a divergence.
        let (mut s, mut ctx, pages) = setup_state(FeatureSelector::Thrifty, 5);
        let gpair = vec![GradientPair { grad: 1.0, hess: 1.0 }];
        s.setup(&mut ctx, &[0.0; 5], &gpair, &pages, 5, 1, 0.0, 0.0, 0);
        let mut n = 0;
        while s.next_feature(&mut ctx, n, &[0.0; 5], &gpair, &pages, 5, 0, 1, 0.0, 0.0).is_some() {
            n += 1;
        }
        assert_eq!(n, 4);
    }

    /// Feature 0 is orthogonal to the gradients (its sums cancel), feature 1
    /// lines up with them exactly. Only feature 1 has a step to take.
    fn ranking_data() -> (DMatrix, CscPages, Vec<GradientPair>) {
        let d = DMatrix::from_dense(&[1.0f32, 1.0, 1.0, -1.0], 2, 2, f32::NAN).unwrap();
        let pages = CscPages::build(&d, None, false);
        let gpair = vec![
            GradientPair { grad: 1.0, hess: 1.0 },
            GradientPair { grad: -1.0, hess: 1.0 },
        ];
        (d, pages, gpair)
    }

    #[test]
    fn thrifty_ranks_the_feature_with_the_largest_step_first() {
        let (_d, pages, gpair) = ranking_data();
        let mut s = Selector::new(FeatureSelector::Thrifty);
        let mut ctx = Context::default();
        s.setup(&mut ctx, &[0.0; 4], &gpair, &pages, 2, 1, 0.0, 0.0, 0);
        assert_eq!(s.ranked[0], vec![1, 0]);
    }

    #[test]
    fn greedy_picks_the_largest_step_every_time() {
        let (_d, pages, gpair) = ranking_data();
        let mut s = Selector::new(FeatureSelector::Greedy);
        let mut ctx = Context::default();
        s.setup(&mut ctx, &[0.0; 4], &gpair, &pages, 2, 1, 0.0, 0.0, 0);
        let f = s.next_feature(&mut ctx, 0, &[0.0; 4], &gpair, &pages, 2, 0, 1, 0.0, 0.0);
        assert_eq!(f, Some(1));
    }
}
