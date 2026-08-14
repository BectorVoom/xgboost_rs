//! The learning-to-rank objectives — LambdaMART.
//!
//! A port of `src/objective/lambdarank_obj.{h,cc}` and the parts of
//! `src/common/ranking_utils.h` it reads. All three `rank:*` objectives share
//! one pair-wise engine and differ only in the *delta metric* a swapped pair
//! would cause:
//!
//! | Objective | delta |
//! |---|---|
//! | `rank:pairwise` | `1` — every mis-ordered pair costs the same |
//! | `rank:ndcg` | change in NDCG |
//! | `rank:map` | change in mean average precision |
//!
//! Rows are grouped into queries by `group`/`qid`; weights are per query, not
//! per row, which is what upstream requires too.

use super::{GradientPair, Objective, fit_intercept, sigmoid};
use crate::data::MetaInfo;
use crate::parameters::{LambdaRankPairMethod, LambdaRankParameters};
use crate::rng::MinStdRand;
use crate::{Error, Result};

/// `obj::Eps64`.
const EPS64: f64 = 1e-16;

/// Which delta metric the pair-wise gradients are weighted by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RankLoss {
    /// `rank:pairwise`.
    Pairwise,
    /// `rank:ndcg`.
    Ndcg,
    /// `rank:map`.
    Map,
}

impl RankLoss {
    const fn name(self) -> &'static str {
        match self {
            Self::Pairwise => "rank:pairwise",
            Self::Ndcg => "rank:ndcg",
            Self::Map => "rank:map",
        }
    }

    /// The metric a fit reports by default. Pairwise reports NDCG too — it
    /// optimises order, and NDCG is how upstream scores order.
    const fn metric_base(self) -> &'static str {
        match self {
            Self::Pairwise | Self::Ndcg => "ndcg",
            Self::Map => "map",
        }
    }
}

/// `ltr::CalcDCGGain` — the exponential gain `2^rel - 1`.
#[inline]
fn dcg_gain(label: f32) -> f64 {
    // Upstream shifts a `uint32`, so a non-integral or huge label is not a
    // gain it can express; clamping keeps the arithmetic finite.
    let rel = label.max(0.0).min(31.0) as u32;
    ((1u64 << rel) - 1) as f64
}

/// `ltr::CalcDCGDiscount`.
#[inline]
fn dcg_discount(idx: usize) -> f64 {
    1.0 / ((idx as f64) + 2.0).log2()
}

/// One query group's working state.
struct GroupRank {
    /// Row indices of the group, sorted by descending prediction.
    rank: Vec<usize>,
    /// Group-local row offset into the whole matrix.
    begin: usize,
}

/// Positions tracked when the truncation level does not bound them.
///
/// `RankingCache::MaxPositionSize` caps an untruncated fit here: the bias
/// decays exponentially down the list, so estimating it past this depth buys
/// nothing.
const MAX_TRACKED_POSITIONS: usize = 32;

/// The position-bias state of an unbiased fit, upstream's `ti_plus_`/`tj_minus_`
/// and the per-round `li`/`lj` accumulators.
///
/// This is the "Unbiased LambdaMART" estimator: examination propensity is
/// re-estimated from every round's own pairs, and the pair gradients are
/// divided by it, so a document that was rarely examined counts for more.
#[derive(Clone, Debug)]
struct PositionBias {
    /// `t_i^+`: propensity that the higher-ranked document was examined.
    ti_plus: Vec<f64>,
    /// `t_j^-`: propensity that the lower-ranked one was.
    tj_minus: Vec<f64>,
    /// Per-position cost accumulated over this round, aggregated across groups.
    li: Vec<f64>,
    lj: Vec<f64>,
    /// `1 / (1 + lambdarank_bias_norm)`, the exponent of the update.
    regularizer: f64,
}

impl PositionBias {
    fn new(size: usize, bias_norm: f64) -> Self {
        Self {
            // A fit starts believing every position was examined equally.
            ti_plus: vec![1.0; size],
            tj_minus: vec![1.0; size],
            li: vec![0.0; size],
            lj: vec![0.0; size],
            regularizer: 1.0 / (1.0 + bias_norm),
        }
    }

    fn size(&self) -> usize {
        self.ti_plus.len()
    }

    /// `LambdaRankUpdatePositionBias` — re-estimate the propensities from the
    /// costs this round accumulated, then clear them for the next one.
    ///
    /// The update normalises position 0 to 1, which costs the values their
    /// meaning as probabilities. That is what the authors specify and what
    /// upstream does, so it is reproduced rather than corrected.
    fn update(&mut self) {
        let (li0, lj0) = (self.li[0], self.lj[0]);
        for i in 0..self.size() {
            if li0 >= EPS64 {
                self.ti_plus[i] = (self.li[i] / li0).powf(self.regularizer);
            }
            if lj0 >= EPS64 {
                self.tj_minus[i] = (self.lj[i] / lj0).powf(self.regularizer);
            }
            debug_assert!(self.ti_plus[i].is_finite(), "propensity diverged at position {i}");
            debug_assert!(self.tj_minus[i].is_finite(), "propensity diverged at position {i}");
        }
        self.li.fill(0.0);
        self.lj.fill(0.0);
    }
}

/// `rank:pairwise`, `rank:ndcg` and `rank:map`.
pub struct LambdaRank {
    loss: RankLoss,
    param: LambdaRankParameters,
    /// Present only for `lambdarank_unbiased`; sized on the first round, when
    /// the group sizes are known.
    bias: Option<PositionBias>,
    /// The round's draw from the session engine, which seeds the per-group pair
    /// sampler. Only `lambdarank_pair_method=mean` samples pairs, so only it
    /// asks for one.
    pair_seed: u32,
}

impl LambdaRank {
    pub fn new(loss: RankLoss, param: LambdaRankParameters) -> Self {
        Self { loss, param, bias: None, pair_seed: 0 }
    }

    /// `RankingCache::MaxPositionSize` — how many positions the bias is
    /// estimated for.
    fn max_position_size(&self, info: &MetaInfo) -> usize {
        if self.has_truncation() {
            return self.num_pair();
        }
        let max_group =
            info.groups().into_iter().map(|(b, e)| e - b).max().unwrap_or(0);
        max_group.min(MAX_TRACKED_POSITIONS)
    }


    /// Pairs per sample, resolving the method-dependent default.
    fn num_pair(&self) -> usize {
        self.param.resolved_num_pair_per_sample() as usize
    }

    fn has_truncation(&self) -> bool {
        self.param.pair_method == LambdaRankPairMethod::TopK
    }

    /// `LambdaRankParam::TopK` — the truncation level, or the whole list.
    fn top_k(&self) -> usize {
        if self.has_truncation() { self.num_pair() } else { usize::MAX }
    }

    /// Sort each group's rows by descending prediction, ties to the lower row
    /// index so the ranking is reproducible.
    fn rank_groups(&self, preds: &[f32], info: &MetaInfo) -> Vec<GroupRank> {
        info.groups()
            .into_iter()
            .map(|(begin, end)| {
                let mut rank: Vec<usize> = (0..end - begin).collect();
                rank.sort_by(|&a, &b| {
                    preds[begin + b]
                        .total_cmp(&preds[begin + a])
                        .then_with(|| a.cmp(&b))
                });
                GroupRank { rank, begin }
            })
            .collect()
    }

    /// `NDCGCache::InitOnCPU` — the inverse ideal DCG of one group.
    fn inv_idcg(&self, labels: &[f32]) -> f64 {
        let mut sorted: Vec<f32> = labels.to_vec();
        sorted.sort_by(|a, b| b.total_cmp(a));
        let topk = self.top_k().min(sorted.len());
        let mut idcg = 0.0f64;
        for (i, &y) in sorted.iter().take(topk).enumerate() {
            idcg += dcg_discount(i) * if self.param.ndcg_exp_gain { dcg_gain(y) } else { y as f64 };
        }
        if idcg == 0.0 { 0.0 } else { 1.0 / idcg }
    }

    /// `cpu_impl::MAPStat` — relevant-document counts and the running
    /// precision sum, both indexed by rank.
    fn map_stat(labels: &[f32], rank: &[usize]) -> (Vec<f64>, Vec<f64>) {
        let n = rank.len();
        let mut n_rel = vec![0.0f64; n];
        let mut acc = vec![0.0f64; n];
        for k in 0..n {
            let y = labels[rank[k]] as f64;
            n_rel[k] = if k == 0 { y } else { n_rel[k - 1] + y };
            // `acc` is upstream's `\sum l_k / k` — the label over its rank, not
            // the precision at that rank. Weighting the term by `n_rel[k]` as
            // well double-counts the relevant documents seen so far; with
            // binary labels the two agree at k = 0 and diverge from k = 1 on,
            // which is why only `rank:map` drifted.
            let term = y / ((k + 1) as f64);
            acc[k] = if k == 0 { term } else { acc[k - 1] + term };
        }
        (n_rel, acc)
    }

    /// `DeltaNDCG`.
    fn delta_ndcg(
        &self,
        y_high: f32,
        y_low: f32,
        rank_high: usize,
        rank_low: usize,
        inv_idcg: f64,
    ) -> f64 {
        let gain = |y: f32| if self.param.ndcg_exp_gain { dcg_gain(y) } else { y as f64 };
        let (dh, dl) = (dcg_discount(rank_high), dcg_discount(rank_low));
        let original = gain(y_high) * dh + gain(y_low) * dl;
        let changed = gain(y_low) * dh + gain(y_high) * dl;
        (original - changed) * inv_idcg
    }

    /// `DeltaMAP`.
    fn delta_map(
        y_high: f32,
        y_low: f32,
        rank_high: usize,
        rank_low: usize,
        n_rel: &[f64],
        acc: &[f64],
    ) -> f64 {
        // The caller orders the ranks; `rank_low` is always the later one, so
        // `rank_low >= 1` and indexing `rank_low - 1` is in range.
        let (r_h, r_l) = ((rank_high + 1) as f64, (rank_low + 1) as f64);
        let total = *n_rel.last().expect("non-empty group");
        if total <= 0.0 {
            return 0.0;
        }
        let m = n_rel[rank_low];
        let n = n_rel[rank_high];
        let b = acc[rank_low - 1] - acc[rank_high];
        if y_high < y_low {
            (m / r_l - (n + 1.0) / r_h - b) / total
        } else {
            (n / r_h - m / r_l + b) / total
        }
    }
}

impl Objective for LambdaRank {
    fn name(&self) -> &'static str {
        self.loss.name()
    }

    fn num_output_group(&self, _info: &MetaInfo) -> usize {
        1
    }

    /// `LambdaRankObj::GetGradient` draws a seed only when it will sample
    /// pairs, which is `mean` alone — `topk` enumerates them.
    fn wants_pair_seed(&self) -> bool {
        !self.has_truncation()
    }

    fn set_pair_seed(&mut self, seed: u32) {
        self.pair_seed = seed;
    }

    fn get_gradient(&mut self, preds: &[f32], info: &MetaInfo, _iter: i32, out: &mut Vec<GradientPair>) {
        out.clear();
        out.resize(preds.len(), GradientPair::default());
        if preds.is_empty() {
            return;
        }
        let groups = self.rank_groups(preds, info);
        let n_groups = groups.len();

        // The propensities are read by the pair loop and rewritten after it, so
        // they are moved out of `self` for the duration: the loop borrows
        // `self` immutably for the loss and its parameters.
        if self.param.unbiased && self.bias.is_none() {
            self.bias =
                Some(PositionBias::new(self.max_position_size(info), self.param.bias_norm));
        }
        let mut bias = self.bias.take();
        // `RankingCache::InitOnCPU`: weights are per query and normalised so
        // an unweighted fit and an all-ones-weighted fit agree.
        let group_weight = |g: usize| -> f32 {
            match &info.weights {
                Some(w) if w.len() == n_groups => w[g],
                _ => 1.0,
            }
        };
        let sum_weights: f64 = (0..n_groups).map(|g| group_weight(g) as f64).sum();
        let weight_norm =
            if sum_weights > 0.0 { n_groups as f64 / sum_weights } else { 1.0 };

        for (g, group) in groups.iter().enumerate() {
            let cnt = group.rank.len();
            if cnt < 2 {
                continue;
            }
            let begin = group.begin;
            let labels = &info.labels[begin..begin + cnt];
            let g_preds = &preds[begin..begin + cnt];
            let g_out = &mut out[begin..begin + cnt];

            let inv_idcg = if self.loss == RankLoss::Ndcg { self.inv_idcg(labels) } else { 0.0 };
            let map = if self.loss == RankLoss::Map {
                Some(Self::map_stat(labels, &group.rank))
            } else {
                None
            };

            let best_score = g_preds[group.rank[0]];
            let worst_score = g_preds[group.rank[cnt - 1]];
            let mut sum_lambda = 0.0f64;

            let pair = |rank_i: usize,
                        rank_j: usize,
                        g_out: &mut [GradientPair],
                        sum_lambda: &mut f64,
                        bias: Option<&mut PositionBias>| {
                let (mut rank_high, mut rank_low) = (rank_i, rank_j);
                let (idx_i, idx_j) = (group.rank[rank_high], group.rank[rank_low]);
                if labels[idx_i] == labels[idx_j] {
                    return;
                }
                if labels[idx_i] < labels[idx_j] {
                    std::mem::swap(&mut rank_high, &mut rank_low);
                }
                let idx_high = group.rank[rank_high];
                let idx_low = group.rank[rank_low];
                let (y_high, y_low) = (labels[idx_high], labels[idx_low]);
                let (s_high, s_low) = (g_preds[idx_high], g_preds[idx_low]);

                let mut delta_metric = match (self.loss, &map) {
                    (RankLoss::Pairwise, _) => 1.0,
                    (RankLoss::Ndcg, _) => {
                        self.delta_ndcg(y_high, y_low, rank_high, rank_low, inv_idcg).abs()
                    }
                    (RankLoss::Map, Some((n_rel, acc))) => {
                        // MAP's delta is defined with the earlier rank first.
                        let (lo, hi) = (rank_high.min(rank_low), rank_high.max(rank_low));
                        let (y_lo, y_hi) = if rank_high <= rank_low {
                            (y_high, y_low)
                        } else {
                            (y_low, y_high)
                        };
                        Self::delta_map(y_lo, y_hi, lo, hi, n_rel, acc).abs()
                    }
                    (RankLoss::Map, None) => unreachable!("map stats are built for rank:map"),
                };

                // `lambdarank_score_normalization`: damp pairs the model
                // already separates well.
                if self.param.score_normalization && best_score != worst_score {
                    delta_metric /= (s_high - s_low).abs() as f64 + 0.01;
                }

                let sig = sigmoid(s_high - s_low) as f64;
                let mut lambda = (sig - 1.0) * delta_metric;
                let mut hessian = (sig * (1.0 - sig)).max(EPS64) * delta_metric * 2.0;

                // `lambdarank_unbiased`: divide the pair out by how likely each
                // of its two documents was to be examined at its position, so
                // a pair from deep in the list — where a click is rarer for
                // reasons that have nothing to do with relevance — counts for
                // more. The positions are the ones on the *input* list, which
                // is assumed ordered by relevance, so the row index is the
                // position.
                if let Some(bias) = bias {
                    let k = bias.size();
                    let cost = (1.0 / (1.0 - sig)).ln() * delta_metric;
                    if idx_high < k && idx_low < k {
                        let (t_plus, t_minus) = (bias.ti_plus[idx_high], bias.tj_minus[idx_low]);
                        if t_minus >= EPS64 && t_plus >= EPS64 {
                            lambda /= t_plus * t_minus;
                            hessian /= t_plus * t_minus;
                        }
                        // Each side's cost is attributed net of the *other*
                        // side's propensity, which is what makes the next
                        // round's estimate an improvement rather than a
                        // restatement.
                        if t_minus >= EPS64 {
                            bias.li[idx_high] += cost / t_minus;
                        }
                        if t_plus >= EPS64 {
                            bias.lj[idx_low] += cost / t_plus;
                        }
                    }
                }

                let pg = GradientPair { grad: lambda as f32, hess: hessian as f32 };
                g_out[idx_high].grad += pg.grad;
                g_out[idx_high].hess += pg.hess;
                // The lower-ranked document takes the opposite push.
                g_out[idx_low].grad -= pg.grad;
                g_out[idx_low].hess += pg.hess;

                *sum_lambda += -2.0 * pg.grad as f64;
            };

            if self.has_truncation() {
                let n = cnt.min(self.num_pair());
                for i in 0..n {
                    for j in i + 1..cnt {
                        pair(i, j, g_out, &mut sum_lambda, bias.as_mut());
                    }
                }
            } else {
                // `mean`: sample pairs across relevance buckets, from
                // `std::minstd_rand rnd(seed + g)` — one engine per group, all
                // of them offset from the single draw the round took out of the
                // session engine.
                let mut rnd = MinStdRand::new(self.pair_seed.wrapping_add(g as u32));
                // Ranks sorted by label, descending: bucket boundaries.
                let mut y_sorted: Vec<usize> = (0..cnt).collect();
                y_sorted.sort_by(|&a, &b| {
                    labels[group.rank[b]].total_cmp(&labels[group.rank[a]]).then_with(|| a.cmp(&b))
                });
                let label_at = |k: usize| labels[group.rank[y_sorted[k]]];

                let mut i = 0usize;
                while i < cnt {
                    let mut j = i + 1;
                    while j < cnt && label_at(i) == label_at(j) {
                        j += 1;
                    }
                    let (n_lefts, n_rights) = (i, cnt - j);
                    if n_lefts + n_rights == 0 {
                        i = j;
                        continue;
                    }
                    for _ in 0..self.num_pair() {
                        for pair_idx in i..j {
                            let mut ridx = rnd.next_below(n_lefts + n_rights);
                            if ridx >= n_lefts {
                                ridx = ridx - i + j;
                            }
                            pair(
                                y_sorted[pair_idx],
                                y_sorted[ridx],
                                g_out,
                                &mut sum_lambda,
                                bias.as_mut(),
                            );
                        }
                    }
                    i = j;
                }
            }

            // `lambdarank_normalization`, the LightGBM-style rescaling.
            let mut norm = 1.0f64;
            if self.param.normalization {
                if self.param.pair_method == LambdaRankPairMethod::Mean {
                    norm = 1.0 / self.num_pair() as f64;
                } else if sum_lambda > 0.0 {
                    norm = (1.0 + sum_lambda).log2() / sum_lambda;
                }
            }
            // Three `f32` multiplies, not one combined scale: upstream applies
            // the normalisation in its own pass (and only when it is not 1),
            // then the group weight and the weight norm in a second, and each
            // `GradientPair::operator*` rounds to `f32`. Folding them into one
            // factor rounds once instead of three times, which is enough to
            // move a split threshold.
            if norm != 1.0 {
                let norm = norm as f32;
                for p in g_out.iter_mut() {
                    p.grad *= norm;
                    p.hess *= norm;
                }
            }
            let (w, w_norm) = (group_weight(g), weight_norm as f32);
            for p in g_out.iter_mut() {
                p.grad = p.grad * w * w_norm;
                p.hess = p.hess * w * w_norm;
            }
        }

        // Every group has contributed its costs; re-estimate the propensities
        // the *next* round will divide by.
        if let Some(bias) = bias.as_mut() {
            bias.update();
        }
        self.bias = bias;
    }

    fn init_estimation(&mut self, info: &MetaInfo) -> Vec<f32> {
        // Ranking scores are relative, so the intercept does not change the
        // induced order — but it does change the margin the first gradient is
        // taken at, and therefore every tree. Upstream fits it from the
        // gradient like any other `FitIntercept` objective; because ranking
        // gradients very nearly cancel within a group, the answer lands within
        // a rounding error of zero rather than on the 0.5 default this used to
        // return. Keeping 0.5 moved every tree — see
        // tests/oracle_string_parameters.rs.
        fit_intercept(self, info)
    }

    /// `ndcg@k` / `map@k` at the truncation level the objective uses.
    fn default_metric(&self) -> String {
        if self.has_truncation() {
            format!("{}@{}", self.loss.metric_base(), self.num_pair())
        } else {
            self.loss.metric_base().to_owned()
        }
    }

    fn validate_data(&self, info: &MetaInfo) -> Result<()> {
        if info.n_targets() != 1 {
            return Err(Error::invalid(
                "num_target",
                "multi-output learning to rank is not supported",
            ));
        }
        let n_groups = info.groups().len();
        if let Some(w) = &info.weights
            && w.len() != n_groups
            && w.len() != info.num_row
        {
            return Err(Error::invalid(
                "weight",
                format!(
                    "a ranking fit takes one weight per query group ({n_groups}), got {}",
                    w.len()
                ),
            ));
        }
        if self.loss != RankLoss::Ndcg
            && info.labels.iter().any(|&y| y < 0.0)
        {
            return Err(Error::invalid(
                "objective",
                format!("`{}` needs non-negative relevance labels", self.name()),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ranked(labels: &[f32], group_sizes: &[usize]) -> MetaInfo {
        let mut ptr = vec![0usize];
        let mut acc = 0;
        for s in group_sizes {
            acc += s;
            ptr.push(acc);
        }
        MetaInfo {
            num_row: labels.len(),
            num_col: 1,
            labels: labels.to_vec(),
            num_target: 1,
            group_ptr: ptr,
            ..Default::default()
        }
    }

    fn gradient(obj: &mut LambdaRank, preds: &[f32], info: &MetaInfo) -> Vec<GradientPair> {
        let mut out = Vec::new();
        obj.get_gradient(preds, info, 0, &mut out);
        out
    }

    #[test]
    fn a_mis_ordered_pair_is_pushed_apart() {
        for loss in [RankLoss::Pairwise, RankLoss::Ndcg, RankLoss::Map] {
            let mut obj = LambdaRank::new(loss, LambdaRankParameters::default());
            // Row 0 is more relevant but scored lower.
            let info = ranked(&[1.0, 0.0], &[2]);
            let g = gradient(&mut obj, &[0.0, 1.0], &info);
            assert!(g[0].grad < 0.0, "{loss:?}: the relevant row should rise, got {g:?}");
            assert!(g[1].grad > 0.0, "{loss:?}: {g:?}");
            assert!(g.iter().all(|p| p.hess > 0.0), "{loss:?}");
        }
    }

    #[test]
    fn a_correctly_ordered_pair_is_barely_touched() {
        let mut obj = LambdaRank::new(RankLoss::Ndcg, LambdaRankParameters::default());
        let info = ranked(&[1.0, 0.0], &[2]);
        let right = gradient(&mut obj, &[8.0, -8.0], &info);
        let wrong = gradient(&mut obj, &[-8.0, 8.0], &info);
        assert!(
            right[0].grad.abs() < wrong[0].grad.abs(),
            "a well-separated pair has a smaller gradient: {right:?} vs {wrong:?}"
        );
    }

    #[test]
    fn equal_labels_produce_no_gradient() {
        let mut obj = LambdaRank::new(RankLoss::Pairwise, LambdaRankParameters::default());
        let info = ranked(&[1.0, 1.0, 1.0], &[3]);
        let g = gradient(&mut obj, &[0.3, 0.1, 0.2], &info);
        assert!(g.iter().all(|p| p.grad == 0.0 && p.hess == 0.0), "{g:?}");
    }

    #[test]
    fn groups_are_independent() {
        let mut obj = LambdaRank::new(RankLoss::Ndcg, LambdaRankParameters::default());
        // Two groups, the second already perfectly ordered.
        let info = ranked(&[1.0, 0.0, 1.0, 0.0], &[2, 2]);
        let g = gradient(&mut obj, &[0.0, 1.0, 9.0, -9.0], &info);
        assert!(g[0].grad < 0.0, "the mis-ordered group moves");
        assert!(g[2].grad.abs() < g[0].grad.abs(), "the ordered group barely moves: {g:?}");
    }

    #[test]
    fn the_mean_pair_method_is_reproducible() {
        let param = LambdaRankParameters {
            pair_method: LambdaRankPairMethod::Mean,
            ..LambdaRankParameters::default()
        };
        let mut obj = LambdaRank::new(RankLoss::Ndcg, param);
        let info = ranked(&[2.0, 1.0, 0.0, 1.0, 2.0, 0.0], &[6]);
        let preds = [0.1f32, 0.5, 0.2, 0.9, 0.3, 0.4];
        assert_eq!(gradient(&mut obj, &preds, &info), gradient(&mut obj, &preds, &info));
    }

    #[test]
    fn the_default_metric_carries_the_truncation_level() {
        let topk = LambdaRank::new(RankLoss::Ndcg, LambdaRankParameters::default());
        assert_eq!(topk.default_metric(), "ndcg@32");

        let mean = LambdaRank::new(
            RankLoss::Map,
            LambdaRankParameters {
                pair_method: LambdaRankPairMethod::Mean,
                ..LambdaRankParameters::default()
            },
        );
        assert_eq!(mean.default_metric(), "map");
    }

    #[test]
    fn group_weights_scale_a_query() {
        let mut obj = LambdaRank::new(RankLoss::Pairwise, LambdaRankParameters::default());
        let mut info = ranked(&[1.0, 0.0, 1.0, 0.0], &[2, 2]);
        let plain = gradient(&mut obj, &[0.0, 1.0, 0.0, 1.0], &info);
        info.weights = Some(vec![3.0, 1.0]);
        let weighted = gradient(&mut obj, &[0.0, 1.0, 0.0, 1.0], &info);
        assert!(
            weighted[0].grad.abs() > plain[0].grad.abs(),
            "the heavier query counts for more: {plain:?} vs {weighted:?}"
        );
    }

    fn unbiased_params(bias_norm: f64) -> LambdaRankParameters {
        LambdaRankParameters {
            unbiased: true,
            bias_norm,
            // Keep the pair set deterministic and the gradients unscaled, so
            // the debiasing is the only thing moving.
            normalization: false,
            score_normalization: false,
            ..LambdaRankParameters::default()
        }
    }

    /// A biased fit carries no propensity state at all; an unbiased one builds
    /// it on the first round.
    #[test]
    fn position_bias_is_only_tracked_when_asked_for() {
        let info = ranked(&[3.0, 2.0, 1.0, 0.0], &[4]);
        let preds = [0.1f32, 0.2, 0.3, 0.4];

        let mut biased = LambdaRank::new(RankLoss::Ndcg, LambdaRankParameters::default());
        gradient(&mut biased, &preds, &info);
        assert!(biased.bias.is_none(), "an ordinary fit tracks no position bias");

        let mut unbiased = LambdaRank::new(RankLoss::Ndcg, unbiased_params(1.0));
        gradient(&mut unbiased, &preds, &info);
        let bias = unbiased.bias.as_ref().expect("an unbiased fit tracks position bias");
        assert_eq!(bias.size(), unbiased.num_pair(), "topk bounds the tracked positions");
        assert_eq!(bias.ti_plus[0], 1.0, "position 0 is the normalisation point");
    }

    /// The propensities start uniform, so the first round's gradients match a
    /// biased fit exactly; they diverge once the first estimate lands.
    #[test]
    fn debiasing_starts_neutral_and_then_changes_the_gradients() {
        let info = ranked(&[3.0, 2.0, 1.0, 0.0], &[4]);
        let preds = [0.4f32, 0.1, 0.3, 0.2];

        let mut biased = LambdaRank::new(RankLoss::Ndcg, {
            LambdaRankParameters { normalization: false, score_normalization: false, ..Default::default() }
        });
        let mut unbiased = LambdaRank::new(RankLoss::Ndcg, unbiased_params(1.0));

        let first_biased = gradient(&mut biased, &preds, &info);
        let first_unbiased = gradient(&mut unbiased, &preds, &info);
        assert_eq!(
            first_biased, first_unbiased,
            "round 0 divides by propensities that are all still 1"
        );

        let second_biased = gradient(&mut biased, &preds, &info);
        let second_unbiased = gradient(&mut unbiased, &preds, &info);
        assert_ne!(
            second_biased, second_unbiased,
            "round 1 must use the propensities round 0 estimated"
        );
    }

    /// Positions further down the list are examined less, so their estimated
    /// propensity falls below the normalised first position.
    #[test]
    fn later_positions_get_a_smaller_propensity() {
        // A long, clearly ordered list: the deeper a document sits, the less
        // its pairs contribute.
        let labels: Vec<f32> = (0..16).map(|i| (15 - i) as f32).collect();
        let info = ranked(&labels, &[16]);
        let preds: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();

        let mut obj = LambdaRank::new(RankLoss::Ndcg, unbiased_params(1.0));
        for _ in 0..5 {
            gradient(&mut obj, &preds, &info);
        }
        let bias = obj.bias.as_ref().unwrap();
        assert_eq!(bias.ti_plus[0], 1.0);
        assert!(
            bias.ti_plus[..16].iter().all(|t| t.is_finite() && *t >= 0.0),
            "propensities stay finite and non-negative: {:?}",
            &bias.ti_plus[..16]
        );
        assert!(
            bias.ti_plus[8] < bias.ti_plus[0],
            "position 8 should be examined less than position 0: {} vs {}",
            bias.ti_plus[8],
            bias.ti_plus[0]
        );
    }

    /// `lambdarank_bias_norm` is the exponent of the propensity update, so it
    /// changes how sharply the estimate moves away from uniform.
    #[test]
    fn bias_norm_controls_how_far_the_estimate_moves() {
        let labels: Vec<f32> = (0..16).map(|i| (15 - i) as f32).collect();
        let info = ranked(&labels, &[16]);
        let preds: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();

        let estimate = |bias_norm: f64| {
            let mut obj = LambdaRank::new(RankLoss::Ndcg, unbiased_params(bias_norm));
            for _ in 0..5 {
                gradient(&mut obj, &preds, &info);
            }
            obj.bias.as_ref().unwrap().ti_plus[8]
        };

        // A larger `bias_norm` means a smaller exponent, which pulls the ratio
        // back towards 1.
        let sharp = estimate(0.0);
        let soft = estimate(9.0);
        assert!(sharp < soft, "bias_norm=0 should move further than 9: {sharp} vs {soft}");
        assert!(soft <= 1.0, "the estimate never exceeds the normalised first position");
    }

    /// An untruncated fit bounds the tracked positions by the longest group and
    /// a hard cap, rather than by the pair count.
    #[test]
    fn the_mean_pair_method_bounds_tracked_positions_by_group_size() {
        let short = ranked(&[3.0, 2.0, 1.0], &[3]);
        let mut obj = LambdaRank::new(
            RankLoss::Ndcg,
            LambdaRankParameters {
                pair_method: LambdaRankPairMethod::Mean,
                ..unbiased_params(1.0)
            },
        );
        gradient(&mut obj, &[0.1, 0.2, 0.3], &short);
        assert_eq!(obj.bias.as_ref().unwrap().size(), 3, "bounded by the group");

        let labels: Vec<f32> = (0..100).map(|i| (i % 5) as f32).collect();
        let long = ranked(&labels, &[100]);
        let mut obj = LambdaRank::new(
            RankLoss::Ndcg,
            LambdaRankParameters {
                pair_method: LambdaRankPairMethod::Mean,
                ..unbiased_params(1.0)
            },
        );
        gradient(&mut obj, &vec![0.5f32; 100], &long);
        assert_eq!(
            obj.bias.as_ref().unwrap().size(),
            MAX_TRACKED_POSITIONS,
            "a long group is capped, because the bias decays anyway"
        );
    }
}
