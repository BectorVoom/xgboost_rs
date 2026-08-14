//! Ranking and classification-ranking metrics: `auc`, `aucpr`, `pre`, `ndcg`,
//! `map` and `ams`.
//!
//! Ports of `src/metric/auc.{h,cc}` and `src/metric/rank_metric.cc`. The
//! ranking scores are computed per query group and averaged with the group
//! weights; without a `group` the whole matrix is one query, which is exactly
//! how upstream treats a binary classification matrix scored with `auc`.

use super::Metric;
use crate::data::MetaInfo;

/// Which curve `auc`/`aucpr` integrates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Curve {
    /// `auc` — receiver operating characteristic.
    Roc,
    /// `aucpr` — precision/recall.
    Pr,
}

/// `auc` and `aucpr`.
///
/// With a `group` set the score is the mean over queries of the within-query
/// AUC, as upstream's ranking AUC is; without one it is the ordinary binary
/// AUC over every row.
#[derive(Clone, Copy, Debug)]
pub struct Auc {
    curve: Curve,
}

impl Auc {
    pub fn roc() -> Self {
        Self { curve: Curve::Roc }
    }

    pub fn pr() -> Self {
        Self { curve: Curve::Pr }
    }
}

/// Row indices ordered by descending prediction, ties to the lower index.
fn by_descending_score(preds: &[f32]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..preds.len()).collect();
    idx.sort_by(|&a, &b| preds[b].total_cmp(&preds[a]).then_with(|| a.cmp(&b)));
    idx
}

/// `TrapezoidArea`.
#[inline]
fn trapezoid(x0: f64, x1: f64, y0: f64, y1: f64) -> f64 {
    (x0 - x1).abs() * (y0 + y1) * 0.5
}

/// `detail::CalcDeltaPRAUC` — the exact area under the interpolated
/// precision/recall segment between two operating points.
fn delta_pr_auc(fp_prev: f64, fp: f64, tp_prev: f64, tp: f64, total_pos: f64) -> f64 {
    let pr_prev = tp_prev / total_pos;
    let pr = tp / total_pos;
    let (a, b) = if tp == tp_prev {
        (1.0, 0.0)
    } else {
        let h = (fp - fp_prev) / (tp - tp_prev);
        (h + 1.0, (fp_prev - h * tp_prev) / total_pos)
    };
    if b != 0.0 {
        (pr - pr_prev - b / a * ((a * pr + b).ln() - (a * pr_prev + b).ln())) / a
    } else {
        (pr - pr_prev) / a
    }
}

/// `BinaryAUC`: sweeps the ranked list accumulating true and false positives,
/// adding one segment's area whenever the score changes.
///
/// Returns `(fp, tp, area)`; the caller divides the area by `fp * tp` for ROC,
/// which is what normalises it into `[0, 1]`.
fn binary_auc(
    preds: &[f32],
    labels: &[f32],
    weights: impl Fn(usize) -> f64,
    area: impl Fn(f64, f64, f64, f64) -> f64,
) -> (f64, f64, f64) {
    let sorted = by_descending_score(preds);
    if sorted.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mut auc = 0.0f64;
    let first = sorted[0];
    let w = weights(first);
    let mut fp = (1.0 - labels[first] as f64) * w;
    let mut tp = labels[first] as f64 * w;
    let (mut tp_prev, mut fp_prev) = (0.0f64, 0.0f64);

    for i in 1..sorted.len() {
        if preds[sorted[i]] != preds[sorted[i - 1]] {
            auc += area(fp_prev, fp, tp_prev, tp);
            tp_prev = tp;
            fp_prev = fp;
        }
        let idx = sorted[i];
        let w = weights(idx);
        fp += (1.0 - labels[idx] as f64) * w;
        tp += labels[idx] as f64 * w;
    }
    auc += area(fp_prev, fp, tp_prev, tp);
    // A degenerate list — all positive or all negative — has no area to report.
    if fp <= 0.0 || tp <= 0.0 {
        return (0.0, 0.0, 0.0);
    }
    (fp, tp, auc)
}

impl Metric for Auc {
    fn name(&self) -> &str {
        match self.curve {
            Curve::Roc => "auc",
            Curve::Pr => "aucpr",
        }
    }

    fn eval(&self, preds: &[f32], info: &MetaInfo) -> f64 {
        let groups = info.groups();
        let ranking = info.group_ptr.len() > 2;

        let score_one = |begin: usize, end: usize, weight: &dyn Fn(usize) -> f64| -> Option<f64> {
            let p = &preds[begin..end];
            let y = &info.labels[begin..end];
            match self.curve {
                Curve::Roc => {
                    let (fp, tp, auc) = binary_auc(p, y, |i| weight(begin + i), trapezoid);
                    if fp * tp <= 0.0 { None } else { Some(auc / (fp * tp)) }
                }
                Curve::Pr => {
                    let (mut total_pos, mut total_neg) = (0.0f64, 0.0f64);
                    for (i, &label) in y.iter().enumerate() {
                        let w = weight(begin + i);
                        total_pos += w * label as f64;
                        total_neg += w * (1.0 - label as f64);
                    }
                    if total_pos <= 0.0 || total_neg <= 0.0 {
                        return None;
                    }
                    let (_, _, auc) = binary_auc(p, y, |i| weight(begin + i), |a, b, c, d| {
                        delta_pr_auc(a, b, c, d, total_pos)
                    });
                    Some(auc)
                }
            }
        };

        if !ranking {
            // One list: the ordinary binary AUC with per-row weights. A
            // single-class matrix has no AUC at all, and `NaN` says so rather
            // than a number that looks like a score. (`f64::min` would turn it
            // back into `1.0`, so the clamp only runs on a real score.)
            let weight = |i: usize| info.weight(i) as f64;
            return match score_one(0, info.num_row, &weight) {
                Some(auc) => auc.min(1.0),
                None => f64::NAN,
            };
        }

        // Ranking: average the per-query AUCs, skipping degenerate queries as
        // upstream does rather than scoring them zero.
        let n_groups = groups.len();
        let group_weight = |g: usize| -> f64 {
            match &info.weights {
                Some(w) if w.len() == n_groups => w[g] as f64,
                _ => 1.0,
            }
        };
        let unit = |_: usize| 1.0f64;
        let mut sum = 0.0f64;
        let mut valid = 0.0f64;
        for (g, &(begin, end)) in groups.iter().enumerate() {
            if let Some(auc) = score_one(begin, end, &unit) {
                sum += auc * group_weight(g);
                valid += group_weight(g);
            }
        }
        if valid == 0.0 { f64::NAN } else { (sum / valid).min(1.0) }
    }
}

// -------------------------------------------------------- ranking scores ----

/// Which ranking score is being computed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Score {
    /// `pre` — precision at the cut-off.
    Precision,
    /// `ndcg` — normalised discounted cumulative gain.
    Ndcg,
    /// `map` — mean average precision.
    Map,
}

/// `pre`, `ndcg` and `map`, with their optional `@n` cut-off and trailing `-`.
///
/// The trailing `-` changes what an empty result list scores: `1` normally,
/// `0` with the minus variant. That is the whole difference, and it is why
/// `ndcg` and `ndcg-` are separate `eval_metric` values.
#[derive(Clone, Debug)]
pub struct RankScore {
    score: Score,
    top_n: Option<u32>,
    minus: bool,
    /// The exponential `2^rel - 1` gain, upstream's `ndcg_exp_gain` default.
    exp_gain: bool,
    name: String,
}

impl RankScore {
    fn new(score: Score, base: &str, top_n: Option<u32>, minus: bool) -> Self {
        let mut name = base.to_owned();
        if let Some(n) = top_n {
            name.push_str(&format!("@{n}"));
        }
        if minus {
            name.push('-');
        }
        Self { score, top_n, minus, exp_gain: true, name }
    }

    pub fn precision(top_n: Option<u32>) -> Self {
        Self::new(Score::Precision, "pre", top_n, false)
    }

    pub fn ndcg(top_n: Option<u32>, minus: bool) -> Self {
        Self::new(Score::Ndcg, "ndcg", top_n, minus)
    }

    pub fn map(top_n: Option<u32>, minus: bool) -> Self {
        Self::new(Score::Map, "map", top_n, minus)
    }

    /// Score relevance linearly rather than as `2^rel - 1`.
    pub fn with_linear_gain(mut self) -> Self {
        self.exp_gain = false;
        self
    }

    fn top_k(&self) -> usize {
        self.top_n.map_or(usize::MAX, |n| n as usize)
    }

    fn gain(&self, label: f32) -> f64 {
        if self.exp_gain {
            ((1u64 << label.max(0.0).min(31.0) as u32) - 1) as f64
        } else {
            label as f64
        }
    }

    /// The score of one query group.
    fn score_group(&self, preds: &[f32], labels: &[f32]) -> f64 {
        let sorted = by_descending_score(preds);
        let n = sorted.len().min(self.top_k());
        match self.score {
            Score::Precision => {
                if n == 0 {
                    return if self.minus { 0.0 } else { 1.0 };
                }
                let hits: f64 = sorted[..n].iter().map(|&i| labels[i] as f64).sum();
                hits / n as f64
            }
            Score::Ndcg => {
                // The ideal ordering, for the denominator.
                let mut ideal: Vec<f32> = labels.to_vec();
                ideal.sort_by(|a, b| b.total_cmp(a));
                let idcg: f64 = ideal[..n]
                    .iter()
                    .enumerate()
                    .map(|(i, &y)| discount(i) * self.gain(y))
                    .sum();
                if idcg <= 0.0 {
                    return if self.minus { 0.0 } else { 1.0 };
                }
                let dcg: f64 = sorted[..n]
                    .iter()
                    .enumerate()
                    .map(|(i, &idx)| discount(i) * self.gain(labels[idx]))
                    .sum();
                dcg / idcg
            }
            Score::Map => {
                let mut n_hits = 0.0f64;
                let mut acc = 0.0f64;
                for (i, &idx) in sorted[..n].iter().enumerate() {
                    let p = labels[idx] as f64;
                    n_hits += p;
                    acc += n_hits / (i + 1) as f64 * p;
                }
                // Documents past the cut-off still count towards the number of
                // relevant ones the average is divided by.
                let mut total_hits = n_hits;
                for &idx in &sorted[n..] {
                    total_hits += labels[idx] as f64;
                }
                if total_hits > 0.0 {
                    acc / total_hits.min(self.top_k() as f64)
                } else if self.minus {
                    0.0
                } else {
                    1.0
                }
            }
        }
    }
}

/// `ltr::CalcDCGDiscount`.
#[inline]
fn discount(idx: usize) -> f64 {
    1.0 / ((idx as f64) + 2.0).log2()
}

impl Metric for RankScore {
    fn name(&self) -> &str {
        &self.name
    }

    fn eval(&self, preds: &[f32], info: &MetaInfo) -> f64 {
        let groups = info.groups();
        let n_groups = groups.len();
        let group_weight = |g: usize| -> f64 {
            match &info.weights {
                Some(w) if w.len() == n_groups => w[g] as f64,
                _ => 1.0,
            }
        };
        let mut sum = 0.0f64;
        let mut sw = 0.0f64;
        for (g, &(begin, end)) in groups.iter().enumerate() {
            let w = group_weight(g);
            sum += self.score_group(&preds[begin..end], &info.labels[begin..end]) * w;
            sw += w;
        }
        if sw <= 0.0 { 0.0 } else { (sum / sw).min(1.0) }
    }
}

// ------------------------------------------------------------------- AMS ----

/// `ams@ratio` — the approximate median significance from the Higgs challenge.
///
/// Unlike every other metric here it is *maximised*, and it scans the ranked
/// list for the threshold that maximises significance rather than scoring a
/// fixed one.
#[derive(Clone, Debug)]
pub struct Ams {
    ratio: f32,
    name: String,
}

impl Ams {
    pub fn new(ratio: f32) -> Self {
        Self { ratio, name: format!("ams@{ratio}") }
    }
}

impl Metric for Ams {
    fn name(&self) -> &str {
        &self.name
    }

    fn eval(&self, preds: &[f32], info: &MetaInfo) -> f64 {
        let ndata = preds.len();
        if ndata == 0 {
            return 0.0;
        }
        let sorted = by_descending_score(preds);
        let mut ntop = (self.ratio as f64 * ndata as f64) as usize;
        if ntop == 0 {
            ntop = ndata;
        }
        // The regularisation term of the AMS formula.
        const BR: f64 = 10.0;
        let (mut s_tp, mut b_fp, mut tams) = (0.0f64, 0.0f64, 0.0f64);
        let ams = |s: f64, b: f64| (2.0 * ((s + b + BR) * (1.0 + s / (b + BR)).ln() - s)).sqrt();

        for i in 0..(ndata - 1).min(ntop) {
            let ridx = sorted[i];
            let wt = info.weight(ridx) as f64;
            if info.labels[ridx] > 0.5 {
                s_tp += wt;
            } else {
                b_fp += wt;
            }
            // Only consider a cut where the score actually changes.
            if preds[ridx] != preds[sorted[i + 1]] {
                tams = tams.max(ams(s_tp, b_fp));
            }
        }
        if ntop == ndata { tams } else { ams(s_tp, b_fp) }
    }
}

// `ams@t` used to be the one metric off against the pinned 3.4.0 oracle. It is
// **not** the arithmetic above, which reproduces upstream to the last digit;
// the fixture was asking an unanswerable question.
//
// The cut is `ratio * n` rows down the list sorted by prediction, and upstream
// sorts with `std::sort` on the prediction *alone*. When the cut lands inside a
// run of equal predictions — which is the normal case for a tree, whose
// predictions take only `n_leaves` distinct values — which rows sit above it is
// decided by libstdc++'s introsort, not by XGBoost, and nothing outside that
// binary can reproduce it. Every candidate `ntop` was checked: for the old
// fixture, upstream's value is not the AMS of *any* prefix in index order,
// which is exactly what a different tie permutation looks like.
//
// The fixture now asks a question with an answer: a deep unregularised fit on
// a drawn (not computed) binary label, cut at a quarter, has distinct
// predictions either side of the cut in every round — and there the two agree
// exactly. See the `ams@` branch of `tools/gen_string_param_fixtures.py`.

#[cfg(test)]
mod tests {
    use super::*;

    fn binary(labels: &[f32]) -> MetaInfo {
        MetaInfo {
            num_row: labels.len(),
            num_col: 1,
            labels: labels.to_vec(),
            num_target: 1,
            ..Default::default()
        }
    }

    fn ranked(labels: &[f32], sizes: &[usize]) -> MetaInfo {
        let mut ptr = vec![0usize];
        let mut acc = 0;
        for s in sizes {
            acc += s;
            ptr.push(acc);
        }
        MetaInfo { group_ptr: ptr, ..binary(labels) }
    }

    #[test]
    fn a_perfect_ranking_scores_one_and_a_reversed_one_zero() {
        let m = Auc::roc();
        let d = binary(&[0.0, 0.0, 1.0, 1.0]);
        assert!((m.eval(&[0.1, 0.2, 0.8, 0.9], &d) - 1.0).abs() < 1e-9);
        assert!(m.eval(&[0.9, 0.8, 0.2, 0.1], &d).abs() < 1e-9);
    }

    #[test]
    fn a_random_ranking_scores_a_half() {
        let m = Auc::roc();
        // Perfectly interleaved: every positive beats exactly half the negatives.
        let d = binary(&[1.0, 0.0, 1.0, 0.0]);
        let got = m.eval(&[4.0, 3.0, 2.0, 1.0], &d);
        assert!((got - 0.75).abs() < 1e-9, "{got}");
        let got = m.eval(&[1.0, 1.0, 1.0, 1.0], &d);
        assert!((got - 0.5).abs() < 1e-9, "all-tied predictions score 0.5, got {got}");
    }

    #[test]
    fn a_degenerate_label_set_has_no_auc() {
        let m = Auc::roc();
        assert!(m.eval(&[0.1, 0.2], &binary(&[1.0, 1.0])).is_nan());
    }

    #[test]
    fn aucpr_rewards_positives_at_the_top() {
        let m = Auc::pr();
        let d = binary(&[1.0, 1.0, 0.0, 0.0]);
        let good = m.eval(&[0.9, 0.8, 0.2, 0.1], &d);
        let bad = m.eval(&[0.1, 0.2, 0.8, 0.9], &d);
        assert!(good > bad, "{good} vs {bad}");
        assert!((good - 1.0).abs() < 1e-6, "a perfect ranking gives 1, got {good}");
    }

    #[test]
    fn ndcg_is_one_for_the_ideal_order() {
        let m = RankScore::ndcg(None, false);
        let d = ranked(&[3.0, 2.0, 1.0, 0.0], &[4]);
        assert!((m.eval(&[4.0, 3.0, 2.0, 1.0], &d) - 1.0).abs() < 1e-9);
        assert!(m.eval(&[1.0, 2.0, 3.0, 4.0], &d) < 1.0);
    }

    #[test]
    fn the_cut_off_limits_what_ndcg_looks_at() {
        let top1 = RankScore::ndcg(Some(1), false);
        let d = ranked(&[1.0, 0.0, 0.0, 0.0], &[4]);
        // The one relevant document is ranked first: perfect at any cut-off.
        assert!((top1.eval(&[4.0, 3.0, 2.0, 1.0], &d) - 1.0).abs() < 1e-9);
        // Ranked last: nothing relevant inside the cut-off.
        assert!(top1.eval(&[1.0, 2.0, 3.0, 4.0], &d).abs() < 1e-9);
    }

    #[test]
    fn the_minus_variant_changes_only_the_empty_case() {
        // No relevant documents at all: the ideal DCG is zero.
        let d = ranked(&[0.0, 0.0], &[2]);
        assert_eq!(RankScore::ndcg(None, false).eval(&[1.0, 2.0], &d), 1.0);
        assert_eq!(RankScore::ndcg(None, true).eval(&[1.0, 2.0], &d), 0.0);
        assert_eq!(RankScore::map(None, false).eval(&[1.0, 2.0], &d), 1.0);
        assert_eq!(RankScore::map(None, true).eval(&[1.0, 2.0], &d), 0.0);
    }

    #[test]
    fn precision_counts_relevant_documents_in_the_cut_off() {
        let m = RankScore::precision(Some(2));
        let d = ranked(&[1.0, 1.0, 0.0, 0.0], &[4]);
        assert_eq!(m.eval(&[4.0, 3.0, 2.0, 1.0], &d), 1.0);
        assert_eq!(m.eval(&[4.0, 1.0, 3.0, 2.0], &d), 0.5);
    }

    #[test]
    fn map_averages_precision_at_each_hit() {
        let m = RankScore::map(None, false);
        let d = ranked(&[1.0, 0.0, 1.0, 0.0], &[4]);
        // Hits at ranks 1 and 3: (1/1 + 2/3) / 2.
        let got = m.eval(&[4.0, 3.0, 2.0, 1.0], &d);
        assert!((got - (1.0 + 2.0 / 3.0) / 2.0).abs() < 1e-9, "{got}");
    }

    #[test]
    fn ranking_metrics_average_over_queries() {
        let m = RankScore::ndcg(None, false);
        // The first query is perfect, the second reversed.
        let d = ranked(&[1.0, 0.0, 1.0, 0.0], &[2, 2]);
        let got = m.eval(&[2.0, 1.0, 1.0, 2.0], &d);
        assert!(got > 0.0 && got < 1.0, "{got}");
    }

    #[test]
    fn ams_is_maximised_by_a_clean_separation() {
        let m = Ams::new(0.15);
        let d = binary(&[1.0, 1.0, 0.0, 0.0]);
        let good = m.eval(&[0.9, 0.8, 0.2, 0.1], &d);
        let bad = m.eval(&[0.1, 0.2, 0.8, 0.9], &d);
        assert!(good > bad, "{good} vs {bad}");
        assert_eq!(m.name(), "ams@0.15");
    }
}
