//! Column sampling — `colsample_bytree`, `colsample_bylevel`, `colsample_bynode`.
//!
//! A port of `xgboost::common::ColumnSampler` (`src/common/random.h`). The
//! three ratios nest: the per-tree sample is drawn from every feature, the
//! per-level sample from the tree's, and the per-node sample from the level's.
//! Each sample keeps `max(1, ratio * n)` features, so a fit never runs out of
//! candidates however small the ratios are.
//!
//! Samples are drawn by shuffling and truncating, then sorted back into
//! ascending feature order — which matters, because split ties are broken
//! towards the lower feature index.
//!
//! When the matrix carries `feature_weights` the shuffle is replaced by
//! Efraimidis–Spirakis weighted sampling without replacement, exactly as
//! `common::WeightedSamplingWithoutReplacement` does it, so a heavier column is
//! likelier to survive every one of the three samples.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::rng::{Mt19937, canonical_f32_from_mt, shuffle};

use super::param::RT_EPS;

/// A shared, ascending list of candidate feature indices.
pub type FeatureSet = Arc<Vec<u32>>;

/// Draws the feature sets a tree, a level and a node may split on.
#[derive(Clone, Debug, Default)]
pub struct ColumnSampler {
    bytree: f32,
    bylevel: f32,
    bynode: f32,
    /// One weight per *feature index*, or empty for uniform sampling.
    feature_weights: Arc<Vec<f32>>,
    tree_set: FeatureSet,
    /// Level samples, cached by depth for the tree currently being grown.
    level_sets: BTreeMap<i32, FeatureSet>,
}

impl ColumnSampler {
    /// Uniform sampling, the `feature_weights`-free case.
    pub fn new(bynode: f32, bylevel: f32, bytree: f32) -> Self {
        Self::weighted(bynode, bylevel, bytree, &[])
    }

    /// Sampling weighted by `feature_weights`, indexed by feature. An empty
    /// slice is the uniform case, as an empty `MetaInfo::feature_weights` is
    /// upstream.
    pub fn weighted(bynode: f32, bylevel: f32, bytree: f32, feature_weights: &[f32]) -> Self {
        Self {
            bytree,
            bylevel,
            bynode,
            feature_weights: Arc::new(feature_weights.to_vec()),
            tree_set: Arc::new(Vec::new()),
            level_sets: BTreeMap::new(),
        }
    }

    /// Whether any of the three ratios actually samples.
    pub fn is_sampling(&self) -> bool {
        self.bytree < 1.0 || self.bylevel < 1.0 || self.bynode < 1.0
    }

    /// Start a new tree: clear the cached level samples and redraw the tree's
    /// own feature set. Called once per tree, as `ColumnSampler::Init` is.
    pub fn reset(&mut self, num_col: usize, rng: &mut Mt19937) {
        self.level_sets.clear();
        let all: Vec<u32> = (0..num_col as u32).collect();
        self.tree_set = self.col_sample(&Arc::new(all), self.bytree, rng);
    }

    /// The features a node at `depth` may split on.
    ///
    /// Must be called exactly once per node, because with `colsample_bynode < 1`
    /// every call draws a fresh sample.
    pub fn feature_set(&mut self, depth: i32, rng: &mut Mt19937) -> FeatureSet {
        if self.bylevel == 1.0 && self.bynode == 1.0 {
            return Arc::clone(&self.tree_set);
        }
        if !self.level_sets.contains_key(&depth) {
            let level = self.col_sample(&self.tree_set, self.bylevel, rng);
            self.level_sets.insert(depth, level);
        }
        let level = Arc::clone(&self.level_sets[&depth]);
        if self.bynode == 1.0 {
            return level;
        }
        self.col_sample(&level, self.bynode, rng)
    }

    /// Keep `max(1, colsample * n)` of `features`, chosen without replacement
    /// and returned in ascending order.
    ///
    /// A ratio of exactly 1 returns the input untouched *and consumes no
    /// randomness*, which is what keeps an unsampled level from shifting the
    /// engine and changing every later draw.
    fn col_sample(&self, features: &FeatureSet, colsample: f32, rng: &mut Mt19937) -> FeatureSet {
        if colsample == 1.0 {
            return Arc::clone(features);
        }
        let n = ((colsample * features.len() as f32) as i32).max(1) as usize;

        // Upstream seeds a fresh engine from the session engine rather than
        // shuffling with the session engine directly.
        let seed = rng.next_u32();
        let mut local = Mt19937::new(seed);

        let mut sampled = if self.feature_weights.is_empty() {
            let mut all = features.as_ref().clone();
            shuffle(&mut all, &mut local);
            all.truncate(n);
            all
        } else {
            weighted_sample(features, &self.feature_weights, n, &mut local)
        };
        sampled.sort_unstable();
        Arc::new(sampled)
    }
}

/// `common::WeightedSamplingWithoutReplacement`: Efraimidis–Spirakis sampling
/// in its logarithmic form.
///
/// Each candidate draws `u ~ U[0, 1)` and takes the key `ln(u) / w`; the `n`
/// largest keys win. Because `ln(u)` is negative, a large weight divides it
/// towards zero and so ranks higher — which is the whole trick, and why the
/// weight floor matters: at `w = 0` the key would be `-inf` (or NaN, at
/// `u = 0`) rather than merely last.
///
/// Ties are broken towards the lower position, matching the stable sort
/// upstream's `ArgSort` runs.
fn weighted_sample(
    features: &FeatureSet,
    feature_weights: &[f32],
    n: usize,
    rng: &mut Mt19937,
) -> Vec<u32> {
    let mut keyed: Vec<(f32, usize)> = features
        .iter()
        .enumerate()
        .map(|(i, &feature)| {
            let w = feature_weights[feature as usize].max(RT_EPS);
            let u = canonical_f32_from_mt(rng);
            (u.ln() / w, i)
        })
        .collect();

    // Descending by key, ties by ascending position. `f32::total_cmp` orders
    // the `-inf` a `u` of exactly zero produces without any NaN special case.
    keyed.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    keyed.truncate(n);
    keyed.into_iter().map(|(_, i)| features[i]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sampler(bynode: f32, bylevel: f32, bytree: f32) -> (ColumnSampler, Mt19937) {
        (ColumnSampler::new(bynode, bylevel, bytree), Mt19937::new(0))
    }

    #[test]
    fn no_sampling_returns_every_feature() {
        let (mut cs, mut rng) = sampler(1.0, 1.0, 1.0);
        assert!(!cs.is_sampling());
        cs.reset(8, &mut rng);
        let set = cs.feature_set(0, &mut rng);
        assert_eq!(*set, (0..8).collect::<Vec<u32>>());
        // And it consumed nothing, so the engine is untouched.
        assert_eq!(rng.next_u32(), Mt19937::new(0).next_u32());
    }

    #[test]
    fn bytree_fixes_one_set_for_the_whole_tree() {
        let (mut cs, mut rng) = sampler(1.0, 1.0, 0.5);
        cs.reset(10, &mut rng);
        let a = cs.feature_set(0, &mut rng);
        let b = cs.feature_set(3, &mut rng);
        assert_eq!(a.len(), 5);
        assert_eq!(*a, *b, "without level or node sampling every node shares the tree set");
        assert!(a.windows(2).all(|w| w[0] < w[1]), "must be ascending and unique");
    }

    #[test]
    fn bylevel_draws_once_per_depth_from_the_tree_set() {
        let (mut cs, mut rng) = sampler(1.0, 0.5, 1.0);
        cs.reset(10, &mut rng);
        let d0 = cs.feature_set(0, &mut rng);
        let d0_again = cs.feature_set(0, &mut rng);
        let d1 = cs.feature_set(1, &mut rng);
        assert_eq!(d0.len(), 5);
        assert_eq!(*d0, *d0_again, "a depth is sampled once, then cached");
        assert_ne!(*d0, *d1, "a new depth draws again");
    }

    #[test]
    fn bynode_draws_for_every_node() {
        let (mut cs, mut rng) = sampler(0.5, 1.0, 1.0);
        cs.reset(20, &mut rng);
        let a = cs.feature_set(0, &mut rng);
        let b = cs.feature_set(0, &mut rng);
        assert_eq!(a.len(), 10);
        assert_ne!(*a, *b, "two nodes at one depth get their own samples");
    }

    #[test]
    fn the_ratios_nest() {
        let (mut cs, mut rng) = sampler(0.5, 0.5, 0.5);
        cs.reset(64, &mut rng);
        let tree = Arc::clone(&cs.tree_set);
        assert_eq!(tree.len(), 32);
        let node = cs.feature_set(0, &mut rng);
        assert_eq!(node.len(), 8, "0.5 * 0.5 * 0.5 of 64");
        assert!(node.iter().all(|f| tree.contains(f)), "a node samples inside its tree set");
    }

    #[test]
    fn a_tiny_ratio_still_leaves_one_feature() {
        let (mut cs, mut rng) = sampler(1.0, 1.0, 0.001);
        cs.reset(10, &mut rng);
        assert_eq!(cs.feature_set(0, &mut rng).len(), 1);
    }

    #[test]
    fn resetting_starts_a_new_tree() {
        let (mut cs, mut rng) = sampler(1.0, 0.5, 1.0);
        cs.reset(10, &mut rng);
        let first_tree_d0 = cs.feature_set(0, &mut rng);
        cs.reset(10, &mut rng);
        let second_tree_d0 = cs.feature_set(0, &mut rng);
        assert_ne!(*first_tree_d0, *second_tree_d0, "level caches must not survive a reset");
    }

    #[test]
    fn sampling_is_reproducible_from_the_engine_state() {
        let draw = |seed: u32| {
            let mut cs = ColumnSampler::new(0.5, 0.5, 0.5);
            let mut rng = Mt19937::new(seed);
            cs.reset(32, &mut rng);
            (*cs.feature_set(0, &mut rng)).clone()
        };
        assert_eq!(draw(1), draw(1));
        assert_ne!(draw(1), draw(2));
    }

    // ------------------------------------------------- feature weights ----

    /// How often each feature survives `trials` independent per-tree samples.
    fn survival_counts(weights: &[f32], bytree: f32, trials: u32) -> Vec<u32> {
        let n = weights.len();
        let mut counts = vec![0u32; n];
        for seed in 0..trials {
            let mut cs = ColumnSampler::weighted(1.0, 1.0, bytree, weights);
            let mut rng = Mt19937::new(seed);
            cs.reset(n, &mut rng);
            for &f in cs.feature_set(0, &mut rng).iter() {
                counts[f as usize] += 1;
            }
        }
        counts
    }

    #[test]
    fn weights_still_produce_a_valid_sample() {
        let weights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut cs = ColumnSampler::weighted(1.0, 1.0, 0.5, &weights);
        let mut rng = Mt19937::new(0);
        cs.reset(8, &mut rng);
        let set = cs.feature_set(0, &mut rng);

        assert_eq!(set.len(), 4, "the ratio decides the size, not the weights");
        assert!(set.windows(2).all(|w| w[0] < w[1]), "ascending and unique: {set:?}");
        assert!(set.iter().all(|&f| f < 8));
    }

    /// The point of the feature: a heavier column is drawn more often. Checked
    /// as an ordering over many trials rather than against a pinned sequence,
    /// because the draw is only statistically defined.
    #[test]
    fn a_heavier_feature_is_sampled_more_often() {
        let weights = [1.0, 2.0, 4.0, 8.0];
        let counts = survival_counts(&weights, 0.5, 400);
        assert!(
            counts.windows(2).all(|w| w[0] < w[1]),
            "survival should rise with the weight, got {counts:?}"
        );
    }

    /// A zero weight is floored at `kRtEps`, not excluded — upstream's
    /// behaviour, and the reason this is a preference rather than a mask.
    #[test]
    fn a_zero_weight_is_last_in_line_but_not_forbidden() {
        let weights = [0.0, 1.0, 1.0, 1.0];
        let counts = survival_counts(&weights, 0.5, 400);
        assert!(counts[0] < counts[1] / 10, "a zero weight should be rare, got {counts:?}");
        assert!(
            counts[1..].iter().all(|&c| c > 0),
            "the weighted features must still be drawn, got {counts:?}"
        );
    }

    /// Equal weights are not the *same* draw as no weights — they take a
    /// different route through the engine — but they must be unbiased.
    #[test]
    fn equal_weights_do_not_favour_any_feature() {
        let counts = survival_counts(&[3.0; 6], 0.5, 1200);
        let expected: i32 = 1200 * 3 / 6;
        for (f, &c) in counts.iter().enumerate() {
            let drift = (c as i32 - expected).abs();
            assert!(drift < expected / 4, "feature {f}: {c} vs ~{expected} ({counts:?})");
        }
    }

    #[test]
    fn weighted_sampling_is_reproducible_and_seed_dependent() {
        let weights = [1.0, 5.0, 2.0, 9.0, 3.0, 7.0, 4.0, 6.0];
        let draw = |seed: u32| {
            let mut cs = ColumnSampler::weighted(0.5, 1.0, 0.5, &weights);
            let mut rng = Mt19937::new(seed);
            cs.reset(8, &mut rng);
            (*cs.feature_set(0, &mut rng)).clone()
        };
        assert_eq!(draw(1), draw(1));
        assert_ne!(draw(1), draw(9));
    }

    /// A ratio of 1 short-circuits before any draw, weighted or not, so
    /// weights never cost randomness a fit would otherwise spend elsewhere.
    #[test]
    fn weights_change_nothing_when_nothing_is_sampled() {
        let mut cs = ColumnSampler::weighted(1.0, 1.0, 1.0, &[1.0, 2.0, 3.0, 4.0]);
        let mut rng = Mt19937::new(0);
        cs.reset(4, &mut rng);
        assert_eq!(*cs.feature_set(0, &mut rng), vec![0, 1, 2, 3]);
        assert_eq!(rng.next_u32(), Mt19937::new(0).next_u32());
    }
}
