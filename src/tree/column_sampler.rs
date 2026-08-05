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

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::rng::{Mt19937, shuffle};

/// A shared, ascending list of candidate feature indices.
pub type FeatureSet = Arc<Vec<u32>>;

/// Draws the feature sets a tree, a level and a node may split on.
#[derive(Clone, Debug, Default)]
pub struct ColumnSampler {
    bytree: f32,
    bylevel: f32,
    bynode: f32,
    tree_set: FeatureSet,
    /// Level samples, cached by depth for the tree currently being grown.
    level_sets: BTreeMap<i32, FeatureSet>,
}

impl ColumnSampler {
    pub fn new(bynode: f32, bylevel: f32, bytree: f32) -> Self {
        Self {
            bytree,
            bylevel,
            bynode,
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
        self.tree_set = col_sample(&Arc::new(all), self.bytree, rng);
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
            let level = col_sample(&self.tree_set, self.bylevel, rng);
            self.level_sets.insert(depth, level);
        }
        let level = Arc::clone(&self.level_sets[&depth]);
        if self.bynode == 1.0 {
            return level;
        }
        col_sample(&level, self.bynode, rng)
    }
}

/// Keep `max(1, colsample * n)` of `features`, chosen uniformly without
/// replacement and returned in ascending order.
///
/// A ratio of exactly 1 returns the input untouched *and consumes no
/// randomness*, which is what keeps an unsampled level from shifting the
/// engine and changing every later draw.
fn col_sample(features: &FeatureSet, colsample: f32, rng: &mut Mt19937) -> FeatureSet {
    if colsample == 1.0 {
        return Arc::clone(features);
    }
    let n = ((colsample * features.len() as f32) as i32).max(1) as usize;

    // Upstream seeds a fresh engine from the session engine rather than
    // shuffling with the session engine directly.
    let seed = rng.next_u32();
    let mut local = Mt19937::new(seed);

    let mut sampled = features.as_ref().clone();
    shuffle(&mut sampled, &mut local);
    sampled.truncate(n);
    sampled.sort_unstable();
    Arc::new(sampled)
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
}
