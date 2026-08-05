//! The two constraint systems that change which splits a node may take:
//! `monotone_constraints` and `interaction_constraints`.
//!
//! Ports of `xgboost::tree::TreeEvaluator` (`src/tree/split_evaluator.h`) and
//! `xgboost::FeatureInteractionConstraintHost` (`src/tree/constraints.cc`).
//!
//! Monotonicity is enforced two ways at once, and needs both to hold:
//!
//! * a split whose children are ordered the wrong way round scores `-inf`, so
//!   it is never chosen;
//! * every node carries a `[lower, upper]` box that its weight is clipped
//!   into, and a split hands each child a box narrowed at the midpoint of the
//!   two child weights. The clip is what stops a *later*, deeper split from
//!   undoing the ordering an earlier one established.

use super::param::{GradStats, TrainParam, calc_gain_given_weight, calc_weight};
use crate::parameters::MonotoneConstraint;

/// Per-node weight bounds plus the per-feature monotonicity directions.
#[derive(Clone, Debug, Default)]
pub struct SplitEvaluator {
    /// `-1`, `0` or `+1` per feature. Empty when nothing is constrained.
    monotone: Vec<i32>,
    lower: Vec<f32>,
    upper: Vec<f32>,
    has_constraint: bool,
}

impl SplitEvaluator {
    /// Build for `n_features`, from the public constraint list.
    ///
    /// A list of all-zero constraints is the same as no constraints, which is
    /// the check `TrainParam::HasMonotone` makes.
    pub fn new(constraints: &[MonotoneConstraint], n_features: usize) -> Self {
        let mut monotone: Vec<i32> = constraints.iter().map(direction).collect();
        monotone.resize(n_features, 0);
        let has_constraint = monotone.iter().any(|&c| c != 0);
        let (lower, upper) = if has_constraint {
            (vec![f32::MIN; 256], vec![f32::MAX; 256])
        } else {
            (Vec::new(), Vec::new())
        };
        Self { monotone, lower, upper, has_constraint }
    }

    /// Whether any feature is constrained.
    pub fn has_constraint(&self) -> bool {
        self.has_constraint
    }

    /// Reset the bounds for a new tree, keeping the allocation.
    pub fn reset(&mut self) {
        self.lower.fill(f32::MIN);
        self.upper.fill(f32::MAX);
    }

    fn ensure_bounds(&mut self, max_nid: usize) {
        let needed = max_nid * 2 + 1;
        if self.lower.len() < needed {
            self.lower.resize(needed, f32::MIN);
            self.upper.resize(needed, f32::MAX);
        }
    }

    /// Clip a weight into node `nid`'s box.
    #[inline]
    fn apply_bounds(&self, nid: usize, w: f32) -> f32 {
        if !self.has_constraint {
            return w;
        }
        w.clamp(self.lower[nid], self.upper[nid])
    }

    /// Optimal weight for `stats` at node `nid`, respecting the node's box.
    #[inline]
    pub fn calc_weight(&self, nid: usize, p: &TrainParam, stats: &GradStats) -> f32 {
        self.apply_bounds(nid, calc_weight(p, stats))
    }

    /// Gain of `stats` evaluated at its own bounded weight.
    #[inline]
    pub fn calc_gain(&self, nid: usize, p: &TrainParam, stats: &GradStats) -> f32 {
        calc_gain_given_weight(p, stats, self.calc_weight(nid, p, stats))
    }

    /// Gain of splitting node `nid` on `fidx` into `left` and `right`, or
    /// `-inf` when the split is not allowed.
    ///
    /// `-inf` covers both rejections upstream folds in here: a child below
    /// `min_child_weight`, and a monotone constraint the child weights violate.
    #[inline]
    pub fn calc_split_gain(
        &self,
        nid: usize,
        fidx: u32,
        p: &TrainParam,
        left: &GradStats,
        right: &GradStats,
    ) -> f32 {
        if !is_valid_split(p, left, right) {
            return f32::NEG_INFINITY;
        }
        let wleft = self.calc_weight(nid, p, left);
        let wright = self.calc_weight(nid, p, right);
        let gain =
            calc_gain_given_weight(p, left, wleft) + calc_gain_given_weight(p, right, wright);

        if !self.has_constraint {
            return gain;
        }
        match self.monotone[fidx as usize] {
            0 => gain,
            c if c > 0 => {
                if wleft <= wright {
                    gain
                } else {
                    f32::NEG_INFINITY
                }
            }
            _ => {
                if wleft >= wright {
                    gain
                } else {
                    f32::NEG_INFINITY
                }
            }
        }
    }

    /// Hand both children of a split their weight boxes.
    ///
    /// `left_weight` and `right_weight` are the child weights *before* the
    /// learning rate is applied, as upstream's `TreeEvaluator::AddSplit`
    /// receives them.
    pub fn add_split(
        &mut self,
        nid: usize,
        left: usize,
        right: usize,
        fidx: u32,
        left_weight: f32,
        right_weight: f32,
    ) {
        if !self.has_constraint {
            return;
        }
        self.ensure_bounds(left.max(right));

        let (lower, upper) = (self.lower[nid], self.upper[nid]);
        self.lower[left] = lower;
        self.upper[left] = upper;
        self.lower[right] = lower;
        self.upper[right] = upper;

        let mid = (left_weight + right_weight) / 2.0;
        debug_assert!(!mid.is_nan(), "a split midpoint cannot be NaN");
        match self.monotone[fidx as usize] {
            c if c < 0 => {
                self.lower[left] = mid;
                self.upper[right] = mid;
            }
            c if c > 0 => {
                self.upper[left] = mid;
                self.lower[right] = mid;
            }
            _ => {}
        }
    }
}

/// `tree::IsValidSplit`: both children must carry hessian at all, and at least
/// `min_child_weight` of it.
///
/// The `> 0` half only bites once some rows carry no hessian — which is exactly
/// what row sampling does — or once `min_child_weight` is set to zero.
#[inline]
fn is_valid_split(p: &TrainParam, left: &GradStats, right: &GradStats) -> bool {
    left.sum_hess > 0.0
        && right.sum_hess > 0.0
        && left.sum_hess >= p.min_child_weight as f64
        && right.sum_hess >= p.min_child_weight as f64
}

/// The `MonotoneConstraint` as the integer XGBoost stores.
fn direction(c: &MonotoneConstraint) -> i32 {
    match c {
        MonotoneConstraint::Decreasing => -1,
        MonotoneConstraint::Unconstrained => 0,
        MonotoneConstraint::Increasing => 1,
    }
}

/// A set of feature indices as a bitmask.
///
/// Membership is asked once per candidate split — the innermost loop of the
/// fit — so the representation is a flat bit array rather than a hash or tree
/// set: a query is one shift and one mask, and a whole set is a handful of
/// words for any realistic feature count.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct FeatureMask {
    words: Vec<u64>,
}

impl FeatureMask {
    fn empty(n_features: usize) -> Self {
        Self { words: vec![0; n_features.div_ceil(64)] }
    }

    fn full(n_features: usize) -> Self {
        let mut mask = Self::empty(n_features);
        for f in 0..n_features as u32 {
            mask.insert(f);
        }
        mask
    }

    #[inline]
    fn insert(&mut self, feature: u32) {
        let (word, bit) = (feature as usize / 64, feature as usize % 64);
        if word < self.words.len() {
            self.words[word] |= 1 << bit;
        }
    }

    #[inline]
    fn contains(&self, feature: u32) -> bool {
        let (word, bit) = (feature as usize / 64, feature as usize % 64);
        self.words.get(word).is_some_and(|w| w & (1 << bit) != 0)
    }

    #[inline]
    fn union_with(&mut self, other: &Self) {
        for (a, b) in self.words.iter_mut().zip(&other.words) {
            *a |= b;
        }
    }

    /// Whether every feature in `self` is also in `other`.
    #[inline]
    fn is_subset_of(&self, other: &Self) -> bool {
        self.words.iter().zip(&other.words).all(|(a, b)| a & !b == 0)
    }
}

/// Which features each node may split on, given the interaction groups.
///
/// A node may use a feature when it is either already on the path from the
/// root, or in a group that contains every feature on that path.
#[derive(Clone, Debug, Default)]
pub struct InteractionConstraints {
    groups: Vec<FeatureMask>,
    /// Features permitted at each node.
    allowed: Vec<FeatureMask>,
    /// Features split on at each node and its ancestors.
    path: Vec<FeatureMask>,
    n_features: usize,
    enabled: bool,
}

impl InteractionConstraints {
    /// Build from the public constraint groups. `None` disables the system.
    pub fn new(groups: Option<&Vec<Vec<u32>>>, n_features: usize) -> Self {
        let to_mask = |group: &Vec<u32>| {
            let mut mask = FeatureMask::empty(n_features);
            for &feature in group {
                mask.insert(feature);
            }
            mask
        };
        let mut out = Self {
            groups: groups.map(|gs| gs.iter().map(to_mask).collect()).unwrap_or_default(),
            allowed: Vec::new(),
            path: Vec::new(),
            n_features,
            enabled: groups.is_some_and(|g| !g.is_empty()),
        };
        out.reset();
        out
    }

    /// Whether any constraint is in force.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Start a new tree: the root may split on anything.
    pub fn reset(&mut self) {
        if !self.enabled {
            return;
        }
        self.allowed = vec![FeatureMask::full(self.n_features)];
        self.path = vec![FeatureMask::empty(self.n_features)];
    }

    /// Whether node `nid` may split on `fidx`.
    #[inline]
    pub fn query(&self, nid: usize, fidx: u32) -> bool {
        if !self.enabled {
            return true;
        }
        self.allowed[nid].contains(fidx)
    }

    /// Record that `nid` split on `fidx`, and derive both children's permitted
    /// feature sets.
    pub fn split(&mut self, nid: usize, fidx: u32, left: usize, right: usize) {
        if !self.enabled {
            return;
        }
        let new_size = left.max(right) + 1;

        let mut path = self.path[nid].clone();
        path.insert(fidx);
        if self.path.len() < new_size {
            self.path.resize(new_size, FeatureMask::empty(self.n_features));
            self.allowed.resize(new_size, FeatureMask::empty(self.n_features));
        }

        let mut allowed = path.clone();
        for group in &self.groups {
            // A group stays relevant only while it covers everything split on
            // so far; once it does not, this subtree can never enter it again.
            if path.is_subset_of(group) {
                allowed.union_with(group);
            }
        }

        self.path[left] = path.clone();
        self.path[right] = path;
        self.allowed[left] = allowed.clone();
        self.allowed[right] = allowed;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn param() -> TrainParam {
        TrainParam { reg_lambda: 1.0, min_child_weight: 0.0, ..Default::default() }
    }

    #[test]
    fn an_unconstrained_evaluator_changes_nothing() {
        let e = SplitEvaluator::new(&[], 3);
        assert!(!e.has_constraint());
        let p = param();
        let s = GradStats::new(4.0, 3.0);
        assert_eq!(e.calc_weight(0, &p, &s), calc_weight(&p, &s));
        // All-zero constraints are the same as none at all.
        assert!(!SplitEvaluator::new(&[MonotoneConstraint::Unconstrained; 3], 3).has_constraint());
    }

    #[test]
    fn a_violating_split_scores_negative_infinity() {
        let e = SplitEvaluator::new(&[MonotoneConstraint::Increasing], 1);
        let p = param();
        // Left weight above right weight contradicts an increasing constraint.
        let heavy_left = GradStats::new(-8.0, 1.0); // weight +4
        let light_right = GradStats::new(8.0, 1.0); // weight -4
        assert_eq!(e.calc_split_gain(0, 0, &p, &heavy_left, &light_right), f32::NEG_INFINITY);
        // The same split the other way round is fine.
        assert!(e.calc_split_gain(0, 0, &p, &light_right, &heavy_left).is_finite());
    }

    #[test]
    fn a_decreasing_constraint_is_the_mirror_image() {
        let e = SplitEvaluator::new(&[MonotoneConstraint::Decreasing], 1);
        let p = param();
        let high = GradStats::new(-8.0, 1.0);
        let low = GradStats::new(8.0, 1.0);
        assert!(e.calc_split_gain(0, 0, &p, &high, &low).is_finite());
        assert_eq!(e.calc_split_gain(0, 0, &p, &low, &high), f32::NEG_INFINITY);
    }

    #[test]
    fn an_unconstrained_feature_is_unaffected_by_its_neighbours() {
        let e = SplitEvaluator::new(
            &[MonotoneConstraint::Increasing, MonotoneConstraint::Unconstrained],
            2,
        );
        let p = param();
        let a = GradStats::new(-8.0, 1.0);
        let b = GradStats::new(8.0, 1.0);
        assert_eq!(e.calc_split_gain(0, 0, &p, &a, &b), f32::NEG_INFINITY);
        assert!(e.calc_split_gain(0, 1, &p, &a, &b).is_finite());
    }

    #[test]
    fn a_split_narrows_its_children_boxes() {
        let mut e = SplitEvaluator::new(&[MonotoneConstraint::Increasing], 1);
        let p = param();
        e.add_split(0, 1, 2, 0, -2.0, 4.0); // midpoint 1.0
        // Increasing: the left child is capped at the midpoint, the right
        // child floored at it.
        assert_eq!(e.calc_weight(1, &p, &GradStats::new(-100.0, 1.0)), 1.0);
        assert_eq!(e.calc_weight(2, &p, &GradStats::new(100.0, 1.0)), 1.0);
        // And weights already inside the box are untouched.
        assert!(e.calc_weight(1, &p, &GradStats::new(1.0, 1.0)) < 1.0);
    }

    #[test]
    fn boxes_are_inherited_and_grow_only_tighter() {
        let mut e = SplitEvaluator::new(&[MonotoneConstraint::Increasing], 1);
        e.add_split(0, 1, 2, 0, -2.0, 4.0); // left upper = 1
        e.add_split(1, 3, 4, 0, -4.0, -2.0); // midpoint -3 inside [MIN, 1]
        let p = param();
        // Node 3 inherits the parent's cap and takes the tighter new one.
        assert_eq!(e.calc_weight(3, &p, &GradStats::new(-100.0, 1.0)), -3.0);
        // Node 4 keeps the grandparent's cap of 1.
        assert_eq!(e.calc_weight(4, &p, &GradStats::new(-100.0, 1.0)), 1.0);
    }

    #[test]
    fn min_child_weight_rejects_before_any_constraint_is_consulted() {
        let e = SplitEvaluator::new(&[], 1);
        let p = TrainParam { min_child_weight: 5.0, ..param() };
        let tiny = GradStats::new(1.0, 1.0);
        let big = GradStats::new(1.0, 100.0);
        assert_eq!(e.calc_split_gain(0, 0, &p, &tiny, &big), f32::NEG_INFINITY);
    }

    /// A child with no hessian at all is rejected even when `min_child_weight`
    /// would allow it — the case row sampling creates.
    #[test]
    fn a_child_without_hessian_is_never_a_split() {
        let e = SplitEvaluator::new(&[], 1);
        let p = TrainParam { min_child_weight: 0.0, ..param() };
        let empty = GradStats::new(0.0, 0.0);
        let full = GradStats::new(1.0, 10.0);
        assert_eq!(e.calc_split_gain(0, 0, &p, &empty, &full), f32::NEG_INFINITY);
        assert_eq!(e.calc_split_gain(0, 0, &p, &full, &empty), f32::NEG_INFINITY);
        assert!(e.calc_split_gain(0, 0, &p, &full, &full).is_finite());
    }

    #[test]
    fn no_interaction_groups_permit_everything() {
        let c = InteractionConstraints::new(None, 5);
        assert!(!c.is_enabled());
        for f in 0..5 {
            assert!(c.query(0, f));
        }
    }

    #[test]
    fn a_split_restricts_children_to_its_own_groups() {
        let groups = vec![vec![0, 1], vec![2, 3, 4]];
        let mut c = InteractionConstraints::new(Some(&groups), 5);
        assert!(c.is_enabled());
        // The root is unrestricted.
        for f in 0..5 {
            assert!(c.query(0, f));
        }

        c.split(0, 0, 1, 2);
        // Having split on feature 0, only its group — plus feature 0 itself —
        // stays available.
        assert!(c.query(1, 0));
        assert!(c.query(1, 1));
        for f in 2..5 {
            assert!(!c.query(1, f), "feature {f} interacts with 0 across groups");
            assert!(!c.query(2, f));
        }
    }

    #[test]
    fn a_feature_in_two_groups_keeps_both_open() {
        let groups = vec![vec![0, 1], vec![0, 2]];
        let mut c = InteractionConstraints::new(Some(&groups), 4);
        c.split(0, 0, 1, 2);
        assert!(c.query(1, 1), "group [0,1] still covers the path");
        assert!(c.query(1, 2), "group [0,2] does too");
        assert!(!c.query(1, 3), "feature 3 is in no group with 0");

        // Descending into feature 1 rules the second group out.
        c.split(1, 1, 3, 4);
        assert!(c.query(3, 0));
        assert!(c.query(3, 1));
        assert!(!c.query(3, 2), "the path {{0,1}} is not covered by [0,2]");
    }

    #[test]
    fn a_feature_outside_every_group_isolates_its_subtree() {
        let groups = vec![vec![0, 1]];
        let mut c = InteractionConstraints::new(Some(&groups), 4);
        c.split(0, 3, 1, 2);
        assert!(c.query(1, 3), "a feature can always be split on again");
        for f in [0u32, 1, 2] {
            assert!(!c.query(1, f), "feature 3 is in no group, so nothing may join it");
        }
    }

    #[test]
    fn resetting_returns_the_root_to_unrestricted() {
        let groups = vec![vec![0, 1]];
        let mut c = InteractionConstraints::new(Some(&groups), 4);
        c.split(0, 0, 1, 2);
        assert!(!c.query(1, 2));
        c.reset();
        assert!(c.query(0, 2));
    }
}
