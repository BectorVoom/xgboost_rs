//! The decision tree model — XGBoost's `RegTree`, single-target.
//!
//! Node 0 is the root. A node is a leaf exactly when it has no left child;
//! leaves store their output in `value`, internal nodes store the split
//! threshold there, matching upstream's union and its JSON encoding (both are
//! serialised under `split_conditions`).

/// Marker for "no child", as upstream's `kInvalidNodeId`.
pub const INVALID_NODE: i32 = -1;

/// `kDeletedNodeMarker`: the split index a pruned-away node is stamped with.
///
/// Upstream writes `UINT32_MAX` into the packed `(default_left, split_index)`
/// word, which reads back as split index `2^31 - 1` with the default-left flag
/// set. Both halves are reproduced so a pruned tree serialises byte for byte.
pub const DELETED_SPLIT_INDEX: u32 = (1 << 31) - 1;

/// One entry of the TreeSHAP "unique path", upstream's `PathElement`.
///
/// `zero_fraction` and `one_fraction` are the fractions of subsets in which the
/// feature is absent and present; `pweight` is the proportion of subsets of a
/// given size that reach this point.
#[derive(Clone, Copy, Debug, Default)]
struct PathElement {
    feature_index: i32,
    zero_fraction: f32,
    one_fraction: f32,
    pweight: f32,
}

/// `ExtendPath`: grow the subset-weight polynomial by one feature.
fn extend_path(
    path: &mut [PathElement],
    unique_depth: usize,
    zero_fraction: f32,
    one_fraction: f32,
    feature_index: i32,
) {
    path[unique_depth] = PathElement {
        feature_index,
        zero_fraction,
        one_fraction,
        pweight: if unique_depth == 0 { 1.0 } else { 0.0 },
    };
    let depth = unique_depth as f32;
    for i in (0..unique_depth).rev() {
        path[i + 1].pweight += one_fraction * path[i].pweight * (i as f32 + 1.0) / (depth + 1.0);
        path[i].pweight = zero_fraction * path[i].pweight * (depth - i as f32) / (depth + 1.0);
    }
}

/// `UnwindPath`: undo one [`extend_path`], removing `path_index`.
fn unwind_path(path: &mut [PathElement], unique_depth: usize, path_index: usize) {
    let one_fraction = path[path_index].one_fraction;
    let zero_fraction = path[path_index].zero_fraction;
    let mut next_one_portion = path[unique_depth].pweight;
    let depth = unique_depth as f32;

    for i in (0..unique_depth).rev() {
        if one_fraction != 0.0 {
            let tmp = path[i].pweight;
            path[i].pweight = next_one_portion * (depth + 1.0) / ((i as f32 + 1.0) * one_fraction);
            next_one_portion =
                tmp - path[i].pweight * zero_fraction * (depth - i as f32) / (depth + 1.0);
        } else if zero_fraction != 0.0 {
            path[i].pweight *= (depth + 1.0) / (zero_fraction * (depth - i as f32));
        }
    }
    for i in path_index..unique_depth {
        path[i].feature_index = path[i + 1].feature_index;
        path[i].zero_fraction = path[i + 1].zero_fraction;
        path[i].one_fraction = path[i + 1].one_fraction;
    }
}

/// `UnwoundPathSum`: the total subset weight the path would have without
/// `path_index`, without actually modifying it.
fn unwound_path_sum(path: &[PathElement], unique_depth: usize, path_index: usize) -> f32 {
    let one_fraction = path[path_index].one_fraction;
    let zero_fraction = path[path_index].zero_fraction;
    let mut next_one_portion = path[unique_depth].pweight;
    let depth = unique_depth as f32;
    let mut total = 0.0f32;

    for i in (0..unique_depth).rev() {
        if one_fraction != 0.0 {
            let tmp = next_one_portion * (depth + 1.0) / ((i as f32 + 1.0) * one_fraction);
            total += tmp;
            next_one_portion = path[i].pweight - tmp * zero_fraction * (depth - i as f32) / (depth + 1.0);
        } else if zero_fraction != 0.0 {
            total += (path[i].pweight / zero_fraction) / ((depth - i as f32) / (depth + 1.0));
        }
    }
    total
}

/// A single tree node.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Node {
    pub parent: i32,
    pub left: i32,
    pub right: i32,
    /// Split feature index; meaningless for leaves.
    pub split_index: u32,
    pub default_left: bool,
    /// Split threshold for internal nodes, output value for leaves.
    pub value: f32,
}

impl Default for Node {
    fn default() -> Self {
        Self {
            parent: INVALID_NODE,
            left: INVALID_NODE,
            right: INVALID_NODE,
            split_index: 0,
            default_left: false,
            value: 0.0,
        }
    }
}

impl Node {
    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.left == INVALID_NODE
    }
}

/// Per-node training statistics, upstream's `RTreeNodeStat`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NodeStat {
    pub loss_chg: f32,
    pub sum_hess: f32,
    pub base_weight: f32,
}

/// A single regression tree.
#[derive(Clone, Debug)]
pub struct RegTree {
    pub nodes: Vec<Node>,
    pub stats: Vec<NodeStat>,
    /// Depth of each node, indexed by node id.
    depths: Vec<i32>,
    /// Whether a node has been pruned away. A deleted node keeps its slot so
    /// the ids around it never move.
    deleted: Vec<bool>,
    num_feature: usize,
}

impl RegTree {
    /// A tree consisting of a single leaf with value `0.0`.
    pub fn new(num_feature: usize) -> Self {
        Self {
            nodes: vec![Node::default()],
            stats: vec![NodeStat::default()],
            depths: vec![0],
            deleted: vec![false],
            num_feature,
        }
    }

    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn num_feature(&self) -> usize {
        self.num_feature
    }

    pub fn depth(&self, nid: usize) -> i32 {
        self.depths[nid]
    }

    /// Number of leaves in the tree, ignoring the slots pruning retired.
    pub fn num_leaves(&self) -> usize {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(nid, n)| n.is_leaf() && !self.deleted[*nid])
            .count()
    }

    /// Set a leaf's output value.
    pub fn set_leaf(&mut self, nid: usize, value: f32) {
        self.nodes[nid].value = value;
    }

    /// Split leaf `nid`, appending its two children.
    ///
    /// `left_leaf_weight`/`right_leaf_weight` are already scaled by the
    /// learning rate; `base_weight` is not.
    #[allow(clippy::too_many_arguments)]
    pub fn expand_node(
        &mut self,
        nid: usize,
        split_index: u32,
        split_value: f32,
        default_left: bool,
        base_weight: f32,
        left_leaf_weight: f32,
        right_leaf_weight: f32,
        loss_change: f32,
        sum_hess: f32,
        left_sum: f32,
        right_sum: f32,
    ) {
        debug_assert!(self.nodes[nid].is_leaf(), "cannot re-split an internal node");
        let pleft = self.alloc_node(nid as i32);
        let pright = self.alloc_node(nid as i32);

        let node = &mut self.nodes[nid];
        node.left = pleft as i32;
        node.right = pright as i32;
        node.split_index = split_index;
        node.default_left = default_left;
        node.value = split_value;

        self.nodes[pleft].value = left_leaf_weight;
        self.nodes[pright].value = right_leaf_weight;

        self.stats[nid] = NodeStat { loss_chg: loss_change, sum_hess, base_weight };
        self.stats[pleft] =
            NodeStat { loss_chg: 0.0, sum_hess: left_sum, base_weight: left_leaf_weight };
        self.stats[pright] =
            NodeStat { loss_chg: 0.0, sum_hess: right_sum, base_weight: right_leaf_weight };
    }

    /// `TreePruner::DoPrune`: collapse every split whose loss reduction is
    /// below `min_split_loss`, or that sits deeper than `max_depth`.
    ///
    /// Pruning is bottom-up and repeats: collapsing a split can leave its own
    /// parent with two leaf children, which may then be prunable in turn. The
    /// return value is the number of nodes removed.
    ///
    /// A removed node keeps its slot, marked deleted, rather than being
    /// renumbered away. That is what upstream does, and it is what keeps a
    /// pruned tree's node ids — which the saved model records — identical to
    /// the reference implementation's.
    pub fn prune(&mut self, min_split_loss: f32, max_depth: i32, learning_rate: f32) -> usize {
        let mut pruned = 0usize;
        for nid in 0..self.nodes.len() {
            if !self.deleted[nid] && self.nodes[nid].is_leaf() {
                pruned +=
                    self.try_prune_leaf(nid, self.depths[nid], min_split_loss, max_depth, learning_rate);
            }
        }
        pruned
    }

    /// `TreePruner::TryPruneLeaf`, walking up from a leaf for as long as each
    /// parent qualifies.
    fn try_prune_leaf(
        &mut self,
        nid: usize,
        depth: i32,
        min_split_loss: f32,
        max_depth: i32,
        learning_rate: f32,
    ) -> usize {
        let mut nid = nid;
        let mut depth = depth;
        let mut pruned = 0usize;
        loop {
            let parent = self.nodes[nid].parent;
            if parent < 0 {
                return pruned;
            }
            let pid = parent as usize;
            let (left, right) = (self.nodes[pid].left, self.nodes[pid].right);
            // Only a parent whose *both* children are already leaves can go.
            let balanced = left != INVALID_NODE
                && right != INVALID_NODE
                && self.nodes[left as usize].is_leaf()
                && self.nodes[right as usize].is_leaf();
            let stat = self.stats[pid];
            let need_prune =
                stat.loss_chg < min_split_loss || (max_depth != 0 && depth > max_depth);
            if !(balanced && need_prune) {
                return pruned;
            }
            self.delete_node(left as usize);
            self.delete_node(right as usize);
            self.nodes[pid].left = INVALID_NODE;
            self.nodes[pid].right = INVALID_NODE;
            self.nodes[pid].value = learning_rate * stat.base_weight;
            pruned += 2;
            nid = pid;
            depth -= 1;
        }
    }

    /// `RegTree::DeleteNode`: retire a node's slot without moving anything.
    fn delete_node(&mut self, nid: usize) {
        self.deleted[nid] = true;
        self.nodes[nid].split_index = DELETED_SPLIT_INDEX;
        self.nodes[nid].default_left = true;
    }

    /// Whether node `nid` has been pruned away.
    #[inline]
    pub fn is_deleted(&self, nid: usize) -> bool {
        self.deleted[nid]
    }

    /// Nodes still occupying a slot but no longer part of the tree.
    pub fn num_deleted(&self) -> usize {
        self.deleted.iter().filter(|d| **d).count()
    }

    /// Mark every node unreachable from the root as deleted, which is how a
    /// loaded model recovers the flags `num_deleted` only counts.
    pub(crate) fn recompute_deleted(&mut self) {
        let mut reachable = vec![false; self.nodes.len()];
        let mut stack = vec![0usize];
        while let Some(nid) = stack.pop() {
            if nid >= self.nodes.len() || reachable[nid] {
                continue;
            }
            reachable[nid] = true;
            if !self.nodes[nid].is_leaf() {
                stack.push(self.nodes[nid].left as usize);
                stack.push(self.nodes[nid].right as usize);
            }
        }
        self.deleted = reachable.into_iter().map(|r| !r).collect();
    }

    /// Recompute node depths from the parent links, after loading a model.
    pub(crate) fn recompute_depths(&mut self) {
        self.deleted.resize(self.nodes.len(), false);
        self.depths = vec![0; self.nodes.len()];
        for nid in 0..self.nodes.len() {
            let parent = self.nodes[nid].parent;
            if parent >= 0 {
                self.depths[nid] = self.depths[parent as usize] + 1;
            }
        }
    }

    fn alloc_node(&mut self, parent: i32) -> usize {
        let nid = self.nodes.len();
        self.nodes.push(Node { parent, ..Default::default() });
        self.stats.push(NodeStat::default());
        self.deleted.push(false);
        let depth = if parent >= 0 { self.depths[parent as usize] + 1 } else { 0 };
        self.depths.push(depth);
        nid
    }

    /// Cover-weighted mean prediction of the subtree at each node,
    /// upstream's `node_mean_values_`.
    ///
    /// This is the expected value SHAP attributions are measured against: the
    /// value the tree would predict for a row about which nothing is known.
    pub fn node_mean_values(&self) -> Vec<f32> {
        let mut out = vec![0.0f32; self.nodes.len()];
        // Children always have a higher id than their parent, so one reverse
        // pass suffices and no recursion is needed.
        for nid in (0..self.nodes.len()).rev() {
            let node = self.nodes[nid];
            if node.is_leaf() {
                out[nid] = node.value;
                continue;
            }
            let (l, r) = (node.left as usize, node.right as usize);
            let (wl, wr) = (self.stats[l].sum_hess, self.stats[r].sum_hess);
            let total = wl + wr;
            out[nid] = if total > 0.0 {
                (out[l] * wl + out[r] * wr) / total
            } else {
                (out[l] + out[r]) / 2.0
            };
        }
        out
    }

    /// Add this tree's exact TreeSHAP contributions for one row to `out`,
    /// whose last slot is the bias.
    ///
    /// The algorithm is the path-dependent TreeSHAP of Lundberg et al., the one
    /// `RegTree::CalculateContributions` runs: the sum over features of what it
    /// adds is `leaf value - expected value`, and the expected value goes into
    /// the bias, so the whole row still sums to the prediction.
    pub fn add_shap_contributions<F>(&self, get: &F, weight: f32, out: &mut [f32])
    where
        F: Fn(u32) -> Option<f32>,
    {
        debug_assert_eq!(out.len(), self.num_feature + 1);
        let means: Vec<f32> = self.node_mean_values().iter().map(|v| v * weight).collect();
        out[self.num_feature] += means[0];
        if self.nodes[0].is_leaf() {
            return;
        }
        let mut path = vec![PathElement::default(); self.max_depth() + 2];
        self.tree_shap(get, weight, 0, 0, &mut path, 1.0, 1.0, -1, out);
    }

    /// Add this tree's approximate ("Saabas") contributions for one row.
    ///
    /// Each split on the path the row takes is credited with the whole change
    /// in the subtree's expected value, which is what `approx_contribs` asks
    /// for: much cheaper than TreeSHAP, and order-dependent rather than exact.
    pub fn add_saabas_contributions<F>(&self, get: &F, weight: f32, out: &mut [f32])
    where
        F: Fn(u32) -> Option<f32>,
    {
        debug_assert_eq!(out.len(), self.num_feature + 1);
        let means: Vec<f32> = self.node_mean_values().iter().map(|v| v * weight).collect();
        out[self.num_feature] += means[0];

        let mut nid = 0usize;
        let mut node_value = means[0];
        while !self.nodes[nid].is_leaf() {
            let node = self.nodes[nid];
            let split_index = node.split_index as usize;
            nid = self.next_node(nid, get(node.split_index));
            let new_value = means[nid];
            out[split_index] += new_value - node_value;
            node_value = new_value;
        }
    }

    /// Depth of the deepest node, used to size the SHAP path buffer.
    fn max_depth(&self) -> usize {
        self.depths.iter().copied().max().unwrap_or(0) as usize
    }

    /// The child a row moves to from `nid`.
    #[inline]
    fn next_node(&self, nid: usize, value: Option<f32>) -> usize {
        let node = self.nodes[nid];
        match value {
            Some(v) => {
                if v < node.value {
                    node.left as usize
                } else {
                    node.right as usize
                }
            }
            None => {
                if node.default_left {
                    node.left as usize
                } else {
                    node.right as usize
                }
            }
        }
    }

    /// The TreeSHAP recursion. `unique_depth` counts the distinct features
    /// already on the path.
    #[allow(clippy::too_many_arguments)]
    fn tree_shap<F>(
        &self,
        get: &F,
        weight: f32,
        node_index: usize,
        unique_depth: usize,
        parent_path: &mut Vec<PathElement>,
        parent_zero_fraction: f32,
        parent_one_fraction: f32,
        parent_feature_index: i32,
        phi: &mut [f32],
    ) where
        F: Fn(u32) -> Option<f32>,
    {
        // Each level works on its own copy of the path, because the two
        // children extend it differently.
        let mut path = parent_path.clone();
        if path.len() < unique_depth + 2 {
            path.resize(unique_depth + 2, PathElement::default());
        }
        extend_path(
            &mut path,
            unique_depth,
            parent_zero_fraction,
            parent_one_fraction,
            parent_feature_index,
        );

        let node = self.nodes[node_index];
        if node.is_leaf() {
            for i in 1..=unique_depth {
                let w = unwound_path_sum(&path, unique_depth, i);
                let element = path[i];
                phi[element.feature_index as usize] +=
                    w * (element.one_fraction - element.zero_fraction) * node.value * weight;
            }
            return;
        }

        let hot = self.next_node(node_index, get(node.split_index));
        let cold = if hot == node.left as usize { node.right as usize } else { node.left as usize };
        let cover = self.stats[node_index].sum_hess;
        let (hot_zero, cold_zero) = if cover > 0.0 {
            (self.stats[hot].sum_hess / cover, self.stats[cold].sum_hess / cover)
        } else {
            (0.5, 0.5)
        };

        let mut incoming_zero_fraction = 1.0f32;
        let mut incoming_one_fraction = 1.0f32;
        let mut unique_depth = unique_depth;

        // Splitting twice on one feature does not add a new path element; the
        // earlier one is unwound and folded into this split instead.
        let mut path_index = 1usize;
        while path_index <= unique_depth {
            if path[path_index].feature_index == node.split_index as i32 {
                break;
            }
            path_index += 1;
        }
        if path_index <= unique_depth {
            incoming_zero_fraction = path[path_index].zero_fraction;
            incoming_one_fraction = path[path_index].one_fraction;
            unwind_path(&mut path, unique_depth, path_index);
            unique_depth -= 1;
        }

        self.tree_shap(
            get,
            weight,
            hot,
            unique_depth + 1,
            &mut path,
            hot_zero * incoming_zero_fraction,
            incoming_one_fraction,
            node.split_index as i32,
            phi,
        );
        self.tree_shap(
            get,
            weight,
            cold,
            unique_depth + 1,
            &mut path,
            cold_zero * incoming_zero_fraction,
            0.0,
            node.split_index as i32,
            phi,
        );
    }

    /// Leaf index reached by a row, given a lookup for feature values.
    ///
    /// `get` returns `None` for missing features, which then follow the node's
    /// default direction.
    #[inline]
    pub fn leaf_index<F>(&self, get: F) -> usize
    where
        F: Fn(u32) -> Option<f32>,
    {
        let mut nid = 0usize;
        loop {
            let node = &self.nodes[nid];
            if node.is_leaf() {
                return nid;
            }
            nid = match get(node.split_index) {
                Some(v) => {
                    if v < node.value {
                        node.left as usize
                    } else {
                        node.right as usize
                    }
                }
                None => {
                    if node.default_left {
                        node.left as usize
                    } else {
                        node.right as usize
                    }
                }
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stump() -> RegTree {
        let mut t = RegTree::new(2);
        t.expand_node(0, 1, 0.5, true, 0.25, -1.0, 2.0, 3.0, 10.0, 4.0, 6.0);
        t
    }

    #[test]
    fn expanding_a_leaf_creates_two_children() {
        let t = stump();
        assert_eq!(t.num_nodes(), 3);
        assert_eq!(t.num_leaves(), 2);
        assert!(!t.nodes[0].is_leaf());
        assert_eq!(t.nodes[1].parent, 0);
        assert_eq!(t.depth(1), 1);
        assert_eq!(t.stats[0], NodeStat { loss_chg: 3.0, sum_hess: 10.0, base_weight: 0.25 });
        assert_eq!(t.stats[2], NodeStat { loss_chg: 0.0, sum_hess: 6.0, base_weight: 2.0 });
    }

    #[test]
    fn traversal_splits_on_less_than() {
        let t = stump();
        assert_eq!(t.leaf_index(|f| if f == 1 { Some(0.4) } else { None }), 1);
        // The boundary value itself goes right: the test is `v < split_value`.
        assert_eq!(t.leaf_index(|f| if f == 1 { Some(0.5) } else { None }), 2);
    }

    #[test]
    fn missing_values_follow_the_default_direction() {
        let mut t = stump();
        assert_eq!(t.leaf_index(|_| None), 1, "default_left sends missing left");
        t.nodes[0].default_left = false;
        assert_eq!(t.leaf_index(|_| None), 2);
    }
}
