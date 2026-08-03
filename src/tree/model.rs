//! The decision tree model — XGBoost's `RegTree`, single-target.
//!
//! Node 0 is the root. A node is a leaf exactly when it has no left child;
//! leaves store their output in `value`, internal nodes store the split
//! threshold there, matching upstream's union and its JSON encoding (both are
//! serialised under `split_conditions`).

/// Marker for "no child", as upstream's `kInvalidNodeId`.
pub const INVALID_NODE: i32 = -1;

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
    num_feature: usize,
}

impl RegTree {
    /// A tree consisting of a single leaf with value `0.0`.
    pub fn new(num_feature: usize) -> Self {
        Self {
            nodes: vec![Node::default()],
            stats: vec![NodeStat::default()],
            depths: vec![0],
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

    /// Number of leaves in the tree.
    pub fn num_leaves(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_leaf()).count()
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

    /// Recompute node depths from the parent links, after loading a model.
    pub(crate) fn recompute_depths(&mut self) {
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
        let depth = if parent >= 0 { self.depths[parent as usize] + 1 } else { 0 };
        self.depths.push(depth);
        nid
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
