//! Model serialisation in XGBoost's JSON format.
//!
//! The layout matches `Booster.save_model(...json)` from XGBoost 3.x, so a
//! model written here loads in the reference implementation and vice versa.
//! Numeric parameters are encoded as strings, as upstream's DMLC parameter
//! serialisation does.

use crate::api::Booster;
use crate::gbm::{GBTree, GBTreeModel};
use crate::learner::Learner;
use crate::tree::model::{INVALID_NODE, Node, NodeStat, RegTree};
use crate::tree::param::TrainParam;
use crate::{Error, Result};
use serde_json::{Value, json};

/// XGBoost writes the root's parent as `kInvalidNodeId` widened to unsigned.
const ROOT_PARENT: i64 = 2147483647;

/// Serialise a booster to XGBoost's JSON model format.
pub fn save_model(booster: &Booster) -> String {
    let learner = booster.learner();
    let model = &learner.gbm().model;

    let trees: Vec<Value> = model.trees.iter().enumerate().map(|(i, t)| tree_to_json(i, t)).collect();
    let iteration_indptr: Vec<usize> = (0..=model.trees.len()).collect();
    let tree_info = vec![0u32; model.trees.len()];

    let doc = json!({
        "learner": {
            "attributes": {},
            "feature_names": [],
            "feature_types": [],
            "gradient_booster": {
                "model": {
                    "gbtree_model_param": {
                        "num_parallel_tree": "1",
                        "num_trees": model.trees.len().to_string(),
                    },
                    "iteration_indptr": iteration_indptr,
                    "tree_info": tree_info,
                    "trees": trees,
                },
                "name": "gbtree",
            },
            "learner_model_param": {
                "base_score": format_f32(learner.base_score()),
                "boost_from_average": "1",
                "num_class": "0",
                "num_feature": model.num_feature.to_string(),
                "num_target": "1",
            },
            "objective": {
                "name": learner.objective().name(),
                "reg_loss_param": { "scale_pos_weight": "1" },
            },
        },
        "version": [3, 0, 5],
    });
    doc.to_string()
}

fn tree_to_json(id: usize, tree: &RegTree) -> Value {
    let n = tree.num_nodes();
    let mut left = Vec::with_capacity(n);
    let mut right = Vec::with_capacity(n);
    let mut parents = Vec::with_capacity(n);
    let mut split_indices = Vec::with_capacity(n);
    let mut split_conditions = Vec::with_capacity(n);
    let mut default_left = Vec::with_capacity(n);
    let mut base_weights = Vec::with_capacity(n);
    let mut loss_changes = Vec::with_capacity(n);
    let mut sum_hessian = Vec::with_capacity(n);

    for (nid, node) in tree.nodes.iter().enumerate() {
        left.push(node.left as i64);
        right.push(node.right as i64);
        parents.push(if node.parent == INVALID_NODE { ROOT_PARENT } else { node.parent as i64 });
        split_indices.push(node.split_index);
        split_conditions.push(node.value);
        default_left.push(u8::from(node.default_left));
        base_weights.push(tree.stats[nid].base_weight);
        loss_changes.push(tree.stats[nid].loss_chg);
        sum_hessian.push(tree.stats[nid].sum_hess);
    }

    json!({
        "base_weights": base_weights,
        "categories": [],
        "categories_nodes": [],
        "categories_segments": [],
        "categories_sizes": [],
        "default_left": default_left,
        "id": id,
        "left_children": left,
        "loss_changes": loss_changes,
        "parents": parents,
        "right_children": right,
        "split_conditions": split_conditions,
        "split_indices": split_indices,
        "split_type": vec![0u8; n],
        "sum_hessian": sum_hessian,
        "tree_param": {
            "num_deleted": "0",
            "num_feature": tree.num_feature().to_string(),
            "num_nodes": n.to_string(),
            "size_leaf_vector": "1",
        },
    })
}

/// Parse a model written by [`save_model`] or by XGBoost.
pub fn load_model(text: &str) -> Result<Booster> {
    let doc: Value = serde_json::from_str(text)
        .map_err(|e| Error::ModelFormat(format!("not valid JSON: {e}")))?;
    let learner_json = doc
        .get("learner")
        .ok_or_else(|| Error::ModelFormat("missing `learner`".into()))?;

    let booster_name = learner_json
        .pointer("/gradient_booster/name")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::ModelFormat("missing gradient_booster name".into()))?;
    if booster_name != "gbtree" {
        return Err(Error::ModelFormat(format!("unsupported booster `{booster_name}`")));
    }

    let param = learner_json
        .get("learner_model_param")
        .ok_or_else(|| Error::ModelFormat("missing learner_model_param".into()))?;
    let base_score = parse_str_f32(param, "base_score")?;
    let num_feature = parse_str_usize(param, "num_feature")?;

    let objective_name = learner_json
        .pointer("/objective/name")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::ModelFormat("missing objective name".into()))?;

    let trees_json = learner_json
        .pointer("/gradient_booster/model/trees")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::ModelFormat("missing trees".into()))?;
    let mut trees = Vec::with_capacity(trees_json.len());
    for t in trees_json {
        trees.push(tree_from_json(t, num_feature)?);
    }

    let obj = crate::objective::create(objective_name)?;
    let metric = crate::metric::create(obj.default_metric())?;
    let mut gbm = GBTree::new(num_feature, TrainParam::default());
    gbm.model = GBTreeModel { trees, num_feature };

    Ok(Booster::from_learner(Learner::from_model(obj, metric, gbm, base_score)))
}

fn tree_from_json(t: &Value, num_feature: usize) -> Result<RegTree> {
    let left = int_array(t, "left_children")?;
    let right = int_array(t, "right_children")?;
    let parents = int_array(t, "parents")?;
    let split_indices = int_array(t, "split_indices")?;
    let split_conditions = float_array(t, "split_conditions")?;
    let default_left = int_array(t, "default_left")?;
    let base_weights = float_array(t, "base_weights")?;
    let loss_changes = float_array(t, "loss_changes")?;
    let sum_hessian = float_array(t, "sum_hessian")?;

    let n = left.len();
    for (name, len) in [
        ("right_children", right.len()),
        ("parents", parents.len()),
        ("split_indices", split_indices.len()),
        ("split_conditions", split_conditions.len()),
        ("default_left", default_left.len()),
        ("base_weights", base_weights.len()),
        ("loss_changes", loss_changes.len()),
        ("sum_hessian", sum_hessian.len()),
    ] {
        if len != n {
            return Err(Error::ModelFormat(format!(
                "tree array `{name}` has {len} entries, expected {n}"
            )));
        }
    }

    let mut tree = RegTree::new(num_feature);
    tree.nodes = (0..n)
        .map(|i| Node {
            parent: if parents[i] == ROOT_PARENT { INVALID_NODE } else { parents[i] as i32 },
            left: left[i] as i32,
            right: right[i] as i32,
            split_index: split_indices[i] as u32,
            default_left: default_left[i] != 0,
            value: split_conditions[i],
        })
        .collect();
    tree.stats = (0..n)
        .map(|i| NodeStat {
            loss_chg: loss_changes[i],
            sum_hess: sum_hessian[i],
            base_weight: base_weights[i],
        })
        .collect();
    tree.recompute_depths();
    Ok(tree)
}

fn format_f32(v: f32) -> String {
    // Rust's shortest round-trip formatting; XGBoost's `strtof` reads it back
    // exactly, which is what matters for a round trip.
    v.to_string()
}

fn parse_str_f32(v: &Value, key: &str) -> Result<f32> {
    v.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| Error::ModelFormat(format!("missing `{key}`")))?
        .parse::<f32>()
        .map_err(|e| Error::ModelFormat(format!("`{key}` is not a float: {e}")))
}

fn parse_str_usize(v: &Value, key: &str) -> Result<usize> {
    v.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| Error::ModelFormat(format!("missing `{key}`")))?
        .parse::<usize>()
        .map_err(|e| Error::ModelFormat(format!("`{key}` is not an integer: {e}")))
}

fn int_array(t: &Value, key: &str) -> Result<Vec<i64>> {
    t.get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| Error::ModelFormat(format!("missing tree array `{key}`")))?
        .iter()
        .map(|x| x.as_i64().ok_or_else(|| Error::ModelFormat(format!("`{key}` holds a non-integer"))))
        .collect()
}

fn float_array(t: &Value, key: &str) -> Result<Vec<f32>> {
    t.get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| Error::ModelFormat(format!("missing tree array `{key}`")))?
        .iter()
        .map(|x| {
            x.as_f64()
                .map(|f| f as f32)
                .ok_or_else(|| Error::ModelFormat(format!("`{key}` holds a non-number")))
        })
        .collect()
}
