//! Model serialisation in XGBoost's JSON format.
//!
//! The layout matches `Booster.save_model(...json)` from XGBoost 3.x, so a
//! model written here loads in the reference implementation and vice versa.
//! Numeric parameters are encoded as strings, as upstream's DMLC parameter
//! serialisation does.

use crate::api::Booster;
use crate::gbm::{Booster as GradientBooster, GBTree, GBTreeModel};
use crate::learner::Learner;
use crate::linear::{GBLinear, GBLinearModel};
use crate::tree::model::{INVALID_NODE, Node, NodeStat, RegTree};
use crate::tree::param::TrainParam;
use crate::{Error, Result};
use serde_json::{Value, json};

/// XGBoost writes the root's parent as `kInvalidNodeId` widened to unsigned.
const ROOT_PARENT: i64 = 2147483647;

/// Serialise a booster to XGBoost's JSON model format.
pub fn save_model(booster: &Booster) -> String {
    let learner = booster.learner();
    match learner.booster() {
        GradientBooster::Tree(_) => save_tree_model(booster),
        GradientBooster::Linear(linear) => save_linear_model(learner, linear),
    }
}

/// Serialise a `gblinear` model. Its `gradient_booster` holds one flat weight
/// array — features then the intercepts — rather than a tree list.
fn save_linear_model(learner: &Learner, linear: &GBLinear) -> String {
    let model = &linear.model;
    let is_multiclass = learner.objective().name().starts_with("multi:");
    let n_groups = model.num_output_group.max(1);
    let (num_class, num_target) = if is_multiclass { (n_groups, 1) } else { (0, n_groups) };

    let doc = json!({
        "learner": {
            "attributes": {},
            "feature_names": [],
            "feature_types": [],
            "gradient_booster": {
                "name": "gblinear",
                "model": {
                    "weights": model.weight,
                    "boosted_rounds": model.num_boosted_rounds,
                },
            },
            "learner_model_param": {
                "base_score": format_base_score(learner.base_scores()),
                "boost_from_average": "1",
                "num_class": num_class.to_string(),
                "num_feature": model.num_feature.to_string(),
                "num_target": num_target.to_string(),
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

fn save_tree_model(booster: &Booster) -> String {
    let learner = booster.learner();
    let model = &learner.gbm().expect("checked by the caller").model;

    let trees: Vec<Value> = model.trees.iter().enumerate().map(|(i, t)| tree_to_json(i, t)).collect();
    // One entry per boosting round, so a round contributes
    // `num_parallel_tree * num_output_group` trees to a single interval.
    let per_round = model.trees_per_round();
    let iteration_indptr: Vec<usize> =
        (0..=model.trees.len() / per_round).map(|i| i * per_round).collect();

    // `num_class` and `num_target` split the output groups the way upstream
    // does: a classification model records classes, anything else targets.
    let is_multiclass = learner.objective().name().starts_with("multi:");
    let n_groups = model.num_output_group.max(1);
    let (num_class, num_target) = if is_multiclass { (n_groups, 1) } else { (0, n_groups) };

    // A DART model is a gbtree model plus the per-tree weights dropout left
    // behind, and it records itself under its own booster name.
    let is_dart = learner.gbm().expect("checked by the caller").is_dart();
    let booster_name = if is_dart { "dart" } else { "gbtree" };
    let weight_drop: Vec<f32> =
        if is_dart { model.tree_weight.clone() } else { Vec::new() };

    let doc = json!({
        "learner": {
            "attributes": {},
            "feature_names": [],
            "feature_types": [],
            "gradient_booster": {
                "model": {
                    "gbtree_model_param": {
                        "num_parallel_tree": model.num_parallel_tree.max(1).to_string(),
                        "num_trees": model.trees.len().to_string(),
                    },
                    "iteration_indptr": iteration_indptr,
                    "tree_info": model.tree_info,
                    "trees": trees,
                },
                "name": booster_name,
                "weight_drop": weight_drop,
            },
            "learner_model_param": {
                "base_score": format_base_score(learner.base_scores()),
                "boost_from_average": "1",
                "num_class": num_class.to_string(),
                "num_feature": model.num_feature.to_string(),
                "num_target": num_target.to_string(),
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

    // `SaveCategoricalSplit`: the bit sets are written out as the category
    // codes they contain, node by node in ascending node order, so the file
    // does not depend on the width of the in-memory bit field.
    let mut categories: Vec<i64> = Vec::new();
    let mut categories_nodes: Vec<i64> = Vec::new();
    let mut categories_segments: Vec<i64> = Vec::new();
    let mut categories_sizes: Vec<i64> = Vec::new();
    let mut split_type = vec![0u8; n];
    for nid in 0..n {
        if !tree.is_categorical_split(nid) {
            continue;
        }
        split_type[nid] = 1;
        let bits = tree.node_categories(nid);
        let begin = categories.len() as i64;
        categories_nodes.push(nid as i64);
        categories_segments.push(begin);
        for code in 0..(bits.len() * 32) as u32 {
            if crate::tree::cat::check_bit(bits, code) {
                categories.push(code as i64);
            }
        }
        categories_sizes.push(categories.len() as i64 - begin);
    }

    json!({
        "base_weights": base_weights,
        "categories": categories,
        "categories_nodes": categories_nodes,
        "categories_segments": categories_segments,
        "categories_sizes": categories_sizes,
        "default_left": default_left,
        "id": id,
        "left_children": left,
        "loss_changes": loss_changes,
        "parents": parents,
        "right_children": right,
        "split_conditions": split_conditions,
        "split_indices": split_indices,
        "split_type": split_type,
        "sum_hessian": sum_hessian,
        // A vector-leaf tree's outputs do not fit in `split_conditions`, which
        // holds one value per node; they ride alongside, as upstream's
        // `MultiTargetTree` writes them.
        "leaf_values": tree.leaf_vectors(),
        "tree_param": {
            "num_deleted": tree.num_deleted().to_string(),
            "num_feature": tree.num_feature().to_string(),
            "num_nodes": n.to_string(),
            "size_leaf_vector": tree.leaf_size().to_string(),
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
    // `dart` is a `gbtree` model plus per-tree weights, so both load here.
    if booster_name == "gblinear" {
        return load_linear_model(learner_json);
    }
    if booster_name != "gbtree" && booster_name != "dart" {
        return Err(Error::ModelFormat(format!("unsupported booster `{booster_name}`")));
    }

    let param = learner_json
        .get("learner_model_param")
        .ok_or_else(|| Error::ModelFormat("missing learner_model_param".into()))?;
    let base_score = parse_base_score(param)?;
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

    let num_parallel_tree = learner_json
        .pointer("/gradient_booster/model/gbtree_model_param/num_parallel_tree")
        .and_then(Value::as_str)
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(1)
        .max(1);

    // Output groups: `num_class` for a classifier, `num_target` otherwise.
    let num_class = parse_str_usize(param, "num_class").unwrap_or(0);
    let num_target = parse_str_usize(param, "num_target").unwrap_or(1);
    let num_output_group = if num_class > 0 { num_class } else { num_target.max(1) };

    let tree_info: Vec<u32> = learner_json
        .pointer("/gradient_booster/model/tree_info")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(|v| v.as_u64()).map(|v| v as u32).collect())
        .unwrap_or_else(|| vec![0; trees.len()]);
    if tree_info.len() != trees.len() {
        return Err(Error::ModelFormat(format!(
            "tree_info has {} entries for {} trees",
            tree_info.len(),
            trees.len()
        )));
    }

    // `dart` records one weight per tree; `gbtree` leaves them all at 1.
    let tree_weight: Vec<f32> = learner_json
        .pointer("/gradient_booster/weight_drop")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_f64).map(|v| v as f32).collect())
        .filter(|w: &Vec<f32>| !w.is_empty())
        .unwrap_or_else(|| vec![1.0; trees.len()]);
    if tree_weight.len() != trees.len() {
        return Err(Error::ModelFormat(format!(
            "weight_drop has {} entries for {} trees",
            tree_weight.len(),
            trees.len()
        )));
    }

    let obj = objective_from_model(objective_name, num_class)?;
    let metric = crate::metric::create(&obj.default_metric())?;
    let mut gbm = GBTree::new(num_feature, TrainParam::default());
    gbm.model =
        GBTreeModel { tree_weight, trees, tree_info, num_feature, num_parallel_tree, num_output_group };

    let learner =
        Learner::from_model(obj, vec![metric], GradientBooster::Tree(gbm), base_score)?;
    Ok(Booster::from_learner(learner))
}

/// Parse a `gblinear` model.
fn load_linear_model(learner_json: &Value) -> Result<Booster> {
    let param = learner_json
        .get("learner_model_param")
        .ok_or_else(|| Error::ModelFormat("missing learner_model_param".into()))?;
    let base_score = parse_base_score(param)?;
    let num_feature = parse_str_usize(param, "num_feature")?;
    let num_class = parse_str_usize(param, "num_class").unwrap_or(0);
    let num_target = parse_str_usize(param, "num_target").unwrap_or(1);
    let num_output_group = if num_class > 0 { num_class } else { num_target.max(1) };

    let objective_name = learner_json
        .pointer("/objective/name")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::ModelFormat("missing objective name".into()))?;

    let weight: Vec<f32> = learner_json
        .pointer("/gradient_booster/model/weights")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::ModelFormat("missing gblinear weights".into()))?
        .iter()
        .map(|v| {
            v.as_f64()
                .map(|f| f as f32)
                .ok_or_else(|| Error::ModelFormat("`weights` holds a non-number".into()))
        })
        .collect::<Result<Vec<f32>>>()?;
    let expected = (num_feature + 1) * num_output_group;
    if weight.len() != expected {
        return Err(Error::ModelFormat(format!(
            "gblinear has {} weights, expected ({num_feature} + 1) * {num_output_group} = \
             {expected}",
            weight.len()
        )));
    }

    let num_boosted_rounds = learner_json
        .pointer("/gradient_booster/model/boosted_rounds")
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize;

    let obj = objective_from_model(objective_name, num_class)?;
    let metric = crate::metric::create(&obj.default_metric())?;
    let model =
        GBLinearModel { weight, num_feature, num_output_group, num_boosted_rounds };
    let learner = Learner::from_model(
        obj,
        vec![metric],
        GradientBooster::Linear(Box::new(GBLinear::from_model(model))),
        base_score,
    )?;
    Ok(Booster::from_learner(learner))
}

/// Rebuild the objective a saved model names, supplying the `num_class` the
/// model header carries rather than the placeholder `from_str` would give.
fn objective_from_model(
    name: &str,
    num_class: usize,
) -> Result<Box<dyn crate::objective::Objective>> {
    use crate::parameters::Objective as Spec;
    let spec: Spec = name.parse()?;
    let spec = match (spec, num_class) {
        (Spec::MultiSoftmax { .. }, k) if k > 0 => Spec::MultiSoftmax { num_class: k as u32 },
        (Spec::MultiSoftprob { .. }, k) if k > 0 => Spec::MultiSoftprob { num_class: k as u32 },
        (other, _) => other,
    };
    crate::objective::create(&spec, 1.0)
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
    // `num_deleted` only counts the retired slots; which ones they are is
    // recovered from the links, since nothing unreachable from the root is
    // part of the tree.
    tree.recompute_deleted();
    load_categorical_split(&mut tree, t, n)?;

    // `size_leaf_vector` above 1 makes this a vector-leaf tree, whose outputs
    // live in their own array rather than in `split_conditions`.
    let leaf_size = t
        .pointer("/tree_param/size_leaf_vector")
        .and_then(|v| match v {
            Value::String(s) => s.parse::<usize>().ok(),
            other => other.as_u64().map(|n| n as usize),
        })
        .unwrap_or(1)
        .max(1);
    if leaf_size > 1 {
        let values = float_array(t, "leaf_values")?;
        if values.len() != n * leaf_size {
            return Err(Error::ModelFormat(format!(
                "`leaf_values` holds {} entries, expected {n} nodes x {leaf_size} outputs",
                values.len()
            )));
        }
        tree.set_leaf_vectors(leaf_size, values);
    }
    Ok(tree)
}

/// `RegTree::LoadCategoricalSplit` — rebuild the per-node category bit sets
/// from the flat code lists the file carries.
///
/// A model with no categorical split omits or empties these arrays, which is
/// what every tree written before this existed looks like, so their absence is
/// not an error.
fn load_categorical_split(tree: &mut RegTree, t: &Value, n: usize) -> Result<()> {
    let split_type = int_array(t, "split_type").unwrap_or_default();
    let nodes = int_array(t, "categories_nodes").unwrap_or_default();
    let segments = int_array(t, "categories_segments").unwrap_or_default();
    let sizes = int_array(t, "categories_sizes").unwrap_or_default();
    let categories = int_array(t, "categories").unwrap_or_default();

    if nodes.is_empty() {
        return Ok(());
    }
    if segments.len() != nodes.len() || sizes.len() != nodes.len() {
        return Err(Error::ModelFormat(format!(
            "categorical arrays disagree: {} nodes, {} segments, {} sizes",
            nodes.len(),
            segments.len(),
            sizes.len()
        )));
    }

    let mut categorical = vec![false; n];
    let mut node_segments = vec![(0u32, 0u32); n];
    let mut storage: Vec<u32> = Vec::new();

    for (i, &nid) in nodes.iter().enumerate() {
        let nid = nid as usize;
        if nid >= n {
            return Err(Error::ModelFormat(format!(
                "categorical split names node {nid} in a tree of {n} nodes"
            )));
        }
        let begin = segments[i] as usize;
        let end = begin + sizes[i] as usize;
        if end > categories.len() {
            return Err(Error::ModelFormat(format!(
                "node {nid} names categories {begin}..{end} of {}",
                categories.len()
            )));
        }
        let codes = &categories[begin..end];
        // The bit field is sized to the largest category it has to hold, which
        // is what upstream reconstructs too — the width is not recorded.
        let max_cat = codes.iter().copied().max().ok_or_else(|| {
            Error::ModelFormat(format!("node {nid} is a categorical split with no categories"))
        })?;
        let base = storage.len() as u32;
        let words = crate::tree::cat::storage_size(max_cat as usize + 1);
        storage.resize(storage.len() + words, 0);
        for &code in codes {
            crate::tree::cat::set_bit(&mut storage[base as usize..], code as u32);
        }
        categorical[nid] = true;
        node_segments[nid] = (base, words as u32);
    }

    // `split_type` is the authority on which nodes are categorical; a
    // disagreement with `categories_nodes` means the file is inconsistent.
    for (nid, &kind) in split_type.iter().enumerate().take(n) {
        if (kind != 0) != categorical[nid] {
            return Err(Error::ModelFormat(format!(
                "node {nid} has split_type {kind} but {} category list",
                if categorical[nid] { "a" } else { "no" }
            )));
        }
    }

    tree.set_categories(categorical, node_segments, storage);
    Ok(())
}

fn format_f32(v: f32) -> String {
    // Rust's shortest round-trip formatting; XGBoost's `strtof` reads it back
    // exactly, which is what matters for a round trip.
    v.to_string()
}

/// `base_score` as `LearnerModelParamLegacy` writes it: a bare number for a
/// single-output model, and the `(a,b,c)` array form when a multi-output fit
/// has one intercept per output.
fn format_base_score(values: &[f32]) -> String {
    match values {
        [] => "0.5".to_owned(),
        [single] => format_f32(*single),
        many => {
            let joined = many.iter().map(|v| format_f32(*v)).collect::<Vec<_>>().join(",");
            format!("({joined})")
        }
    }
}

/// Parse either spelling of `base_score`.
fn parse_base_score(v: &Value) -> Result<Vec<f32>> {
    let text = v
        .get("base_score")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::ModelFormat("missing `base_score`".into()))?;
    let body = text.trim();
    let body = body.strip_prefix('(').and_then(|b| b.strip_suffix(')')).unwrap_or(body);
    body.split(',')
        .map(|part| {
            part.trim()
                .parse::<f32>()
                .map_err(|e| Error::ModelFormat(format!("`base_score` is not a float: {e}")))
        })
        .collect()
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
