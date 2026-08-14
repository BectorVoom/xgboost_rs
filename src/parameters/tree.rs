//! Parameters for the tree boosters (`gbtree` and `dart`).
//!
//! Field-for-field this is XGBoost's `tree::TrainParam`
//! (`src/tree/param.h`), plus the pieces the same fit reads from
//! `GBTreeTrainParam` (`src/gbm/gbtree.h`), `GBTreeModelParam`
//! (`src/gbm/gbtree_model.h`), `HistMakerTrainParam`
//! (`src/tree/hist/hist_param.h`), `ColMakerTrainParam`
//! (`src/tree/updater_colmaker.cc`) and `LearnerTrainParam::multi_strategy`
//! (`src/learner.cc`).

use std::fmt;

use serde::{Deserialize, Serialize};

use super::config::{ConfigEntry, ToConfig, push, push_bool, push_opt};
use super::device::Device;
use super::str_enum::str_enum;
use super::validate;
use crate::error::{Error, Result};

str_enum! {
    /// Tree construction algorithm. `Auto` resolves to `Hist`, as it has since
    /// XGBoost 2.0.
    pub enum TreeMethod: "tree_method" {
        /// Let XGBoost choose; currently identical to `hist`.
        Auto = "auto",
        /// Exact greedy: enumerate every split candidate. CPU only.
        Exact = "exact",
        /// Quantile sketch recomputed per iteration from the current hessians.
        Approx = "approx",
        /// Histogram over fixed quantile bins. The default and the only method
        /// supporting vector-leaf multi-target trees.
        Hist = "hist",
    }
    default = Auto;
}

str_enum! {
    /// Which node the growth driver expands next.
    pub enum GrowPolicy: "grow_policy" {
        /// Split nodes closest to the root first.
        DepthWise = "depthwise",
        /// Split the node with the highest loss change first (LightGBM style).
        LossGuide = "lossguide",
    }
    default = DepthWise;
}

str_enum! {
    /// How rows are drawn when `subsample < 1`.
    pub enum SamplingMethod: "sampling_method" {
        /// Uniform without replacement.
        Uniform = "uniform",
        /// Sample proportionally to gradient magnitude (CatBoost style, GOSS
        /// flavoured). Supported by the `hist` updaters on both CPU and GPU;
        /// rejected by `exact`.
        GradientBased = "gradient_based",
    }
    default = Uniform;
}

str_enum! {
    /// Whether the fit grows new trees or revisits existing ones.
    pub enum ProcessType: "process_type" {
        /// Grow new trees (normal boosting).
        Default = "default",
        /// Re-process the trees already in the model. Requires an explicit
        /// `updater` sequence made only of tree-modifying updaters.
        Update = "update",
    }
    default = Default;
}

str_enum! {
    /// Multi-target training strategy.
    pub enum MultiStrategy: "multi_strategy" {
        /// One single-output tree per target per round.
        OneOutputPerTree = "one_output_per_tree",
        /// One vector-leaf tree covering every target. `hist` only.
        MultiOutputTree = "multi_output_tree",
    }
    default = OneOutputPerTree;
}

str_enum! {
    /// A tree updater, i.e. one stage of the per-round tree pipeline.
    pub enum TreeUpdaterName: "updater" {
        /// Exact greedy, CPU.
        GrowColMaker = "grow_colmaker",
        /// `approx`, CPU.
        GrowHistMaker = "grow_histmaker",
        /// `hist`, CPU.
        GrowQuantileHistMaker = "grow_quantile_histmaker",
        /// `hist`, SYCL plugin.
        GrowQuantileHistMakerSycl = "grow_quantile_histmaker_sycl",
        /// `hist`, CUDA.
        GrowGpuHist = "grow_gpu_hist",
        /// `approx`, CUDA.
        GrowGpuApprox = "grow_gpu_approx",
        /// Post-prune splits whose gain falls below `gamma`.
        Prune = "prune",
        /// Refresh node statistics (and optionally leaf values) from new data.
        Refresh = "refresh",
    }
    default = GrowQuantileHistMaker;
}

/// The kind of processor an updater is built for.
///
/// Only the *growers* have one. `prune` and `refresh` rewrite a finished tree
/// out of node statistics and run wherever the fit does, which is why
/// [`TreeUpdaterName::device_class`] returns `None` for them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UpdaterDevice {
    /// `grow_colmaker`, `grow_histmaker`, `grow_quantile_histmaker`.
    Cpu,
    /// `grow_gpu_hist`, `grow_gpu_approx`.
    Cuda,
    /// `grow_quantile_histmaker_sycl`.
    Sycl,
}

impl UpdaterDevice {
    /// Whether an updater of this class can run on `device`.
    pub fn matches(self, device: Device) -> bool {
        match self {
            Self::Cpu => device.is_cpu(),
            Self::Cuda => device.is_cuda(),
            Self::Sycl => device.is_sycl(),
        }
    }
}

impl fmt::Display for UpdaterDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Cpu => "the CPU",
            Self::Cuda => "CUDA",
            Self::Sycl => "SYCL",
        })
    }
}

impl TreeUpdaterName {
    /// Whether this updater rewrites trees that already exist, rather than
    /// growing new ones. Mirrors `TreeUpdater::CanModifyTree`; it is what
    /// decides which updaters `process_type=update` accepts.
    pub const fn can_modify_tree(self) -> bool {
        matches!(self, Self::Prune | Self::Refresh)
    }

    /// The processor this updater is built for, or `None` if it runs anywhere.
    pub const fn device_class(self) -> Option<UpdaterDevice> {
        match self {
            Self::GrowColMaker | Self::GrowHistMaker | Self::GrowQuantileHistMaker => {
                Some(UpdaterDevice::Cpu)
            }
            Self::GrowGpuHist | Self::GrowGpuApprox => Some(UpdaterDevice::Cuda),
            Self::GrowQuantileHistMakerSycl => Some(UpdaterDevice::Sycl),
            Self::Prune | Self::Refresh => None,
        }
    }
}

str_enum! {
    /// Where a node sends rows whose split feature is missing, for the `exact`
    /// updater only. The histogram updaters always learn the direction.
    pub enum DefaultDirection: "default_direction" {
        /// Learn the direction from the data.
        Learn = "learn",
        /// Always route missing values left.
        Left = "left",
        /// Always route missing values right.
        Right = "right",
    }
    default = Learn;
}

str_enum! {
    /// Per-feature monotonicity constraint.
    pub enum MonotoneConstraint: "monotone_constraints" {
        /// Predictions must not increase with the feature.
        Decreasing = "-1",
        /// No constraint.
        Unconstrained = "0",
        /// Predictions must not decrease with the feature.
        Increasing = "1",
    }
    default = Unconstrained;
}

/// Parameters for the `gbtree` booster.
///
/// Every default matches upstream. Construct with
/// [`TreeBoosterParameters::builder`]; the fields are public so an already
/// validated value can be tweaked and re-[`validate`](Self::validate)d.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TreeBoosterParameters {
    // ---- core boosting / regularisation ----
    /// Learning rate, upstream alias `learning_rate`. Shrinks each new tree's
    /// leaf weights.
    pub eta: f32,
    /// Minimum loss reduction required to split, upstream alias
    /// `min_split_loss`.
    pub gamma: f32,
    /// Maximum tree depth. `0` means unlimited, which is only legal together
    /// with a non-zero `max_leaves`.
    pub max_depth: u32,
    /// Minimum sum of hessian in a child.
    pub min_child_weight: f32,
    /// Cap on each leaf weight's magnitude. `0` means uncapped.
    pub max_delta_step: f32,
    /// L2 regularisation on leaf weights, upstream alias `reg_lambda`.
    pub lambda: f32,
    /// L1 regularisation on leaf weights, upstream alias `reg_alpha`.
    pub alpha: f32,

    // ---- sampling ----
    /// Row subsample ratio, drawn once per boosting round.
    pub subsample: f32,
    /// How rows are drawn when `subsample < 1`.
    pub sampling_method: SamplingMethod,
    /// Column subsample ratio, drawn once per tree.
    pub colsample_bytree: f32,
    /// Column subsample ratio, drawn once per depth level.
    pub colsample_bylevel: f32,
    /// Column subsample ratio, drawn once per split. Unsupported by `exact`.
    pub colsample_bynode: f32,

    // ---- growth ----
    /// Tree construction algorithm.
    pub tree_method: TreeMethod,
    /// Explicit updater pipeline. `None` derives it from `tree_method` and the
    /// device via [`resolved_updaters`](Self::resolved_updaters), which is what
    /// XGBoost does and what you almost always want.
    pub updater: Option<Vec<TreeUpdaterName>>,
    /// Which node the growth driver expands next.
    pub grow_policy: GrowPolicy,
    /// Maximum leaf count. `0` means unlimited.
    pub max_leaves: u32,
    /// Maximum histogram bins per feature, for `hist` and `approx`.
    pub max_bin: u32,
    /// Trees grown per boosting round; `> 1` turns the fit into a boosted
    /// random forest.
    pub num_parallel_tree: u32,
    /// Whether the fit grows new trees or revisits existing ones.
    pub process_type: ProcessType,
    /// Whether the `refresh` updater also rewrites leaf values.
    pub refresh_leaf: bool,
    /// Multi-target training strategy.
    pub multi_strategy: MultiStrategy,

    // ---- constraints ----
    /// One entry per feature; an empty vector means "no constraints".
    pub monotone_constraints: Vec<MonotoneConstraint>,
    /// Groups of feature indices allowed to interact, e.g.
    /// `[[0, 1], [2, 3, 4]]`. `None` means "no constraints".
    pub interaction_constraints: Option<Vec<Vec<u32>>>,

    // ---- categorical features ----
    /// Use one-hot splits for categorical features with at most this many
    /// categories; partition-based splits above it.
    pub max_cat_to_onehot: u32,
    /// Maximum categories considered for a partition-based split.
    pub max_cat_threshold: u32,

    // ---- `hist`-specific ----
    /// Histogram cache size in nodes. `None` uses XGBoost's device-dependent
    /// default, resolved by
    /// [`max_cached_hist_nodes`](Self::max_cached_hist_nodes).
    pub max_cached_hist_node: Option<u64>,
    /// Density below which a feature is treated as sparse by the CPU `hist`
    /// updater.
    pub sparse_threshold: f64,
    /// Verify that every distributed worker built an identical tree. Debug aid;
    /// costs a synchronisation per tree.
    pub debug_synchronize: bool,

    // ---- `exact` / colmaker-specific ----
    /// Dense-column speed optimisation threshold for the `exact` updater.
    pub opt_dense_col: f32,
    /// Fixed missing-value direction for the `exact` updater.
    pub default_direction: DefaultDirection,
}

impl Default for TreeBoosterParameters {
    fn default() -> Self {
        Self {
            eta: 0.3,
            gamma: 0.0,
            max_depth: 6,
            min_child_weight: 1.0,
            max_delta_step: 0.0,
            lambda: 1.0,
            alpha: 0.0,
            subsample: 1.0,
            sampling_method: SamplingMethod::Uniform,
            colsample_bytree: 1.0,
            colsample_bylevel: 1.0,
            colsample_bynode: 1.0,
            tree_method: TreeMethod::Auto,
            updater: None,
            grow_policy: GrowPolicy::DepthWise,
            max_leaves: 0,
            max_bin: 256,
            num_parallel_tree: 1,
            process_type: ProcessType::Default,
            refresh_leaf: true,
            multi_strategy: MultiStrategy::OneOutputPerTree,
            monotone_constraints: Vec::new(),
            interaction_constraints: None,
            max_cat_to_onehot: 4,
            max_cat_threshold: 64,
            max_cached_hist_node: None,
            sparse_threshold: 0.2,
            debug_synchronize: false,
            opt_dense_col: 1.0,
            default_direction: DefaultDirection::Learn,
        }
    }
}

/// `HistMakerTrainParam::CpuDefaultNodes`.
const CPU_DEFAULT_CACHED_HIST_NODES: u64 = 1 << 16;
/// `HistMakerTrainParam::CudaDefaultNodes` — smaller, GPU memory is scarcer.
const CUDA_DEFAULT_CACHED_HIST_NODES: u64 = 1 << 12;

impl TreeBoosterParameters {
    /// Start from XGBoost's defaults.
    pub fn builder() -> TreeBoosterParametersBuilder {
        TreeBoosterParametersBuilder::default()
    }

    /// The updater pipeline this configuration will actually run.
    ///
    /// Mirrors `MapTreeMethodToUpdaters` in `src/gbm/gbtree.cc`: it is the one
    /// place where the CPU and GPU fits diverge, and the reason `device` has to
    /// be threaded in. An explicitly set `updater` wins, exactly as upstream.
    pub fn resolved_updaters(&self, device: Device) -> Result<Vec<TreeUpdaterName>> {
        use TreeUpdaterName::*;

        if let Some(updater) = &self.updater {
            for named in updater {
                if let Some(wanted) = named.device_class()
                    && !wanted.matches(device)
                {
                    return Err(Error::invalid(
                        "updater",
                        format!(
                            "the `{named}` updater runs on {wanted}, but device is `{device}`; \
                             name an updater for this device, or set `device` to match"
                        ),
                    ));
                }
            }
            return Ok(updater.clone());
        }
        Ok(match self.tree_method {
            TreeMethod::Auto | TreeMethod::Hist => {
                if device.is_cuda() {
                    vec![GrowGpuHist]
                } else if device.is_sycl() {
                    vec![GrowQuantileHistMakerSycl]
                } else {
                    vec![GrowQuantileHistMaker]
                }
            }
            TreeMethod::Approx => {
                if device.is_cuda() {
                    vec![GrowGpuApprox]
                } else if device.is_sycl() {
                    return Err(Error::invalid(
                        "tree_method",
                        "the `approx` tree method has no SYCL updater; use `hist` or `device=cpu`",
                    ));
                } else {
                    vec![GrowHistMaker]
                }
            }
            TreeMethod::Exact => {
                if !device.is_cpu() {
                    return Err(Error::invalid(
                        "tree_method",
                        format!("the `exact` tree method is not supported on device `{device}`"),
                    ));
                }
                vec![GrowColMaker, Prune]
            }
        })
    }

    /// Histogram cache size in nodes, resolving `None` to XGBoost's
    /// device-dependent default (`HistMakerTrainParam::MaxCachedHistNodes`).
    pub fn max_cached_hist_nodes(&self, device: Device) -> u64 {
        self.max_cached_hist_node.unwrap_or(if device.is_cpu() {
            CPU_DEFAULT_CACHED_HIST_NODES
        } else {
            CUDA_DEFAULT_CACHED_HIST_NODES
        })
    }

    /// Whether the resolved pipeline includes the `exact` updater. Several
    /// upstream restrictions key off this rather than off `tree_method`,
    /// because `updater` can select the exact updater directly.
    fn uses_exact(&self) -> bool {
        match &self.updater {
            Some(updater) => updater.contains(&TreeUpdaterName::GrowColMaker),
            None => self.tree_method == TreeMethod::Exact,
        }
    }

    /// Validate every field and every device-independent combination.
    ///
    /// Device-dependent rules (`exact` on GPU) are checked by
    /// [`resolved_updaters`](Self::resolved_updaters) and by
    /// [`BoosterParameters::validate`](super::BoosterParameters::validate),
    /// which is the first place a device is known.
    pub fn validate(&self) -> Result<()> {
        validate::ge("eta", self.eta, 0.0)?;
        validate::ge("gamma", self.gamma, 0.0)?;
        validate::ge("min_child_weight", self.min_child_weight, 0.0)?;
        validate::ge("max_delta_step", self.max_delta_step, 0.0)?;
        validate::ge("lambda", self.lambda, 0.0)?;
        validate::ge("alpha", self.alpha, 0.0)?;

        validate::ratio("subsample", self.subsample)?;
        validate::ratio("colsample_bytree", self.colsample_bytree)?;
        validate::ratio("colsample_bylevel", self.colsample_bylevel)?;
        validate::ratio("colsample_bynode", self.colsample_bynode)?;

        // `RegTree` node indices are `int32`, and a depthwise tree holds
        // `2^(d+1) - 1` nodes: upstream refuses to go past 30.
        validate::le("max_depth", self.max_depth, 30)?;
        validate::ge("max_bin", self.max_bin, 2)?;
        validate::ge("num_parallel_tree", self.num_parallel_tree, 1)?;
        validate::ge("max_cat_to_onehot", self.max_cat_to_onehot, 1)?;
        validate::ge("max_cat_threshold", self.max_cat_threshold, 1)?;
        validate::closed("sparse_threshold", self.sparse_threshold, 0.0, 1.0)?;
        validate::closed("opt_dense_col", self.opt_dense_col, 0.0, 1.0)?;
        if let Some(nodes) = self.max_cached_hist_node {
            validate::ge("max_cached_hist_node", nodes, 1)?;
        }

        if self.max_depth == 0 && self.max_leaves == 0 {
            return Err(Error::invalid(
                "max_depth",
                "max_depth and max_leaves cannot both be 0 (unconstrained); \
                 set a depth limit or a leaf limit",
            ));
        }

        if let Some(updater) = &self.updater
            && updater.is_empty()
        {
            return Err(Error::invalid("updater", "the updater sequence must not be empty"));
        }

        if let Some(groups) = &self.interaction_constraints {
            for group in groups {
                if group.is_empty() {
                    return Err(Error::invalid(
                        "interaction_constraints",
                        "interaction groups must not be empty",
                    ));
                }
            }
        }

        self.validate_process_type()?;
        self.validate_exact_restrictions()?;

        if self.multi_strategy == MultiStrategy::MultiOutputTree
            && !matches!(self.tree_method, TreeMethod::Hist | TreeMethod::Auto)
        {
            return Err(Error::invalid(
                "multi_strategy",
                format!(
                    "`multi_output_tree` needs tree_method `hist` or `auto`, got `{}`",
                    self.tree_method
                ),
            ));
        }

        Ok(())
    }

    /// `GBTree::BoostNewTrees` requires the first updater to grow trees under
    /// `process_type=default`, and every updater to modify existing trees under
    /// `process_type=update`.
    fn validate_process_type(&self) -> Result<()> {
        match self.process_type {
            ProcessType::Default => {
                if let Some(first) = self.updater.as_ref().and_then(|u| u.first())
                    && first.can_modify_tree()
                {
                    return Err(Error::invalid(
                        "updater",
                        format!(
                            "`{first}` only modifies existing trees, so it cannot lead the \
                             updater sequence under process_type=default"
                        ),
                    ));
                }
            }
            ProcessType::Update => {
                let Some(updater) = &self.updater else {
                    return Err(Error::invalid(
                        "process_type",
                        "process_type=update needs an explicit `updater` of tree-modifying \
                         updaters (`refresh` and/or `prune`); the tree_method default grows \
                         new trees instead",
                    ));
                };
                if let Some(grower) = updater.iter().find(|u| !u.can_modify_tree()) {
                    return Err(Error::invalid(
                        "updater",
                        format!("`{grower}` grows new trees, which process_type=update forbids"),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Restrictions `ColMaker::Update` enforces at fit time.
    fn validate_exact_restrictions(&self) -> Result<()> {
        if !self.uses_exact() {
            return Ok(());
        }
        if self.colsample_bynode != 1.0 {
            return Err(Error::invalid(
                "colsample_bynode",
                "column sampling by node is not supported by the `exact` tree method",
            ));
        }
        if self.sampling_method != SamplingMethod::Uniform {
            return Err(Error::invalid(
                "sampling_method",
                format!(
                    "the `exact` tree method only supports `uniform` sampling, got `{}`",
                    self.sampling_method
                ),
            ));
        }
        Ok(())
    }
}

impl ToConfig for TreeBoosterParameters {
    fn collect_config(&self, out: &mut Vec<ConfigEntry>) {
        push(out, "eta", self.eta);
        push(out, "gamma", self.gamma);
        push(out, "max_depth", self.max_depth);
        push(out, "min_child_weight", self.min_child_weight);
        push(out, "max_delta_step", self.max_delta_step);
        push(out, "lambda", self.lambda);
        push(out, "alpha", self.alpha);

        push(out, "subsample", self.subsample);
        push(out, "sampling_method", self.sampling_method);
        push(out, "colsample_bytree", self.colsample_bytree);
        push(out, "colsample_bylevel", self.colsample_bylevel);
        push(out, "colsample_bynode", self.colsample_bynode);

        push(out, "tree_method", self.tree_method);
        if let Some(updater) = &self.updater {
            let joined =
                updater.iter().map(|u| u.as_str()).collect::<Vec<_>>().join(",");
            push(out, "updater", joined);
        }
        push(out, "grow_policy", self.grow_policy);
        push(out, "max_leaves", self.max_leaves);
        push(out, "max_bin", self.max_bin);
        push(out, "num_parallel_tree", self.num_parallel_tree);
        push(out, "process_type", self.process_type);
        push_bool(out, "refresh_leaf", self.refresh_leaf);
        push(out, "multi_strategy", self.multi_strategy);

        if !self.monotone_constraints.is_empty() {
            let joined = self
                .monotone_constraints
                .iter()
                .map(|c| c.as_str())
                .collect::<Vec<_>>()
                .join(",");
            push(out, "monotone_constraints", format!("({joined})"));
        }
        if let Some(groups) = &self.interaction_constraints {
            // `serde_json` cannot fail on a `Vec<Vec<u32>>`.
            let json = serde_json::to_string(groups).expect("Vec<Vec<u32>> is serialisable");
            push(out, "interaction_constraints", json);
        }

        push(out, "max_cat_to_onehot", self.max_cat_to_onehot);
        push(out, "max_cat_threshold", self.max_cat_threshold);
        push_opt(out, "max_cached_hist_node", self.max_cached_hist_node);
        push(out, "sparse_threshold", self.sparse_threshold);
        push_bool(out, "debug_synchronize", self.debug_synchronize);

        push(out, "opt_dense_col", self.opt_dense_col);
        push(out, "default_direction", self.default_direction);
    }
}

/// Consuming builder for [`TreeBoosterParameters`].
#[derive(Clone, Debug, Default)]
pub struct TreeBoosterParametersBuilder {
    inner: TreeBoosterParameters,
}

macro_rules! setter {
    ($(#[$meta:meta])* $name:ident: $ty:ty) => {
        $(#[$meta])*
        pub fn $name(mut self, $name: $ty) -> Self {
            self.inner.$name = $name;
            self
        }
    };
}

impl TreeBoosterParametersBuilder {
    setter!(
        /// Learning rate (upstream alias `learning_rate`).
        eta: f32
    );
    setter!(
        /// Minimum loss reduction required to split (alias `min_split_loss`).
        gamma: f32
    );
    setter!(
        /// Maximum tree depth; `0` is unlimited and requires `max_leaves > 0`.
        max_depth: u32
    );
    setter!(
        /// Minimum sum of hessian in a child.
        min_child_weight: f32
    );
    setter!(
        /// Cap on leaf weight magnitude; `0` is uncapped.
        max_delta_step: f32
    );
    setter!(
        /// L2 regularisation on leaf weights (alias `reg_lambda`).
        lambda: f32
    );
    setter!(
        /// L1 regularisation on leaf weights (alias `reg_alpha`).
        alpha: f32
    );
    setter!(
        /// Row subsample ratio per boosting round.
        subsample: f32
    );
    setter!(
        /// How rows are drawn when `subsample < 1`.
        sampling_method: SamplingMethod
    );
    setter!(
        /// Column subsample ratio per tree.
        colsample_bytree: f32
    );
    setter!(
        /// Column subsample ratio per depth level.
        colsample_bylevel: f32
    );
    setter!(
        /// Column subsample ratio per split.
        colsample_bynode: f32
    );
    setter!(
        /// Tree construction algorithm.
        tree_method: TreeMethod
    );
    setter!(
        /// Which node the growth driver expands next.
        grow_policy: GrowPolicy
    );
    setter!(
        /// Maximum leaf count; `0` is unlimited.
        max_leaves: u32
    );
    setter!(
        /// Maximum histogram bins per feature.
        max_bin: u32
    );
    setter!(
        /// Trees grown per boosting round.
        num_parallel_tree: u32
    );
    setter!(
        /// Grow new trees or revisit existing ones.
        process_type: ProcessType
    );
    setter!(
        /// Whether `refresh` also rewrites leaf values.
        refresh_leaf: bool
    );
    setter!(
        /// Multi-target training strategy.
        multi_strategy: MultiStrategy
    );
    setter!(
        /// One monotonicity constraint per feature.
        monotone_constraints: Vec<MonotoneConstraint>
    );
    setter!(
        /// Maximum categories for one-hot categorical splits.
        max_cat_to_onehot: u32
    );
    setter!(
        /// Maximum categories considered for a partition-based split.
        max_cat_threshold: u32
    );
    setter!(
        /// Density below which the CPU `hist` updater treats a feature as sparse.
        sparse_threshold: f64
    );
    setter!(
        /// Verify that distributed workers built identical trees.
        debug_synchronize: bool
    );
    setter!(
        /// Dense-column optimisation threshold for `exact`.
        opt_dense_col: f32
    );
    setter!(
        /// Fixed missing-value direction for `exact`.
        default_direction: DefaultDirection
    );

    /// Set an explicit updater pipeline, overriding `tree_method`.
    pub fn updater(mut self, updater: impl Into<Vec<TreeUpdaterName>>) -> Self {
        self.inner.updater = Some(updater.into());
        self
    }

    /// Restrict feature interactions to the given groups of feature indices.
    pub fn interaction_constraints(mut self, groups: impl Into<Vec<Vec<u32>>>) -> Self {
        self.inner.interaction_constraints = Some(groups.into());
        self
    }

    /// Histogram cache size in nodes; leave unset for the device default.
    pub fn max_cached_hist_node(mut self, nodes: u64) -> Self {
        self.inner.max_cached_hist_node = Some(nodes);
        self
    }

    /// Validate and produce the parameters.
    pub fn build(self) -> Result<TreeBoosterParameters> {
        self.inner.validate()?;
        Ok(self.inner)
    }
}

str_enum! {
    /// How DART picks the trees to drop.
    pub enum DartSampleType: "sample_type" {
        /// Drop trees uniformly at random.
        Uniform = "uniform",
        /// Drop trees with probability proportional to their weight.
        Weighted = "weighted",
    }
    default = Uniform;
}

str_enum! {
    /// How DART rescales the trees that survive a dropout.
    pub enum DartNormalizeType: "normalize_type" {
        /// New tree gets weight `1 / (k + learning_rate)`.
        Tree = "tree",
        /// New tree gets weight `1 / (1 + learning_rate)`.
        Forest = "forest",
    }
    default = Tree;
}

/// Parameters for the `dart` booster: every `gbtree` parameter plus dropout.
///
/// Mirrors `DartTrainParam` in `src/gbm/gbtree.h`.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DartParameters {
    /// The underlying tree parameters; DART grows ordinary trees.
    pub tree: TreeBoosterParameters,
    /// How dropped trees are chosen.
    pub sample_type: DartSampleType,
    /// How surviving trees are rescaled.
    pub normalize_type: DartNormalizeType,
    /// Fraction of trees dropped per round.
    pub rate_drop: f32,
    /// Always drop at least one tree when dropping.
    pub one_drop: bool,
    /// Probability of skipping dropout entirely in a round.
    pub skip_drop: f32,
}

impl DartParameters {
    /// Start from XGBoost's defaults.
    pub fn builder() -> DartParametersBuilder {
        DartParametersBuilder::default()
    }

    /// Validate the tree parameters and the dropout parameters.
    pub fn validate(&self) -> Result<()> {
        self.tree.validate()?;
        validate::closed("rate_drop", self.rate_drop, 0.0, 1.0)?;
        validate::closed("skip_drop", self.skip_drop, 0.0, 1.0)?;
        Ok(())
    }
}

impl ToConfig for DartParameters {
    fn collect_config(&self, out: &mut Vec<ConfigEntry>) {
        self.tree.collect_config(out);
        push(out, "sample_type", self.sample_type);
        push(out, "normalize_type", self.normalize_type);
        push(out, "rate_drop", self.rate_drop);
        push_bool(out, "one_drop", self.one_drop);
        push(out, "skip_drop", self.skip_drop);
    }
}

/// Consuming builder for [`DartParameters`].
#[derive(Clone, Debug, Default)]
pub struct DartParametersBuilder {
    inner: DartParameters,
}

impl DartParametersBuilder {
    /// Set the underlying tree parameters wholesale.
    pub fn tree(mut self, tree: TreeBoosterParameters) -> Self {
        self.inner.tree = tree;
        self
    }

    /// How dropped trees are chosen.
    pub fn sample_type(mut self, sample_type: DartSampleType) -> Self {
        self.inner.sample_type = sample_type;
        self
    }

    /// How surviving trees are rescaled.
    pub fn normalize_type(mut self, normalize_type: DartNormalizeType) -> Self {
        self.inner.normalize_type = normalize_type;
        self
    }

    /// Fraction of trees dropped per round.
    pub fn rate_drop(mut self, rate_drop: f32) -> Self {
        self.inner.rate_drop = rate_drop;
        self
    }

    /// Always drop at least one tree when dropping.
    pub fn one_drop(mut self, one_drop: bool) -> Self {
        self.inner.one_drop = one_drop;
        self
    }

    /// Probability of skipping dropout entirely in a round.
    pub fn skip_drop(mut self, skip_drop: f32) -> Self {
        self.inner.skip_drop = skip_drop;
        self
    }

    /// Validate and produce the parameters.
    pub fn build(self) -> Result<DartParameters> {
        self.inner.validate()?;
        Ok(self.inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_xgboost() {
        let params = TreeBoosterParameters::default();
        assert_eq!(params.eta, 0.3);
        assert_eq!(params.max_depth, 6);
        assert_eq!(params.lambda, 1.0);
        assert_eq!(params.alpha, 0.0);
        assert_eq!(params.gamma, 0.0);
        assert_eq!(params.min_child_weight, 1.0);
        assert_eq!(params.max_bin, 256);
        assert_eq!(params.tree_method, TreeMethod::Auto);
        assert_eq!(params.grow_policy, GrowPolicy::DepthWise);
        params.validate().unwrap();
    }

    #[test]
    fn maps_tree_method_to_cpu_and_gpu_updaters() {
        use TreeUpdaterName::*;
        let hist = TreeBoosterParameters::builder().tree_method(TreeMethod::Hist).build().unwrap();
        assert_eq!(hist.resolved_updaters(Device::Cpu).unwrap(), vec![GrowQuantileHistMaker]);
        assert_eq!(hist.resolved_updaters(Device::cuda(0)).unwrap(), vec![GrowGpuHist]);

        let approx =
            TreeBoosterParameters::builder().tree_method(TreeMethod::Approx).build().unwrap();
        assert_eq!(approx.resolved_updaters(Device::Cpu).unwrap(), vec![GrowHistMaker]);
        assert_eq!(approx.resolved_updaters(Device::cuda(0)).unwrap(), vec![GrowGpuApprox]);

        // `auto` is `hist` since XGBoost 2.0.
        let auto = TreeBoosterParameters::default();
        assert_eq!(auto.resolved_updaters(Device::Cpu).unwrap(), vec![GrowQuantileHistMaker]);

        let exact = TreeBoosterParameters::builder().tree_method(TreeMethod::Exact).build().unwrap();
        assert_eq!(exact.resolved_updaters(Device::Cpu).unwrap(), vec![GrowColMaker, Prune]);
        assert!(exact.resolved_updaters(Device::cuda(0)).is_err());
    }

    #[test]
    fn resolves_device_dependent_hist_cache() {
        let params = TreeBoosterParameters::default();
        assert_eq!(params.max_cached_hist_nodes(Device::Cpu), 65536);
        assert_eq!(params.max_cached_hist_nodes(Device::cuda(0)), 4096);

        let pinned = TreeBoosterParameters::builder().max_cached_hist_node(7).build().unwrap();
        assert_eq!(pinned.max_cached_hist_nodes(Device::Cpu), 7);
        assert_eq!(pinned.max_cached_hist_nodes(Device::cuda(0)), 7);
    }

    #[test]
    fn rejects_out_of_range_values() {
        let cases: Vec<(&str, TreeBoosterParametersBuilder)> = vec![
            ("eta", TreeBoosterParameters::builder().eta(-0.1)),
            ("gamma", TreeBoosterParameters::builder().gamma(-1.0)),
            ("subsample", TreeBoosterParameters::builder().subsample(0.0)),
            ("subsample", TreeBoosterParameters::builder().subsample(1.5)),
            ("colsample_bytree", TreeBoosterParameters::builder().colsample_bytree(0.0)),
            ("max_bin", TreeBoosterParameters::builder().max_bin(1)),
            ("max_depth", TreeBoosterParameters::builder().max_depth(31)),
            ("num_parallel_tree", TreeBoosterParameters::builder().num_parallel_tree(0)),
            ("min_child_weight", TreeBoosterParameters::builder().min_child_weight(f32::NAN)),
            ("sparse_threshold", TreeBoosterParameters::builder().sparse_threshold(1.5)),
        ];
        for (name, builder) in cases {
            let err = builder.build().unwrap_err().to_string();
            assert!(err.contains(name), "expected {name} in {err}");
        }
    }

    #[test]
    fn rejects_unconstrained_depth_and_leaves() {
        let err = TreeBoosterParameters::builder().max_depth(0).build().unwrap_err();
        assert!(err.to_string().contains("max_leaves"));
        // Either limit alone is fine.
        TreeBoosterParameters::builder().max_depth(0).max_leaves(31).build().unwrap();
        TreeBoosterParameters::builder().max_depth(3).build().unwrap();
    }

    #[test]
    fn enforces_exact_restrictions() {
        let err = TreeBoosterParameters::builder()
            .tree_method(TreeMethod::Exact)
            .colsample_bynode(0.5)
            .build()
            .unwrap_err();
        assert!(err.to_string().contains("colsample_bynode"));

        let err = TreeBoosterParameters::builder()
            .tree_method(TreeMethod::Exact)
            .sampling_method(SamplingMethod::GradientBased)
            .build()
            .unwrap_err();
        assert!(err.to_string().contains("sampling_method"));

        // The same restrictions apply when `exact` is selected via `updater`.
        let err = TreeBoosterParameters::builder()
            .updater([TreeUpdaterName::GrowColMaker])
            .colsample_bynode(0.5)
            .build()
            .unwrap_err();
        assert!(err.to_string().contains("colsample_bynode"));

        // Gradient-based sampling is supported by the histogram updaters.
        TreeBoosterParameters::builder()
            .sampling_method(SamplingMethod::GradientBased)
            .subsample(0.5)
            .build()
            .unwrap();
    }

    #[test]
    fn enforces_process_type_updater_pairing() {
        use TreeUpdaterName::*;

        let err = TreeBoosterParameters::builder()
            .process_type(ProcessType::Update)
            .build()
            .unwrap_err();
        assert!(err.to_string().contains("process_type"));

        let err = TreeBoosterParameters::builder()
            .process_type(ProcessType::Update)
            .updater([GrowQuantileHistMaker])
            .build()
            .unwrap_err();
        assert!(err.to_string().contains("grow_quantile_histmaker"));

        TreeBoosterParameters::builder()
            .process_type(ProcessType::Update)
            .updater([Refresh, Prune])
            .build()
            .unwrap();

        let err = TreeBoosterParameters::builder().updater([Prune]).build().unwrap_err();
        assert!(err.to_string().contains("prune"));
    }

    #[test]
    fn restricts_vector_leaf_to_hist() {
        let err = TreeBoosterParameters::builder()
            .multi_strategy(MultiStrategy::MultiOutputTree)
            .tree_method(TreeMethod::Approx)
            .build()
            .unwrap_err();
        assert!(err.to_string().contains("multi_strategy"));

        TreeBoosterParameters::builder()
            .multi_strategy(MultiStrategy::MultiOutputTree)
            .tree_method(TreeMethod::Hist)
            .build()
            .unwrap();
    }

    #[test]
    fn emits_upstream_parameter_names() {
        let config = TreeBoosterParameters::builder()
            .eta(0.1)
            .max_depth(8)
            .monotone_constraints(vec![
                MonotoneConstraint::Increasing,
                MonotoneConstraint::Unconstrained,
                MonotoneConstraint::Decreasing,
            ])
            .interaction_constraints(vec![vec![0, 1], vec![2, 3, 4]])
            .updater([TreeUpdaterName::GrowGpuHist])
            .build()
            .unwrap()
            .to_config_map();

        assert_eq!(config["eta"], "0.1");
        assert_eq!(config["max_depth"], "8");
        assert_eq!(config["monotone_constraints"], "(1,0,-1)");
        assert_eq!(config["interaction_constraints"], "[[0,1],[2,3,4]]");
        assert_eq!(config["updater"], "grow_gpu_hist");
        assert_eq!(config["refresh_leaf"], "1");
        // Unset optionals stay out so XGBoost keeps its own default.
        assert!(!config.contains_key("max_cached_hist_node"));
    }

    #[test]
    fn dart_defaults_and_validation() {
        let dart = DartParameters::default();
        assert_eq!(dart.sample_type, DartSampleType::Uniform);
        assert_eq!(dart.normalize_type, DartNormalizeType::Tree);
        assert_eq!(dart.rate_drop, 0.0);
        assert_eq!(dart.skip_drop, 0.0);
        assert!(!dart.one_drop);
        assert_eq!(dart.tree.eta, 0.3);
        dart.validate().unwrap();

        let err = DartParameters::builder().rate_drop(1.5).build().unwrap_err();
        assert!(err.to_string().contains("rate_drop"));

        let config = DartParameters::builder()
            .rate_drop(0.2)
            .one_drop(true)
            .build()
            .unwrap()
            .to_config_map();
        assert_eq!(config["rate_drop"], "0.2");
        assert_eq!(config["one_drop"], "1");
        assert_eq!(config["eta"], "0.3");
    }
}
