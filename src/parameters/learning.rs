//! Learning-task parameters: the objective, its tuning knobs, the evaluation
//! metrics, and the model-level settings the objective feeds.
//!
//! Sources: `LearnerTrainParam` and `LearnerModelParamLegacy`
//! (`src/learner.cc`), `Context::seed*` (`include/xgboost/context.h`), and the
//! per-objective parameter structs `RegLossParam`
//! (`src/objective/regression_param.h`), `SoftmaxMultiClassParam`
//! (`src/objective/multiclass_param.h`), `PoissonRegressionParam` and
//! `TweedieRegressionParam` (`src/objective/regression_obj.cu`),
//! `PseudoHuberParam` (`src/common/pseudo_huber.h`), `QuantileLossParam`
//! (`src/common/quantile_loss_utils.h`), `ExpectileLossParam`
//! (`src/common/expectile_loss_utils.h`), `AFTParam`
//! (`src/common/survival_util.h`) and `LambdaRankParam`
//! (`src/common/ranking_utils.h`).

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::config::{ConfigEntry, ToConfig, push, push_bool, push_f32_list, push_opt};
use super::str_enum::str_enum;
use super::validate;
use crate::error::{Error, Result};

str_enum! {
    /// Noise distribution for the accelerated failure time model.
    pub enum AftDistribution: "aft_loss_distribution" {
        /// Normal.
        Normal = "normal",
        /// Logistic.
        Logistic = "logistic",
        /// Extreme value (Gumbel).
        Extreme = "extreme",
    }
    default = Normal;
}

str_enum! {
    /// How LambdaMART builds document pairs.
    pub enum LambdaRankPairMethod: "lambdarank_pair_method" {
        /// Sample pairs from the whole list.
        Mean = "mean",
        /// Restrict pairs to the truncation level.
        TopK = "topk",
    }
    default = TopK;
}

/// Tuning parameters shared by the three `rank:*` objectives.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LambdaRankParameters {
    /// How document pairs are built.
    pub pair_method: LambdaRankPairMethod,
    /// Pairs per sample. `None` uses XGBoost's method-dependent default: `32`
    /// for `topk`, `1` for `mean`.
    pub num_pair_per_sample: Option<u32>,
    /// Debias click position with extended inverse propensity weighting.
    pub unbiased: bool,
    /// Normalise leaf values.
    pub normalization: bool,
    /// Normalise the delta by the prediction score difference.
    pub score_normalization: bool,
    /// Lp regularisation for unbiased LambdaMART.
    pub bias_norm: f64,
    /// Label gain is `2^rel - 1` when true, plain `rel` when false. NDCG only.
    pub ndcg_exp_gain: bool,
}

impl Default for LambdaRankParameters {
    fn default() -> Self {
        Self {
            pair_method: LambdaRankPairMethod::TopK,
            num_pair_per_sample: None,
            unbiased: false,
            normalization: true,
            score_normalization: true,
            bias_norm: 1.0,
            ndcg_exp_gain: true,
        }
    }
}

/// `LambdaRankParam::DefaultK`.
const LAMBDARANK_DEFAULT_TOP_K: u32 = 32;
/// `LambdaRankParam::DefaultSamplePairs`.
const LAMBDARANK_DEFAULT_SAMPLE_PAIRS: u32 = 1;

impl LambdaRankParameters {
    /// Pairs per sample, resolving `None` the way `LambdaRankParam::NumPair`
    /// does.
    pub fn resolved_num_pair_per_sample(&self) -> u32 {
        self.num_pair_per_sample.unwrap_or(match self.pair_method {
            LambdaRankPairMethod::TopK => LAMBDARANK_DEFAULT_TOP_K,
            LambdaRankPairMethod::Mean => LAMBDARANK_DEFAULT_SAMPLE_PAIRS,
        })
    }

    /// Validate every field.
    pub fn validate(&self) -> Result<()> {
        if let Some(pairs) = self.num_pair_per_sample {
            validate::ge("lambdarank_num_pair_per_sample", pairs, 1)?;
        }
        validate::ge("lambdarank_bias_norm", self.bias_norm, 0.0)?;
        Ok(())
    }
}

impl ToConfig for LambdaRankParameters {
    fn collect_config(&self, out: &mut Vec<ConfigEntry>) {
        push(out, "lambdarank_pair_method", self.pair_method);
        push_opt(out, "lambdarank_num_pair_per_sample", self.num_pair_per_sample);
        push_bool(out, "lambdarank_unbiased", self.unbiased);
        push_bool(out, "lambdarank_normalization", self.normalization);
        push_bool(out, "lambdarank_score_normalization", self.score_normalization);
        push(out, "lambdarank_bias_norm", self.bias_norm);
        push_bool(out, "ndcg_exp_gain", self.ndcg_exp_gain);
    }
}

/// The `objective` parameter, together with the parameters that only that
/// objective reads.
///
/// Carrying the tuning knobs inside the variant is what stops a fit from
/// silently ignoring, say, `tweedie_variance_power` set alongside
/// `reg:squarederror`.
///
/// ```
/// use xgboost_rs::parameters::{Objective, ToConfig};
///
/// let objective = Objective::RegTweedie { tweedie_variance_power: 1.2 };
/// let config = objective.to_config_map();
/// assert_eq!(config["objective"], "reg:tweedie");
/// assert_eq!(config["tweedie_variance_power"], "1.2");
/// ```
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum Objective {
    /// `reg:squarederror` — squared loss. XGBoost's default.
    #[default]
    #[serde(rename = "reg:squarederror")]
    RegSquaredError,
    /// `reg:squaredlogerror` — squared log loss.
    #[serde(rename = "reg:squaredlogerror")]
    RegSquaredLogError,
    /// `reg:logistic` — logistic regression on a continuous target in `[0, 1]`.
    #[serde(rename = "reg:logistic")]
    RegLogistic,
    /// `reg:pseudohubererror` — Pseudo-Huber loss.
    #[serde(rename = "reg:pseudohubererror")]
    RegPseudoHuberError {
        /// The delta term. Must be positive.
        huber_slope: f32,
    },
    /// `reg:absoluteerror` — L1 loss with leaf-value refresh.
    #[serde(rename = "reg:absoluteerror")]
    RegAbsoluteError,
    /// `reg:quantileerror` — quantile (pinball) loss.
    #[serde(rename = "reg:quantileerror")]
    RegQuantileError {
        /// Quantiles to fit, ascending, each in `[0, 1]`. One output per entry.
        quantile_alpha: Vec<f32>,
    },
    /// `reg:expectileerror` — expectile loss.
    #[serde(rename = "reg:expectileerror")]
    RegExpectileError {
        /// Expectiles to fit, ascending, each in `[0, 1]`.
        expectile_alpha: Vec<f32>,
    },
    /// `reg:gamma` — gamma regression with a log link.
    #[serde(rename = "reg:gamma")]
    RegGamma,
    /// `reg:tweedie` — Tweedie regression with a log link.
    #[serde(rename = "reg:tweedie")]
    RegTweedie {
        /// Variance power in `[1, 2)`.
        tweedie_variance_power: f32,
    },
    /// `reg:linear` — deprecated spelling of `reg:squarederror`, kept because
    /// old configs and saved models still carry it.
    #[serde(rename = "reg:linear")]
    RegLinear,
    /// `count:poisson` — Poisson regression with a log link.
    #[serde(rename = "count:poisson")]
    CountPoisson {
        /// Poisson-specific cap on leaf weights; upstream defaults it to `0.7`
        /// rather than the tree booster's `0`, to keep the log link stable.
        max_delta_step: f32,
    },
    /// `survival:cox` — Cox proportional hazards.
    #[serde(rename = "survival:cox")]
    SurvivalCox,
    /// `survival:aft` — accelerated failure time.
    #[serde(rename = "survival:aft")]
    SurvivalAft {
        /// Noise distribution.
        aft_loss_distribution: AftDistribution,
        /// Scale of the noise distribution. Must be positive.
        aft_loss_distribution_scale: f32,
    },
    /// `binary:logistic` — binary classification, outputs probability.
    #[serde(rename = "binary:logistic")]
    BinaryLogistic,
    /// `binary:logitraw` — binary classification, outputs the margin.
    #[serde(rename = "binary:logitraw")]
    BinaryLogitRaw,
    /// `binary:hinge` — hinge loss, outputs 0 or 1.
    #[serde(rename = "binary:hinge")]
    BinaryHinge,
    /// `multi:softmax` — multiclass, outputs the predicted class.
    #[serde(rename = "multi:softmax")]
    MultiSoftmax {
        /// Number of classes.
        num_class: u32,
    },
    /// `multi:softprob` — multiclass, outputs per-class probabilities.
    #[serde(rename = "multi:softprob")]
    MultiSoftprob {
        /// Number of classes.
        num_class: u32,
    },
    /// `rank:pairwise` — LambdaMART with pairwise loss.
    #[serde(rename = "rank:pairwise")]
    RankPairwise(LambdaRankParameters),
    /// `rank:ndcg` — LambdaMART optimising NDCG.
    #[serde(rename = "rank:ndcg")]
    RankNdcg(LambdaRankParameters),
    /// `rank:map` — LambdaMART optimising MAP.
    #[serde(rename = "rank:map")]
    RankMap(LambdaRankParameters),
}

impl Objective {
    /// The XGBoost `objective` string.
    pub const fn name(&self) -> &'static str {
        match self {
            Self::RegSquaredError => "reg:squarederror",
            Self::RegSquaredLogError => "reg:squaredlogerror",
            Self::RegLogistic => "reg:logistic",
            Self::RegPseudoHuberError { .. } => "reg:pseudohubererror",
            Self::RegAbsoluteError => "reg:absoluteerror",
            Self::RegQuantileError { .. } => "reg:quantileerror",
            Self::RegExpectileError { .. } => "reg:expectileerror",
            Self::RegGamma => "reg:gamma",
            Self::RegTweedie { .. } => "reg:tweedie",
            Self::RegLinear => "reg:linear",
            Self::CountPoisson { .. } => "count:poisson",
            Self::SurvivalCox => "survival:cox",
            Self::SurvivalAft { .. } => "survival:aft",
            Self::BinaryLogistic => "binary:logistic",
            Self::BinaryLogitRaw => "binary:logitraw",
            Self::BinaryHinge => "binary:hinge",
            Self::MultiSoftmax { .. } => "multi:softmax",
            Self::MultiSoftprob { .. } => "multi:softprob",
            Self::RankPairwise(_) => "rank:pairwise",
            Self::RankNdcg(_) => "rank:ndcg",
            Self::RankMap(_) => "rank:map",
        }
    }

    /// `num_class`, for the multiclass objectives only.
    pub const fn num_class(&self) -> Option<u32> {
        match self {
            Self::MultiSoftmax { num_class } | Self::MultiSoftprob { num_class } => Some(*num_class),
            _ => None,
        }
    }

    /// The LambdaMART parameters, for the `rank:*` objectives only.
    pub const fn lambdarank(&self) -> Option<&LambdaRankParameters> {
        match self {
            Self::RankPairwise(p) | Self::RankNdcg(p) | Self::RankMap(p) => Some(p),
            _ => None,
        }
    }

    /// Validate the objective's own parameters.
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::RegPseudoHuberError { huber_slope } => {
                // Upstream declares no bound, but the slope is the loss's delta
                // term and a zero or negative delta makes the gradient
                // undefined, so reject it here rather than at the first row.
                validate::gt("huber_slope", *huber_slope, 0.0)?;
            }
            Self::RegQuantileError { quantile_alpha } => {
                validate_alpha_list("quantile_alpha", quantile_alpha)?;
            }
            Self::RegExpectileError { expectile_alpha } => {
                validate_alpha_list("expectile_alpha", expectile_alpha)?;
            }
            Self::RegTweedie { tweedie_variance_power } => {
                validate::half_open("tweedie_variance_power", *tweedie_variance_power, 1.0, 2.0)?;
            }
            Self::CountPoisson { max_delta_step } => {
                validate::ge("max_delta_step", *max_delta_step, 0.0)?;
            }
            Self::SurvivalAft { aft_loss_distribution_scale, .. } => {
                validate::gt("aft_loss_distribution_scale", *aft_loss_distribution_scale, 0.0)?;
            }
            Self::MultiSoftmax { num_class } | Self::MultiSoftprob { num_class } => {
                validate::ge("num_class", *num_class, 1)?;
            }
            Self::RankPairwise(p) | Self::RankNdcg(p) | Self::RankMap(p) => p.validate()?,
            _ => {}
        }
        Ok(())
    }
}

/// `QuantileLossParam::Validate` / `ExpectileLossParam::Validate`.
fn validate_alpha_list(name: &'static str, alphas: &[f32]) -> Result<()> {
    if alphas.is_empty() {
        return Err(Error::invalid(name, "at least one value is required"));
    }
    for alpha in alphas {
        validate::closed(name, *alpha, 0.0, 1.0)?;
    }
    if alphas.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err(Error::invalid(name, "values must be sorted in ascending order"));
    }
    Ok(())
}

impl fmt::Display for Objective {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for Objective {
    type Err = Error;

    /// Parse an `objective` string into the matching variant with upstream's
    /// default tuning parameters. `multi:*` has no default `num_class`
    /// upstream either, so it parses to `num_class = 1`, the declared lower
    /// bound; set it explicitly afterwards.
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s {
            "reg:squarederror" => Self::RegSquaredError,
            "reg:squaredlogerror" => Self::RegSquaredLogError,
            "reg:logistic" => Self::RegLogistic,
            "reg:pseudohubererror" => Self::RegPseudoHuberError { huber_slope: 1.0 },
            "reg:absoluteerror" => Self::RegAbsoluteError,
            "reg:quantileerror" => Self::RegQuantileError { quantile_alpha: Vec::new() },
            "reg:expectileerror" => Self::RegExpectileError { expectile_alpha: Vec::new() },
            "reg:gamma" => Self::RegGamma,
            "reg:tweedie" => Self::RegTweedie { tweedie_variance_power: 1.5 },
            "reg:linear" => Self::RegLinear,
            "count:poisson" => Self::CountPoisson { max_delta_step: 0.7 },
            "survival:cox" => Self::SurvivalCox,
            "survival:aft" => Self::SurvivalAft {
                aft_loss_distribution: AftDistribution::Normal,
                aft_loss_distribution_scale: 1.0,
            },
            "binary:logistic" => Self::BinaryLogistic,
            "binary:logitraw" => Self::BinaryLogitRaw,
            "binary:hinge" => Self::BinaryHinge,
            "multi:softmax" => Self::MultiSoftmax { num_class: 1 },
            "multi:softprob" => Self::MultiSoftprob { num_class: 1 },
            "rank:pairwise" => Self::RankPairwise(LambdaRankParameters::default()),
            "rank:ndcg" => Self::RankNdcg(LambdaRankParameters::default()),
            "rank:map" => Self::RankMap(LambdaRankParameters::default()),
            other => {
                return Err(Error::parse("objective", other, "unknown objective"));
            }
        })
    }
}

impl ToConfig for Objective {
    fn collect_config(&self, out: &mut Vec<ConfigEntry>) {
        push(out, "objective", self.name());
        match self {
            Self::RegPseudoHuberError { huber_slope } => push(out, "huber_slope", huber_slope),
            Self::RegQuantileError { quantile_alpha } => {
                push_f32_list(out, "quantile_alpha", quantile_alpha)
            }
            Self::RegExpectileError { expectile_alpha } => {
                push_f32_list(out, "expectile_alpha", expectile_alpha)
            }
            Self::RegTweedie { tweedie_variance_power } => {
                push(out, "tweedie_variance_power", tweedie_variance_power)
            }
            Self::CountPoisson { max_delta_step } => push(out, "max_delta_step", max_delta_step),
            Self::SurvivalAft { aft_loss_distribution, aft_loss_distribution_scale } => {
                push(out, "aft_loss_distribution", aft_loss_distribution);
                push(out, "aft_loss_distribution_scale", aft_loss_distribution_scale);
            }
            Self::MultiSoftmax { num_class } | Self::MultiSoftprob { num_class } => {
                push(out, "num_class", num_class)
            }
            Self::RankPairwise(p) | Self::RankNdcg(p) | Self::RankMap(p) => p.collect_config(out),
            _ => {}
        }
    }
}

/// An `eval_metric` value.
///
/// Parameterised metrics carry their argument, so `error@0.7` and `ndcg@5-`
/// are values rather than strings that have to be spelled correctly.
///
/// ```
/// use xgboost_rs::parameters::EvalMetric;
///
/// assert_eq!(EvalMetric::ErrorAt(0.7).to_string(), "error@0.7");
/// let ndcg: EvalMetric = "ndcg@5-".parse().unwrap();
/// assert_eq!(ndcg, EvalMetric::Ndcg { top_n: Some(5), minus: true });
/// ```
#[derive(Clone, Debug, PartialEq)]
pub enum EvalMetric {
    /// `rmse` — root mean squared error.
    Rmse,
    /// `rmsle` — root mean squared log error.
    Rmsle,
    /// `mae` — mean absolute error.
    Mae,
    /// `mape` — mean absolute percentage error.
    Mape,
    /// `mphe` — mean Pseudo-Huber error.
    Mphe,
    /// `logloss` — negative log likelihood.
    Logloss,
    /// `error` — binary error rate at the default 0.5 threshold.
    Error,
    /// `error@t` — binary error rate at threshold `t`.
    ErrorAt(f32),
    /// `merror` — multiclass error rate.
    MError,
    /// `mlogloss` — multiclass negative log likelihood.
    MLogloss,
    /// `auc` — area under the ROC curve.
    Auc,
    /// `aucpr` — area under the precision/recall curve.
    Aucpr,
    /// `pre` / `pre@n` — precision, optionally at cut-off `n`.
    Pre(Option<u32>),
    /// `ndcg` / `ndcg@n` / `ndcg-` / `ndcg@n-` — normalised discounted
    /// cumulative gain. `minus` scores an empty list as 0 instead of 1.
    Ndcg {
        /// Truncation level.
        top_n: Option<u32>,
        /// The trailing `-` variant.
        minus: bool,
    },
    /// `map` / `map@n` / `map-` / `map@n-` — mean average precision.
    Map {
        /// Truncation level.
        top_n: Option<u32>,
        /// The trailing `-` variant.
        minus: bool,
    },
    /// `poisson-nloglik`.
    PoissonNegLogLik,
    /// `gamma-nloglik`.
    GammaNegLogLik,
    /// `cox-nloglik`.
    CoxNegLogLik,
    /// `gamma-deviance`.
    GammaDeviance,
    /// `tweedie-nloglik@p` — the variance power is mandatory upstream.
    TweedieNegLogLik(f32),
    /// `aft-nloglik`.
    AftNegLogLik,
    /// `interval-regression-accuracy`.
    IntervalRegressionAccuracy,
    /// `quantile` — pinball loss.
    Quantile,
    /// `expectile` — expectile loss.
    Expectile,
    /// `ams@t` — the Higgs-challenge AMS metric at threshold `t`.
    Ams(f32),
    /// Any metric this crate does not know, e.g. one from a plugin build.
    ///
    /// [`FromStr`] deliberately never produces this — an unrecognised metric
    /// name is far more often a typo than a plugin — so construct it
    /// explicitly. Deserialisation *does* fall back to it, so a `Custom`
    /// metric survives a serde round-trip.
    Custom(String),
}

impl EvalMetric {
    /// Validate the metric's argument.
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::ErrorAt(threshold) => validate::closed("eval_metric", *threshold, 0.0, 1.0)?,
            Self::TweedieNegLogLik(power) => {
                validate::half_open("eval_metric", *power, 1.0, 2.0)?
            }
            Self::Ams(threshold) => validate::ge("eval_metric", *threshold, 0.0)?,
            Self::Pre(Some(top_n)) => validate::ge("eval_metric", *top_n, 1)?,
            Self::Ndcg { top_n: Some(top_n), .. } | Self::Map { top_n: Some(top_n), .. } => {
                validate::ge("eval_metric", *top_n, 1)?
            }
            Self::Custom(name) => {
                if name.is_empty() {
                    return Err(Error::invalid("eval_metric", "custom metric name is empty"));
                }
                // `Learner::SetParam` rejects whitespace in configuration values.
                if name.chars().any(char::is_whitespace) {
                    return Err(Error::invalid(
                        "eval_metric",
                        format!("custom metric `{name}` must not contain whitespace"),
                    ));
                }
            }
            _ => {}
        }
        Ok(())
    }
}

/// Render `base@n` / `base@n-` style names.
fn fmt_ranking(f: &mut fmt::Formatter<'_>, base: &str, top_n: Option<u32>, minus: bool) -> fmt::Result {
    f.write_str(base)?;
    if let Some(top_n) = top_n {
        write!(f, "@{top_n}")?;
    }
    if minus {
        f.write_str("-")?;
    }
    Ok(())
}

impl fmt::Display for EvalMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rmse => f.write_str("rmse"),
            Self::Rmsle => f.write_str("rmsle"),
            Self::Mae => f.write_str("mae"),
            Self::Mape => f.write_str("mape"),
            Self::Mphe => f.write_str("mphe"),
            Self::Logloss => f.write_str("logloss"),
            Self::Error => f.write_str("error"),
            Self::ErrorAt(threshold) => write!(f, "error@{threshold}"),
            Self::MError => f.write_str("merror"),
            Self::MLogloss => f.write_str("mlogloss"),
            Self::Auc => f.write_str("auc"),
            Self::Aucpr => f.write_str("aucpr"),
            Self::Pre(None) => f.write_str("pre"),
            Self::Pre(Some(top_n)) => write!(f, "pre@{top_n}"),
            Self::Ndcg { top_n, minus } => fmt_ranking(f, "ndcg", *top_n, *minus),
            Self::Map { top_n, minus } => fmt_ranking(f, "map", *top_n, *minus),
            Self::PoissonNegLogLik => f.write_str("poisson-nloglik"),
            Self::GammaNegLogLik => f.write_str("gamma-nloglik"),
            Self::CoxNegLogLik => f.write_str("cox-nloglik"),
            Self::GammaDeviance => f.write_str("gamma-deviance"),
            Self::TweedieNegLogLik(power) => write!(f, "tweedie-nloglik@{power}"),
            Self::AftNegLogLik => f.write_str("aft-nloglik"),
            Self::IntervalRegressionAccuracy => f.write_str("interval-regression-accuracy"),
            Self::Quantile => f.write_str("quantile"),
            Self::Expectile => f.write_str("expectile"),
            Self::Ams(threshold) => write!(f, "ams@{threshold}"),
            Self::Custom(name) => f.write_str(name),
        }
    }
}

impl FromStr for EvalMetric {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        let bad = |reason: String| Error::parse("eval_metric", s, reason);

        // A ranking metric may carry a trailing `-`; strip it before splitting
        // off the `@` argument so `ndcg@5-` parses.
        let (body, minus) = match s.strip_suffix('-') {
            Some(body) if matches!(body.split('@').next(), Some("ndcg" | "map")) => (body, true),
            _ => (s, false),
        };
        let (name, arg) = match body.split_once('@') {
            Some((name, arg)) => (name, Some(arg)),
            None => (body, None),
        };

        let float = |arg: Option<&str>| -> Result<f32> {
            let arg = arg.ok_or_else(|| bad(format!("`{name}` needs an `@<value>` argument")))?;
            arg.parse::<f32>().map_err(|_| bad(format!("`{arg}` is not a number")))
        };
        let top_n = |arg: Option<&str>| -> Result<Option<u32>> {
            arg.map(|arg| {
                arg.parse::<u32>().map_err(|_| bad(format!("`{arg}` is not a positive integer")))
            })
            .transpose()
        };
        let no_arg = |metric: EvalMetric| -> Result<EvalMetric> {
            match arg {
                None => Ok(metric),
                Some(arg) => Err(bad(format!("`{name}` takes no argument, got `@{arg}`"))),
            }
        };

        let metric = match name {
            "rmse" => no_arg(Self::Rmse)?,
            "rmsle" => no_arg(Self::Rmsle)?,
            "mae" => no_arg(Self::Mae)?,
            "mape" => no_arg(Self::Mape)?,
            "mphe" => no_arg(Self::Mphe)?,
            "logloss" => no_arg(Self::Logloss)?,
            "error" => match arg {
                None => Self::Error,
                Some(_) => Self::ErrorAt(float(arg)?),
            },
            "merror" => no_arg(Self::MError)?,
            "mlogloss" => no_arg(Self::MLogloss)?,
            "auc" => no_arg(Self::Auc)?,
            "aucpr" => no_arg(Self::Aucpr)?,
            "pre" => Self::Pre(top_n(arg)?),
            "ndcg" => Self::Ndcg { top_n: top_n(arg)?, minus },
            "map" => Self::Map { top_n: top_n(arg)?, minus },
            "poisson-nloglik" => no_arg(Self::PoissonNegLogLik)?,
            "gamma-nloglik" => no_arg(Self::GammaNegLogLik)?,
            "cox-nloglik" => no_arg(Self::CoxNegLogLik)?,
            "gamma-deviance" => no_arg(Self::GammaDeviance)?,
            "tweedie-nloglik" => Self::TweedieNegLogLik(float(arg)?),
            "aft-nloglik" => no_arg(Self::AftNegLogLik)?,
            "interval-regression-accuracy" => no_arg(Self::IntervalRegressionAccuracy)?,
            "quantile" => no_arg(Self::Quantile)?,
            "expectile" => no_arg(Self::Expectile)?,
            "ams" => Self::Ams(float(arg)?),
            other => {
                return Err(bad(format!(
                    "unknown metric `{other}`; use EvalMetric::Custom for plugin metrics"
                )));
            }
        };
        metric.validate()?;
        Ok(metric)
    }
}

impl Serialize for EvalMetric {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for EvalMetric {
    /// Falls back to [`EvalMetric::Custom`] for names [`FromStr`] rejects, so a
    /// plugin metric survives a round-trip. The custom name is still validated.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let text = String::deserialize(deserializer)?;
        match text.parse() {
            Ok(metric) => Ok(metric),
            Err(known_metric_error) => {
                let custom = Self::Custom(text);
                custom.validate().map_err(|_| serde::de::Error::custom(known_metric_error))?;
                Ok(custom)
            }
        }
    }
}

/// Learning-task parameters.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LearningTaskParameters {
    /// The loss being minimised, plus its own tuning parameters.
    pub objective: Objective,
    /// Initial prediction (in margin space). `None` lets XGBoost estimate the
    /// intercept from the training labels.
    pub base_score: Option<f32>,
    /// Metrics evaluated on the watchlist each round. Empty means "just the
    /// objective's default metric".
    pub eval_metric: Vec<EvalMetric>,
    /// PRNG seed, upstream alias `random_state`.
    pub seed: i64,
    /// Reseed the PRNG from the iteration number each round.
    pub seed_per_iteration: bool,
    /// Number of output targets. `None` lets XGBoost infer it from the labels.
    pub num_target: Option<u32>,
    /// Estimate the intercept from the training data. Ignored when
    /// `base_score` is set.
    pub boost_from_average: bool,
    /// Weight multiplier for positive examples; the usual lever for class
    /// imbalance. XGBoost's documentation lists this under the tree booster,
    /// but it is read by the objective (`RegLossParam`), so it lives here.
    pub scale_pos_weight: f32,
}

impl Default for LearningTaskParameters {
    fn default() -> Self {
        Self {
            objective: Objective::RegSquaredError,
            base_score: None,
            eval_metric: Vec::new(),
            seed: 0,
            seed_per_iteration: false,
            num_target: None,
            boost_from_average: true,
            scale_pos_weight: 1.0,
        }
    }
}

impl LearningTaskParameters {
    /// Start from XGBoost's defaults.
    pub fn builder() -> LearningTaskParametersBuilder {
        LearningTaskParametersBuilder::default()
    }

    /// Validate every field, the objective's own parameters, and every metric.
    pub fn validate(&self) -> Result<()> {
        self.objective.validate()?;
        for metric in &self.eval_metric {
            metric.validate()?;
        }
        if let Some(base_score) = self.base_score {
            // `LearnerConfiguration::ConfigureModelParam` rejects a non-finite
            // intercept.
            validate::finite("base_score", base_score)?;
        }
        if let Some(num_target) = self.num_target {
            validate::ge("num_target", num_target, 1)?;
        }
        validate::ge("scale_pos_weight", self.scale_pos_weight, 0.0)?;

        // `LearnerModelParam`'s constructor: multi-class multi-target is not
        // implemented upstream.
        if let (Some(num_class), Some(num_target)) = (self.objective.num_class(), self.num_target)
            && num_class > 1
            && num_target > 1
        {
            return Err(Error::invalid(
                "num_target",
                format!(
                    "multi-target multi-class is not supported: num_class={num_class}, \
                     num_target={num_target}"
                ),
            ));
        }
        Ok(())
    }
}

impl ToConfig for LearningTaskParameters {
    fn collect_config(&self, out: &mut Vec<ConfigEntry>) {
        self.objective.collect_config(out);
        push_opt(out, "base_score", self.base_score);
        for metric in &self.eval_metric {
            push(out, "eval_metric", metric);
        }
        push(out, "seed", self.seed);
        push_bool(out, "seed_per_iteration", self.seed_per_iteration);
        push_opt(out, "num_target", self.num_target);
        push_bool(out, "boost_from_average", self.boost_from_average);
        push(out, "scale_pos_weight", self.scale_pos_weight);
    }
}

/// Consuming builder for [`LearningTaskParameters`].
#[derive(Clone, Debug, Default)]
pub struct LearningTaskParametersBuilder {
    inner: LearningTaskParameters,
}

impl LearningTaskParametersBuilder {
    /// The loss being minimised.
    pub fn objective(mut self, objective: Objective) -> Self {
        self.inner.objective = objective;
        self
    }

    /// Pin the intercept instead of estimating it.
    pub fn base_score(mut self, base_score: f32) -> Self {
        self.inner.base_score = Some(base_score);
        self
    }

    /// Metrics evaluated on the watchlist each round.
    pub fn eval_metric(mut self, metrics: impl Into<Vec<EvalMetric>>) -> Self {
        self.inner.eval_metric = metrics.into();
        self
    }

    /// PRNG seed.
    pub fn seed(mut self, seed: i64) -> Self {
        self.inner.seed = seed;
        self
    }

    /// Reseed the PRNG from the iteration number each round.
    pub fn seed_per_iteration(mut self, seed_per_iteration: bool) -> Self {
        self.inner.seed_per_iteration = seed_per_iteration;
        self
    }

    /// Number of output targets; leave unset to infer from the labels.
    pub fn num_target(mut self, num_target: u32) -> Self {
        self.inner.num_target = Some(num_target);
        self
    }

    /// Estimate the intercept from the training data.
    pub fn boost_from_average(mut self, boost_from_average: bool) -> Self {
        self.inner.boost_from_average = boost_from_average;
        self
    }

    /// Weight multiplier for positive examples.
    pub fn scale_pos_weight(mut self, scale_pos_weight: f32) -> Self {
        self.inner.scale_pos_weight = scale_pos_weight;
        self
    }

    /// Validate and produce the parameters.
    pub fn build(self) -> Result<LearningTaskParameters> {
        self.inner.validate()?;
        Ok(self.inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_xgboost() {
        let params = LearningTaskParameters::default();
        assert_eq!(params.objective, Objective::RegSquaredError);
        assert_eq!(params.base_score, None);
        assert_eq!(params.seed, 0);
        assert!(params.boost_from_average);
        assert_eq!(params.scale_pos_weight, 1.0);
        params.validate().unwrap();
    }

    #[test]
    fn objective_names_round_trip() {
        for name in [
            "reg:squarederror",
            "reg:squaredlogerror",
            "reg:logistic",
            "reg:pseudohubererror",
            "reg:absoluteerror",
            "reg:quantileerror",
            "reg:expectileerror",
            "reg:gamma",
            "reg:tweedie",
            "reg:linear",
            "count:poisson",
            "survival:cox",
            "survival:aft",
            "binary:logistic",
            "binary:logitraw",
            "binary:hinge",
            "multi:softmax",
            "multi:softprob",
            "rank:pairwise",
            "rank:ndcg",
            "rank:map",
        ] {
            let objective: Objective = name.parse().unwrap();
            assert_eq!(objective.name(), name);
        }
        assert!("reg:nonsense".parse::<Objective>().is_err());
    }

    #[test]
    fn objective_carries_its_own_parameters() {
        let config = Objective::SurvivalAft {
            aft_loss_distribution: AftDistribution::Logistic,
            aft_loss_distribution_scale: 2.0,
        }
        .to_config_map();
        assert_eq!(config["objective"], "survival:aft");
        assert_eq!(config["aft_loss_distribution"], "logistic");
        assert_eq!(config["aft_loss_distribution_scale"], "2");

        let config = Objective::MultiSoftprob { num_class: 5 }.to_config_map();
        assert_eq!(config["num_class"], "5");

        let config =
            Objective::RegQuantileError { quantile_alpha: vec![0.1, 0.5, 0.9] }.to_config_map();
        assert_eq!(config["quantile_alpha"], "(0.1,0.5,0.9)");

        // A squared-error fit never emits another objective's knobs.
        let config = Objective::RegSquaredError.to_config_map();
        assert_eq!(config.len(), 1);
    }

    #[test]
    fn objective_parameters_are_validated() {
        assert!(
            Objective::RegTweedie { tweedie_variance_power: 2.0 }.validate().is_err(),
            "the variance power range is [1, 2)"
        );
        assert!(Objective::RegTweedie { tweedie_variance_power: 1.0 }.validate().is_ok());
        assert!(Objective::RegPseudoHuberError { huber_slope: 0.0 }.validate().is_err());
        assert!(Objective::MultiSoftmax { num_class: 0 }.validate().is_err());
        assert!(
            Objective::RegQuantileError { quantile_alpha: vec![0.9, 0.1] }.validate().is_err(),
            "alphas must be ascending"
        );
        assert!(Objective::RegQuantileError { quantile_alpha: vec![] }.validate().is_err());
        assert!(
            Objective::RegExpectileError { expectile_alpha: vec![1.5] }.validate().is_err(),
            "alphas must be in [0, 1]"
        );
    }

    #[test]
    fn lambdarank_defaults_and_resolution() {
        let params = LambdaRankParameters::default();
        assert_eq!(params.pair_method, LambdaRankPairMethod::TopK);
        assert_eq!(params.resolved_num_pair_per_sample(), 32);
        assert!(params.normalization && params.score_normalization && params.ndcg_exp_gain);
        assert!(!params.unbiased);

        let mean = LambdaRankParameters {
            pair_method: LambdaRankPairMethod::Mean,
            ..LambdaRankParameters::default()
        };
        assert_eq!(mean.resolved_num_pair_per_sample(), 1);

        let config = Objective::RankNdcg(params).to_config_map();
        assert_eq!(config["objective"], "rank:ndcg");
        assert_eq!(config["lambdarank_pair_method"], "topk");
        assert_eq!(config["ndcg_exp_gain"], "1");
        assert!(!config.contains_key("lambdarank_num_pair_per_sample"));
    }

    #[test]
    fn eval_metric_names_round_trip() {
        for name in [
            "rmse",
            "rmsle",
            "mae",
            "mape",
            "mphe",
            "logloss",
            "error",
            "error@0.7",
            "merror",
            "mlogloss",
            "auc",
            "aucpr",
            "pre",
            "pre@3",
            "ndcg",
            "ndcg@5",
            "ndcg-",
            "ndcg@5-",
            "map",
            "map@10",
            "map-",
            "map@10-",
            "poisson-nloglik",
            "gamma-nloglik",
            "cox-nloglik",
            "gamma-deviance",
            "tweedie-nloglik@1.5",
            "aft-nloglik",
            "interval-regression-accuracy",
            "quantile",
            "expectile",
            "ams@0.15",
        ] {
            let metric: EvalMetric = name.parse().unwrap();
            assert_eq!(metric.to_string(), name, "round trip for {name}");
        }
    }

    #[test]
    fn eval_metric_rejects_bad_spellings() {
        for name in ["", "rmsee", "rmse@1", "tweedie-nloglik", "ams", "error@x", "ndcg@x"] {
            assert!(name.parse::<EvalMetric>().is_err(), "should reject {name:?}");
        }
        assert!(EvalMetric::Custom("my metric".into()).validate().is_err());
        assert!(EvalMetric::Custom("my-metric".into()).validate().is_ok());
        assert!(EvalMetric::TweedieNegLogLik(2.0).validate().is_err());
    }

    #[test]
    fn repeated_eval_metrics_survive_the_map() {
        let params = LearningTaskParameters::builder()
            .objective(Objective::BinaryLogistic)
            .eval_metric([EvalMetric::Logloss, EvalMetric::Auc, EvalMetric::ErrorAt(0.7)])
            .build()
            .unwrap();
        assert_eq!(
            params.to_config().iter().filter(|(k, _)| k == "eval_metric").count(),
            3
        );
        assert_eq!(params.to_config_map()["eval_metric"], "logloss,auc,error@0.7");
    }

    #[test]
    fn rejects_multi_target_multi_class() {
        let err = LearningTaskParameters::builder()
            .objective(Objective::MultiSoftprob { num_class: 3 })
            .num_target(2)
            .build()
            .unwrap_err();
        assert!(err.to_string().contains("num_target"));

        LearningTaskParameters::builder()
            .objective(Objective::MultiSoftprob { num_class: 3 })
            .num_target(1)
            .build()
            .unwrap();
    }

    #[test]
    fn emits_upstream_parameter_names() {
        let config = LearningTaskParameters::builder()
            .objective(Objective::BinaryLogistic)
            .base_score(0.5)
            .seed(42)
            .scale_pos_weight(3.0)
            .build()
            .unwrap()
            .to_config_map();
        assert_eq!(config["objective"], "binary:logistic");
        assert_eq!(config["base_score"], "0.5");
        assert_eq!(config["seed"], "42");
        assert_eq!(config["scale_pos_weight"], "3");
        assert_eq!(config["boost_from_average"], "1");
        assert!(!config.contains_key("num_target"));
    }
}
