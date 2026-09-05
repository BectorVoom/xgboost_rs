//! Evaluation metrics.
//!
//! Every `eval_metric` XGBoost accepts is implemented here, including the
//! parameterised spellings: `error@0.7`, `tweedie-nloglik@1.5`, `ndcg@5-`,
//! `map@10`, `pre@3`, `ams@0.15`. A metric is constructed from the same string
//! the parameter surface emits, so an `eval_metric` that builds is an
//! `eval_metric` that runs.
//!
//! # What a metric sees
//!
//! Predictions arrive after the objective's `EvalTransform`, so a metric never
//! has to know which objective produced them: `mlogloss` always sees class
//! probabilities and `aft-nloglik` always sees the untransformed margin. The
//! number of outputs per row is `preds.len() / info.num_row`, which is how the
//! elementwise metrics stay correct for multi-output fits.

pub mod elementwise;
pub mod multiclass;
pub mod rank;
pub mod survival;

use crate::data::MetaInfo;
use crate::parameters::EvalMetric;
use crate::Result;

/// A metric evaluated on transformed predictions.
pub trait Metric {
    /// The `eval_metric` spelling this metric reports under, including any
    /// `@argument`.
    fn name(&self) -> &str;

    /// Score `preds` against `info`. Lower is better unless the metric's
    /// documentation says otherwise.
    fn eval(&self, preds: &[f32], info: &MetaInfo) -> f64;
}

/// Sum of `(value, weight)` over every `(row, output)`, the shape every
/// elementwise metric reduces to.
///
/// The output count is taken from the prediction length, so a multi-output fit
/// scores each output against its own label column and a quantile fit — several
/// outputs against one label column — still lines up.
pub(crate) fn elementwise_reduce(
    preds: &[f32],
    info: &MetaInfo,
    row: impl Fn(f32, f32) -> f64 + Sync,
) -> (f64, f64) {
    use rayon::prelude::*;
    if info.num_row == 0 {
        return (0.0, 0.0);
    }
    let n_groups = (preds.len() / info.num_row).max(1);
    let n_targets = info.n_targets();
    // Blocks of rows summed in parallel, the block sums added in row order:
    // the same number whatever the thread count, and a pass that was 3 ms of
    // a 22 ms round at 500 000 rows on one core is a fraction of that.
    // Within a block, `REDUCE_LANES` interleaved lanes each summed in row
    // order and then added in lane order: the shape a device reduction takes
    // (`gpu::objective::rmse_lanes_kernel`), so the two give one number.
    let blocks: Vec<(f64, f64)> = crate::threading::install(|| {
        (0..info.num_row.div_ceil(REDUCE_ROWS))
            .into_par_iter()
            .map(|b| {
                let (mut esum, mut wsum) = (0.0f64, 0.0f64);
                let end = ((b + 1) * REDUCE_ROWS).min(info.num_row);
                for lane in 0..REDUCE_LANES {
                    let (mut le, mut lw) = (0.0f64, 0.0f64);
                    let mut i = b * REDUCE_ROWS + lane;
                    while i < end {
                        let w = info.weight(i) as f64;
                        for t in 0..n_groups {
                            let label = info.label(i, t.min(n_targets - 1));
                            le += row(label, preds[i * n_groups + t]) * w;
                            lw += w;
                        }
                        i += REDUCE_LANES;
                    }
                    esum += le;
                    wsum += lw;
                }
                (esum, wsum)
            })
            .collect()
    });
    blocks.iter().fold((0.0, 0.0), |(e, w), (be, bw)| (e + be, w + bw))
}

/// Rows per block of [`elementwise_reduce`]'s parallel sum.
pub(crate) const REDUCE_ROWS: usize = 4096;
/// Interleaved partial sums within a block; see [`elementwise_reduce`].
pub(crate) const REDUCE_LANES: usize = 64;

/// `wsum == 0 ? esum : esum / wsum`, the ending most metrics share.
#[inline]
pub(crate) fn mean(esum: f64, wsum: f64) -> f64 {
    if wsum == 0.0 { esum } else { esum / wsum }
}

/// Construct a metric from its XGBoost name.
///
/// The name is parsed with the same [`EvalMetric`] grammar the parameter
/// surface uses, so `error@0.7` and `ndcg@5-` are accepted here exactly where
/// they are accepted there.
pub fn create(name: &str) -> Result<Box<dyn Metric>> {
    let metric: EvalMetric = name.parse()?;
    create_from(&metric)
}

/// Construct a metric from a typed [`EvalMetric`].
pub fn create_from(metric: &EvalMetric) -> Result<Box<dyn Metric>> {
    use EvalMetric::*;
    use elementwise::{Elementwise, Kind};
    Ok(match metric {
        Rmse => Box::new(Elementwise::new(Kind::Rmse)),
        Rmsle => Box::new(Elementwise::new(Kind::Rmsle)),
        Mae => Box::new(Elementwise::new(Kind::Mae)),
        Mape => Box::new(Elementwise::new(Kind::Mape)),
        Mphe => Box::new(Elementwise::new(Kind::Mphe { slope: 1.0 })),
        Logloss => Box::new(Elementwise::new(Kind::Logloss)),
        Error => Box::new(Elementwise::new(Kind::Error { threshold: 0.5 })),
        ErrorAt(threshold) => Box::new(Elementwise::new(Kind::Error { threshold: *threshold })),
        PoissonNegLogLik => Box::new(Elementwise::new(Kind::PoissonNegLogLik)),
        GammaNegLogLik => Box::new(Elementwise::new(Kind::GammaNegLogLik)),
        GammaDeviance => Box::new(Elementwise::new(Kind::GammaDeviance)),
        TweedieNegLogLik(rho) => Box::new(Elementwise::new(Kind::TweedieNegLogLik { rho: *rho })),
        Quantile => Box::new(elementwise::PinballLoss::quantile()),
        Expectile => Box::new(elementwise::PinballLoss::expectile()),
        MError => Box::new(multiclass::MultiClass::error()),
        MLogloss => Box::new(multiclass::MultiClass::logloss()),
        Auc => Box::new(rank::Auc::roc()),
        Aucpr => Box::new(rank::Auc::pr()),
        Pre(top_n) => Box::new(rank::RankScore::precision(*top_n)),
        Ndcg { top_n, minus } => Box::new(rank::RankScore::ndcg(*top_n, *minus)),
        Map { top_n, minus } => Box::new(rank::RankScore::map(*top_n, *minus)),
        Ams(ratio) => Box::new(rank::Ams::new(*ratio)),
        CoxNegLogLik => Box::new(survival::CoxNegLogLik),
        AftNegLogLik => Box::new(survival::AftNegLogLik::default()),
        IntervalRegressionAccuracy => Box::new(survival::IntervalRegressionAccuracy),
        // `EvalMetric::Error` is in scope from the glob, so the crate error
        // type has to be named through its path here.
        Custom(name) => {
            return Err(crate::Error::invalid(
                "eval_metric",
                format!("`{name}` is a plugin metric with no implementation in this crate"),
            ));
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(labels: &[f32]) -> MetaInfo {
        MetaInfo {
            num_row: labels.len(),
            num_col: 1,
            labels: labels.to_vec(),
            num_target: 1,
            ..Default::default()
        }
    }

    /// Every metric the parameter surface can spell must construct and run.
    #[test]
    fn every_metric_spelling_constructs_and_scores() {
        let d = info(&[1.0, 0.0, 1.0, 0.0]);
        let preds = [0.9f32, 0.2, 0.6, 0.1];
        for name in [
            "rmse",
            "rmsle",
            "mae",
            "mphe",
            "logloss",
            "error",
            "error@0.7",
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
            "gamma-deviance",
            "tweedie-nloglik@1.5",
            "quantile",
            "expectile",
            "ams@0.15",
        ] {
            let metric = create(name).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(metric.name(), name, "a metric must report the name it was built from");
            let score = metric.eval(&preds, &d);
            assert!(score.is_finite(), "{name} scored {score}");
        }

        // `mape` divides by the label, so it needs one that is not zero — the
        // same restriction it has upstream.
        let positive = info(&[2.0, 4.0, 2.0, 4.0]);
        let mape = create("mape").unwrap();
        assert_eq!(mape.name(), "mape");
        assert!(mape.eval(&preds, &positive).is_finite());
    }

    #[test]
    fn multiclass_and_survival_metrics_construct() {
        for name in
            ["merror", "mlogloss", "cox-nloglik", "aft-nloglik", "interval-regression-accuracy"]
        {
            let metric = create(name).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(metric.name(), name);
        }
    }

    #[test]
    fn a_misspelled_metric_is_rejected() {
        for name in ["rmsee", "rmse@1", "tweedie-nloglik", "ams", "", "ndcg@x"] {
            assert!(create(name).is_err(), "{name:?} should be rejected");
        }
    }

    #[test]
    fn a_plugin_metric_has_no_implementation() {
        // `Metric` is not `Debug`, so the error is matched out by hand.
        match create_from(&EvalMetric::Custom("my-metric".into())) {
            Ok(_) => panic!("a plugin metric should not resolve"),
            Err(e) => assert!(e.to_string().contains("my-metric"), "{e}"),
        }
    }
}
