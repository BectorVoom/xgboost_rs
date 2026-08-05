//! Every objective and every evaluation metric, exercised through the public
//! `train` API by its *effect*: an objective must fit data its loss is designed
//! for better than the intercept alone, and a metric must move in the direction
//! its definition says it should.
//!
//! An objective that is accepted and then quietly behaves like squared error is
//! worse than one that is rejected, so nothing here checks only that a fit
//! succeeds.

use xgboost_rs::parameters::{
    AftDistribution, BoosterParameters, BoosterType, EvalMetric, GeneralParameters,
    LambdaRankPairMethod, LambdaRankParameters, LearningTaskParameters, Objective,
    TrainingParameters, TreeBoosterParameters, VerboseEval, Verbosity,
};
use xgboost_rs::{Booster, DMatrix, api};

/// A deterministic pseudo-random stream, so every fixture is reproducible.
struct Rng(u64);

impl Rng {
    fn new() -> Self {
        Self(0x1234_5678_9abc_def0)
    }

    fn next(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32 / (1u32 << 24) as f32
    }
}

fn params(objective: Objective, rounds: u32) -> TrainingParameters {
    TrainingParameters {
        booster: BoosterParameters {
            booster: BoosterType::Gbtree(TreeBoosterParameters {
                max_depth: 4,
                ..Default::default()
            }),
            general: GeneralParameters { verbosity: Verbosity::Silent, ..Default::default() },
            learning: LearningTaskParameters { objective, ..Default::default() },
        },
        num_boost_round: rounds,
        verbose_eval: VerboseEval::Silent,
        ..Default::default()
    }
}

fn train(p: &TrainingParameters, d: &DMatrix) -> Booster {
    api::train(p, d, &[]).expect("training failed").0
}

/// Features that carry a real signal, plus the raw feature values.
fn features(rows: usize, cols: usize) -> (Vec<f32>, DMatrix) {
    let mut rng = Rng::new();
    let x: Vec<f32> = (0..rows * cols).map(|_| rng.next()).collect();
    let d = DMatrix::from_dense(&x, rows, cols, f32::NAN).unwrap();
    (x, d)
}

/// Mean squared error of `preds` against `labels`.
fn mse(preds: &[f32], labels: &[f32]) -> f64 {
    preds
        .iter()
        .zip(labels)
        .map(|(p, y)| {
            let d = (p - y) as f64;
            d * d
        })
        .sum::<f64>()
        / labels.len() as f64
}

// ------------------------------------------------------ regression losses --

/// Each regression objective must beat its own intercept on data its link fits.
#[test]
fn every_regression_objective_learns_its_own_link() {
    let (x, mut d) = features(600, 3);
    // A positive, skewed target: valid for the log-link objectives and for the
    // squared-error family alike.
    let y: Vec<f32> = (0..600)
        .map(|r| {
            let row = &x[r * 3..(r + 1) * 3];
            (1.0 + 2.0 * row[0] + row[1]).exp().min(50.0)
        })
        .collect();
    d.set_labels(&y).unwrap();

    let baseline = mse(&vec![y.iter().sum::<f32>() / y.len() as f32; y.len()], &y);

    for objective in [
        Objective::RegSquaredError,
        Objective::RegSquaredLogError,
        Objective::RegPseudoHuberError { huber_slope: 1.0 },
        Objective::RegAbsoluteError,
        Objective::RegGamma,
        Objective::RegTweedie { tweedie_variance_power: 1.5 },
        Objective::CountPoisson { max_delta_step: 0.7 },
    ] {
        let name = objective.name();
        let booster = train(&params(objective, 25), &d);
        let preds = booster.predict(&d);
        assert_eq!(preds.len(), d.num_row(), "{name}");
        assert!(preds.iter().all(|p| p.is_finite()), "{name} produced a non-finite prediction");
        assert!(
            mse(&preds, &y) < baseline,
            "{name} did not beat the intercept: {} vs {baseline}",
            mse(&preds, &y)
        );
    }
}

/// The log-link objectives predict on the response scale, not the margin.
#[test]
fn the_log_link_objectives_predict_positive_values() {
    let (_, mut d) = features(300, 2);
    let y: Vec<f32> = (0..300).map(|i| (i % 7) as f32 + 1.0).collect();
    d.set_labels(&y).unwrap();

    for objective in [
        Objective::RegGamma,
        Objective::CountPoisson { max_delta_step: 0.7 },
        Objective::RegTweedie { tweedie_variance_power: 1.5 },
    ] {
        let name = objective.name();
        let booster = train(&params(objective, 5), &d);
        assert!(booster.predict(&d).iter().all(|p| *p > 0.0), "{name} predicted a non-positive");
        // The margin is the log of it, so the two must differ.
        let margin = booster.predict_margin(&d);
        let value = booster.predict(&d);
        assert!(margin[0] != value[0], "{name} applied no transform");
    }
}

/// `reg:logistic` keeps its predictions inside `[0, 1]`; `binary:logitraw`
/// deliberately does not.
#[test]
fn the_logistic_objectives_differ_in_what_they_report() {
    let (_, mut d) = features(400, 2);
    let y: Vec<f32> = (0..400).map(|i| f32::from(i % 3 == 0)).collect();
    d.set_labels(&y).unwrap();

    for objective in [Objective::RegLogistic, Objective::BinaryLogistic] {
        let name = objective.name();
        let preds = train(&params(objective, 10), &d).predict(&d);
        assert!(preds.iter().all(|p| (0.0..=1.0).contains(p)), "{name} left the unit interval");
    }

    let raw = train(&params(Objective::BinaryLogitRaw, 10), &d).predict(&d);
    assert!(raw.iter().any(|p| !(0.0..=1.0).contains(p)), "logitraw should report a margin");

    // Hinge reports a hard class.
    let hinge = train(&params(Objective::BinaryHinge, 10), &d).predict(&d);
    assert!(hinge.iter().all(|p| *p == 0.0 || *p == 1.0), "hinge must predict 0 or 1");
}

/// A label the objective's link cannot represent is rejected before the fit,
/// not silently turned into a NaN.
#[test]
fn out_of_range_labels_are_rejected_per_objective() {
    let (_, mut d) = features(50, 2);
    d.set_labels(&vec![-1.0f32; 50]).unwrap();

    for objective in [
        Objective::RegGamma,
        Objective::CountPoisson { max_delta_step: 0.7 },
        Objective::RegTweedie { tweedie_variance_power: 1.5 },
        Objective::BinaryLogistic,
        Objective::RegSquaredLogError,
    ] {
        let name = objective.name();
        match api::train(&params(objective, 1), &d, &[]) {
            Ok(_) => panic!("{name} accepted a label its link cannot take"),
            Err(e) => assert!(e.to_string().contains(name), "{name}: {e}"),
        }
    }
}

// -------------------------------------------------------- multi-output ----

/// `multi:softprob` produces one probability per class, and `multi:softmax`
/// the class index — from the same trees.
#[test]
fn multiclass_grows_one_tree_per_class_and_predicts_every_class() {
    let (x, mut d) = features(600, 4);
    // Three classes, separable on the first feature.
    let y: Vec<f32> = (0..600)
        .map(|r| {
            let v = x[r * 4];
            if v < 0.33 {
                0.0
            } else if v < 0.66 {
                1.0
            } else {
                2.0
            }
        })
        .collect();
    d.set_labels(&y).unwrap();

    let rounds = 5;
    let prob = train(&params(Objective::MultiSoftprob { num_class: 3 }, rounds), &d);
    assert_eq!(prob.num_output_group(), 3);
    assert_eq!(prob.num_trees(), rounds as usize * 3, "one tree per class per round");
    assert_eq!(prob.boosted_rounds(), rounds as usize, "a round still counts once");

    let preds = prob.predict(&d);
    assert_eq!(preds.len(), 600 * 3);
    for row in preds.chunks(3) {
        let sum: f32 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "probabilities must sum to 1, got {row:?}");
    }

    let class = train(&params(Objective::MultiSoftmax { num_class: 3 }, rounds), &d);
    let labels = class.predict(&d);
    assert_eq!(labels.len(), 600, "softmax reports one class per row");
    assert!(labels.iter().all(|c| (0.0..3.0).contains(c)));
    // The fit actually learned the split, not just the majority class.
    let correct = labels.iter().zip(&y).filter(|(p, t)| *p == *t).count();
    assert!(correct > 500, "only {correct}/600 rows classified correctly");
}

/// A multiclass model survives a save/load round trip with its groups intact.
#[test]
fn a_multiclass_model_round_trips_through_json() {
    let (x, mut d) = features(300, 3);
    let y: Vec<f32> = (0..300).map(|r| f32::from(x[r * 3] > 0.5)).collect();
    d.set_labels(&y).unwrap();

    let booster = train(&params(Objective::MultiSoftprob { num_class: 2 }, 4), &d);
    let json: serde_json::Value = serde_json::from_str(&booster.save_model()).unwrap();
    assert_eq!(json["learner"]["learner_model_param"]["num_class"], "2");
    let tree_info = json["learner"]["gradient_booster"]["model"]["tree_info"].as_array().unwrap();
    assert_eq!(tree_info.len(), 8);
    // A round boosts every group, so the groups interleave round by round.
    assert_eq!(tree_info.iter().map(|v| v.as_u64().unwrap()).collect::<Vec<_>>(), [
        0, 1, 0, 1, 0, 1, 0, 1
    ]);

    let reloaded = Booster::load_model(&booster.save_model()).unwrap();
    assert_eq!(reloaded.num_output_group(), 2);
    assert_eq!(reloaded.boosted_rounds(), 4);
    assert_eq!(reloaded.predict(&d), booster.predict(&d));
}

/// `num_target > 1` is a real multi-output fit: one tree per target per round,
/// and each target learns its own signal.
#[test]
fn multi_target_regression_learns_each_target_separately() {
    let (x, mut d) = features(500, 3);
    // Target 0 follows feature 0, target 1 follows the negation of feature 1.
    let mut y = Vec::with_capacity(1000);
    for r in 0..500 {
        y.push(x[r * 3]);
        y.push(1.0 - x[r * 3 + 1]);
    }
    d.set_labels_multi(&y, 2).unwrap();

    let booster = train(&params(Objective::RegSquaredError, 20), &d);
    assert_eq!(booster.num_output_group(), 2);
    assert_eq!(booster.num_trees(), 40);

    let preds = booster.predict(&d);
    assert_eq!(preds.len(), 1000);
    let first: Vec<f32> = preds.iter().step_by(2).copied().collect();
    let second: Vec<f32> = preds.iter().skip(1).step_by(2).copied().collect();
    let want_first: Vec<f32> = y.iter().step_by(2).copied().collect();
    let want_second: Vec<f32> = y.iter().skip(1).step_by(2).copied().collect();
    assert!(mse(&first, &want_first) < 0.01, "target 0 was not learned");
    assert!(mse(&second, &want_second) < 0.01, "target 1 was not learned");
}

/// The quantile loss fits one output per `quantile_alpha`, and the outputs come
/// back in ascending order.
#[test]
fn quantile_regression_fits_every_requested_quantile() {
    let (x, mut d) = features(800, 2);
    // Heteroscedastic noise, so the quantiles genuinely differ.
    let mut rng = Rng::new();
    let y: Vec<f32> = (0..800).map(|r| x[r * 2] * 2.0 + rng.next()).collect();
    d.set_labels(&y).unwrap();

    let alphas = vec![0.1, 0.5, 0.9];
    let booster = train(&params(Objective::RegQuantileError { quantile_alpha: alphas }, 30), &d);
    assert_eq!(booster.num_output_group(), 3);

    let preds = booster.predict(&d);
    assert_eq!(preds.len(), 800 * 3);
    for row in preds.chunks(3) {
        assert!(row[0] <= row[1] && row[1] <= row[2], "quantiles crossed: {row:?}");
    }

    // The 10% quantile really does sit below 10% of the labels far more often
    // than the 90% one does.
    let below_low = (0..800).filter(|&i| y[i] < preds[i * 3]).count();
    let below_high = (0..800).filter(|&i| y[i] < preds[i * 3 + 2]).count();
    assert!(below_low < below_high, "{below_low} vs {below_high}");
    assert!(below_low < 300, "the 10% quantile is too high: {below_low}/800 labels below it");
    assert!(below_high > 500, "the 90% quantile is too low: {below_high}/800 labels below it");
}

/// The expectile loss also produces one non-crossing output per alpha.
#[test]
fn expectile_regression_fits_ascending_expectiles() {
    let (x, mut d) = features(500, 2);
    let mut rng = Rng::new();
    let y: Vec<f32> = (0..500).map(|r| x[r * 2] + rng.next()).collect();
    d.set_labels(&y).unwrap();

    let booster =
        train(&params(Objective::RegExpectileError { expectile_alpha: vec![0.2, 0.5, 0.8] }, 20), &d);
    assert_eq!(booster.num_output_group(), 3);
    let preds = booster.predict(&d);
    for row in preds.chunks(3) {
        assert!(row[0] <= row[1] && row[1] <= row[2], "expectiles crossed: {row:?}");
    }
}

// ------------------------------------------------------------- survival ----

/// `survival:aft` learns from a censoring interval, for each distribution.
#[test]
fn aft_learns_from_censored_intervals() {
    let (x, mut d) = features(500, 2);
    // Survival time rises with feature 0; half the rows are right-censored.
    let time: Vec<f32> = (0..500).map(|r| 1.0 + 20.0 * x[r * 2]).collect();
    let lower = time.clone();
    let upper: Vec<f32> =
        time.iter().enumerate().map(|(i, t)| if i % 2 == 0 { *t } else { f32::INFINITY }).collect();
    d.set_labels(&time).unwrap();
    d.set_label_bounds(&lower, &upper).unwrap();

    for dist in [AftDistribution::Normal, AftDistribution::Logistic, AftDistribution::Extreme] {
        let objective = Objective::SurvivalAft {
            aft_loss_distribution: dist,
            aft_loss_distribution_scale: 1.0,
        };
        let booster = train(&params(objective, 25), &d);
        let preds = booster.predict(&d);
        assert!(preds.iter().all(|p| p.is_finite() && *p > 0.0), "{dist} predicted {:?}", preds[0]);

        // Predicted survival time must rise with the feature it depends on.
        let low: f32 = (0..500).filter(|r| x[r * 2] < 0.2).map(|r| preds[r]).sum();
        let high: f32 = (0..500).filter(|r| x[r * 2] > 0.8).map(|r| preds[r]).sum();
        let n_low = (0..500).filter(|r| x[r * 2] < 0.2).count() as f32;
        let n_high = (0..500).filter(|r| x[r * 2] > 0.8).count() as f32;
        assert!(high / n_high > low / n_low, "{dist} learned no signal");
    }
}

/// AFT needs its censoring interval, and says so.
#[test]
fn aft_without_bounds_is_rejected() {
    let (_, mut d) = features(50, 2);
    d.set_labels(&vec![1.0f32; 50]).unwrap();
    let objective = Objective::SurvivalAft {
        aft_loss_distribution: AftDistribution::Normal,
        aft_loss_distribution_scale: 1.0,
    };
    match api::train(&params(objective, 1), &d, &[]) {
        Ok(_) => panic!("AFT should need its bounds"),
        Err(e) => assert!(e.to_string().contains("set_label_bounds"), "{e}"),
    }
}

/// `survival:cox` ranks risk: rows that failed early must get a higher hazard
/// than rows that survived.
#[test]
fn cox_ranks_early_failures_above_late_ones() {
    let (x, mut d) = features(400, 2);
    // Negative labels are censored; time falls as feature 0 rises, so a high
    // feature 0 means a high hazard.
    let y: Vec<f32> = (0..400)
        .map(|r| {
            let t = 1.0 + 10.0 * (1.0 - x[r * 2]);
            if r % 3 == 0 { -t } else { t }
        })
        .collect();
    d.set_labels(&y).unwrap();

    let booster = train(&params(Objective::SurvivalCox, 20), &d);
    let preds = booster.predict(&d);
    assert!(preds.iter().all(|p| p.is_finite() && *p > 0.0));

    let hazard_low: f32 =
        (0..400).filter(|r| x[r * 2] < 0.2).map(|r| preds[r]).sum::<f32>().max(1e-9);
    let hazard_high: f32 =
        (0..400).filter(|r| x[r * 2] > 0.8).map(|r| preds[r]).sum::<f32>().max(1e-9);
    assert!(hazard_high > hazard_low, "cox learned no risk ordering: {hazard_high} vs {hazard_low}");
}

// -------------------------------------------------------------- ranking ----

/// The three `rank:*` objectives learn to order documents within a query.
#[test]
fn every_ranking_objective_learns_to_order_a_query() {
    // 40 queries of 5 documents; relevance follows feature 0.
    let (queries, docs) = (40usize, 5usize);
    let rows = queries * docs;
    let mut x = Vec::with_capacity(rows * 2);
    let mut y = Vec::with_capacity(rows);
    let mut rng = Rng::new();
    for _ in 0..queries {
        for k in 0..docs {
            let relevance = (docs - 1 - k) as f32;
            x.push(relevance / docs as f32 + rng.next() * 0.05);
            x.push(rng.next());
            y.push(relevance);
        }
    }
    let mut d = DMatrix::from_dense(&x, rows, 2, f32::NAN).unwrap();
    d.set_labels(&y).unwrap();
    d.set_group(&vec![docs; queries]).unwrap();

    for objective in [
        Objective::RankPairwise(LambdaRankParameters::default()),
        Objective::RankNdcg(LambdaRankParameters::default()),
        Objective::RankMap(LambdaRankParameters::default()),
    ] {
        let name = objective.name();
        let booster = train(&params(objective, 20), &d);
        let preds = booster.predict(&d);
        assert_eq!(preds.len(), rows, "{name}");

        // Within each query the most relevant document must score highest.
        let mut correct = 0;
        for q in 0..queries {
            let slice = &preds[q * docs..(q + 1) * docs];
            let best = slice
                .iter()
                .enumerate()
                .fold((0usize, f32::MIN), |acc, (i, &v)| if v > acc.1 { (i, v) } else { acc })
                .0;
            if best == 0 {
                correct += 1;
            }
        }
        assert!(correct > queries * 3 / 4, "{name} ordered only {correct}/{queries} queries");
    }
}

/// The `mean` pair method is a different sampler, so it must give a different
/// model — and still a reproducible one.
#[test]
fn the_lambdarank_pair_method_changes_the_fit() {
    let (queries, docs) = (20usize, 4usize);
    let rows = queries * docs;
    let mut rng = Rng::new();
    let x: Vec<f32> = (0..rows * 2).map(|_| rng.next()).collect();
    let y: Vec<f32> = (0..rows).map(|i| (i % docs) as f32).collect();
    let mut d = DMatrix::from_dense(&x, rows, 2, f32::NAN).unwrap();
    d.set_labels(&y).unwrap();
    d.set_group(&vec![docs; queries]).unwrap();

    let topk = train(&params(Objective::RankNdcg(LambdaRankParameters::default()), 5), &d);
    let mean_param = LambdaRankParameters {
        pair_method: LambdaRankPairMethod::Mean,
        ..LambdaRankParameters::default()
    };
    let p = params(Objective::RankNdcg(mean_param), 5);
    let mean = train(&p, &d);

    assert_ne!(topk.save_model(), mean.save_model(), "the pair method must reach the fit");
    assert_eq!(mean.save_model(), train(&p, &d).save_model(), "sampling must be reproducible");
}

/// `lambdarank_unbiased` re-weights pairs by an estimated examination
/// propensity, so it must reach the fit — and `lambdarank_bias_norm` must
/// change how far that estimate moves.
#[test]
fn unbiased_lambdamart_and_its_bias_norm_reach_the_fit() {
    let (queries, docs) = (30usize, 8usize);
    let rows = queries * docs;
    let mut rng = Rng::new();
    let mut x = Vec::with_capacity(rows * 2);
    let mut y = Vec::with_capacity(rows);
    for _ in 0..queries {
        for k in 0..docs {
            // Documents arrive in relevance order, which is the position the
            // debiasing attributes examination to.
            let relevance = (docs - 1 - k) as f32;
            x.push(relevance / docs as f32 + rng.next() * 0.1);
            x.push(rng.next());
            y.push(relevance);
        }
    }
    let mut d = DMatrix::from_dense(&x, rows, 2, f32::NAN).unwrap();
    d.set_labels(&y).unwrap();
    d.set_group(&vec![docs; queries]).unwrap();

    let fit = |param: LambdaRankParameters| {
        train(&params(Objective::RankNdcg(param), 12), &d).save_model()
    };

    let biased = fit(LambdaRankParameters::default());
    let unbiased =
        fit(LambdaRankParameters { unbiased: true, ..LambdaRankParameters::default() });
    assert_ne!(biased, unbiased, "`lambdarank_unbiased` must change the model");

    // The propensity update raises its ratio to `1 / (1 + bias_norm)`, so two
    // different norms give two different re-weightings.
    let sharp = fit(LambdaRankParameters {
        unbiased: true,
        bias_norm: 0.0,
        ..LambdaRankParameters::default()
    });
    let soft = fit(LambdaRankParameters {
        unbiased: true,
        bias_norm: 8.0,
        ..LambdaRankParameters::default()
    });
    assert_ne!(sharp, soft, "`lambdarank_bias_norm` must change the model");

    // And the parameter is not merely perturbing the fit: it still ranks.
    let booster = train(
        &params(
            Objective::RankNdcg(LambdaRankParameters {
                unbiased: true,
                ..LambdaRankParameters::default()
            }),
            30,
        ),
        &d,
    );
    let preds = booster.predict(&d);
    let mut correct = 0;
    for q in 0..queries {
        let slice = &preds[q * docs..(q + 1) * docs];
        let best = slice
            .iter()
            .enumerate()
            .fold((0usize, f32::MIN), |acc, (i, &v)| if v > acc.1 { (i, v) } else { acc })
            .0;
        if best == 0 {
            correct += 1;
        }
    }
    assert!(correct > queries * 3 / 4, "an unbiased fit ordered only {correct}/{queries} queries");
}

/// `lambdarank_unbiased` is state that evolves across rounds, so the same
/// configuration must still reproduce exactly.
#[test]
fn an_unbiased_ranking_fit_is_reproducible() {
    let (queries, docs) = (20usize, 6usize);
    let rows = queries * docs;
    let mut rng = Rng::new();
    let x: Vec<f32> = (0..rows * 2).map(|_| rng.next()).collect();
    let y: Vec<f32> = (0..rows).map(|i| (i % docs) as f32).collect();
    let mut d = DMatrix::from_dense(&x, rows, 2, f32::NAN).unwrap();
    d.set_labels(&y).unwrap();
    d.set_group(&vec![docs; queries]).unwrap();

    let p = params(
        Objective::RankNdcg(LambdaRankParameters {
            unbiased: true,
            ..LambdaRankParameters::default()
        }),
        10,
    );
    assert_eq!(train(&p, &d).save_model(), train(&p, &d).save_model());
}

/// A ranking fit reports the truncation level in its default metric name.
#[test]
fn the_ranking_default_metric_carries_the_truncation_level() {
    let (queries, docs) = (10usize, 4usize);
    let rows = queries * docs;
    let mut rng = Rng::new();
    let x: Vec<f32> = (0..rows).map(|_| rng.next()).collect();
    let y: Vec<f32> = (0..rows).map(|i| (i % docs) as f32).collect();
    let mut d = DMatrix::from_dense(&x, rows, 1, f32::NAN).unwrap();
    d.set_labels(&y).unwrap();
    d.set_group(&vec![docs; queries]).unwrap();

    let param =
        LambdaRankParameters { num_pair_per_sample: Some(3), ..LambdaRankParameters::default() };
    let p = params(Objective::RankNdcg(param), 2);
    let (_, history) = api::train(&p, &d, &[(&d, "train")]).unwrap();
    assert_eq!(history[0][0].0, "train-ndcg@3");
}

// -------------------------------------------------------------- metrics ----

/// Every metric runs inside a real fit and reports under its own name.
#[test]
fn every_metric_reports_from_a_fit() {
    let (x, mut d) = features(400, 3);
    let y: Vec<f32> = (0..400).map(|r| f32::from(x[r * 3] > 0.5)).collect();
    d.set_labels(&y).unwrap();

    let metrics = vec![
        EvalMetric::Rmse,
        EvalMetric::Rmsle,
        EvalMetric::Mae,
        EvalMetric::Mphe,
        EvalMetric::Logloss,
        EvalMetric::Error,
        EvalMetric::ErrorAt(0.7),
        EvalMetric::Auc,
        EvalMetric::Aucpr,
        EvalMetric::Pre(Some(3)),
        EvalMetric::Ndcg { top_n: Some(5), minus: true },
        EvalMetric::Map { top_n: None, minus: false },
        EvalMetric::Ams(0.15),
        EvalMetric::Quantile,
    ];
    let names: Vec<String> = metrics.iter().map(EvalMetric::to_string).collect();

    let mut p = params(Objective::BinaryLogistic, 3);
    p.booster.learning.eval_metric = metrics;
    let (_, history) = api::train(&p, &d, &[(&d, "train")]).unwrap();

    let reported: Vec<String> =
        history[0].iter().map(|(n, _)| n.trim_start_matches("train-").to_owned()).collect();
    assert_eq!(reported, names);
    assert!(history[0].iter().all(|(_, v)| v.is_finite()), "{:?}", history[0]);
}

/// The objective's default metric is used when none is configured, and it
/// carries the objective's own parameters where it has any.
#[test]
fn the_default_metric_follows_the_objective() {
    let (_, mut d) = features(200, 2);
    d.set_labels(&(0..200).map(|i| (i % 5) as f32 + 1.0).collect::<Vec<_>>()).unwrap();

    for (objective, expected) in [
        (Objective::RegSquaredError, "rmse"),
        (Objective::RegAbsoluteError, "mae"),
        (Objective::RegGamma, "gamma-deviance"),
        (Objective::CountPoisson { max_delta_step: 0.7 }, "poisson-nloglik"),
        (Objective::RegTweedie { tweedie_variance_power: 1.2 }, "tweedie-nloglik@1.2"),
        (Objective::RegPseudoHuberError { huber_slope: 1.0 }, "mphe"),
    ] {
        let name = objective.name();
        let p = params(objective, 2);
        let (_, history) = api::train(&p, &d, &[(&d, "eval")]).unwrap();
        assert_eq!(history[0][0].0, format!("eval-{expected}"), "wrong default metric for {name}");
    }
}

/// A classification fit reports the classification defaults.
#[test]
fn classification_defaults_to_its_own_metrics() {
    let (x, mut d) = features(200, 2);
    let y: Vec<f32> = (0..200).map(|r| f32::from(x[r * 2] > 0.5)).collect();
    d.set_labels(&y).unwrap();

    for (objective, expected) in [
        (Objective::BinaryLogistic, "logloss"),
        (Objective::BinaryHinge, "error"),
        (Objective::MultiSoftprob { num_class: 2 }, "mlogloss"),
    ] {
        let name = objective.name();
        let p = params(objective, 2);
        let (_, history) = api::train(&p, &d, &[(&d, "eval")]).unwrap();
        assert_eq!(history[0][0].0, format!("eval-{expected}"), "wrong default metric for {name}");
    }
}

/// A metric must actually improve as a fit that optimises it progresses.
#[test]
fn metrics_track_the_progress_of_a_fit() {
    let (x, mut d) = features(500, 3);
    let y: Vec<f32> = (0..500).map(|r| f32::from(x[r * 3] > 0.4)).collect();
    d.set_labels(&y).unwrap();

    let mut p = params(Objective::BinaryLogistic, 15);
    p.booster.learning.eval_metric = vec![EvalMetric::Logloss, EvalMetric::Auc];
    let (_, history) = api::train(&p, &d, &[(&d, "train")]).unwrap();

    let logloss_first = history[0][0].1;
    let logloss_last = history.last().unwrap()[0].1;
    assert!(logloss_last < logloss_first, "logloss should fall: {logloss_first} -> {logloss_last}");

    let auc_first = history[0][1].1;
    let auc_last = history.last().unwrap()[1].1;
    assert!(auc_last >= auc_first, "auc should rise: {auc_first} -> {auc_last}");
    assert!(auc_last > 0.9, "a separable problem should score well: {auc_last}");
}

// -------------------------------------------------------- base_score ------

/// `base_score` is set and read on the *prediction* scale, whatever the link.
#[test]
fn base_score_is_in_prediction_space_for_every_link() {
    let (_, mut d) = features(200, 2);
    d.set_labels(&vec![2.0f32; 200]).unwrap();

    for objective in [
        Objective::RegSquaredError,
        Objective::RegGamma,
        Objective::CountPoisson { max_delta_step: 0.7 },
    ] {
        let name = objective.name();
        // A zero learning rate makes every leaf zero, so the prediction is the
        // intercept and nothing else.
        let mut p = params(objective, 1);
        if let BoosterType::Gbtree(tree) = &mut p.booster.booster {
            tree.eta = 0.0;
        }
        p.booster.learning.base_score = Some(2.0);
        let booster = train(&p, &d);
        assert_eq!(booster.base_score(), 2.0, "{name}");
        // With no rounds the prediction is the intercept itself.
        let preds = booster.predict(&d);
        assert!((preds[0] - 2.0).abs() < 1e-4, "{name} predicted {}", preds[0]);
    }
}

/// `boost_from_average = false` leaves the intercept at XGBoost's untrained
/// default instead of estimating it.
#[test]
fn boost_from_average_can_be_turned_off() {
    let (_, mut d) = features(200, 2);
    d.set_labels(&vec![7.0f32; 200]).unwrap();

    let estimated = train(&params(Objective::RegSquaredError, 1), &d);
    assert!((estimated.base_score() - 7.0).abs() < 1e-4, "{}", estimated.base_score());

    let mut p = params(Objective::RegSquaredError, 1);
    p.booster.learning.boost_from_average = false;
    assert_eq!(train(&p, &d).base_score(), 0.5);
}

/// A base score the link cannot represent is rejected rather than producing a
/// NaN margin.
#[test]
fn an_impossible_base_score_is_rejected() {
    let (_, mut d) = features(50, 2);
    d.set_labels(&vec![1.0f32; 50]).unwrap();

    let mut p = params(Objective::RegGamma, 1);
    p.booster.learning.base_score = Some(0.0);
    match api::train(&p, &d, &[]) {
        Ok(_) => panic!("a zero base score has no logarithm"),
        Err(e) => assert!(e.to_string().contains("base_score"), "{e}"),
    }

    let mut p = params(Objective::BinaryLogistic, 1);
    p.booster.learning.base_score = Some(1.5);
    match api::train(&p, &d, &[]) {
        Ok(_) => panic!("a probability above 1 is not a base score"),
        Err(e) => assert!(e.to_string().contains("base_score"), "{e}"),
    }
}
