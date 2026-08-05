//! Run-wide settings that are not part of the model: thread count, the random
//! seed, and the random engine those seeds drive.
//!
//! This is XGBoost's `Context` (`include/xgboost/context.h`) reduced to the
//! parts a CPU `hist` fit reads. It is threaded through the booster rather than
//! stored globally because the random engine is *stateful*: every column and
//! row sample advances it, so where a draw happens in the sequence is part of
//! the model's definition.

use crate::parameters::{GeneralParameters, LearningTaskParameters, Verbosity};
use crate::rng::Mt19937;

/// `LearnerImpl::kRandSeedMagic`, the multiplier `seed_per_iteration` uses.
pub(crate) const RAND_SEED_MAGIC: i64 = 127;

/// Per-run context: threads, seed, verbosity, and the session random engine.
#[derive(Clone, Debug)]
pub struct Context {
    /// Requested thread count; `0` means "every core available".
    pub nthread: u32,
    /// The `seed` parameter.
    pub seed: i64,
    /// Reseed the engine at the start of every boosting round.
    pub seed_per_iteration: bool,
    /// How much the fit reports on stderr.
    pub verbosity: Verbosity,
    rng: Mt19937,
}

impl Default for Context {
    fn default() -> Self {
        Self::new(&GeneralParameters::default(), &LearningTaskParameters::default())
    }
}

impl Context {
    /// Build from the two public parameter groups that own these settings.
    pub fn new(general: &GeneralParameters, learning: &LearningTaskParameters) -> Self {
        Self {
            nthread: general.nthread,
            seed: learning.seed,
            seed_per_iteration: learning.seed_per_iteration,
            verbosity: general.verbosity,
            // `Learner::Configure` seeds the engine from `seed`; the narrowing
            // to the engine's 32-bit seed type is C++'s and is reproduced here.
            rng: Mt19937::new(learning.seed as u32),
        }
    }

    /// The session random engine, upstream's `Context::Rng()`.
    pub fn rng(&mut self) -> &mut Mt19937 {
        &mut self.rng
    }

    /// Reseed for a new boosting round, as `LearnerImpl::UpdateOneIter` does
    /// when `seed_per_iteration` is set. A no-op otherwise.
    pub fn seed_for_iteration(&mut self, boosted_rounds: usize) {
        if self.seed_per_iteration {
            let seed = self
                .seed
                .wrapping_mul(RAND_SEED_MAGIC)
                .wrapping_add(boosted_rounds as i64);
            self.rng = Mt19937::new(seed as u32);
        }
    }

    /// Threads this run will use, resolving `0` to the available parallelism.
    pub fn threads(&self) -> usize {
        crate::threading::resolve(self.nthread as usize)
    }

    /// Whether messages at `level` should be printed.
    pub fn logs(&self, level: Verbosity) -> bool {
        verbosity_rank(self.verbosity) >= verbosity_rank(level)
    }
}

/// Verbosity as the integer XGBoost compares against.
fn verbosity_rank(v: Verbosity) -> u8 {
    match v {
        Verbosity::Silent => 0,
        Verbosity::Warning => 1,
        Verbosity::Info => 2,
        Verbosity::Debug => 3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_seed_parameter_drives_the_engine() {
        let learning = LearningTaskParameters { seed: 7, ..Default::default() };
        let mut a = Context::new(&GeneralParameters::default(), &learning);
        let mut b = Context::new(&GeneralParameters::default(), &learning);
        assert_eq!(a.rng().next_u32(), b.rng().next_u32());

        let other = LearningTaskParameters { seed: 8, ..Default::default() };
        let mut c = Context::new(&GeneralParameters::default(), &other);
        assert_ne!(
            Context::new(&GeneralParameters::default(), &learning).rng().next_u32(),
            c.rng().next_u32()
        );
    }

    #[test]
    fn seed_per_iteration_reseeds_only_when_asked() {
        let learning = LearningTaskParameters { seed: 3, ..Default::default() };
        let mut fixed = Context::new(&GeneralParameters::default(), &learning);
        let first = fixed.rng().next_u32();
        fixed.seed_for_iteration(1);
        assert_ne!(fixed.rng().next_u32(), first, "an unseeded engine keeps advancing");

        let learning =
            LearningTaskParameters { seed: 3, seed_per_iteration: true, ..Default::default() };
        let mut per_iter = Context::new(&GeneralParameters::default(), &learning);
        per_iter.seed_for_iteration(1);
        let round_one = per_iter.rng().next_u32();
        let mut again = Context::new(&GeneralParameters::default(), &learning);
        again.seed_for_iteration(1);
        assert_eq!(again.rng().next_u32(), round_one, "round 1 is reproducible");

        again.seed_for_iteration(2);
        assert_ne!(again.rng().next_u32(), round_one, "round 2 differs from round 1");
    }

    #[test]
    fn verbosity_gates_messages_by_level() {
        let quiet = Context {
            verbosity: Verbosity::Silent,
            ..Context::default()
        };
        assert!(!quiet.logs(Verbosity::Warning));

        let normal = Context::default();
        assert!(normal.logs(Verbosity::Warning));
        assert!(!normal.logs(Verbosity::Info));

        let loud = Context { verbosity: Verbosity::Debug, ..Context::default() };
        assert!(loud.logs(Verbosity::Info));
        assert!(loud.logs(Verbosity::Debug));
    }
}
