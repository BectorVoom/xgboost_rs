//! The random number generation XGBoost's sampling decisions are drawn from.
//!
//! Row and column sampling are the only parts of a fit that are not a pure
//! function of the data, so reproducing them means reproducing the exact
//! generators upstream uses:
//!
//! | Upstream | Here |
//! |---|---|
//! | `xgboost::RandomEngine` = `std::mt19937` (`Context::Rng`) | [`Mt19937`] |
//! | `std::linear_congruential_engine<uint64_t, 16807, 0, 1<<63>` (`RandomReplace::EngineT`) | [`Lcg63`] |
//! | `std::generate_canonical` | [`canonical_f64`], [`canonical_f32`] |
//! | `std::bernoulli_distribution` | [`bernoulli`] |
//! | `std::shuffle` | [`shuffle`] |
//!
//! # How far parity goes
//!
//! `std::mt19937`, `std::linear_congruential_engine` and
//! `std::generate_canonical` are all specified exactly by the C++ standard, and
//! `std::bernoulli_distribution` is specified in terms of `generate_canonical`.
//! Everything built from those — in particular **uniform row sampling** — is
//! bit-for-bit what upstream draws.
//!
//! `std::shuffle` and `std::uniform_int_distribution` are *not* specified
//! beyond their distribution, so [`shuffle`] reproduces libstdc++'s algorithm
//! rather than something guaranteed by the standard. Column sampling is
//! therefore reproducible run to run here, and statistically identical to
//! upstream, but a fit with `colsample_* < 1` is not guaranteed to pick the
//! same columns as a given XGBoost build.

/// `std::mt19937`, the engine behind XGBoost's `Context::Rng()`.
///
/// Used for the per-tree and per-node column samples, and to seed the row
/// sampler once per tree.
#[derive(Clone, Debug)]
pub struct Mt19937 {
    state: [u32; Self::N],
    /// Index of the next word to output; `N` means "twist first".
    index: usize,
}

impl Mt19937 {
    const N: usize = 624;
    const M: usize = 397;
    /// Bits of `state[i]` that belong to the "lower" half.
    const LOWER_MASK: u32 = (1 << 31) - 1;
    const UPPER_MASK: u32 = !Self::LOWER_MASK;
    const MATRIX_A: u32 = 0x9908_b0df;
    const INIT_MULTIPLIER: u32 = 1_812_433_253;

    /// Seed as `std::mt19937::seed(sd)` does.
    pub fn new(seed: u32) -> Self {
        let mut state = [0u32; Self::N];
        state[0] = seed;
        for i in 1..Self::N {
            let prev = state[i - 1];
            state[i] = Self::INIT_MULTIPLIER
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i as u32);
        }
        Self { state, index: Self::N }
    }

    /// The next 32-bit output, i.e. C++'s `engine()`.
    pub fn next_u32(&mut self) -> u32 {
        if self.index >= Self::N {
            self.twist();
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    fn twist(&mut self) {
        for i in 0..Self::N {
            let x = (self.state[i] & Self::UPPER_MASK)
                | (self.state[(i + 1) % Self::N] & Self::LOWER_MASK);
            let mut next = x >> 1;
            if x & 1 != 0 {
                next ^= Self::MATRIX_A;
            }
            self.state[i] = self.state[(i + Self::M) % Self::N] ^ next;
        }
        self.index = 0;
    }
}

impl Default for Mt19937 {
    /// `std::mt19937`'s default seed, 5489.
    fn default() -> Self {
        Self::new(5489)
    }
}

/// `std::linear_congruential_engine<uint64_t, 16807, 0, 1 << 63>`, upstream's
/// `RandomReplace::EngineT`.
///
/// One value is consumed per row during row sampling. Because the multiplier
/// and modulus are known, the state after `n` steps is available in closed form
/// ([`Lcg63::skip`]), which is how row `i`'s decision stays independent of how
/// the rows were divided between threads.
#[derive(Clone, Copy, Debug)]
pub struct Lcg63 {
    state: u64,
}

impl Lcg63 {
    /// `RandomReplace::kBase`.
    pub const BASE: u64 = 16807;
    /// `RandomReplace::kMod`.
    pub const MODULUS: u64 = 1 << 63;

    /// Seed as `linear_congruential_engine::seed` does: reduce modulo the
    /// modulus, and substitute `1` for a zero state, which would otherwise be
    /// absorbing.
    pub fn new(seed: u64) -> Self {
        let state = seed % Self::MODULUS;
        Self { state: if state == 0 { 1 } else { state } }
    }

    /// The state `exponent` steps on from `seed`, upstream's
    /// `RandomReplace::SimpleSkip`.
    pub fn skip(exponent: u64, seed: u64) -> u64 {
        let modulus = Self::MODULUS as u128;
        let mut result: u128 = 1;
        let mut base = Self::BASE as u128;
        let mut exponent = exponent;
        while exponent > 0 {
            if exponent % 2 == 1 {
                result = (result * base) % modulus;
            }
            base = (base * base) % modulus;
            exponent >>= 1;
        }
        ((result * seed as u128) % modulus) as u64
    }

    /// Advance and return, i.e. C++'s `engine()`.
    pub fn next_u64(&mut self) -> u64 {
        self.state =
            ((self.state as u128 * Self::BASE as u128) % Self::MODULUS as u128) as u64;
        self.state
    }
}

impl Mt19937 {
    /// `std::uniform_real_distribution<double>{0, 1}`, which the standard
    /// defines as `generate_canonical<double, 53>`.
    ///
    /// A 32-bit engine needs `ceil(53 / 32) = 2` draws, combined
    /// low-word-first; reproducing the draw count matters because it is what
    /// keeps the engine's stream aligned with C++'s.
    pub fn next_f64(&mut self) -> f64 {
        let lo = self.next_u32() as u64;
        let hi = self.next_u32() as u64;
        let value = (lo as f64 + hi as f64 * 4_294_967_296.0) / 18_446_744_073_709_551_616.0;
        // The division can round up to exactly 1, which the half-open range
        // forbids; libstdc++ caps it at the predecessor of 1.
        if value >= 1.0 { F64_JUST_BELOW_ONE } else { value }
    }
}

/// `std::minstd_rand`, i.e.
/// `linear_congruential_engine<uint_fast32_t, 48271, 0, 2147483647>`.
///
/// The pair sampler in the `mean` LambdaMART method is seeded per query group
/// from this engine upstream, so the same generator is reproduced here rather
/// than substituting a different one — the pairs a query trains on are part of
/// what the model is.
#[derive(Clone, Copy, Debug)]
pub struct MinStdRand {
    state: u32,
}

impl MinStdRand {
    const MULTIPLIER: u64 = 48271;
    const MODULUS: u64 = 2_147_483_647;

    pub fn new(seed: u32) -> Self {
        let state = (seed as u64 % Self::MODULUS) as u32;
        // A zero state is absorbing, so the standard substitutes 1.
        Self { state: if state == 0 { 1 } else { state } }
    }

    pub fn next_u32(&mut self) -> u32 {
        self.state = ((self.state as u64 * Self::MULTIPLIER) % Self::MODULUS) as u32;
        self.state
    }

    /// A uniform integer in `[0, n)`, following libstdc++'s
    /// `uniform_int_distribution` rejection scheme. Returns `0` for `n == 0`.
    pub fn next_below(&mut self, n: usize) -> usize {
        if n <= 1 {
            return 0;
        }
        let range = n as u64;
        let engine_range = Self::MODULUS - 1;
        let scaling = engine_range / range;
        let past = range * scaling;
        let mut draw = self.next_u32() as u64;
        while draw >= past {
            draw = self.next_u32() as u64;
        }
        (draw / scaling) as usize
    }
}

/// The largest `f64` below 1, `std::nextafter(1.0, 0.0)`.
const F64_JUST_BELOW_ONE: f64 = f64::from_bits(0x3fef_ffff_ffff_ffff);
/// The largest `f32` below 1, `std::nextafterf(1.0f, 0.0f)`.
const F32_JUST_BELOW_ONE: f32 = f32::from_bits(0x3f7f_ffff);

/// `std::generate_canonical<double, 53, Lcg63>`: one draw mapped into `[0, 1)`.
///
/// The standard picks `k = max(1, ceil(bits / log2(range)))` draws; with 53
/// mantissa bits and a 63-bit engine range that is one draw, so this is the
/// whole algorithm rather than a simplification of it.
///
/// The division can round up to exactly 1 for the largest draws, which would
/// break the half-open range the result is documented to live in. libstdc++
/// caps it at the predecessor of 1 (the fix for LWG 2524) and so does this.
#[inline]
pub fn canonical_f64(x: u64) -> f64 {
    let u = x as f64 / Lcg63::MODULUS as f64;
    if u >= 1.0 { F64_JUST_BELOW_ONE } else { u }
}

/// `std::generate_canonical<float, 24, Lcg63>`. One draw, as for [`canonical_f64`].
#[inline]
pub fn canonical_f32(x: u64) -> f32 {
    let u = x as f32 / Lcg63::MODULUS as f32;
    if u >= 1.0 { F32_JUST_BELOW_ONE } else { u }
}

/// `std::bernoulli_distribution{p}`, which the standard defines as
/// `generate_canonical<double>(g) < p`.
#[inline]
pub fn bernoulli(x: u64, p: f64) -> bool {
    canonical_f64(x) < p
}

/// A uniform integer in `[0, n)`, following libstdc++'s
/// `uniform_int_distribution` rejection scheme.
///
/// Panics if `n` is zero, which no caller here can produce: every sampled set
/// holds at least one element.
fn uniform_int_below(rng: &mut Mt19937, n: u64) -> u64 {
    assert!(n > 0, "uniform_int_below needs a non-empty range");
    let urange = n - 1;
    let urngrange = u32::MAX as u64;
    if urngrange > urange {
        // Reject the tail that would bias the low values, then scale down.
        let scaling = urngrange / (urange + 1);
        let past = (urange + 1) * scaling;
        let mut draw = rng.next_u32() as u64;
        while draw >= past {
            draw = rng.next_u32() as u64;
        }
        draw / scaling
    } else {
        // The engine cannot cover the range in one draw. Unreachable for the
        // feature counts XGBoost supports (`bst_feature_t` is 32-bit), but the
        // fallback keeps the function total.
        let hi = rng.next_u32() as u64;
        let lo = rng.next_u32() as u64;
        ((hi << 32) | lo) % n
    }
}

/// `std::shuffle`: Fisher-Yates, walking forwards and swapping each element
/// with a uniformly chosen one at or before it.
pub fn shuffle<T>(items: &mut [T], rng: &mut Mt19937) {
    for i in 1..items.len() {
        let j = uniform_int_below(rng, i as u64 + 1) as usize;
        items.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference output of `std::mt19937` seeded with 5489, which every
    /// standard library agrees on.
    #[test]
    fn mt19937_matches_the_standard_reference_values() {
        let mut rng = Mt19937::default();
        assert_eq!(rng.next_u32(), 3_499_211_612);
        assert_eq!(rng.next_u32(), 581_869_302);
        assert_eq!(rng.next_u32(), 3_890_346_734);

        // The standard pins the 10000th output of the default-seeded engine.
        let mut rng = Mt19937::default();
        let mut last = 0;
        for _ in 0..10_000 {
            last = rng.next_u32();
        }
        assert_eq!(last, 4_123_659_995);
    }

    #[test]
    fn mt19937_seeding_is_sensitive_to_the_seed() {
        assert_ne!(Mt19937::new(0).next_u32(), Mt19937::new(1).next_u32());
        // Seeding is deterministic, so two engines with one seed agree forever.
        let (mut a, mut b) = (Mt19937::new(7), Mt19937::new(7));
        for _ in 0..1000 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    /// `skip(n, seed)` must equal `n` steps of the engine, which is the whole
    /// reason row sampling does not depend on the thread count.
    #[test]
    fn lcg_skip_agrees_with_stepping() {
        for seed in [1u64, 42, 1 << 40, (1 << 63) - 1] {
            let mut stepped = Lcg63::new(seed);
            let mut state = seed;
            for n in 1..50u64 {
                state = Lcg63::skip(n, seed);
                assert_eq!(stepped.next_u64(), state, "seed {seed} step {n}");
            }
            assert_eq!(Lcg63::skip(49, seed), state);
        }
    }

    #[test]
    fn lcg_never_settles_on_zero() {
        // A zero seed would be absorbing, so the engine substitutes one.
        let mut rng = Lcg63::new(0);
        assert_eq!(rng.next_u64(), Lcg63::BASE);
        // Every state stays inside the modulus.
        let mut rng = Lcg63::new((1 << 63) - 1);
        for _ in 0..1000 {
            assert!(rng.next_u64() < Lcg63::MODULUS);
        }
    }

    /// The range is half-open, including for the largest draws — where the
    /// division rounds up to 1 and the cap has to catch it.
    #[test]
    fn canonical_values_stay_in_the_unit_interval() {
        for x in [0u64, 1, 1 << 32, Lcg63::MODULUS - 1] {
            let u = canonical_f64(x);
            assert!((0.0..1.0).contains(&u), "f64: {x} -> {u}");
            let u = canonical_f32(x);
            assert!((0.0..1.0).contains(&u), "f32: {x} -> {u}");
        }
        assert_eq!(canonical_f64(0), 0.0);
        assert_eq!(canonical_f64(Lcg63::MODULUS / 2), 0.5);
        assert_eq!(canonical_f32(Lcg63::MODULUS - 1), F32_JUST_BELOW_ONE);
        assert_eq!(canonical_f64(Lcg63::MODULUS - 1), F64_JUST_BELOW_ONE);
    }

    #[test]
    fn bernoulli_is_never_true_at_zero_and_always_true_at_one() {
        let mut rng = Lcg63::new(12345);
        for _ in 0..200 {
            let x = rng.next_u64();
            assert!(!bernoulli(x, 0.0));
            assert!(bernoulli(x, 1.0));
        }
    }

    #[test]
    fn bernoulli_frequency_tracks_the_probability() {
        let mut rng = Lcg63::new(2024);
        let n = 200_000;
        let hits = (0..n).filter(|_| bernoulli(rng.next_u64(), 0.25)).count();
        let rate = hits as f64 / n as f64;
        assert!((rate - 0.25).abs() < 0.01, "rate {rate}");
    }

    #[test]
    fn shuffle_is_a_permutation_and_depends_on_the_engine() {
        let mut a: Vec<u32> = (0..64).collect();
        let mut rng = Mt19937::new(0);
        shuffle(&mut a, &mut rng);
        let mut sorted = a.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..64).collect::<Vec<_>>());
        assert_ne!(a, sorted, "a 64-element shuffle should move something");

        let mut b: Vec<u32> = (0..64).collect();
        shuffle(&mut b, &mut Mt19937::new(0));
        assert_eq!(a, b, "the same engine state gives the same permutation");
    }

    #[test]
    fn shuffle_leaves_degenerate_inputs_alone() {
        let mut rng = Mt19937::new(1);
        let mut empty: Vec<u32> = Vec::new();
        shuffle(&mut empty, &mut rng);
        assert!(empty.is_empty());
        let mut one = vec![7u32];
        shuffle(&mut one, &mut rng);
        assert_eq!(one, vec![7]);
    }

    #[test]
    fn uniform_int_covers_its_whole_range() {
        let mut rng = Mt19937::new(3);
        let mut seen = [false; 5];
        for _ in 0..1000 {
            let v = uniform_int_below(&mut rng, 5) as usize;
            assert!(v < 5);
            seen[v] = true;
        }
        assert!(seen.iter().all(|&s| s));
        // A single-element range consumes a draw but can only answer zero.
        assert_eq!(uniform_int_below(&mut rng, 1), 0);
    }
}
