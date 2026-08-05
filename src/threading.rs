//! Thread pool control for the CPU training path.
//!
//! Parallelism here is *deterministic*: work is split into fixed-size blocks
//! that do not depend on the thread count, and partial results are always
//! reduced in block order. Training the same data twice — or on a machine with
//! a different core count — produces bit-identical models.
//!
//! XGBoost's `nthread` is a per-fit parameter, so pools are cached per thread
//! count instead of a single process-wide pool: two fits in one process may ask
//! for different counts and both must get what they asked for.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

static REQUESTED_THREADS: AtomicUsize = AtomicUsize::new(0);

/// Pools by thread count. Leaked rather than dropped: a pool outlives every fit
/// that uses it, and there is one per distinct count, so the set is tiny.
fn pools() -> &'static Mutex<HashMap<usize, &'static rayon::ThreadPool>> {
    static POOLS: OnceLock<Mutex<HashMap<usize, &'static rayon::ThreadPool>>> = OnceLock::new();
    POOLS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Set the default thread count for fits that do not carry their own `nthread`
/// (XGBoost's `nthread`, where `0` means "all available cores").
///
/// A fit whose parameters set `nthread` explicitly uses that instead.
pub fn set_num_threads(n: usize) {
    REQUESTED_THREADS.store(n, Ordering::Relaxed);
}

/// The default thread count: whatever [`set_num_threads`] last requested, or
/// every available core.
pub fn num_threads() -> usize {
    resolve(REQUESTED_THREADS.load(Ordering::Relaxed))
}

/// Resolve a requested count, where `0` means "all available cores" and falls
/// back to the process-wide default before the machine's parallelism.
pub(crate) fn resolve(requested: usize) -> usize {
    if requested > 0 {
        return requested;
    }
    let default = REQUESTED_THREADS.load(Ordering::Relaxed);
    if default > 0 {
        return default;
    }
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

/// The pool for `n` threads, created on first use.
pub(crate) fn pool(n: usize) -> &'static rayon::ThreadPool {
    let n = n.max(1);
    let mut pools = pools().lock().expect("the pool registry is never poisoned");
    pools.entry(n).or_insert_with(|| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .thread_name(move |i| format!("xgboost-rs-{i}"))
            .build()
            .expect("thread pool creation cannot fail with a valid thread count");
        Box::leak(Box::new(pool))
    })
}

/// Run `f` on the pool for the default thread count.
#[inline]
pub(crate) fn install<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    install_with(num_threads(), f)
}

/// Run `f` on the pool for `n` threads.
///
/// Always installs, even for a single thread: nested `par_iter` calls pick up
/// whichever pool is current, so running `f` inline would silently hand the
/// work to rayon's global pool and ignore the configured thread count.
#[inline]
pub(crate) fn install_with<R: Send>(n: usize, f: impl FnOnce() -> R + Send) -> R {
    pool(n).install(f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_requested_count_wins_over_the_machine() {
        assert_eq!(resolve(3), 3);
        assert_eq!(resolve(1), 1);
    }

    #[test]
    fn zero_means_every_core() {
        assert!(resolve(0) >= 1);
    }

    #[test]
    fn each_thread_count_gets_its_own_pool() {
        assert_eq!(pool(2).current_num_threads(), 2);
        assert_eq!(pool(3).current_num_threads(), 3);
        // The registry is a cache, not a fresh pool per call.
        assert!(std::ptr::eq(pool(2), pool(2)));
        assert_eq!(install_with(2, rayon::current_num_threads), 2);
    }
}
