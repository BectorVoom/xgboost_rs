//! Thread pool control for the CPU training path.
//!
//! Parallelism here is *deterministic*: work is split into fixed-size blocks
//! that do not depend on the thread count, and partial results are always
//! reduced in block order. Training the same data twice — or on a machine with
//! a different core count — produces bit-identical models.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

static REQUESTED_THREADS: AtomicUsize = AtomicUsize::new(0);
static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();

/// Request a thread count for training (XGBoost's `nthread`).
///
/// `0` means "all available cores". Must be called before the first training
/// run; afterwards the pool is fixed for the process.
pub fn set_num_threads(n: usize) {
    REQUESTED_THREADS.store(n, Ordering::Relaxed);
}

/// Threads the pool will use.
pub fn num_threads() -> usize {
    let requested = REQUESTED_THREADS.load(Ordering::Relaxed);
    if requested > 0 {
        return requested;
    }
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

/// The training thread pool, created on first use.
pub(crate) fn pool() -> &'static rayon::ThreadPool {
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads())
            .thread_name(|i| format!("xgboost-rs-{i}"))
            .build()
            .expect("thread pool creation cannot fail with a valid thread count")
    })
}

/// Run `f` on the training pool.
///
/// Always installs, even for a single thread: nested `par_iter` calls pick up
/// whichever pool is current, so running `f` inline would silently hand the
/// work to rayon's global pool and ignore the configured thread count.
#[inline]
pub(crate) fn install<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    pool().install(f)
}
