//! Launch-shape overrides for measurement.
//!
//! The kernels' geometry is decided by the runtime's reported hardware and
//! the constants beside each kernel. Those constants were chosen by measuring,
//! and re-measuring them on a new device means rebuilding for every value
//! unless they can be moved from outside. `XGB_GPU_TUNE` is that outside: a
//! comma-separated `key=value` list, read once, that a launch site consults
//! through [`get`]. An unset key is the constant. Nothing in a fit depends on
//! these values but its speed — every kernel gives the same answer at every
//! geometry — so they are a benchmark's knob, not a parameter.
//!
//! Keys the crate reads:
//!
//! | key | site | meaning |
//! | --- | --- | --- |
//! | `hist_block` | `HistogramEngine::build_into_batch` | units per cube of the shared-memory histogram |
//! | `hist_bps` | same | cubes per SM one node may have |
//! | `hist_cap` | same | cube cap per node, overriding `hist_bps × SMs` |
//! | `hist_contig` | same | `1` gives each unit consecutive items of the tile (one row load per row) instead of strided ones |
//! | `hist_probe` | same | `1` skips the accumulation, `2` also the bin decode — diagnostic shapes for `bench`, wrong by design |
//! | `hist_global` | `HistogramBuilder::build` | `1` forces the global-memory path |
//! | `hist_smem64` | same | `1` accumulates in shared memory as native `i64` (wrong on CUDA through cubecl 0.10; see `HistogramBuilder`) |
//! | `hist_shmem` | same | shared-memory budget per cube, in bytes |

use std::collections::HashMap;
use std::sync::OnceLock;

fn table() -> &'static HashMap<String, u64> {
    static TABLE: OnceLock<HashMap<String, u64>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let Ok(spec) = std::env::var("XGB_GPU_TUNE") else {
            return HashMap::new();
        };
        spec.split(',')
            .filter_map(|kv| {
                let (k, v) = kv.split_once('=')?;
                let v = v.trim().parse::<u64>().ok()?;
                Some((k.trim().to_owned(), v))
            })
            .collect()
    })
}

/// The override for `key`, if the environment named one.
pub fn get(key: &str) -> Option<u64> {
    table().get(key).copied()
}

/// [`get`], as a `u32`, falling back to `default`.
pub fn get_or(key: &str, default: u32) -> u32 {
    get(key).map_or(default, |v| v.min(u32::MAX as u64) as u32)
}
