//! Hardware-adaptive launch geometry.
//!
//! Every kernel in `histogram.rs`, `quantiser.rs`, `evaluate_splits.rs` and
//! `row_partitioner.rs` was written with `CubeDim::new_1d(256)` and a
//! hand-rolled `div_ceil` cube count. 256 is a defensible guess for a
//! discrete NVIDIA GPU — eight 32-lane warps — but it is a guess, and it is
//! wrong in three ways that matter:
//!
//! * **It is not the device's plane size.** A workgroup should be a whole
//!   number of warps/wavefronts, or the tail plane runs with idle SIMD lanes.
//!   256 is a whole number of 32-lane warps and of 64-lane wavefronts, so it
//!   happens to be safe on NVIDIA and AMD — but not on a runtime that reports
//!   `plane_size_max == 1` (the CPU runtime), where a "unit" is an OS thread
//!   and 256 of them per workgroup is thread-pool thrash, not parallelism.
//! * **A `div_ceil` in X alone can exceed the backend's grid limit.** wgpu caps
//!   each grid dimension at 65535. [`HistogramEngine::zeroed`] clears a whole
//!   depth-10 frontier — ~26M `u32` words — which is 102,540 cubes of 256 in X.
//!   [`cubecl::calculate_cube_count_elemwise`] folds the overflow into Y and Z
//!   instead; `ABSOLUTE_POS` is a whole-grid flattened index
//!   (`x + y*gridX + z*gridX*gridY`), so the kernels do not notice.
//! * **Scalar loads leave the memory path idle.** The clear and subtract
//!   kernels are pure bandwidth, and the device can move 128 bits per
//!   instruction if asked in `Vector` widths rather than one word at a time.
//!
//! [`HistogramEngine::zeroed`]: super::histogram::HistogramEngine::zeroed

use cubecl::prelude::*;

/// Units per workgroup to aim for on a runtime with planes. Four 64-lane
/// wavefronts or eight 32-lane warps; within every backend's per-cube limit.
const PREFERRED_BLOCK: u32 = 256;

/// Scalar operations a unit should be worth before a plane-less runtime is
/// asked to dispatch another OS thread for it.
const WORK_PER_CPU_UNIT: usize = 32 * 1024;

/// Ceiling on units per workgroup on a plane-less runtime, against hosts that
/// report an anomalous core count.
const CPU_CUBE_DIM_MAX: u32 = 64;

/// Whether the runtime executes on hardware SIMD planes (GPU warps or
/// wavefronts) rather than on operating-system threads.
///
/// The distinction is what makes a shared-memory staging loop worth writing: on
/// a plane-less runtime `SharedMemory` is ordinary heap and `sync_cube` is a
/// thread-pool barrier, so the staging costs more than the global-memory
/// traffic it avoids.
pub fn has_planes<R: Runtime>(client: &ComputeClient<R>) -> bool {
    client.properties().hardware.plane_size_max > 1
}

/// Geometry for a kernel that maps one unit to one element and indexes with
/// `ABSOLUTE_POS`.
///
/// The workgroup is [`block_1d`]-sized — a whole number of planes, adapted to
/// the runtime family — and `calculate_cube_count_elemwise` then spreads the
/// cube count across X/Y/Z so it stays inside `max_cube_count`.
///
/// `lanes` is the number of units wanted — elements for a scalar kernel,
/// `Vector`s for a vectorised one, never the underlying scalar count.
///
/// Deliberately 1-D, rather than the 2-D shape `CubeDim::new` derives. A
/// plane-shaped `CubeDim(plane, planes_per_cube)` is the right default on a
/// real GPU, but it is a pessimisation on a software rasteriser: measured on
/// lavapipe, clearing 64 MB ran at 29.0 GB/s at `CubeDim(256)` and 16.1 GB/s at
/// the `CubeDim(64, 8)` that `CubeDim::new` picks here, because a wider
/// workgroup halves the number of independently schedulable cubes. Whole
/// planes of 256 units is the shape that is right on both.
pub fn elementwise<R: Runtime>(client: &ComputeClient<R>, lanes: usize) -> (CubeCount, CubeDim) {
    elementwise_with_work(client, lanes, 1)
}

/// [`elementwise`], for a kernel whose units are not all the same size.
///
/// `work_per_lane` is the number of scalar operations one unit performs; it
/// only matters on a plane-less runtime, where it decides whether a second OS
/// thread is worth dispatching at all.
pub fn elementwise_with_work<R: Runtime>(
    client: &ComputeClient<R>,
    lanes: usize,
    work_per_lane: usize,
) -> (CubeCount, CubeDim) {
    if lanes == 0 {
        // `calculate_cube_count_elemwise` answers `Static(0, 0, 0)` here.
        // Every caller's kernel is bounds-checked against its own length, so a
        // single idle cube is the cheaper thing to hand a backend that dislikes
        // an empty dispatch.
        return (CubeCount::Static(1, 1, 1), CubeDim::new_1d(1));
    }
    let cube_dim = CubeDim::new_1d(units_per_cube(client, lanes, work_per_lane));
    (cubecl::calculate_cube_count_elemwise(client, lanes, cube_dim), cube_dim)
}

/// Units per workgroup for `lanes` units of `work_per_lane` scalar operations.
fn units_per_cube<R: Runtime>(client: &ComputeClient<R>, lanes: usize, work_per_lane: usize) -> u32 {
    if has_planes(client) {
        return block_1d(client, PREFERRED_BLOCK);
    }
    // No planes: a "unit" is an OS thread costing ~1 us to dispatch, which at
    // 3 GHz buys tens of thousands of scalar operations. Asking for a second
    // thread below that threshold loses to the thread pool. Untested here --
    // this crate builds against wgpu and CUDA, both of which have planes --
    // but it is what keeps the helper honest if `cubecl/cpu` is ever added.
    let hardware = &client.properties().hardware;
    let cores = hardware.num_cpu_cores.unwrap_or(1).max(1) as usize;
    let total = lanes.saturating_mul(work_per_lane.max(1));
    let units = (total / WORK_PER_CPU_UNIT).clamp(1, cores.min(lanes));
    (units as u32).min(CPU_CUBE_DIM_MAX)
}

/// A 1-D workgroup size for kernels that index with `CUBE_POS_X * CUBE_DIM_X +
/// UNIT_POS_X` and so cannot accept the 2-D shape [`elementwise`] returns.
///
/// Rounds `preferred` down to a whole number of planes and clamps it to the
/// device's maximum units per cube, leaving a power of two on every device
/// whose plane size is one (all of them, in practice) — which the scan loops in
/// `evaluate_splits` and `row_partitioner` require.
pub fn block_1d<R: Runtime>(client: &ComputeClient<R>, preferred: u32) -> u32 {
    let hardware = &client.properties().hardware;
    let plane = hardware.plane_size_max.max(1);
    let limit = hardware.max_units_per_cube.max(plane);
    // Round the limit down to a whole number of planes too: clamping to a
    // bare `limit` that isn't itself plane-aligned would hand back a block
    // that is a whole number of planes right up until this `.min`, then isn't.
    let limit = (limit / plane) * plane;
    // Whole planes, never more than the device allows, never fewer than one.
    (preferred / plane).max(1).saturating_mul(plane).min(limit)
}

/// Cube-count geometry for a kernel that maps one whole *cube* — not one unit
/// — to one item of work, identified by the flattened cube index
/// `CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_Z * CUBE_COUNT_X *
/// CUBE_COUNT_Y`.
///
/// Spreads `cubes` across X, then Y, then Z so no axis exceeds the device's
/// per-axis grid limit ([`elementwise`] does the equivalent for per-unit
/// kernels). A partially-filled last row/plane can overprovision cubes past
/// `cubes`, so the kernel must bounds-check the flattened index itself —
/// there is no per-cube analogue of `ABSOLUTE_POS`'s implicit bound.
pub fn cubes_1d<R: Runtime>(client: &ComputeClient<R>, cubes: u32) -> CubeCount {
    if cubes == 0 {
        return CubeCount::Static(1, 1, 1);
    }
    let max = client.properties().hardware.max_cube_count;
    let x = cubes.min(max.0.max(1));
    let rows = cubes.div_ceil(x);
    let y = rows.min(max.1.max(1));
    let z = rows.div_ceil(y).min(max.2.max(1));
    CubeCount::Static(x, y, z)
}

/// The widest hardware-preferred vector width for `T` that divides every one of
/// `extents` exactly.
///
/// A partial trailing vector would read or write past the end of the
/// allocation, so the length *and* every offset the kernel adds to an index
/// must be a multiple of the width — passing the offsets in is what lets a
/// kernel that writes at `out_off + i` be vectorised at all.
pub fn line_size_for<R: Runtime, T>(client: &ComputeClient<R>, extents: &[usize]) -> usize {
    if extents.contains(&0) {
        return 1;
    }
    client
        .io_optimized_vector_sizes(core::mem::size_of::<T>())
        .find(|width| extents.iter().all(|e| e.is_multiple_of(*width)))
        .unwrap_or(1)
}
