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
//! The plane-less case stopped being hypothetical when the CubeCL CPU runtime
//! became a supported backend, and there it splits the kernels in two:
//!
//! * A kernel that **synchronises** gets a single unit per cube
//!   ([`block_1d`], `CPU_SYNC_CUBE_UNITS`), because a `sync_cube` at
//!   `cube_dim == cores` costs a scheduler quantum — and so gets no parallelism
//!   at all.
//! * A kernel that **does not** gets the whole machine ([`elementwise`]),
//!   because there the units are just threads with work to do. That is worth
//!   26× on this host, which is why [`cooperative`] exists: it lets a kernel be
//!   written once and launched in the shape whose cost the runtime can pay.
//!
//! One more rule falls out of the CPU backend being a **JIT**: `CubeDim` is
//! part of CubeCL's kernel cache key, so a launcher that varies the workgroup
//! width with the problem size recompiles the kernel at ~65 ms a time. Geometry
//! has to be *stable across launches*, not merely well-sized for one — see
//! `units_per_cube`, which is why it ignores `lanes`.
//!
//! All of it comes from the runtime's own reported properties, so a GPU build
//! is unaffected.
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

/// Units per workgroup on a plane-less runtime, for a kernel that
/// synchronises: one.
///
/// Since every kernel in this module grew a serial shape (see [`cooperative`])
/// no launch on a plane-less runtime synchronises any more, and this constant
/// only guards a cooperative launch that finds its way there — `hist_kernel`,
/// which an atomic-free runtime never runs. It stays at one for the reason
/// below, which is not a tuning choice but a failure mode. The CubeCL CPU runtime
/// spawns one OS thread per unit and implements `sync_cube` as a *spin* barrier
/// every unit of the cube must arrive at. While the cube is narrower than the
/// machine that barrier is nearly free; at `cube_dim == cores` there is no
/// spare core for the last arrival to be scheduled on, and it costs a scheduler
/// quantum. Timed on an 8-core host, 512 cubes, one barrier each:
///
/// | units per cube | 1 | 2 | 4 | 8 |
/// | --- | --- | --- | --- | --- |
/// | cost per barrier | free | free | ~10 µs | **~15 ms** |
/// | `gpu_row_partition::dense_root_split` | 0.44 s | 0.37 s | 1.58 s | 128.5 s |
///
/// A cube of one unit skips the barrier outright (`BARRIER_TARGET <= 1` returns
/// immediately), so the cost stops depending on how the host happens to
/// schedule. The width that would be *fast* here is a narrow one just below the
/// core count, which is exactly the width that has no parallelism left to give.
///
/// So this constant is not where a plane-less runtime finds its threads: a
/// kernel that has to synchronise cannot have them. Parallelism comes from
/// writing the kernel so it does **not** synchronise — see [`cooperative`] —
/// after which [`units_per_cube`] gives it the whole machine. Measured on the
/// same host, a barrier-free kernel at `cube_dim = 8` runs 26x the `cube_dim = 1`
/// version.
const CPU_SYNC_CUBE_UNITS: u32 = 1;

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

/// [`elementwise`], for a kernel whose unit takes a *run* of consecutive
/// lanes rather than one: returns the geometry and the run length.
///
/// On a runtime with planes the run is 1 and the geometry is
/// [`elementwise_with_work`]'s — one unit per lane, which is what fills a GPU.
/// On a plane-less runtime every unit thread executes the kernel body once
/// per cube of the grid, inside the runtime's own loop over
/// `CUBE_COUNT_{Z,Y,X}`, and that loop costs a few dozen instructions per
/// trip: address arithmetic for the builtins, the bounds test, the scalar
/// argument reloads. A kernel that does one load-subtract-store per unit pays
/// all of it per element, and the batch subtraction measured 12–20 GB/s on a
/// 70 GB/s host — with its buffers *in cache* — for exactly that reason. So
/// here the grid is a single cube of [`units_per_cube`] units and each unit
/// walks `lanes / units` consecutive lanes in a plain loop, the way a rayon
/// chunk would.
///
/// The kernel's contract is `first = ABSOLUTE_POS * run`, then lanes
/// `first .. min(first + run, lanes)`, so the GPU shape is the old one and the
/// CPU shape is one static split of the range per unit. The run is a runtime
/// scalar, not a comptime one: it tracks the problem size, and a comptime
/// argument that did so would recompile the kernel every level.
pub fn elementwise_runs<R: Runtime>(
    client: &ComputeClient<R>,
    lanes: usize,
    work_per_lane: usize,
) -> (CubeCount, CubeDim, u32) {
    if has_planes(client) || lanes == 0 {
        let (count, dim) = elementwise_with_work(client, lanes, work_per_lane);
        return (count, dim, 1);
    }
    let units = units_per_cube(client, lanes, work_per_lane);
    let run = lanes.div_ceil(units as usize) as u32;
    (CubeCount::Static(1, 1, 1), CubeDim::new_1d(units), run)
}

/// The run length a plane-less runtime wants for `lanes` split over `units`
/// (see [`elementwise_runs`]); 1 on a runtime with planes. For kernels whose
/// grid axes are spoken for and so build their own cube count.
pub fn run_length<R: Runtime>(client: &ComputeClient<R>, lanes: usize, units: u32) -> u32 {
    if has_planes(client) {
        1
    } else {
        lanes.div_ceil(units.max(1) as usize).max(1) as u32
    }
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
///
/// Deliberately **independent of `lanes`** on a plane-less runtime, which is
/// not what a first reading suggests. A unit there is an OS thread, so sizing
/// the workgroup to the work looks right — but `CubeDim` is part of CubeCL's
/// kernel cache key, and that runtime is a JIT: every distinct width is a fresh
/// MLIR compilation, measured at ~65 ms. A launcher that scales the width with
/// the frontier therefore recompiles as the tree deepens, which costs far more
/// than the thread dispatch it was avoiding (~1 µs). One width, and the cube
/// *count* absorbs the size instead.
///
/// `work_per_lane` survives only as the "is this worth any threads at all"
/// test, which is a single extra width (1) rather than one per size.
fn units_per_cube<R: Runtime>(client: &ComputeClient<R>, lanes: usize, work_per_lane: usize) -> u32 {
    if has_planes(client) {
        return block_1d(client, PREFERRED_BLOCK);
    }
    let hardware = &client.properties().hardware;
    let cores = (hardware.num_cpu_cores.unwrap_or(1).max(1)).min(CPU_CUBE_DIM_MAX) as usize;
    let total = lanes.saturating_mul(work_per_lane.max(1));
    if total < WORK_PER_CPU_UNIT {
        return 1;
    }
    // Not `cores.min(lanes)`: clamping to the lane count would make the width
    // vary again for every small launch. A cube wider than its work is free —
    // every kernel here bounds-checks its own index — while a second width is
    // another compilation. So a plane-less runtime only ever sees two.
    cores as u32
}

/// A 1-D workgroup size for kernels that index with `CUBE_POS_X * CUBE_DIM_X +
/// UNIT_POS_X` and so cannot accept the 2-D shape [`elementwise`] returns.
///
/// Rounds `preferred` down to a whole number of planes and clamps it to the
/// device's maximum units per cube.
///
/// On a plane-less runtime it returns `CPU_SYNC_CUBE_UNITS` instead, because
/// every caller of this function synchronises and a barrier there costs a
/// scheduler round. A narrower cube reaches the same answer — these kernels
/// accumulate exact `i64`, so their result does not depend on how the work was
/// split across units — without the thrash. See `CPU_SYNC_CUBE_UNITS` for the
/// measurements behind the number.
pub fn block_1d<R: Runtime>(client: &ComputeClient<R>, preferred: u32) -> u32 {
    let hardware = &client.properties().hardware;
    let plane = hardware.plane_size_max.max(1);
    if plane == 1 {
        return preferred.min(CPU_SYNC_CUBE_UNITS).max(1);
    }
    let limit = hardware.max_units_per_cube.max(plane);
    // Round the limit down to a whole number of planes too: clamping to a
    // bare `limit` that isn't itself plane-aligned would hand back a block
    // that is a whole number of planes right up until this `.min`, then isn't.
    let limit = (limit / plane) * plane;
    // Whole planes, never more than the device allows, never fewer than one.
    (preferred / plane).max(1).saturating_mul(plane).min(limit)
}

/// [`block_1d`] for a kernel that indexes with `CUBE_POS_X * CUBE_DIM_X +
/// UNIT_POS_X` but **never synchronises** — `subtract_batch_kernel`, whose
/// grid axes are spoken for and so cannot take [`elementwise`]'s shape.
///
/// Such a kernel has nothing to lose from a wide cube on a plane-less runtime,
/// and everything to gain: it is a bandwidth loop, and at one unit it runs on
/// one core. So it gets [`serial_width`] there and a plane-aligned block
/// elsewhere.
pub fn free_block_1d<R: Runtime>(client: &ComputeClient<R>, preferred: u32) -> u32 {
    if has_planes(client) { block_1d(client, preferred) } else { serial_width(client) }
}

/// The cube width the serial shape runs at: every core on a plane-less
/// runtime, a plane-aligned block elsewhere.
///
/// For a host that sizes a work table to the machine — the histogram's row
/// chunks, which should number at least a few per unit — rather than to the
/// problem. It is the width [`elementwise`] gives a launch worth threading.
pub fn serial_width<R: Runtime>(client: &ComputeClient<R>) -> u32 {
    units_per_cube(client, WORK_PER_CPU_UNIT, 1)
}

/// Whether a kernel should use its *cooperative* shape — a whole cube on one
/// work item, exchanging partial results through shared memory and
/// `sync_cube` — rather than its serial one, where a single unit owns a whole
/// work item and the launch is [`elementwise`].
///
/// True exactly when the runtime has planes, which is what makes a barrier
/// cheap. See the module docs for the measurements: on the CubeCL CPU runtime a
/// barrier at `cube_dim == cores` costs a scheduler round (~15 ms), so a
/// cooperative kernel there has to shrink its cube to avoid it and gives up all
/// its parallelism; the serial shape has no barrier at all and so can run
/// `cube_dim = cores` wide.
pub fn cooperative<R: Runtime>(client: &ComputeClient<R>) -> bool {
    has_planes(client)
}

/// [`block_1d`] for a kernel whose units cooperate through a shared-memory
/// **scan or tree reduction** — `scan_tile` and `reduce_best` in
/// `evaluate_splits`, `scan_flags` in `row_partitioner`.
///
/// Those loops step a stride by doubling up from 1 or halving down from
/// `block / 2`, so they only cover the whole workgroup when the block is a
/// power of two: at `block = 6` a `reduce_best` compares slots 0..3 and then
/// 0..1, and slot 2's candidate is never read. Plane sizes are powers of two,
/// so this is a no-op on a GPU; it is the plane-less branch of [`block_1d`],
/// which clamps to a core count that is not, that needs rounding down.
pub fn scan_block_1d<R: Runtime>(client: &ComputeClient<R>, preferred: u32) -> u32 {
    let block = block_1d(client, preferred);
    // `prev_power_of_two`: 1 << floor(log2(block)), and never 0.
    1u32 << (u32::BITS - 1 - block.max(1).leading_zeros())
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
///
/// Zero is a multiple of every width and needs no special case: an offset of
/// zero is the first slot, and a length of zero is a kernel with nothing to
/// do at any width. An earlier version answered 1 whenever an extent was
/// zero, meaning to guard empty buffers, and since the first slot of every
/// batch sits at offset 0 that quietly turned the batch subtraction scalar on
/// every launch — the "vectorised, neutral" result in the design manual was a
/// kernel that had never been vectorised.
pub fn line_size_for<R: Runtime, T>(client: &ComputeClient<R>, extents: &[usize]) -> usize {
    client
        .io_optimized_vector_sizes(core::mem::size_of::<T>())
        .find(|width| extents.iter().all(|e| e.is_multiple_of(*width)))
        .unwrap_or(1)
}
