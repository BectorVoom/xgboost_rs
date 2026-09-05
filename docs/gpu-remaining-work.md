# What the GPU path still does not do

`device=cuda` trains and matches the CPU fit bit-for-bit on a single round
(`tests/gpu_training.rs`), and every one of the 99 string-parameter oracle
cases produces the *identical* tree on both devices
(`every_string_parameter_fits_the_same_on_cpu_and_on_the_device`,
`tests/oracle_string_parameters.rs`). Since the CPU side of those cases is
pinned to XGBoost 3.4.0, that is what pins the device fit to XGBoost on a
machine with no NVIDIA GPU.

Two things are left, and both are refused or bounded by name rather than
silently mis-fitted, so nothing here is a correctness risk.

## 1. Categorical splits (implemented)

The device evaluator scans a feature's bins in ascending order, which is
meaningless for category codes, so a categorical feature is masked out of
`evaluate_feature_kernel` (`GpuHistGrower::evaluate`, `src/gpu/grower.rs`) the
same way column sampling masks out a feature the node may not use. It is
scored instead by `src/gpu/categorical.rs`, the host-side readback route this
doc used to describe as the cheap alternative to a device-side sort:

- The node's histogram is read back once per batch — only when the batch has
  at least one categorical candidate — and decoded into the same
  `GradientPairInt64` pairs `HistogramEngine::read` produces.
- `categorical::enumerate` dispatches `common::UseOneHot` exactly as
  `HistGrower::evaluate_one` does (`n_bins_feature < max_cat_to_onehot`,
  default 4) to `enumerate_one_hot` or `enumerate_partition` — line-for-line
  ports of the CPU functions of the same name (`src/tree/hist.rs`), except the
  running sums stay in the histogram's native quantised `i64` form and are
  decoded only where the gain formula needs a float, matching how the numeric
  kernel path is already exact.
- The winning categorical candidate is merged against the device's
  numeric-only winner with the same tie-break `SplitEntry::need_replace` uses
  (`categorical::replaces`), then carried through `ExpandEntry::cat_bits` to
  `RegTree::expand_categorical` and to the row partitioner's `SegmentSplit`.

The row partitioner already handled categorical splits before this
(`goes_left` tests the bit set, `tests/gpu_row_partition.rs` covers it); the
gap was only the evaluator and the bit-set plumbing feeding it, both now
covered by `tests/gpu_training.rs`'s `categorical_splits_match_the_cpu_fit_*`
tests, which check bit-for-bit parity with the CPU fit the same way every
other GPU training test does.

## 2. `multi_strategy=multi_output_tree` (implemented)

A vector leaf grows one histogram per target. The device histogram kernel stays
scalar and is simply run once per target, over the same row sets and into the
target's own slice of the node's frontier slot — the shape `HistGrower` already
used on the CPU, where `build_hists` runs its scalar kernel once per gradient
column. The target dimension this doc used to ask the kernel for turned out to
be unnecessary; only the frontier layout learned about targets
(`GpuHistGrower::frontier_bins`, `src/gpu/grower.rs`).

The split search is where the shape genuinely changes, because one candidate is
now scored from `n_targets` running sums at once:

- The scalar evaluator accumulates as it scans, so its running sum *is* the
  candidate. Holding `n_targets` of those in shared memory would blow the 16 KiB
  workgroup budget the portable backends allow, so the scan is split out:
  `prefix_scan_kernel` materialises the inclusive prefix sums once per
  `(node, feature, target)` and `evaluate_feature_multi_kernel` reads whichever
  sums a candidate bin needs. Both are exact `i64`, so — as for the scalar
  path — the result does not depend on how the work was split across threads.
  The backward scan needs no second pass: the suffix sum at bin `i` is
  `total - prefix[i - 1]` in fixed point.
- The gain is a line-for-line port of `HistGrower::multi_split_gain`: each
  target's own regularised gain at its own bounded weight, summed in `f32` in
  target order, rejected as a whole when the children's *mean* hessian fails
  `IsValidSplit`. The monotone direction check is absent for the same reason it
  is a no-op on the CPU — it is applied to mean-hessian children whose gradient
  sum is zero, so both weights come out equal. The bounds still shape the gain,
  through each target's real sums.
- A candidate carries one `(grad, hess)` pair summed over targets, as
  `SplitEntry` does on the CPU, so the winner's per-target child sums are read
  back out of the same prefix sums by `multi_child_sums_kernel` — the device
  counterpart of `HistGrower::multi_child_sums`, down to how a `-1` condition
  (a threshold matching no cut) is interpreted.

One thing the CPU does not have to worry about: the fixed point has to stay
addable *across* targets, because the node's cover and a candidate's summed
child sums add the targets together in `i64` and each target alone already fills
the 62 bits below the sign. `GradientQuantiser::new_multi` therefore bounds by
the sum of the per-target bounds rather than the largest of them, at a cost of
`log2(n_targets)` bits. Bounding by the largest overflows from three targets up,
which is what the `matches_across_depth_and_grow_policy` case caught.

Categorical features are refused, on the GPU exactly as on the CPU: a vector
leaf has no categorical split there either.

## 3. `tree_method=approx` (implemented)

What makes a fit `approx` is the per-round re-sketch of the quantile cuts,
weighted by the current hessians — not which processor grows the tree from
them. So `approx_cuts` (`src/gbm/mod.rs`) is shared by both devices and only
the grower differs: `grow_gpu_approx` builds a fresh ELLPACK from those cuts
each round, which is the cost `approx` pays either way. Under a constant
hessian both devices skip the re-sketch, as upstream does.

## 4. `gblinear` (implemented)

Note the spelling: upstream **rejects** `updater=gpu_coord_descent`, deprecated
since 2.0.0 in favour of `device=cuda` with `updater=coord_descent`, so there
is no new updater name here either. `src/gpu/linear.rs` runs the two O(nnz)
passes a coordinate step is made of — the column's `(g·x, h·x²)` and the
correction to that column's residuals — and the weights come out *bit-identical*
to the CPU's, because the sum is folded in the same fixed 4096-entry blocks and
the residual update is forced to round at every `f32` operation.

`shotgun` is hogwild and has no device version upstream or here. It is
accepted, as upstream accepts it, runs on the CPU, and says so through
`BoosterParameters::warnings` — otherwise `device` would be a parameter that
silently did nothing.

## 5. The CubeCL CPU runtime (implemented, with two named gaps)

The kernels are generic over `R: Runtime` and now resolve that through a
feature ladder — `cuda` over `vulkan` over `cpu` — with `gpu::BACKEND` naming
the winner. The bottom rung, `cubecl/cpu`, is a pure-Rust MLIR JIT with no
system dependency, and it is the default: a plain `cargo test` builds and runs
every kernel in this module on a machine with no GPU toolchain, and
`tests/gpu_training.rs` passes there in full — a `device=cuda` fit on the CPU
runtime is bit-identical to the CPU fit, exactly as it is on a real device.

That runtime does not execute kernels the way a GPU does, and two of the
differences reach into the kernel bodies rather than staying in the launcher:

- **Cubes are not concurrent.** The compiled kernel body sits inside `for` loops
  over `CUBE_POS_{Z,Y,X}`, one OS thread runs per *unit*, and `SharedMemory` is
  allocated **once for the whole launch** — so every cube reuses the same
  buffer. A unit that finished cube `i` and looped on to cube `i + 1` would
  overwrite shared memory a slower unit was still reading. Every kernel here
  that declares `SharedMemory` therefore ends its cooperative shape with a
  trailing `sync_cube` (`hist_kernel`, `evaluate_feature_kernel`,
  `reduce_candidates_kernel`, `prefix_scan_kernel`,
  `evaluate_feature_multi_kernel`, `count_tile_kernel`, `scatter_tile_kernel`).
  It is one redundant barrier per cube on a GPU and it is what makes the answer
  right at all on a host that runs that shape.
- **A "unit" is an OS thread, and `sync_cube` is a spin barrier over all of
  them.** A synchronising kernel can only be given a single unit per cube on a
  plane-less runtime (`launch::block_1d`, `CPU_SYNC_CUBE_UNITS`): at four units
  per cube `gpu_row_partition::dense_root_split` took 71 s and at one 1.4 s,
  because that kernel launched a cube per *tile of rows* and every cube paid for
  every barrier in it. That is why no kernel launched on a plane-less runtime
  synchronises any more — see "Getting parallelism out of it" below.

Two things that runtime cannot do are refused by name rather than mis-fitted:

- **No atomics.** `cubecl-cpu` 0.10 registers no atomic types at all, and
  rejects them where they are *written* rather than where they are used — a
  comptime-dead `SharedMemory::<Atomic<u32>>` declaration is still a compile
  error. So the atomic-free scheme is a second kernel,
  `histogram::hist_atomic_free_kernel`, rather than a comptime branch, and it
  is the histogram's serial shape: a work item is one row chunk of one node
  (`CHUNK_ROWS = 4096`, capped per node at twice the core count), one unit
  accumulates it into a private histogram in global memory, and
  `merge_partials_kernel` sums a node's chunks bin by bin into the frontier —
  the CPU grower's own lane scheme. Neither launch has a barrier or a shared
  buffer, so there is no budget to fit and the matrix is always one group. The
  sums are exact `i64` either way, so the histograms are bit-identical to the
  atomic path — `tests/kernels.rs` asserts that against the same CPU oracle.
  What has no atomic-free analogue is the *global-memory* accumulation path,
  so `HistogramBuilder::force_global` there is an `Error::NoGlobalHistogramPath`
  rather than a silent fallback to a path that does not exist.
- **The sparse ELLPACK layout.** It is the only layout whose `goes_left` has to
  *search* a row for the feature's entry, its bins being global rather than
  feature-local, and that search loop inside the guarded body of
  `count_tile_kernel` is a shape the CPU runtime's MLIR pipeline cannot lower
  ("operation with block successors must terminate its parent block"). The
  failure happens in a worker thread at kernel-compile time and takes that
  worker down with it, which would leave every later launch in the process
  waiting forever — so `RowPartitioner::partition` refuses it up front with
  `Error::SparseEllpackUnsupported`. No fit is affected: `build_ellpack` only
  ever produces `Dense` or `DenseCompressed`, and `Sparse` reaches the
  partitioner only from a hand-built `EllpackMatrix`, which is to say from
  `tests/gpu_row_partition.rs`.

### Getting parallelism out of it, without forking the kernels

The CPU runtime executes cubes **sequentially** — the compiled body sits inside
`for` loops over `CUBE_POS_{Z,Y,X}` and one OS thread runs per *unit* — so the
grid is not a parallel axis there and `cube_dim` is the only one. Measured on an
8-core host:

| | 1 unit | 2 | 4 | 8 |
| --- | --- | --- | --- | --- |
| barrier-free kernel, fixed total work | 266 ms | 76 ms | 21.8 ms | **10.2 ms** |
| cost of one `sync_cube` | free | free | ~10 µs | **~15 ms** |

Both axes are steep and they point opposite ways. A barrier is nearly free while
the cube is narrower than the machine and costs a scheduler quantum once it
fills it, because a spinning unit that has been preempted cannot arrive.
So **a kernel that synchronises cannot be given the machine, and a kernel that
does not, can** — there is no width that is good at both.

That is a property of the kernel, not of the runtime, which is what makes it
fixable from here. The lever is a comptime *cooperation width*: how many units
share one work item.

- **Cooperative** (`CUBE_POS` selects the item, units split its inner loop and
  exchange partial results through shared memory). One work item is only a few
  hundred bins, so this is the only shape that fills a GPU.
- **Serial** (`ABSOLUTE_POS` selects the item, one unit owns it end to end).
  No scan, no reduction, no barrier; the shared arrays degrade to per-unit
  scratch. `elementwise` can then hand the launch every core.

Both are one kernel body with `#[comptime] coop`, so a build emits only one, and
the two agree *bit for bit* rather than approximately: the candidates are scored
by the same code off exact `i64` prefix sums, and the winner is chosen by a total
order on `(gain, rank)`, so it cannot depend on the order they were visited in
or on how they were divided across units. `launch::cooperative` picks the shape
from `has_planes`.

Every kernel on the training path now has both shapes, and what a serial work
item is follows the CPU grower in each case:

| kernel | cooperative item | serial item | serial merge |
| --- | --- | --- | --- |
| `evaluate_feature_kernel`, `evaluate_feature_multi_kernel` | a cube per `(node, feature)`, bins split across units, `reduce_best` | a unit per pair, a running accumulator | none: each unit writes its own candidate |
| `reduce_candidates_kernel` | a cube per node, tree reduction over features | a unit per node walking the features | none |
| `prefix_scan_kernel` | a cube per `(node, feature, target)`, tiled Hillis–Steele | a unit per run, a running sum | none |
| `count_tile_kernel`, `scatter_tile_kernel` | a cube per 256-row tile, `scan_flags` | a unit per 4096-row tile (`SERIAL_TILE_ROWS`), a running count that *is* the rank | `scan_tiles_kernel`, unchanged |
| `hist_atomic_free_kernel` | — (`hist_kernel` is the cooperative shape, and needs atomics) | a unit per 4096-row chunk of a node, a private histogram in global memory | `merge_partials_kernel`, a unit per `(node, bin)` |

Measured on an 8-core Apple M1 (`train_bench`, 200 000 rows × 32 features,
256 bins, 5 rounds, 8 threads; models bit-identical throughout, and to the
`device=cpu` fit):

| per round | rayon `device=cpu` | CPU runtime, before | after the partitioner | after the histogram | after split evaluation and the subtraction width |
| --- | --- | --- | --- | --- | --- |
| depth 6 | 10.3 ms | 167.8 ms | 100.3 ms | 32.5 ms | 37.1 ms |
| depth 10 | 35.9 ms | 347.5 ms | 219.3 ms | 113.0 ms | 83.3 ms |

The last column also widens `subtract_batch_kernel`'s cube
(`launch::free_block_1d`): it has no barrier, and at one unit it was a serial
pass over the whole frontier. At 1 000 000 rows × 32 features and depth 8 the
CPU runtime is 225 ms per round against the rayon path's 87 ms. The histogram
kernel alone (`bench`, 1 M rows × 32 features) went from 117.7 ms per build to
21.6 ms dense and from 323 ms to 38.4 ms at sparsity 0.5.

A second pass took the same fit to 17.0 ms per round at depth 6 and 55.6 ms at
depth 10 (the histogram build alone to 9.9 ms dense), and at 1 M rows × depth 8
to 115 ms against the rayon path's 61 ms measured back to back — 1.65× and
1.55× behind it; a third, profiled per launch, to 17.7 and 53.9 ms (an exact
skip of empty bins in the split search, a frontier buffer pool, a vectorised
subtraction), and located the rest of the gap in the deep levels' per-node
dense histograms, which the rayon grower shares. A fourth pass, on a quiet
host, took it to 14.5–16.4 ms at depth 6 and 37.6–38.9 ms at depth 10 (rayon
in the same window: 7.1–7.5 and 28.6–29.7), and 84 ms against 66 at 1 M rows
× depth 8 — from 1.8× behind to 1.3× at depth — with three changes: a run of
lanes per unit in the elementwise kernels, whose one-element-per-unit shape
paid the CPU runtime's per-cube loop per element (the batch subtraction went
from 12.6 to 42 GB/s); a blocked row loop in the histogram kernel that issues
a block of scattered rows' cache misses together (−30–40% on deep-level
shapes); and a closed-form split gain that no longer computes two child
weights the configuration never reads (−15% on the deep evaluator launches).
It also found that the "vectorised" subtraction had launched scalar all along
(`launch::line_size_for` treated an offset of zero as an empty extent). A
fifth pass removed two launches per level — the partitioner's commit copy,
replaced by swapping its two buffers with a host-side table of which holds
each row, and the partial-histogram merge, folded into the subtraction — for
14.0–14.5 ms per depth-6 round and 82 ms at 1 M rows × depth 8; depth 10 is
unchanged within noise. A sixth pass viewed each histogram bin as one
`Vector<i64, 2>` instead of two words (one load, add and store per visit
instead of two of each — a third off the histogram kernel) and handed the
per-round gradient upload to the client as an owned buffer instead of a
copied slice: **10.3–10.8 ms** per round at depth 6 (rayon 7.4–8.5),
**29.8–30.7** at depth 10 (rayon 28.9–29.6) and **56.3 ms** at 1 M rows ×
depth 8 (rayon 55.7) — parity at depth and at scale, 1.3× behind on a
shallow tree, where the rest is the runtime's launch floor. The first passes
were profiled with CubeCL's
own per-kernel logger
(`CUBECL_DEBUG_LOG`) and with phase timers against the rayon grower, took it
further; `docs/cpu-kernel-design-manual.md` is the write-up, with the
measurements behind each rule. In short: `launch_unchecked` (the checked mode's
per-access bounds test was 2.2× on the histogram), single-chunk nodes
accumulating straight into their frontier slot so the atomic-free path never
clears a frontier, the partitioner's tile scan folded into the scatter pass,
one packed readback for the split search, the per-row cut lookups hoisted
out of the partition predicate, and — the largest single step, 25 → 17 ms at
depth 6 — a feature-major copy of the bin matrix (`DeviceEllpack::gidx_t`) for
that predicate to read, so a tile's decisions walk one column instead of one
cache line per row. Two things it ruled out by measurement: the
JIT's LLVM optimisation level (`cubecl-cpu` uses 0; patched to 3, the
histogram moved 9.9 → 8.5 ms and the fit under 5%) and unrolling the histogram's
inner loop (noise). What bounds the CPU runtime from here is the cost of a
launch — a thread hand-off of 20–150 µs, nine of them and two readbacks per
level. The 4-byte bin index was tried at 8 and 9 bits and is a GPU-only win
(`docs/cpu-kernel-design-manual.md` §7): the device matrix is now bit-packed
on runtimes with planes (`gpu::ellpack::device_bits`) and left as whole words
on the CPU runtime, where the narrower index cost more instructions than it
saved bytes.

**A JIT changes what "good geometry" means.** `CubeDim` is part of CubeCL's
kernel cache key, and the CPU backend compiles through MLIR at run time, so
every distinct workgroup width is a fresh ~65 ms compilation. A launcher that
sizes the width to the work — the obvious thing, and free on a GPU — recompiles
as the frontier grows and loses far more than the ~1 µs thread dispatch it was
avoiding. `launch::units_per_cube` therefore ignores the lane count on a
plane-less runtime and offers exactly two widths, letting the cube *count*
absorb the size. The same rule shapes the serial work items: a tile is a
constant 4096 rows and a row chunk is a constant 4096 rows, so the comptime
arguments never change with the frontier and each kernel compiles once per
shape.

**Speed, in perspective.** None of this makes the CPU runtime a fast trainer;
the crate's hand-written CPU path (`device=cpu`) remains the fast host route and
the one pinned to XGBoost by the oracle tests, and nothing here touches it. What
it buys is that the *device* code is no longer pathologically slow when run on a
host, which is what makes it testable at a realistic size.

## 6. Metal / `wgpu-msl` (builds and runs the integer kernels; cannot fit)

`--features metal` is wgpu with Metal Shading Language rather than SPIR-V. It is
the same `WgpuRuntime` as `--features vulkan` — only the shader compiler
differs, and wgpu picks it from the adapter, which `AutoGraphicsApi` resolves to
Metal on macOS. Unlike `vulkan` it needs **nothing installed**: the Vulkan SDK
check in `cubecl-wgpu`'s build script is gated on the `spirv` feature alone.

What runs there is genuinely useful coverage. Metal registers `I64` and `u32`
atomics, so the ELLPACK, histogram (both the shared and global paths, and the
u32-carry scheme), subtraction trick and row partitioner all pass — and unlike
the CPU runtime it executes cubes **concurrently**, which is the exact hazard
class the trailing `sync_cube()` work in §5 is about. It is the only backend
available without a toolchain that can check that.

What does not run is the fit. `cubecl-cpp`'s Metal dialect emits `#error type
double not supported!` for `Elem::F64`, because Metal Shading Language has no
`double` — and `register_types` in `cubecl-wgpu`'s Metal backend registers `I64`
but not `F64`, so the backend says so up front. The `f64` in these kernels is
not incidental:

- `consider_split` and everything under it (`child_weight`,
  `calc_gain_given_weight`, `threshold_l1`) are ports of XGBoost's `double` gain
  arithmetic, and are what make the device fit agree with the CPU fit to the
  1e-5 the oracle tests demand.
- `quantise_gpair_kernel` converts to fixed point through `f64`.
- `column_sums_kernel` (`gblinear`) accumulates in `f64` precisely so the device
  reproduces the host's model bit for bit.

Demoting them to `f32` would be a different model, not the same one computed
differently, so `gpu::supports_f64` is checked at `SplitEvaluatorGpu::new`
(which `GpuHistGrower` builds) and in the linear updater's `configure_device`,
and a fit on such a backend is `Error::NoF64Support` — refused by name, at
construction, rather than a wgpu validation panic several launches later.
`gpu_training::a_backend_without_f64_refuses_the_fit` pins both halves of that.
The suite is green on `--features metal`: the `f64`-dependent cases skip, and
say why.

## 7. SYCL (refused, and out of scope)

`Device::Sycl` is rejected outright. CubeCL has no SYCL backend here, so there
is nothing to run it on.

## Not a feature gap, but outstanding

**A vector leaf under a monotone box parts company with the CPU past depth 4.**
Both devices score a sum of `n_targets` clipped-weight terms, and a clipped
term is a difference of large quantities, so candidates near the bottom of a
deep tree are all but tied — and the device's `f32` arithmetic, which the
backend evaluates wide and narrows once rather than rounding at every
operation, is enough to swap which one wins. Identical to depth 4, no worse a
fit beyond (`multi_output_tree_matches_with_regularisation_and_constraints`).
The `Array<f32>` round-trip that fixed the same problem in `src/gpu/linear.rs`
would probably fix this too.

**Speed.** `docs/gpu-benchmarks.md` records the device path against
XGBoost's `gpu_hist` on a Kaggle T4: with both processes warm it is faster
on all 12 benchmark cases (1.5–2.8×; sparse input 1.6×, loss-guided growth
1.5×), and with each fit in a fresh process 1.0–1.7× when the driver's
context creation (`gpu::warm_up`) overlaps the data load, 0.6–1.25× when
it does not. A round at 500 000 × 50 is 7 ms, with the gradients, the
prediction update and the metric on the device (`gpu::objective`) for the
objectives it covers — `reg:squarederror` so far. Every other objective, a
vector-leaf fit, row sampling and DART take the host round, which is 6 ms a
round slower on that VM's cores.

The two places it loses are worth knowing before reaching for `device=cuda`:
`gblinear` (0.61×, because coordinate descent needs the previous feature's
residuals and so pays a launch and a readback per feature per group per round)
and anything whose extra work is host-side rather than in the kernels — `dart`
and `approx` both come out level with the CPU.
