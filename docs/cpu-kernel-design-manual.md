# Designing CubeCL kernels that are fast on the CPU runtime too

A manual for writing one `#[cube]` kernel body that fills a GPU *and* runs at
core count, with rayon-class throughput, on CubeCL's CPU runtime. Everything
in it was measured in this crate on an 8-core Apple M1 with `cubecl` 0.10.0,
and the numbers are the ones the rules came from. Where a rule is a guess it
says so.

The short version: **the CPU runtime is not a slow GPU. It has the opposite
cost model, and a kernel written for one is pathological on the other.** The
way to serve both is not two kernels but one body with two *shapes*, chosen at
comptime from the runtime's own properties, over arithmetic exact enough that
the shape cannot change the answer.

Contents:

1. [What the CPU runtime actually is](#1-what-the-cpu-runtime-actually-is)
2. [The cost model, side by side](#2-the-cost-model-side-by-side)
3. [The rules](#3-the-rules)
4. [Worked examples](#4-worked-examples)
5. [Profiling](#5-profiling)
6. [What it bought](#6-what-it-bought)
7. [What is still on the table](#7-what-is-still-on-the-table)
8. [The GPU side: what this manual does not claim](#8-the-gpu-side-what-this-manual-does-not-claim)
9. [Checklist](#9-checklist)

---

## 1. What the CPU runtime actually is

Read from `cubecl-cpu` 0.10.0's source, not inferred. File paths are inside
that crate.

| | |
| --- | --- |
| A unit | one OS thread, spawned per unit of `cube_dim` (`compute/worker.rs`), blocking on an `mpsc` channel between launches |
| A cube | *not* a unit of parallelism. Every unit thread runs the whole grid: the kernel body is emitted inside nested `scf.for` loops over `CUBE_COUNT_{Z,Y,X}` (`compiler/visitor/mod.rs::insert_builtin_loop`) |
| `sync_cube` | a process-global spin barrier over three atomics (`compute/compute_task.rs::sync_cube`). Free while the cube is narrower than the machine; a scheduler quantum (~15 ms measured) once spinning threads ≥ cores |
| `SharedMemory` | heap, allocated **once per launch**, reused by every cube a unit loops through |
| Atomics | none. `Atomic<T>` fails at kernel compile time, even in a dead comptime branch |
| Plane ops | `panic!("… is not supported on CPU")`; `plane_size_max == 1` |
| `Vector<T, N>` | real MLIR `vector` dialect, so NEON/AVX for elementwise work. Float horizontal sums are strict left-to-right chains (no fast-math anywhere in CubeCL) |
| Compilation | MLIR JIT, ~15–60 ms per distinct `(kernel, comptime args, cube_dim)`; no on-disk cache. The training path compiles nine variants before its first round — about 280 ms — and the split evaluator twice, once per cube width (R5) |
| LLVM opt level | 0 (`compiler/module.rs:100`, `ExecutionEngine::new(&module, 0, ..)`), after MLIR canonicalize / sccp / mem2reg / cse / inliner. **Measured to matter little**: patched to 3, the histogram build went 9.9 → 8.5 ms and the fit moved under 5%. Two things follow from 0 that do matter: the fast register allocator keeps every SSA value on the stack, so there is no register to hoist a load *into* (R9); and nothing removes arithmetic that is dead only along a branch (R15) |
| Checked mode | the default `launch` wraps every array access in a bounds test. Measured at **2.2×** on the histogram kernel; `launch_unchecked` removes it |
| A launch | the client thread enqueues into a 32-slot batched channel (`cubecl-common`, `device/handle/channel.rs`); the device thread drains it (spin → yield → 150 µs sleeps when idle) and for each launch sends one task per unit to the worker threads and waits for them all |
| The cube loop | every unit thread runs the kernel body once per cube of the grid, inside `scf.for` loops over Z, Y and X that recompute the builtins (`ABSOLUTE_POS`, the per-axis positions) and reload the kernel's scalar arguments each trip. A few dozen instructions per cube per unit — nothing next to a 4096-row tile, everything next to one vector subtract (R13) |

The hand-offs are what a launch costs. Microbenchmarked on this host:

| | |
| --- | --- |
| launch, 1 unit, no wait | 20 µs |
| launch, 8 units, no wait | ~100–150 µs (worker wake-ups) |
| launch of a tiny kernel followed by a readback of it | 160 µs |
| readback alone, 16 bytes, queue idle | 7 µs |
| `client.sync()` | 13 µs |
| `create_from_slice`, 256 bytes | 1 µs |
| `client.empty`, 1 MiB | 0.5 µs |

So a kernel that takes 50 µs of work costs 150 µs to run, and a level that
launches nine kernels and reads back three results spends a millisecond or two
on hand-offs before any work happens. That is the floor rayon does not have.
It is visible in the per-kernel profile as a **floor of 115–140 µs** under
every eight-unit launch, however little it does: at 200 000 rows the three
partition passes, the merge and the commit all sit on it, and the one-unit
candidate reduction sits at 16–25 µs.

## 2. The cost model, side by side

| | GPU | CPU runtime | rayon |
| --- | --- | --- | --- |
| Parallel axis | `cube_count` × `cube_dim` | `cube_dim` only; `cube_count` is a serial trip count | work-stealing over blocks |
| Cost of a barrier | ~free | a scheduler round at full width | no barriers: join |
| Cost of a launch | µs, queued | 20 µs + ~15 µs per unit woken | ~1 µs (`par_iter` split) |
| Cost of a cube | none: hardware schedules it | a few dozen instructions, per unit | — |
| Memory-level parallelism | thousands of threads in flight | one thread's reorder window per core: ~2 rows of a histogram loop | same, but rustc `-O` keeps the loop short |
| Shared memory | on-chip, per cube | heap, per launch | stack/heap |
| Atomics | yes | no | yes (unused here) |
| Vectors | 128-bit | NEON/AVX for elementwise; scalar for gathers | same, via `-O` autovectorisation |
| Code quality | driver compiler | MLIR passes + LLVM -O0, measured near enough | rustc `-O` |
| Memory bandwidth | device | host, ~70 GB/s here | same host |
| What fills it | thousands of cubes | `cores` units with ≥ 32 K ops each | ≥ 4096-row blocks |

## 3. The rules

### R1. One body, two shapes

Every kernel that cooperates through shared memory gets a `#[comptime] coop:
bool`. The *cooperative* shape is what a GPU wants: a cube owns one work item,
its units split the item's inner loop and exchange partials through
`SharedMemory`, with scans, tree reductions and atomics as needed. The
*serial* shape is what a runtime of OS threads wants: one *unit* owns a whole
work item end to end, indexed by `ABSOLUTE_POS`, with no barrier and no
shared access. The host chooses with `launch::cooperative(client)`, which is
`plane_size_max > 1`.

```rust
#[cube(launch_unchecked)]
pub fn kernel(/* .. */, n_items: u32, #[comptime] coop: bool, #[comptime] block: usize) {
    // Which item: the cube's coordinates, or the flattened unit index. An
    // elementwise grid overprovisions, so a dead unit is folded onto item 0
    // rather than branched away — every index below stays in range, and its
    // result is simply not written.
    let raw = if coop { CUBE_POS_X } else { ABSOLUTE_POS as u32 };
    let live = coop || raw < n_items;
    let item = if live { raw } else { 0u32.into() };

    // Shared arrays exist in both shapes; in the serial one they are per-unit
    // scratch that no other unit addresses, indexed by UNIT_POS_X.
    let mut s_acc = SharedMemory::<i64>::new(block);

    // Units share the item's loop cooperatively, or one unit takes it all.
    let first = if coop { UNIT_POS_X } else { 0u32.into() };
    let stride = if coop { CUBE_DIM_X } else { 1u32.into() };
    /* .. walk first, first + stride, .. into s_acc[UNIT_POS_X] .. */

    if coop {
        reduce_in_shared(&mut s_acc, block); // sync_cube inside
    }
    let slot = (if coop { 0u32.into() } else { UNIT_POS_X }) as usize;
    let writes = if coop { UNIT_POS_X == 0u32 } else { live };
    if writes { /* out[item] = s_acc[slot] */ }

    // A runtime that loops cubes reuses shared memory across them: hold the
    // cube together before any unit moves on. The serial shape shares nothing.
    if coop { sync_cube(); }
}
```

Two things make this one kernel rather than two: the per-element arithmetic
(`consider_split`, the histogram add) is the same function in both branches,
and the comptime `if` on `coop` erases the other branch entirely, so each
build emits exactly one shape.

Applied here: `evaluate_feature_kernel`, `evaluate_feature_multi_kernel`,
`reduce_candidates_kernel`, `prefix_scan_kernel`, `count_tile_kernel`,
`scatter_tile_kernel` (all in `src/gpu/`); the histogram's serial shape is a
separate kernel only because its cooperative shape needs `Atomic<u32>`, which
the CPU compiler rejects where it is *written*.

### R2. A work item is what one thread can own end to end

Pick the serial shape's work item by asking what the rayon path would give a
thread, because it has already solved this problem:

| kernel | cooperative item (a cube) | serial item (a unit) | rayon's equivalent |
| --- | --- | --- | --- |
| histogram | a grid-strided tile of `(row, feature)` over a feature group | a 4096-row chunk of one node, into a private histogram | `build_hists_target`: lanes over `BLOCK_ROWS` |
| row partition | a 256-row tile with a shared-memory scan | a 4096-row tile with a running count | `partition_blocked` |
| split search | a `(node, feature)` pair, bins split across units | a `(node, feature)` pair, one running sum | `evaluate_one` per task |
| candidate reduction | a node, tree reduction over features | a node, a loop over features | `update_entry` |

Sizes follow from the hand-off cost in §1: a unit should be worth at least
~32 K scalar operations (`WORK_PER_CPU_UNIT`), and a level should yield at
least a few items per core. 4096 rows × 32 features clears the first; a cap of
`max(16, 2 × cores)` chunks per node keeps a single root node from
under-filling the machine while a level of 256 tiny nodes still gets one item
each.

### R3. Exact arithmetic is what makes the shape a free choice

The two shapes give the same model bit for bit only because nothing in the
kernels depends on the order work is done in: histogram bins are quantised
`i64` (the `GradientQuantiser`), prefix sums over them are exact, and every
"best" is chosen by a total order on `(gain, rank)` where rank encodes the
order a sequential scan would have met the candidates. Then any decomposition
— across cubes, across units, across chunks merged in any order — produces the
same bits.

Do not add a `f32` accumulator to a kernel that has two shapes. The moment a
sum's value depends on its order, the serial and cooperative shapes disagree,
and the 99-case oracle parity that pins the device fit to the CPU fit is gone.

### R4. In the serial shape, never share; and write, don't add

A serial unit writes only to slices no other unit addresses. Where a result
genuinely needs contributions from several units — a node with more than one
row chunk — they go to *private* slices and a separate elementwise pass merges
them, one unit per output cell. Where a node has a single chunk (every node
below the top few levels), the unit accumulates straight into the final slot.

Then make the final pass **write** rather than add. The histogram merge sets
`hist[cell] = sum`; the subtraction sets `sibling = parent − built`; the
single-chunk build zeroes its own slot before accumulating. Nothing on the
serial path needs a pre-zeroed buffer, so the atomic-free frontier is
`client.empty` rather than a clearing launch. At depth 10 the frontier was
tens of megabytes and the clear ran at memory bandwidth (~1 ms per level);
not doing it is the whole saving.

### R5. Launch discipline

* **Two widths, ever.** `CubeDim` is part of the kernel cache key and the CPU
  backend is a JIT, so a width that tracks the problem size recompiles the
  kernel (~25–60 ms) as the tree grows. `launch::units_per_cube` offers `1`
  (work under 32 K ops) or `cores`, and the cube *count* absorbs the size.
  Even two widths are two compiles: the split evaluator meets both (levels 0
  and 1 are under 32 K ops) and pays a second 57 ms compile at startup to
  save ~80 µs of wake-ups on two launches per round — break-even near 350
  rounds. Acceptable, but it is why the profile shows two big first launches
  for that kernel (§5), and why a third width would need a reason.
* **Comptime arguments are constants.** Tile rows (4096), chunk rows (4096),
  scratch slots (`SERIAL_SLOTS = 64`) never change with the frontier. In the
  serial shape `block` is only an upper bound on the per-unit scratch, so it
  is pinned rather than sized.
* **Per-axis builtins only.** `CUBE_DIM` and `CUBE_COUNT` (the aggregate
  forms) panic on this backend at lowering time, inside a worker thread, and
  the launch silently produces nothing. Use `CUBE_DIM_X * CUBE_DIM_Y * ..`.
* **Geometry from the runtime's properties**, through `launch::elementwise`
  (which uses `calculate_cube_count_elemwise` to spread an X overflow into Y
  and Z on wgpu), `launch::cubes_1d` for cube-per-item grids, and
  `launch::free_block_1d` for a barrier-free kernel whose grid axes are
  spoken for. Never a hand-rolled `div_ceil` in X.
* **Bounds-guard once, then launch unchecked.** Every kernel guards the one
  index that can run past the end (its flattened unit or cube index against
  the item count) and derives everything else from the tables it is handed.
  With that contract stated per kernel, `launch_unchecked` is sound and worth
  2.2× on the histogram. The `SAFETY` note at each site points at the `gpu`
  module docs.
* **A trailing `sync_cube` in the cooperative shape** of every shared-memory
  kernel, because a runtime that loops cubes reuses the buffer; none in the
  serial shape.

### R6. Fewer launches, fewer readbacks

Given §1's numbers, a launch that saves 50 µs of work and costs 150 µs of
hand-off is a loss. Fold what you can:

* The partitioner's tile scan was a launch of its own; now each scatter tile
  sums its segment's counts itself (a few dozen loads at the root, one or two
  at depth), and the segment's first tile records the total.
* The split evaluator's two result buffers (`f32` and `i64`) are one `i64`
  buffer with the floats carried as `u32::reinterpret` bit patterns: one
  readback per level, not two.
* The partitioner's third pass — copying the scattered segments back from the
  scratch buffer — is gone: the two buffers swap roles instead, and a
  per-row table on the host remembers which one holds each row. Every
  segment a level reads was written by the previous level, so the depth-wise
  path never mixes the buffers in one launch; the copy kernel survives only
  for the loss-guided queue, which can revisit a node written several
  partitions ago (`RowPartitioner::catch_up`), and `read` assembles the
  final index from both. 800 KB of copying at 200 000 rows was never the
  cost; the ~130 µs launch under it was.
* The merge of a multi-chunk node's partial histograms rides in the
  subtraction: `build_into_batch_deferred` leaves the partials and their
  per-node ranges as a `PendingMerge`, and `subtract_batch_kernel` sums them,
  writes the built slot and subtracts in one pass over the lines it was
  visiting anyway. A node of one chunk has an empty range and reads its slot
  as before, so it is one kernel variant, not two. Only the root, which has
  no sibling, still launches `merge_partials_kernel`.
* Per level the training path now costs 2 launches for the partition, 1–2 for
  the histogram, 1 for the subtraction (merge included), 2 for the split
  search, and 2 readbacks: 178 launches per five depth-6 rounds where there
  were 232. Measured together, 15.2–16.4 → 14.0–14.5 ms per depth-6 round;
  depth 10 within noise (36–41 against 37.6–38.9), where the two launches
  were a smaller share; 1 M rows × depth 8, 84 → 82 ms. The remaining
  candidates are in §7.

Do not fuse where it changes the decomposition against you: folding the
candidate reduction into the split search would make a unit own a whole
node's features, which serialises the root of the tree.

### R7. Compute what a kernel moves, then check what it achieves

Before touching a streaming kernel, compute its bytes and divide by its time.
At depth 10 a level's frontier is 2 × 256 slots × 8192 bins × 16 bytes, and
the clear, the subtraction and the merge each move it once per level; on this
host the bandwidth is ~70 GB/s, so each pass has a floor of ~1.5 ms that only
R4 — not making the pass — gets under.

A pass that measures well *below* that floor is not memory-bound, whatever it
looks like. The batch subtraction ran at 17 GB/s at depth 10 and this manual
first called it "bandwidth-bound" and later "the frontier's size"; timed in
isolation it ran at **20.8 GB/s with its buffers in cache** (8 nodes, 3 MB),
which no memory system explains. It was the cube loop, one trip per vector
(R13). The vectorised version had also never been vectorised: `line_size_for`
answered 1 whenever an extent was zero, and the first slot of every batch is
at offset 0, so "vectorised, neutral" was a scalar kernel measured twice.
Check the width a kernel actually launched with (`CUBECL_DEBUG_MLIR` names the
dump directory `subtract_batch_kernel_n_1`) before believing a result about
vectors.

### R8. Where SIMD does and does not help

`Vector<T, N>` with the width from `launch::line_size_for` (the widest
`io_optimized_vector_sizes` entry that divides every extent and offset) is
right for pure streams: the frontier clear (`zero_u32_kernel`), subtraction,
quantisation. It lowers to 128-bit stores on a GPU and to NEON/AVX on the CPU
runtime — where, once the kernel takes a run per unit (R13), it was measured
neutral: `Vector<i64, 8>` against scalar at 256 nodes, 2.42 vs 2.57 ms, both
at 40–60 GB/s. On the CPU runtime the run is the lever and the vector is the
GPU's shape carried along for free.

It does not help the three things this workload is made of:

* **Gathers** (`gidx[row * stride + f]` for a row that came from an index
  array, `partials[bin]` for a bin that came from the data) — no contiguous
  lanes to load.
* **Scans with a carried dependency** (the running sum of a split search).
* **Float horizontal reductions** — `vector_sum` on floats is a serial chain
  on every backend. Integer reductions do become add trees, and every
  reduction here is `i64`.

The rayon path's advantage was not SIMD either; it was layout: a 1-byte bin
index (`BinStorage::U8`) against this crate's 4-byte ELLPACK, and a
column-major copy (`ColumnIndex`) for the partition predicate. The second is
now mirrored on device (`DeviceEllpack::gidx_t`) and was the single largest
win of the second pass (§6). The first was tried and turned out to be a
GPU-only win — on the CPU runtime the narrower index costs more instructions
than it saves bytes (§7). Locality paid; density did not.

### R9. Keep loops plain, hoist by hand

The JIT does not hoist loop-invariant loads: `goes_left` used to look up
`cut_ptrs[fidx]` and `cut_ptrs[fidx + 1]` per row, now the tile does it once
and passes them in. Manual unrolling of the histogram's feature loop by four
was measured at 10.0 vs 10.4 ms — noise — and reverted; the loop bookkeeping
is not where the time is. `RuntimeCell` reads and stores are promoted to
registers by `mem2reg`, so a counter in one costs nothing.

Scalar kernel *arguments* are the one invariant the generated code does
reload every iteration (§5, reading the dump). Copying them into a
`RuntimeCell` at the top of the kernel — `let row_stride =
RuntimeCell::<u32>::new(row_stride).read();` — looked like it should turn
each into a register, and two earlier attempts under a loaded host were
suggestive. Measured cleanly (load average under 3, A/B/A, per-launch series),
it is a **loss**: with four scalars hoisted in the histogram kernel and six in
the evaluator, the deep levels ran 8–15% slower in both kernels and returned
exactly when the change was reverted. The reason is in §1: at LLVM `-O0` the
fast register allocator spills every value to the stack between uses, so a
"hoisted" scalar is a stack reload where the original was an argument-buffer
reload, plus a longer live range. There are no registers to hoist into. The
recipe is retired; what does work is fewer *operations* per trip, not fewer
loads (R10, R15).

### R10. Skip what the reference would reject anyway

When the per-entry arithmetic is already at its floor, look for work whose
*result is known in advance*. The split search scores one candidate per bin
in scan order and keeps the first of two equal candidates. An empty bin adds
nothing to the running sums, so its candidate is the previous bin's split at a
later rank — it can never win, in either direction, and the sequential
reference only ever rejects it. Not scoring it (`bin_is_live`) is therefore
exact, costs a two-word compare the loop was loading anyway, and at depth,
where a node of a few hundred rows is spread over thousands of bins, removes
most of the search. The same test applies to the cooperative shape: a
unit whose bin is empty holds its left neighbour's prefix, which that
neighbour scores at a lower rank.

The proof obligation is the tie-break: the skip is sound only because
`better` is a strict total order that prefers the earlier rank, and because
both words are tested — a zero-hessian row still carries a gradient.

Measured: 12% off the split-search kernel at depth 10 on the benchmark's
uniform synthetic data, where a 200-row node still fills about half of a
feature's 256 bins; skewed real data leaves far more bins empty and gains
more. The micro-harness of §5 cannot show it at all — its histogram has no
empty bin — which is a reminder to benchmark on data with the reference's
shape.

### R11. Reuse big buffers; a fresh page costs more than the bandwidth to fill it

A buffer that is allocated, written once by a kernel and dropped pays a page
fault per page on its first touch, and on this host that is most of its
cost: allocating and clearing 64 MB measured 3.1 ms against the ~1 ms its
bandwidth takes, and 5.6 ms with other large buffers live. CubeCL's pool
does not hide it, because a buffer that changes size every level is a new
allocation every level. The frontier — one histogram slot per child of the
level, 67 MB at depth 10 — was exactly that, and it showed up as a
"bandwidth-bound" subtraction running at 17 GB/s and a histogram kernel that
got slower per row at depth. Vectorising the subtraction changed nothing,
which was the tell.

The fix is at the host: keep the buffers (`GpuHistGrower::frontiers`) and
hand one out again when nothing else holds it. `Handle::can_mut` says
exactly that, because every consumer of a level's histograms reaches them
through the handles its expand entries carry, and those drop when the level
is done. On the atomic path the reused buffer is re-zeroed by the vector
kernel; on the atomic-free path it is handed out as it is, since every bin
is written.

Measured: 2–3% off the round. Less than the microbenchmark promised, and the
per-launch series says why: a tree's levels grow, so within one tree every
level is a new size and a new allocation regardless, and reuse only starts
in the second round — yet the deepest level's subtraction costs the same in
every round. The faults were real but small next to what turned out to be
the cube loop (R7, R13). The rule stands (the pool is free), the expectation
is corrected.

### R12. Measure against the rayon path, phase by phase

The comparison that matters is not "GPU code on CPU vs GPU" but "this kernel
vs what rayon does with the same rows". With temporary timers around the three
phases of both growers (partition, histogram, split search), synced after each
on the device side, the picture at depth 6 was: partition 15× slower,
histogram 3.6×, split search 15× — and the 15× figures turned out to be
hand-offs and readbacks, not kernels, because the same kernels in isolation
were 3–5× (§5).

### R13. An elementwise kernel takes a run of lanes per unit, not one

The canonical GPU elementwise kernel — `i = ABSOLUTE_POS; if i < n { out[i]
= f(in[i]) }` — is the worst shape the CPU runtime can be given. Every unit
thread executes the body once per cube of the grid inside the runtime's own
loop (§1), so with one lane per unit the cube loop's bookkeeping is paid per
element, and a kernel whose body is one load, one subtract and one store is
almost entirely bookkeeping. The batch subtraction measured 12.6 GB/s at 256
nodes and 20.8 GB/s at 8 nodes *in cache*; the merge of partial histograms had
the same shape.

The fix is what rayon does: give each unit a contiguous run and loop inside
the body. `launch::elementwise_runs(client, lanes, work)` returns the geometry
and a run length — 1 and the ordinary grid on a runtime with planes, a single
cube of `cores` units and `lanes / cores` on the CPU runtime — and the kernel
follows one contract in both shapes:

```rust
let first = ABSOLUTE_POS as u32 * run;
let mut end = first + run;
if end > n_lanes { end = n_lanes; }
let i = RuntimeCell::<u32>::new(first);
while i.read() < end { /* lane i */ i.store(i.read() + 1u32); }
```

`run` is a runtime scalar, not a comptime one, because it tracks the problem
size and a comptime argument that did so would recompile the kernel every
level (R5). A kernel whose grid axes are spoken for — the subtraction uses
`CUBE_POS_Y` for the node — gets `launch::run_length` and builds its own X
count as `n_lines / (block × run)`, which is one cube per node.

Measured, batch subtraction in isolation, then the fit:

| | 8 nodes | 64 nodes | 256 nodes |
| --- | --- | --- | --- |
| one line per unit | 0.151 ms, 20.8 GB/s | 1.50 ms, 16.7 GB/s | 8.00 ms, 12.6 GB/s |
| a run per unit | 0.088 ms, 35.8 GB/s | 0.44 ms, 57.0 GB/s | 2.42 ms, 41.7 GB/s |

Depth-10 round: 54.2 → 46.7 ms from this change alone. The tile kernels and
the histogram never had the problem because a 4096-row tile or chunk *is* a
run; the split evaluator's 256 bins per pair amortise it well enough. The
rule is the same one as R2, stated for the kernels that looked too simple to
need it.

### R14. Issue a block's cache misses together

At depth a node's rows are a sparse, sorted subset of the matrix — one row
in 512 at depth 9 of a 200 000-row fit — so every row of the histogram loop
is a cache miss the prefetcher cannot anticipate, and the plain loop (a row,
then its 32 features, then the next row) overlaps only as many misses as one
core's reorder window holds: about two rows' worth of work. The serial shape
was 0.36 ns per visit on one node over sequential rows and 1.05–1.15 on 256
nodes over random ones, and the difference was latency, not bytes: the same
rows laid out contiguously ran at 0.57.

The kernel cannot prefetch (there is no intrinsic, and a dead load is removed
by `canonicalize`), but it can order its real work so the misses overlap:
visit the first bin of every cache line of a *block* of rows first (`touch`
rows × 2 lines each, all independent, all in flight together), then the
remaining bins of the same rows, which are now in cache. Both phases do
real histogram adds, so nothing is wasted and — because the adds are exact
`i64` into a private slot — the order cannot change a sum (R3).

Measured on 100 000 random rows in `n` nodes, 32 features, per build:

| nodes × rows | plain loop | blocks of 16 |
| --- | --- | --- |
| 1 × 100 000 | 1.39 ms | 1.33 |
| 16 × 6 250 | 3.31 | 2.00 |
| 64 × 1 562 | 2.80 | 1.75 |
| 256 × 390 | 3.68 | 2.53 |
| 512 × 195 | 4.73 | 3.47 |

Blocks of 8, 16 and 32 were within noise of each other, which says the
outstanding-miss budget saturates early; 16 is `TOUCH_ROWS`. In the fit the
histogram build went from ~11.4 to ~9.0 ms per depth-6 round and the round
from 46.7 to 38.8 ms at depth 10. The single-node, near-sequential case is
unchanged, so the block is on for every item on a plane-less runtime; a GPU
passes `0` and keeps the plain loop.

### R15. Comptime-elide what the JIT will not remove

MLIR's `canonicalize` deletes a pure operation with no uses — but only within
a block. Arithmetic behind a runtime branch whose result flows out through a
block argument survives, used or not. The evaluator's `split_gain` computed
both children's Newton weights (two `f64` divisions) and then, with neither
`max_delta_step` nor a monotone constraint in force, took the closed-form
gain and never read them; the final MLIR still had eight divisions per bin
pair, four per scan direction. Guarding the weights behind the *comptime*
`has_constraint || has_mds` — the configuration that actually needs them —
removed two divisions per live bin with the model hash unchanged, and took
the depth-9 evaluator launch from 5.05 to 4.20 ms (−15%).

The general form: when a comptime flag selects between a cheap and an
expensive formula, make sure the expensive one's *inputs* are behind the same
flag. Read the op counts of the final pass (§5) rather than assuming.

### R16. A bin is one vector, not two words

The histogram's `[grad, hess]` pair was two `i64` words at `slot` and
`slot + 1`, and a visit was two loads, two adds and two stores plus the
address arithmetic for each. Bound as `Array<Vector<i64, 2>>` — the same
bytes, viewed as 128-bit lanes — a visit is one load, one add and one store,
the slot index is the bin itself, the row's gradient pair is one load, and
the per-item zeroing loop is half as long. Nothing about the layout or the
arithmetic changes, so the result is bit-identical; what changes is the
instruction count, and on `-O0` code every instruction is paid in full
(§1). The `N: Size` parameter is fixed at 2 by the launch.

Measured, per build, single-threaded harness: root shape (200 000 sequential
rows) 2.2 → 1.65 ms; 16 nodes × 6 250 random rows 2.0 → 1.66; 256 × 390
2.53 → 2.23; 512 × 195 3.47 → 3.29. In the fit the histogram kernel's share
fell by a third at both depths (27.9 vs 43.6 ms per five depth-6 rounds, 58
vs 86 at depth 10), and the round went 14.2 → 11.7 ms at depth 6 and 37 →
30.5 at depth 10. It is the largest single step since the serial shapes, and
it came from counting the operations in a loop that was already "what you
would write" (§5): on this runtime the loop that looks minimal in source is
not minimal in instructions.

The same view was then tried on the evaluator's two histogram loads per bin,
and it was a **loss**: the depth-9 launch went 4.2 → 5.8 ms (+38%) and the
round 30 → 34 ms, reverted and confirmed. The difference is what the vector
replaces. In the histogram the pair is *operated on* as a pair — one add,
one store — so the vector removes operations. In the evaluator the two words
are consumed separately (one feeds the gradient sum, the other the hessian
sum and the validity test), so the vector load has to be followed by two lane
extracts, and at `-O0` an extract plus its spill costs more than the scalar
load it replaced. The rule is a count of operations after the view, not a
count of loads.

### R17. Hand the client the buffer

`create_from_slice` copies the slice into a fresh vector before the device
thread sees it. For the per-level tables that is nothing; for the gradient
pairs — the one upload of every round large enough to matter, 3.2 MB at
200 000 rows and 16 MB at a million — it is a copy of every byte the
quantiser just wrote. `client.create(Bytes::from_elems(vec))` hands the
allocation over instead (`HistogramEngine::upload_gpairs_owned`): measured
0.80 → 0.33 ms per upload at 200 000 rows and 3.9 → 1.75 ms at a million,
and the round at depth 6 went 11.7 → 10.5 ms, more than the microbenchmark
promised, which says the copy was also evicting something. The root's target
sums are folded before the buffers move.

Two things measured around it and not kept: running the quantiser's
fixed-point pass on the rayon pool (the pass was 0.12–0.17 ms of the 0.75;
parallel or not it did not move the round), and parallelising the
quantiser's bound — which is a sequential `f64` sum whose rounding decides
the fixed-point scale, so it is not free to reorder (R3).

## 4. Worked examples

### The histogram

Cooperative shape (`hist_kernel`, unchanged from the CUDA port): grid
`(tiles, feature groups, nodes)`, each cube privatises its group's bins in
`SharedMemory<Atomic<u32>>`, accumulates a grid-strided tile of
`(row, feature)`, flushes with global atomics.

Serial shape (`hist_atomic_free_kernel`): the host cuts each node into 4096-row
chunks and builds a job table. A unit takes one chunk:

```rust
let item = ABSOLUTE_POS as u32;
if item < n_items {
    let pbase = item_dst_base[item as usize];        // its own slice, in i64 words
    /* zero the slice */
    for each block of `touch` rows in the chunk {
        for each row in the block {                  // phase 1: one bin per cache line
            for fidx in (0..row_stride).step_by(LINE_BINS) { hist_add(row, fidx) }
        }
        for each row in the block {                  // phase 2: the rest, now in cache
            for each line { for fidx in line_start + 1 .. line_end { hist_add(row, fidx) } }
        }
    }
}
// hist_add: bin = gidx[row_begin + fidx]; if dense || bin != null {
//   slot = pbase + global_bin(bin, fidx) * 2; dst[slot] += grad; dst[slot + 1] += hess }
```

With `touch == 0` the two phases collapse to the plain row loop, which is
what a GPU gets. A node with one chunk points `item_dst_base` at its frontier
slot; a node with several points at a private `partials` buffer, and
`merge_partials_kernel` (a run of `(node, bin)` lanes per unit, R13) *writes*
their sum into the slot. The job table is
`HistogramEngine::build_private_batch`.

Result on 1 M rows × 32 features: 117.7 ms → 21.6 (serial shape) → 9.9
(unchecked) ms per build; the rayon path's `build_dense` on the same rows is
in the same range per core. On the shapes a fit actually has — hundreds of
nodes of a few hundred scattered rows — the blocked loop is the difference
between 1.15 and 0.79 ns per visit (R14).

### The subtraction

`sibling = parent − built` over a whole level in one launch, grid `(X, nodes,
Z)` with the node on Y. Cooperative shape: a unit per `Vector<i64, N>`.
Serial shape: the same kernel with `run = n_lines / cores` and one cube in X,
so a unit subtracts a contiguous `1/cores` of the node's slot in a plain loop
and the three offset loads happen once per unit instead of once per vector.
Same body, same bits; 8.0 → 2.4 ms at 256 nodes (R13).

### The row partition

Both shapes keep the two-pass structure (count left rows per tile, then
scatter) and the same tile table. Cooperative: a 256-row tile per cube, ranks
from a Hillis–Steele scan over the units' flags. Serial: a 4096-row tile per
unit, ranks from a running count — the count *is* the rank when one unit walks
the rows in order, so `scan_flags` and its 2·log₂(256) barriers vanish:

```rust
let rank = RuntimeCell::<u32>::new(0u32);
for k in 0..n {
    let off = toff + k;
    let left = goes_left(..);
    out[scatter_slot(left, begin, base, n_left, off, rank.read())] = row;
    rank.store(rank.read() + u32::cast_from(left));
}
```

`scatter_slot` is the one function both shapes use, so a row lands in the same
place whichever computed its rank. Neither shape copies the result back: the
buffer the scatter wrote becomes `ridx`, and the host's per-row side table
says where every segment not rewritten this level still lives (R6).

### The split search

`evaluate_feature_kernel` was the first conversion and the template for R1:
cooperative, a cube per `(node, feature)` scanning bins in tiles through
`s_scan`/`s_carry` and reducing with `reduce_best`; serial, a unit per pair
with two `RuntimeCell` accumulators and no barrier. 81.8 ms → 18.3 ms at 256
nodes × 64 features × 256 bins, identical winners.

## 5. Profiling

**Per-kernel time.** CubeCL has a profiler built in:

```bash
CUBECL_DEBUG_LOG=/tmp/prof.log CUBECL_DEBUG_OPTION=profile-medium \
  ./target/release/train_bench --rows 200000 --features 32 --rounds 5 --depth 10 --device cuda
```

Every launch logs `| <duration> | <KernelName> | CpuRuntime`. Three things to
know before reading it. The **first launch of each kernel variant includes
its JIT compile** (15–60 ms) — *variant*, not kernel: a kernel launched at two
cube widths compiles twice, and skipping only the first line credited the
evaluator with 45% of a depth-6 fit in which it does 8%. Profile mode
serialises the pipeline, so the sums run above the un-profiled wall time —
read the *shares* and the per-launch series, not the totals. And what the
kernels add up to is not the round: at depth 6 they sum to ~13.6 ms of a
15.4 ms round, the rest being readbacks and host work between launches. An
aggregation script that finds compiles by size rather than position:

```python
import re, sys, collections, statistics
runs = collections.defaultdict(list)
for line in open(sys.argv[1]):
    m = re.match(r"\| ([0-9.]+)(ns|µs|ms|s)\s+\| (\w+)", line)
    if not m: continue
    v = float(m.group(1)) * {"ns": 1e-6, "µs": 1e-3, "ms": 1, "s": 1e3}[m.group(2)]
    runs[m.group(3)].append(v)
tot = {}
for k, vs in runs.items():
    med = statistics.median(vs)
    jit = [v for v in vs if v > 5 * med and v > 5]     # a compile dwarfs any launch
    body = [v for v in vs if v not in jit]
    tot[k] = (sum(body), len(body), jit)
T = sum(t[0] for t in tot.values())
for k, (v, n, jit) in sorted(tot.items(), key=lambda kv: -kv[1][0]):
    print(f"{k:26} {v:9.1f} ms {100*v/T:5.1f}% {n:5d} launches {1000*v/n:8.0f} us avg"
          f"   compiles: {', '.join(f'{j:.0f} ms' for j in jit)}")
```

Then print one kernel's series in launch order — `awk` on the name — and
read it level by level: a depth-10 round is ten launches, and a change that
helps level 9 and hurts level 0 shows there and nowhere else.

**The profiler can under-report.** The root histogram build read 0.9–1.0 ms
in every profile of this fit; a wall-clock timer with a `client.sync()` on
each side of it read 2.1 ms, and the isolated harness agreed with the timer.
The launches that follow a readback seem to be timed from somewhere inside
their own execution. So: use the profile for *shares and series*, and put a
synced `Instant` around any single launch whose absolute time matters —
which is how the histogram's real share (61% of a depth-6 round, not the
profile's 40%) was found, and with it R16.

**Phase timers, once more.** With `Instant`s around the level loop's three
phases and around the stages outside it (quantise and upload, root, loop,
final read), a 14.2 ms depth-6 round decomposed as: quantise + upload 0.7,
root 2.8, level loop 10.3 (partition 1.6, everything else 8.7), read 0.1.
Inside the loop, host work and readback bubbles were ~0.7 ms in total. That
is what settled two items of §7 without building them: merging the two
per-level readbacks could recover at most half of 0.7 ms, and the deep
levels' "dense frontier" is not the wrong asymptote at these sizes — 390
rows over 8192 bins leaves 78% of the bins non-empty.

**The shape harness.** The kernel microbenchmark (`bench`) times one node over
every row, which is the one shape a fit has once. What found R14 was ten
minutes of a throwaway `#[ignore]` test that built `HistogramEngine` on
`reference::random_matrix`, took a random half of the rows, split it into
1, 16, 64, 256 and 512 equal nodes *sorted within each node* (what a stable
partition of random data produces), and timed `build_into_batch` on each with
`client.sync()` around ten iterations. The same harness with `(parent,
built, out)` slot triples timed `subtract_batch` and gave R13 its table.
Write it again on the target host; do not keep it. Two traps in it: `cargo
test` runs the harness's tests *concurrently* unless told
`--test-threads=1`, and a run that forgets produces numbers 3× too high with
no other symptom; and `random_matrix`'s `Dense` layout is global bins while
the fit's is feature-local — the histogram cost the same on both here, but
check with data built through `build_cuts` and `build_ellpack` before
believing a difference.

**Runtime costs.** The §1 hand-off table came from a throwaway integration
test that timed `engine.zeroed(4)` in a loop with and without a readback,
`client.sync()`, and `create_from_slice`. Ten lines; write it again rather
than trust the numbers on a different machine.

**Phase timers against rayon.** Temporary `Instant`s around the partition,
histogram and split-search calls of *both* growers, printed per level behind
an env var and aggregated. On the device side, `client.sync()` after the
histogram launches, or the next phase's readback absorbs their time. Remove
them afterwards; they are not observability, they are an experiment.

**Kernels in isolation.** `SplitEvaluatorGpu::evaluate` and
`RowPartitioner::partition` are public; time them on synthetic inputs of the
sizes a level really has (1, 8, 32, 256 nodes). The difference between that
and the phase timer is host work plus hand-offs, and it is what §1 explains.

**Reading the generated code.** `cubecl-cpu` can dump every pass of its
MLIR pipeline, behind its `mlir-dump` cargo feature:

```toml
# temporarily, in Cargo.toml
[dependencies.cubecl-cpu]
version = "0.10.0"
features = ["mlir-dump"]
```

```bash
CUBECL_DEBUG_MLIR=/tmp/mlir ./target/release/bench   # one directory per kernel
```

The directory must already exist (the dump creates only the per-kernel
subdirectory, silently doing nothing otherwise), and a small fit — 20 000
rows, depth 4, two rounds — compiles every training kernel. Per kernel it
writes `cubecl.ir.txt` (the kernel as CubeCL sees it), `cubecl-opt.ir.txt`
(after CubeCL's own SSA optimiser: the thing to read for what your source
became), and under `builtin_module_no-symbol-name/` one `.mlir` per pass
down to `12_cse.mlir`, the final LLVM dialect. The directory name carries the
comptime arguments — `subtract_batch_kernel_n_1` is how the never-vectorised
subtraction was caught (R7). Counting ops in the final pass is a two-line
`grep -o "llvm\.[a-z]*" | sort | uniq -c`, and it is how R15 was found: eight
`llvm.fdiv` in a kernel that needed four. For the histogram's inner loop the
final form was one `gidx` load, one `cut_ptrs` load, an add and a shift, two
64-bit read-modify-writes, and a loop bound **reloaded from the
scalar-argument buffer on every iteration** — every scalar kernel argument is
a load at each use, because nothing after `mem2reg` hoists, and (R9) copying
it out does not help at `-O0`. That is the codegen-quality gap against
rustc, and it is small: the loop is otherwise what you would write. Revert
the manifest afterwards (and the lock file); the feature is not a dependency
this crate wants.

**Check the machine first.** `uptime` before every measurement. One
afternoon's numbers here were taken with a load average of 24 from two other
compilations on the same host, and made a harmless change look 2× slower and
a later revert look no better; a `cargo build` of this crate leaves a load
average above 10 for a minute after it finishes. A battery that polls `sysctl
vm.loadavg` and starts only under 3 is worth the ten lines; so is running the
*same* binary twice, since a difference between identical builds is the
noise floor (±1 ms per round at 200 000 rows here), and so is A/B/A — a
change that helps must un-help when reverted, which is what retired R9's
hoisting.

**What did not work.** `xctrace` under this harness never returned;
`cargo flamegraph` needs `sudo` for dtrace. Neither shows JIT frames anyway.
The built-in profiler plus phase timers was enough.

## 6. What it bought

`train_bench`, 200 000 rows × 32 features, 256 bins, 5 rounds, 8 threads, best
of three. Models bit-identical throughout, and to the `device=cpu` fit.

| ms per round | rayon `device=cpu` | start | serial shapes | + unchecked | + direct writes, no clear, fewer launches | + column-major partition index |
| --- | --- | --- | --- | --- | --- | --- |
| depth 6 | 10.3 | 167.8 | 37.1 | 25.7 | 24.6–26.3 | **17.0** |
| depth 10 | 35.9 | 347.5 | 83.3 | 79.3 | 66.7–68.8 | **55.6** |

Ranges are over separate runs; the host's run-to-run noise is about ±1.5 ms
at this size. At 1 M rows × 32 features and depth 8, measured back to back:
rayon 61 ms per round, CPU runtime 115 ms.

Histogram kernel, 1 M rows × 32 features: 117.7 → 21.6 → 9.9 ms per build
(dense); 323 → 38.4 → 25–28 ms at sparsity 0.5. From 16× behind the rayon path
at depth 6 to 1.65×, and from 9.7× to 1.55× at depth 10.

The last column is the layout change R8 predicted: the partition predicate
reads a feature-major copy of the bin matrix (`DeviceEllpack::gidx_t`), so a
tile's decisions walk one column instead of touching one cache line per row.
It cost a second copy of the matrix on device and ~40 ms of upload at 200 000
rows, and took 8 ms off a 25 ms round — locality, not SIMD, was the gap.

### A third, profiled pass: what the last 15% cost

With the per-kernel logger and the per-launch series it prints (one line per
launch, so a level's cost can be read off directly), the round at depth 10
decomposed into: histogram ~25 ms, split search ~10.5 ms, subtraction ~13 ms,
partition ~1.6 ms. Against the rayon grower's phases — histogram *and*
subtraction 16 ms, split search 10.5 ms, partition 3 ms — the split search
was already at parity and the partition ahead; the whole gap was the
histogram build and the subtraction, and both of those are the deep levels,
where a frontier of 512 slots × 8192 bins × 16 B is written, subtracted and
read once each.

Four things were tried against that, each measured in a quiet window:

| change | evaluator kernel | fit, depth 6 / 10 | kept |
| --- | --- | --- | --- |
| validity-first scoring, running best in registers (R9) | neutral | neutral | yes, as a cleaner shape |
| skip empty bins (R10) | −12% at depth 10 | neutral at depth 6, −1 ms at depth 10 | yes: exact, and real data is sparser than the benchmark's uniform bins |
| vectorised subtraction, `Vector<i64, 8>` | — | neutral | yes: no cost, and it is the right shape on a GPU |
| frontier buffer pool (R11) | — | 17.6–17.8 / 53.4–54.4, −2–3% | yes |

And one was ruled out by the series: the deepest levels' subtraction costs
the same in round five as in round one, so it is not first-touch page faults
— the buffers are reused from round two on. It is the frontier's size. The
rayon grower has the same per-node dense histograms and pays the same
traffic; what it does better there is per-thread code quality on a streaming
loop, which is the JIT's to close, not the kernel's.

Standing after the third pass, this host, 200 000 rows × 32 features:
17.7 ms per round at depth 6 (rayon 10.3, 1.7×) and 53.9 ms at depth 10
(rayon 35.9, 1.5×); 1 M rows × depth 8: 112 ms against 61 (1.8×).

### A fourth pass: the cube loop, the gather, and the dead divisions

Profiled on a quiet host (load average under 3 throughout, rayon re-measured
in the same window at 7.1–7.5 ms and 28.6–29.7 ms per round), starting from
16.4 and 54.2 ms. Per-kernel shares at depth 10, compiles excluded:

| kernel | before | after | what changed |
| --- | --- | --- | --- |
| histogram build | 120 ms / 5 rounds | 90 | blocked row loop (R14) |
| split evaluator | 108 (one compile still counted) | 102 | closed-form gain without the weights (R15) |
| subtraction | 63 | 27 | a run per unit (R13) |
| partition, three passes | 19 | 19 | at the launch floor |
| merge | 5 | 4 | a run per unit |

| ms per round | rayon (this window) | before | + runs (R13) | + blocked gather (R14) | + R15 |
| --- | --- | --- | --- | --- | --- |
| depth 6 | 7.1–7.5 | 16.4 | 16.4–17.2 | 14.7 | **15.2–16.4** |
| depth 10 | 28.6–29.7 | 54.2 | 46.7 | 38.8 | **38.0–38.9** |
| 1 M rows, depth 8 | 66 | 112 (earlier window) | — | — | **84** |

Models bit-identical to the rayon fit throughout (`train_bench --hash`).
From 2.2× behind at depth 6 to 2.1×, from 1.8× to **1.3×** at depth 10, and
from 1.8× to 1.27× at 1 M rows: the deep levels, which were the whole gap,
are now close, and what is left is concentrated at the top of the tree,
where the round is short and the hand-offs are not.

Three things were measured and not kept, each with the reason in its rule:
hoisting scalar arguments (R9: 8–15% slower, `-O0` has no registers),
vectorising the subtraction once it had runs (R8: neutral at 40–60 GB/s),
and launching the histogram at sixteen units on eight cores so the scheduler
could rebalance the efficiency cores' share (§7: −8–12% in isolation, neutral
in the fit, since eight more wake-ups per launch cost what the balance
saved).

### A fifth pass: two launches a level

Taking the first item of the list below: the partitioner's commit copy
replaced by a buffer swap, and the partial-histogram merge folded into the
subtraction (R6). Same host, same window as the fourth pass, three runs each:

| ms per round | rayon | after the fourth pass | after the fifth |
| --- | --- | --- | --- |
| depth 6 | 7.5–7.7 | 15.2–16.4 | **14.0–14.5** |
| depth 10 | 29.7–31.8 | 37.6–38.9 | **36.1–41.1** |
| 1 M rows, depth 8 | 64 | 84 | **82** |

Launches per five rounds: 232 → 178 at depth 6, 380 → 302 at depth 10; the
commit kernel no longer appears in the profile at all and the merge four
times (the root of each round after the first). Models bit-identical. The
gain is exactly the two launch floors per level and no more, which is the
shape §1 predicts; at depth 10 it is inside the run-to-run noise of a 38 ms
round, at depth 6 it is 6%. What remains per level is six or seven launches
and two readbacks, each of them doing work.

### A sixth pass: the loop that looked minimal

Started from the phase timers above, which put the histogram at 61% of the
depth-6 round and the profiler's number for its root launch at half the
truth. Same host, same window, three runs each:

| ms per round | rayon | after the fifth pass | + vector bins (R16) | + owned upload (R17) |
| --- | --- | --- | --- | --- |
| depth 6 | 7.4–8.5 | 14.0–14.5 | 11.6–11.8 | **10.3–10.8** |
| depth 10 | 28.9–29.6 | 36.1–41.1 | 29.8–30.7 | **29.8–30.7** |
| 1 M rows, depth 8 | 55.7 | 82 | 62.9 | **56.3** |

Models bit-identical throughout. At depth 10 and at a million rows the CPU
runtime is now within 3% of the rayon grower; at depth 6 it is 1.3× behind,
and the rest of that is the ~1 ms of launch floors a six-level tree pays
(§7). Per-kernel shares at depth 6 after the pass: histogram 29%, partition
7%, subtraction 3%, evaluator the remainder once its compile is discounted
(~1 ms per round). At depth 10 the evaluator is 28% of the round and the
histogram 25%.

Measured and not kept this pass: a parallel fixed-point pass (R17), the
vector view on the evaluator's loads (R16, +38% on the deep launches), a
device-side merge of the two readbacks (assessed from the phase timers, not
built), and a sparse deep-level frontier (assessed: not the asymptote at
these sizes).

What is left inside the crate is the launch floor on shallow trees and the
JIT's code quality, both the runtime's; `cubecl` 0.11.0-pre.3 is the release
that changes the first, and it is a pre-release.

## 7. What is still on the table

Measured, in order of expected value:

1. **Hand-offs.** 6–7 launches and 2 readbacks per level at 20–150 µs each
   is ~1 ms of the ~1.7 ms a depth-6 level now costs, and the profile shows
   it directly: every eight-unit launch sits on a 115–140 µs floor, so the
   two partition passes are ~0.25 ms per level whatever the row count. The
   commit and the merge were the launches that did the least work and they
   are gone (R6); what is left each does something a level needs, and the
   next fold would cost a barrier (count into scatter) or a decomposition
   (candidate reduction into the split search, R6's warning). Merging the
   two readbacks was assessed with the phase timers (§5) at under half a
   millisecond per round and not built. This is now the whole of the
   depth-6 gap, and the fix is the runtime's thread pool: `cubecl-cpu` 0.11
   (pre-release) is a rewrite with a persistent pool, schedulers and CPU
   affinity — the right upgrade when it stabilises, and one that should be
   measured against §1 first.
2. **Bin width — done, and a lesson.** The device matrix is now bit-packed
   at the narrowest width that holds it (`ellpack::pack_bins`, read through
   the comptime-specialised `load_bin`: 8 bits for 256-bin data, 9 with a
   missing sentinel, a two-word read where the width does not divide a word).
   On the CPU runtime it was a **loss**: the histogram build went 10.0 →
   16.3 ms at 8 bits and 25 → 53 ms on the two-word path, because that kernel
   is instruction-bound and the shift-and-mask is more instructions. So the
   width is a per-runtime choice (`ellpack::device_bits`): packed on a runtime
   with planes, whole words on a plane-less one, where `load_bin` is the plain
   load and measured free. The GPU side of the trade — a 3.5–4× cut in the
   histogram kernel's dominant stream — is the thing to measure on a T4; it
   is untested here beyond the packed read paths, which the CPU-runtime tests
   run at forced widths (`DeviceEllpack::upload_with_bits`).
3. **Heterogeneous cores.** This host has four performance and four
   efficiency cores, and a launch of eight statically assigned units finishes
   when its slowest thread does. Capping the width at four was measured
   slower (histogram 11.1–12.0 vs 10.5–10.9 ms; fit 19.4–21.0 vs 19.9 ms per
   round), so the efficiency cores still pay for themselves. Going the other
   way — sixteen units for the histogram launch, so the scheduler could move
   a waiting thread onto a performance core that finished early — was
   8–12% faster on every shape in isolation and neutral in the fit, A/B
   interleaved: the eight extra wake-ups per launch cost what the balance
   saved, and it would be a third compile. Run-to-run noise of ±1 ms at this
   size is the scheduler's placement, not the kernels.
4. **The deep levels' structure.** A level of 512 nodes of 200 rows still
   zeroes 33 MB of slots, subtracts across 100 MB and scans 67 MB of bins,
   because every node's histogram is dense over 8192 bins. The rayon grower
   shares it, and at these sizes it is the right choice: 390 rows over 8192
   bins leave 78% of the bins non-empty, so a sparse representation would
   pay its indirection for a fifth of the entries. It becomes the asymptote
   only past depth 12 or so at 200 000 rows, which this benchmark does not
   reach; the exact empty-bin skip (R10) is the right concession until then.
5. **The evaluator's per-bin cost.** ~20 cycles per live bin after R15, of
   which two `f32` divisions are the reference's arithmetic and the rest is
   `-O0` codegen around them. It is the largest kernel at depth 10 (28% of
   the round) and at parity with the rayon grower's scan. R16's view on its
   two loads was tried and reverted (+38%); nothing exact is left to remove
   from the loop short of the JIT's optimisation level.
6. **Not** the JIT opt level, not the kernels' inner loops, not hoisting
   scalars, not narrower bins, not oversubscription. All measured.

## 8. The GPU side: what this manual does not claim

Everything above was measured on the CPU runtime, on a host with no GPU. The
kernels are shared — one `#[cube]` body per kernel, the cooperative shape for
hardware with planes and the serial shape for the CPU runtime, selected at
comptime from the runtime's properties; the histogram is the one exception,
with a separate atomic-free kernel because the CPU compiler rejects `Atomic`
types even in a dead branch. That sharing is what makes the following true,
and what makes it a caveat:

* **Nothing in passes four to six has been run on a GPU.** It compiles for
  the cooperative shape, and the oracle tests pass on the CPU runtime, but the
  GPU paths were not executed. Before relying on a GPU build, run the kernel
  oracle and the training comparison there — `KAGGLE.md` and
  `tools/compare_gpu.py` are the recipes.
* **Several of the changes reach the GPU path**, by design, and each should
  help or be neutral there — that is the claim to check first:
  the partitioner's buffer swap (one launch fewer per level on any runtime,
  R6); the closed-form split gain (two divisions fewer per candidate, R15);
  the batch subtraction now actually launching vectorised (R7's bug fix —
  before it, `line_size_for` had made it scalar everywhere); `run = 1` in
  the subtraction and merge, which adds a one-trip loop to the GPU shape
  (R13); and the pending-merge fold, which on a runtime with atomics is a
  no-op (the atomic build returns no pending merge, the item ranges are
  empty). The blocked histogram loop (R14) and the vector bins (R16) live in
  the atomic-free kernel, which a GPU with atomics never runs.
* **The last recorded GPU measurement is a loss.** On a Colab T4
  (`docs/gpu-benchmarks.md`, before any of this work) this crate ran 5× to
  14× slower than XGBoost's `gpu_hist`, with a flat ~5 s floor per fit that
  the document attributes to the per-level host round trips rather than to
  the kernels. Nothing here was aimed at that floor; the launch-count
  reductions of R6 nibble at it and no more. Closing it is a separate
  investigation on GPU hardware.
* **Some GPU wins are known and untested.** The bit-packed bin matrix
  (`ellpack::pack_bins`, 8 bits for 256-bin data) is expected to cut the
  histogram kernel's dominant stream 3.5–4× on a GPU and measured as a loss
  on the CPU runtime, so it is enabled only on runtimes with planes
  (`device_bits`) — and has never been timed where it is enabled. The
  P100 kernel oracle (`KAGGLE.md`) also found native 64-bit global atomics
  beating the privatised shared-memory histogram on dense input, while
  `HistogramBuilder` still prefers shared; that choice should be measured,
  not assumed.
* **The rules themselves are one-sided.** R13, R14, R16 and R17 are about a
  runtime whose units are OS threads and whose code is `-O0`; a GPU has
  hardware scheduling, thousands of threads to hide latency, and a driver
  compiler. Every rule here says which shape it is for. Where a change to a
  shared body was made for the CPU runtime, the GPU shape was kept as it was
  (comptime `touch = 0`, `run = 1`, the atomic path untouched), so the
  expected GPU effect of this work is small either way — but *expected* is
  the word, until it is measured.

## 9. Checklist

Before a kernel is done:

- [ ] One body; `#[comptime] coop` selects the shape; the per-element
      arithmetic is one function in both branches.
- [ ] Every accumulator is exact integer or a total-order reduction; no
      order-dependent float sums.
- [ ] Serial branch: no `sync_cube`, no shared access across units, writes
      only to slices the unit owns; merges are separate elementwise passes that
      *write*.
- [ ] Cooperative branch ends with `sync_cube`.
- [ ] Work item ≥ ~32 K ops; sizes are constants; comptime args never track
      the frontier; at most two cube widths per kernel.
- [ ] Per-axis builtins; geometry from `launch::*`; the flattened index is
      bounds-guarded; `launch_unchecked` with a `SAFETY` note.
- [ ] No loop-invariant loads inside a row loop — but no `RuntimeCell`
      copies of scalar arguments either; `-O0` has nowhere to put them.
- [ ] Elementwise kernels take a run of lanes per unit
      (`launch::elementwise_runs`); one lane per unit is a GPU shape.
- [ ] A loop that gathers scattered rows issues a block's misses together
      before it finishes any row.
- [ ] Expensive arithmetic a comptime configuration never reads is behind
      that comptime flag, not left for the JIT to remove; op counts checked
      in the final MLIR.
- [ ] Vectorised only if it is a stream; check the width it actually launched
      at, and expect it to be neutral on the CPU runtime once it has runs —
      *except* where the vector replaces two scalar operations on adjacent
      words in an instruction-bound loop (R16), which is a count, not a width.
- [ ] Large per-round uploads hand the client an owned buffer (R17).
- [ ] Any single launch whose absolute time matters has been wall-clocked
      with a sync on each side; the profiler's number for it is a share.
- [ ] Profiled: per-kernel shares (compiles found by size, one per *width*),
      the per-launch series level by level, a shape harness on the node sizes
      a fit really has, and a hand-off microbenchmark on the target host.
- [ ] Oracle tests pass on the CPU runtime, and the training tests still say
      bit-identical.
- [ ] If the body is shared, the GPU shape is unchanged or the change has been
      run on a GPU (§8); "compiles for the cooperative shape" is not a result.
