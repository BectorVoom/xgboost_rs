# Performance report: `multi_strategy=multi_output_tree` on the GPU

Companion to `docs/gpu-benchmarks.md`, covering only what the vector-leaf path
added. It is written against the implementation in "Implement GPU
multi_output_tree".

**Read the measurement caveat first.** There is no local NVIDIA GPU here, so
nothing below is a CUDA timing. The tables are wgpu/**lavapipe** — a software
Vulkan rasteriser — where launch and round-trip costs are nothing like a real
device's. They are included for the *shape* of the scaling (how cost grows with
targets, levels and rows), not for throughput, and no conclusion here rests on
their absolute values. The counts and sizes in "What the path costs" are exact
and backend-independent; those are the substance of this report. An absolute
comparison needs the Colab T4 route `docs/gpu-benchmarks.md` documents.

## 1. A dispatch grid overflow that crashes the wgpu backend (defect)

Not a slowdown — a panic. `multi_strategy=multi_output_tree` on `device=cuda`
in a **non-`cuda`-feature build** aborts the fit at moderate depth:

```
wgpu error: Validation Error
  In a dispatch command, indirect:false
    Each current dispatch group size dimension ([65536, 1, 1]) must be less or
    equal to 65535
```

Located to `HistogramEngine::zeroed` (`src/gpu/histogram.rs`), which zeroes a
level's frontier with a **1-D** grid of `frontier_words / 256` workgroups.
Instrumenting the dispatch gives the failing value directly: `grid_x = 65536`,
one over the limit.

Since `frontier_words = 4 * n_bins * n_targets * 2E` for a level with `E`
expandable nodes, the fit survives only while

```
n_bins * n_targets * E  <=  2_097_120
```

That predicts the measured boundary. A `max_depth=d` tree expands parents at
depths `0..=d-2`, so its widest level has `E <= 2^(d-2)`. At 16 features and
`max_bin=256` (`n_bins ≈ 4096`) with 4 targets the ceiling is `E <= 128`, so
depth 8 (`E <= 64`) cannot reach it and depth 10 (`E <= 256`) can:

```
scalar  depth  8, 16f/256b  ->  ok
scalar  depth 10, 16f/256b  ->  ok
scalar  depth 12, 16f/256b  ->  ok          (peak grid_x 50688, 77% of limit)
multi/2 depth 10, 16f/256b  ->  ok
multi/4 depth  8, 16f/256b  ->  ok
multi/4 depth 10, 16f/256b  ->  DISPATCH FAILURE
multi/4 depth 10,  4f/ 64b  ->  ok          (small n_bins buys the headroom back)
```

The instrumented dispatch agrees with the arithmetic throughout: depth 8 peaks
at `grid_x = 32768` (`E = 64`, half the limit), the depth-10 vector-leaf fit
reaches `65536` on a level with `E = 128`, and scalar depth 12 reaches `50688`
on a level with `E = 396` — real trees do not saturate, which is why the scalar
path has never tripped it.

Three things worth separating:

- **The ceiling is pre-existing.** The scalar path shares the same 1-D dispatch
  and already reached 77% of the limit at depth 12. What the vector-leaf path
  did was *divide the ceiling by `n_targets`*, which moves it from "unreachable
  in practice" to "reachable at default settings".
- **CUDA is unaffected.** `gridDim.x` there is `2^31 - 1`. This bites the
  wgpu/Vulkan backend only — which is, however, the backend every test in the
  repository runs on, and the one that makes the path testable without a GPU.
- **The benchmark shapes are close to it.** The `baseline` case in
  `gpu-benchmarks.md` is 50 features at `max_bin=256` (`n_bins ≈ 12800`). At 4
  targets that allows only `E <= 40`, which a depth-8 vector-leaf tree already
  exceeds.

The existing test matrix misses this because the vector-leaf cases top out at
`max_depth=6`. Fix and regression test are both small: give `zeroed` a 2-D grid
(or a grid-stride loop with a capped workgroup count), and add a deep,
wide-`n_bins` vector-leaf case. Nothing else in the path has a
frontier-proportional 1-D dispatch — `subtract_batch` and `prefix_scan` already
spread across `y`/`z`.

## 2. What the path costs, per level

Exact counts, for a level with `E` expandable nodes and `K` targets. The
"scalar" column is the same level in a one-output-per-tree fit.

| per level                | scalar | vector leaf | note |
|--------------------------|--------|-------------|------|
| kernel launches          | 9      | `K + 10`    | `K` histogram builds instead of 1, plus the prefix scan and the child-sums pass |
| **host round trips**     | **3**  | **4**       | the extra one is `multi_child_sums` |
| frontier bytes           | `F`    | `K·F`       | one histogram per `(node, target)` |
| peak device working set  | `F`    | `2·K·F`     | the prefix buffer is exactly frontier-sized |
| extra global traffic     | —      | `2·K·F`     | the prefix pass reads the frontier and writes an equal buffer |

The round trip is the one that should worry us. `gpu-benchmarks.md` already
concludes, from a fit cost that stays flat as rows, features and bins change,
that **per-level host round trips are the dominant remaining cost** of the
scalar GPU path, and names getting from three to one as the next thing to try.
The vector-leaf path went the other way and made it four. Each is a full
pipeline drain.

It is a deliberate trade, and the reason is in the design: a candidate carries
one `(grad, hess)` pair summed over targets — as `SplitEntry` does on the CPU —
so the winner's per-target child sums have to be recovered afterwards. The
alternative, carrying `K` pairs through the split reduction, needs
`block * 2K` `i64` of shared memory and blows the 16 KiB workgroup budget the
portable backends allow from about `K = 3`. Two ways out that do not:

- Fold the child-sums pass into `reduce_candidates_kernel` and read one packed
  buffer, taking the count back to 3. The reduction already knows the winning
  bin and direction; it currently throws both away and the host re-derives the
  bin from the threshold via `find_split_condition`.
- Keep the split entries on device entirely, which is the same fix the scalar
  path wants for its own three.

## 3. Where the frontier and prefix buffers go

`2·K·F` peak is the headline number, and `F` is itself
`16 · n_bins · 2E` bytes. On the CUDA path — where §1's grid ceiling does not
apply — the `gpu-benchmarks.md` `baseline` shape (50 features, `max_bin=256`,
so `n_bins ≈ 12800`) with 4 targets and a level of `E = 64` holds a 100 MiB
frontier and another 100 MiB of prefix sums. That fits a 15 GB T4 and does not
fit much further along either axis; both terms are linear in `n_bins`,
`n_targets` and the level's node count at once.

The prefix buffer is `client.empty(...)` per `evaluate_multi` call, so it leans
entirely on CubeCL's allocator reusing the block across levels; nothing in this
crate pools it.

## 4. Host-side allocation churn

Per tree, `grow` builds `K` gradient columns and `K` quantised columns —
`24 · K · n_rows` bytes, allocated and dropped every tree. At 500k rows and 8
targets that is 96 MB per tree, 20 times over a 20-round fit. The scalar path
already did this at `K = 1`; the vector-leaf path multiplies it.

Per node, `ExpandEntry` now carries three `Vec<GradientPairInt64>`
(`target_sums`, `left_target_sums`, `right_target_sums`). It is cloned on every
`evaluate` (`..node.clone()`) and moves through the expansion queue, so a
depth-10 tree pays thousands of `K`-element heap allocations. `MultiNodeInput`
clones the parent sums again per node per level, `build_children` clones
`targets` per child, and the histogram launch clones `hist_jobs` once per
target. None of this is on the device critical path, but it is all avoidable
with one flat `Vec<GradientPairInt64>` indexed `node * K + t`, which is exactly
how `HistGrower` stores `snode_targets` on the CPU.

## 5. Ruled out: redundant loads in the split kernel

Worth recording because it is the natural suspicion and it is wrong.
`consider_split_multi` reads the forward prefix, the feature total and the
predecessor bin, and picks between them with `if` expressions whose indices are
clamped to stay in bounds — which reads as though every candidate issues six
loads per target where two or four are used.

Dumping the generated SPIR-V (`CUBECL_DEBUG_LOG=<file>`) and counting access
chains into the `prefix` binding shows **8** for the whole kernel: 2 for the
forward scan, 4 for the backward scan, 2 for the missing-value test. That is the
minimum the algorithm needs. CubeCL emits real branches rather than evaluating
both arms, so the defensive clamping costs nothing and there is no redundant
traffic to remove here.

## 6. Measured scaling (lavapipe — shape only, not throughput)

16 features, `max_bin=256`, 3 rounds, `reg:squarederror`. Seconds.

Targets, at 100k rows and depth 6. `multi/1pt` is the vector-leaf fit over the
device's own alternative, one tree per output:

```
targets  gpu-multi  cpu-multi  gpu-1pertree  multi/1pt
      1      0.099      0.047         0.141      0.70x
      2      0.166      0.054         0.196      0.85x
      4      0.219      0.082         0.335      0.65x
      8      0.340      0.132         0.607      0.56x
```

Depth, at 100k rows and 4 targets:

```
 depth  gpu-multi  cpu-multi  gpu/cpu
     2      0.153      0.045    3.44x
     4      0.161      0.056    2.85x
     6      0.248      0.082    3.00x
     8      0.297      0.144    2.07x
```

Rows, at depth 6 and 4 targets:

```
    rows  gpu-multi  cpu-multi  gpu/cpu
   25000      0.128      0.042    3.05x
   50000      0.151      0.063    2.40x
  100000      0.215      0.086    2.50x
  200000      0.370      0.128    2.89x
```

What survives the caveat:

- **Against the device's own alternative, the vector leaf wins, and wins more
  as `K` grows** — 0.85× at 2 targets down to 0.56× at 8. Growing one tree
  instead of `K` amortises the per-level fixed cost across the targets, which is
  the behaviour the design intended. Cost grows about 3.4× from `K = 1` to
  `K = 8`, well under the 6.1× the one-tree-per-output path pays.
- **It still loses to the CPU grower here, by 2–3.4×**, in line with what
  `gpu-benchmarks.md` already records for the scalar path on real hardware
  (5–14×). The vector-leaf path is not a new regression; it inherits the
  standing gap.
- **No usable signal on the round-trip hypothesis.** The `gpu/cpu` ratio is flat
  in rows and *falls* with depth, which is the opposite of what a round-trip
  bound would give — but a software rasteriser does not price a pipeline drain
  the way a T4 does, so this says nothing either way. Testing it needs the CUDA
  build.

## 7. Suggested order of work

1. **The grid overflow (§1).** A defect, not a slowdown, and cheap to fix. Do
   this first, with a regression test at depth ≥ 10 and a large `n_bins`.
2. **The fourth round trip (§2)**, by having the reduction keep the winning bin
   and direction so the child sums ride back in the same buffer. This is the one
   with real upside, given what `gpu-benchmarks.md` concluded about round trips.
3. **Flatten the per-node target vectors (§4).** Mechanical, and it removes the
   allocation churn wholesale.
4. **Re-measure on the T4** via `tools/compare_gpu.py`, extended with
   vector-leaf cases, and fold the result into `docs/gpu-benchmarks.md`. Until
   then this crate has no absolute number for the vector-leaf path at all.

## Reproducing

The tables came from a temporary integration test driving `api::train` on both
devices; it is not committed.

The grid overflow reproduces on a build **without** `--features cuda` with a
`device=cuda`, `multi_strategy=multi_output_tree` fit at 60 000 rows, 16
features, `max_bin=256`, 4 targets, `max_depth=10`, one round. Note that the
committed vector-leaf tests do *not* reach it: they top out at `max_depth=6`,
and their 5-feature matrices give `n_bins ≈ 1280`, which allows `E <= 409` —
far above anything a depth-6 tree produces. A regression test needs both a
larger `n_bins` and a deeper tree.

The `grid_x` figures come from an `eprintln!` in `HistogramEngine::zeroed`.

The SPIR-V count in §5 is
`CUBECL_DEBUG_LOG=/tmp/cubecl.log cargo test --release --test gpu_training
multi_output_tree_matches_the_cpu_fit`, then counting `OpAccessChain`
instructions naming the `prefix` binding inside `EvaluateFeatureMultiKernel`.
