# What each parameter costs

Which parameters change how much work a fit does, by how much, and on which
device. Produced by `src/bin/param_bench.rs`:

```text
cargo run --release --bin param_bench -- \
    --rows 200000 --features 40 --rounds 10 --repeats 3 --threads 8 --device cpu
```

Every row is timed best-of-`--repeats` **whole fits**, so each pays its own
quantile sketch and binning pass and none benefits from a cache another cannot
use. `rel` is against the first row of its group; the sweep also reports the
train metric, because a parameter that looks cheap by *fitting less* is not
cheap, and the two have to be read together.

Absolute times are machine-dependent — an AMD Ryzen AI 7 350, 8 threads — so
read the ratios.

## CPU

200 000 rows x 40 features, 10 rounds, best of 3.

| Group | What moves | Ratio | Fit |
|---|---|---|---|
| `max_bin` | 16 -> 512 | **2.07x** | 0.091 -> 0.057 |
| `max_depth` | 4 -> 10 | **2.89x** | 0.102 -> 0.037 |
| `grow_policy` | depthwise -> lossguide, 64 leaves | 1.33x | level |
| `subsample` | uniform 0.5 | 0.99x | level |
| | `gradient_based` 0.5 | 1.20x | level |
| `colsample` | `bynode` 0.25 | 0.90x | 0.056 -> 0.191 |
| `num_parallel_tree` | 1 -> 4 | **1.84x** | — |
| `constraints` | monotone, all features | 0.96x | 0.056 -> 0.144 |
| `max_cached_hist_node` | 65536 -> 1 | 1.05x | identical |
| `objective` | squarederror -> poisson | 1.20x | — |
| `num_class` | 2 -> 8 | **2.82x** | — |
| `eval_metric` | rmse -> ndcg | 1.49x | — |
| `dart` | `rate_drop` 0 -> 0.5 | **5.68x** | 0.056 -> 0.174 |
| `tree_method` | hist -> approx, constant hessian | 1.12x | identical |
| | hist -> approx, varying hessian | **6.94x** | level |
| | hist -> exact | **11.85x** | 0.0557 -> 0.0546 |
| `opt_dense_col` | 1.0 -> 0.5, `exact` | 0.56x | level |
| `gblinear` | cyclic -> thrifty | 1.13x | level |
| | cyclic -> greedy | **5.95x** | 0.691 -> 0.436 |
| `categorical` | numeric -> partition, 64 cats | 1.16x | 0.200 -> 0.016 |
| | partition -> one-hot, 64 cats | 0.81x | 0.016 -> 0.379 |
| `sparse_threshold` | 0 -> 1 | 0.99x | identical |
| `multi_strategy` | one tree per output -> vector leaf, 4 targets | 1.05x | level |
| `nthread` | 1 -> 8 | 0.50x | identical |

Several of those are the *point* of the parameter rather than an artefact:

* **`approx` costs almost nothing extra under a constant hessian, and 6.9x
  under a varying one.** Its sketch is weighted by the hessian, and
  `reg:squarederror`'s hessian is `1` for every row in every round — so the
  sketch is the one `hist` already built and it is not rebuilt. Give the
  hessian something to vary and the full per-round re-sketch and re-bin
  appears. Both rows fit identically to `hist`, so the cost buys nothing on
  this data; it buys accuracy on data whose bins should move.
* **`exact` is an order of magnitude slower** and fits 2% better. That is the
  trade it makes: every distinct value is a split candidate instead of one per
  bin, and there is no binned matrix to compress the scan.
* **`greedy` is quadratic in the feature count.** It re-scans the whole matrix
  once per feature it selects; `thrifty` ranks every feature in a single pass
  and then cycles that order, which is why it lands at 1.13x rather than 6x —
  for a fit within 1% of greedy's.
* **The categorical enumerators trade cost against what they can express.**
  With 64 categories, one-hot is the *cheaper* of the two and fits 23x worse —
  0.379 RMSE against 0.016 — because a one-hot split can only isolate one
  category at a time. Paying 16% for a split the data actually has is the whole
  reason the partition path exists.
* **Column sampling is not a speed knob.** Every `colsample_*` row is within
  10% of no sampling, and the fit degrades by up to 8x. It is a variance knob.
* **`gradient_based` sampling costs 20%, uniform costs nothing.** Uniform
  zeroes a row's gradient from one draw; gradient-based has to compute each
  row's sampling probability from its own gradient first.
* **`sparse_threshold` is a memory knob, and its extreme costs memory.** Every
  row of that group fits the identical model. On the 40-feature matrix the
  default `0.2` stores 7 of 40 columns sparsely for 13.4 MiB against 15.3
  dense, while `1.0` stores 39 of 40 for **22.7 MiB** — larger than storing
  nothing sparsely at all. A sparse column pays 4 bytes of row id per entry to
  save a 1-byte bin, so past roughly 20% density it is a loss, which is exactly
  where upstream puts the default. Time does not move (0.99x–1.03x).
* **`max_cached_hist_node` is within 5% across four orders of magnitude.** It
  bounds resident memory on a wide tree rather than buying speed.

## CPU against the device, on real hardware

The same sweep run twice on one Kaggle VM — **Tesla T4** (SM 7.5, 15 GB)
against the **4 vCPU Intel Xeon @ 2.00 GHz** that hosts it — at 200 000 rows x
40 features, 10 rounds, best of 2. Same machine, back to back, so the ratio
between the two columns means something even though neither absolute number
belongs to the laptop the CPU table above was measured on.

Reproduce with `tools/kaggle/push.sh --wait param-bench-metadata.json`.

### Does the device win?

Wall clock for the same fit, host CPU against T4:

| Case | CPU | T4 | T4 speedup |
|---|---|---|---|
| `hist`, depth 6 (the baseline) | 0.693 s | 0.545 s | 1.27x |
| `max_depth=10` | 2.208 s | 0.616 s | **3.58x** |
| `lossguide`, 64 leaves | 0.958 s | 0.667 s | 1.44x |
| `num_class=8` | 2.605 s | 2.074 s | 1.26x |
| `multi_output_tree`, 4 targets | 1.307 s | 0.804 s | 1.63x |
| `num_parallel_tree=4` | 1.363 s | 0.932 s | 1.46x |
| `dart`, `rate_drop=0.5` | 2.210 s | 2.146 s | 1.03x |
| `approx`, varying hessian | 4.084 s | 3.737 s | 1.09x |
| `exact` | 8.276 s | refused | — |
| `gblinear/coord_descent` | 0.554 s | 0.901 s | **0.61x** |
| `gblinear/greedy` | 7.401 s | 8.754 s | 0.85x |

**The device wins where the tree is deep and loses where the work is serial.**
`max_depth=10` is the standout at 3.6x: the CPU pays 3.9x going from depth 4 to
10 and the T4 pays 1.27x, because the extra levels are more parallel work of
exactly the kind it already had. At the other end, `gblinear` is *slower* on
the device — see below — and `dart` and `approx` barely move, because both
spend their extra time rebuilding host-side state (the dropped-tree rescale,
the per-round re-sketch) rather than in the kernels.

### Which parameters cost the device something different

Ratios within each device, so the host's speed cancels out:

| Group | What moves | CPU | T4 |
|---|---|---|---|
| `max_depth` | 4 -> 10 | 3.91x | **1.27x** |
| `max_bin` | 16 -> 512 | 1.70x | 1.31x |
| `num_class` | 2 -> 8 | 2.71x | 2.48x |
| `grow_policy` | depthwise/16 -> lossguide/64 | 1.68x | **1.38x** |
| `num_parallel_tree` | 1 -> 4 | 1.94x | 1.71x |
| `multi_strategy` | one tree per output -> vector leaf, 4 targets | 1.01x | **0.86x** |
| `dart` | `rate_drop` 0 -> 0.5 | 3.19x | 3.99x |
| `tree_method` | hist -> approx, varying hessian | 5.89x | 6.86x |
| | hist -> exact | 11.94x | **refused** |
| `gblinear` | gbtree -> `coord_descent` | 0.79x | **1.65x** |
| | cyclic -> `greedy` | 13.4x | **9.7x** |
| `nthread` | 1 -> 4 | 0.72x | **1.01x** |

* **Depth is the parameter the device changes most.** 3.9x on the CPU against
  1.3x on the T4. A deeper tree is more nodes per level, which is more
  independent work for a device that already launches a level at a time. The
  same mechanism is why `lossguide` — which the device used to be blamed for —
  comes out *cheaper* there (1.38x against 1.68x) once the nodes are big
  enough: what the device dislikes is small launches, not the policy.
* **A vector leaf is a win on the device and free on the CPU** — 0.86x against
  1.01x at four targets, with the fit level in both columns, so the saving is
  not bought by fitting less. One tree a round instead of one per target saves
  the device three quarters of its per-tree launches and frontier setup.
* **`nthread` does nothing on the device**, which is the expected shape and
  worth having measured: the CPU column drops to 0.72x from 1 to 4 threads and
  the device column does not move, so the work really is off the host.
* **`gblinear` is the one place the device is the wrong tool.** Coordinate
  descent updates one feature at a time and each step needs the previous
  step's residuals, so the device pays a launch and a readback *per feature per
  group per round* — 1.65x the gbtree reference where the CPU solver is 0.79x.
  The serial dependency is upstream's; what this measures is that the round
  trips dominate. It is the right answer bit-for-bit
  (`tests/gpu_linear.rs`) and the wrong shape for this matrix.
* **`exact` is refused**, as upstream refuses it: there is no GPU column-scan
  updater.

### What the lavapipe stand-in got wrong

This document previously carried a device column measured on **lavapipe**, a
software Vulkan implementation on the local CPU, because this machine has no
NVIDIA GPU. It is a faithful way to run the device *code path* and it was
useful for correctness, but as a performance model it was wrong in both
directions and is recorded here so the next person does not trust it again:

Compared at the size lavapipe was measured at — 50 000 x 20, 5 rounds — with
each column's own CPU beside it, because that is the comparison the claims were
made from:

| Ratio | lavapipe CPU | lavapipe device | T4's CPU | T4 |
|---|---|---|---|---|
| `grow_policy`, depthwise/16 -> lossguide/64 | 2.12x | **4.90x** | 1.95x | 2.47x |
| `num_parallel_tree` 1 -> 4 | 2.05x | **3.75x** | 2.20x | 2.08x |
| `gblinear`, gbtree -> `coord_descent` | 0.58x | **3.35x** | 0.47x | 2.13x |
| `multi_strategy`, vector leaf, 4 targets | 1.17x | 0.65x | 0.96x | 0.75x |

So of the four things that column was used to claim:

* **"`lossguide` is the single largest device-only penalty" was wrong.** The
  penalty is real but roughly half the size, and at the larger configuration it
  **inverts** — 1.38x on the T4 against 1.68x on its CPU, so a loss-guided tree
  is *cheaper* on the device once there is enough work per node.
* **"`num_parallel_tree` costs the device nearly twice what it costs the CPU"
  was wrong**; it costs slightly less.
* **`gblinear` was right in direction and about 60% too pessimistic.**
* **The vector-leaf win was right in direction and overstated.**

The pattern is that lavapipe exaggerates anything launch-bound, because a
"launch" there is CPU work rather than a submission to a real queue — and it
exaggerates most where the per-launch work is smallest, which is why the
overstatement grows as the configuration shrinks. Its reliable signal was the
*sign* of a difference, not its size, and even the sign flipped once. Nothing
about correctness is affected — the fits are identical either way, which is
what `tests/gpu_training.rs` and `tests/gpu_linear.rs` assert — but no timing
claim from it survives.

For the comparison against XGBoost's own `gpu_hist` rather than against this
crate's CPU path, see `docs/gpu-benchmarks.md`, and
`docs/gpu-multi-output-performance.md` for the vector-leaf kernels.
