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

## CPU against the device

The same sweep at 50 000 rows x 20 features, 5 rounds, run twice — once with
`--device cpu`, once with `--device cuda`. Only the groups where the two
*differ* are worth listing; everything else lands within noise of the CPU
column's ratio.

| Group | What moves | CPU | device |
|---|---|---|---|
| `grow_policy` | depthwise -> lossguide, 64 leaves | 2.12x | **4.90x** |
| `num_parallel_tree` | 1 -> 4 | 2.05x | **3.75x** |
| `multi_strategy` | one tree per output -> vector leaf, 4 targets | 1.17x | **0.65x** |
| `dart` | `rate_drop` 0 -> 0.5 | 2.19x | 1.42x |
| `tree_method` | hist -> approx, varying hessian | 3.61x | 1.66x |
| | hist -> exact | 4.67x | **refused** |
| `gblinear` | gbtree -> `coord_descent` | 0.58x | **3.35x** |
| | gbtree -> `shotgun` | 0.61x | 0.34x |
| `nthread` | 1 -> 8 | 0.51x | 1.00x |

What those say about the *shape* of the device path — none of it dependent on
which GPU is underneath:

* **`lossguide` costs the device more than twice what it costs the CPU.** The
  device grows a *batch* of nodes per launch, and loss-guide's queue yields one
  node at a time (`ExpandQueue::pop_batch`), so every launch does one node's
  work. Depth-wise fills the batch with a whole level. This is the single
  largest device-only penalty in the sweep.
* **A vector leaf is a *win* on the device (0.65x) and a small loss on the CPU
  (1.17x).** One tree per round instead of one per target saves the device a
  per-tree round of launches and frontier setup, which is a cost the CPU
  barely has. The fit is level in both columns, so the saving is not bought by
  fitting less. `multi_strategy=multi_output_tree` is worth reaching for on the
  device in a way it is not on the CPU.
* **`nthread` does nothing on the device**, which is the expected shape and
  worth having measured: the CPU column scales 2x from 1 to 8 threads, the
  device column is flat, so the work really is off the host.
* **`gblinear`'s device solver is 3.4x the gbtree reference where the CPU
  solver is 0.6x.** Coordinate descent updates one feature at a time and each
  step needs the previous step's residuals, so the device pays a launch and a
  readback *per feature per group per round*. That serial dependency is
  upstream's too; what this measures is that the round trips dominate at this
  size. It is the right answer bit-for-bit (`tests/gpu_linear.rs`) and the wrong
  shape for a small matrix.
* **`exact` is refused**, as upstream refuses it: there is no GPU column-scan
  updater.

## What the device column is and is not

This machine has no NVIDIA GPU. The `device` column is the device *code path*
running on **lavapipe**, a software Vulkan implementation on the same CPU, so
it measures what that path *does* — how many launches, how much crosses the
host boundary, how the work scales with the parameter — and **not how fast a
GPU would do it**. Nothing above is a claim about GPU speed, and the ratios
quoted are only ever device-against-device.

For real hardware see `docs/gpu-benchmarks.md` (Tesla T4, against XGBoost's own
`gpu_hist`, across rows, features, depth, `max_bin`, sparsity, `grow_policy`
and round count) and `docs/gpu-multi-output-performance.md`. `param_bench`
takes `--device cuda`, so this sweep is what to run there.
