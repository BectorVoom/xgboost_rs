# What the GPU path still does not do

`device=cuda` trains and matches the CPU fit bit-for-bit on a single round
(`tests/gpu_training.rs`). Three things are still refused or absent. Each is
refused by name at the API boundary rather than silently mis-fitted, so nothing
here is a correctness risk — only missing capability.

## 1. Categorical splits (refused)

`src/api.rs` rejects a categorical `DMatrix` on `device=cuda`:

> categorical splits are not implemented on the GPU; fit on `device=cpu`, or
> one-hot encode the categories yourself

The device evaluator scans a feature's bins in ascending order, which is
meaningless for category codes. The CPU reference is `HistGrower::
enumerate_one_hot` and `enumerate_partition` (`src/tree/hist.rs`).

**What it needs.** `common::UseOneHot` picks between two algorithms on
`n_bins_feature < max_cat_to_onehot` (default 4):

- *One-hot* — for each category bin, two candidates: the category alone goes
  right with the feature's missing rows left, then again with missing right.
  The chosen category rides in `split_value`, and `cat::set_bit` turns it into
  a one-bit set. Straightforward to port: no sort, and the existing
  `consider_split` already does the scoring.
- *Partition* — sort the feature's bins by `CalcWeightCat` (the *unconstrained*
  weight; categories carry no monotonicity), then scan that order both
  directions, capped at `max_cat_threshold` (default 64) steps. The head of the
  order is the right-hand side either way. The winning split's bit set names
  the first `partition` categories of the sorted order.

The partition path is the hard part: it needs the sort on device. A bitonic
sort in shared memory covers `n_bins_feature <= block`; beyond that it needs
multiple elements per thread. Two cheaper routes, if the full port is not
wanted:

- Evaluate *only* the categorical features host-side — read back those
  features' bins for the node, reuse the CPU enumeration, and merge the result
  with the device's numeric best via `SplitEntry::update_entry`. Correct, and
  the readback is proportional to the categorical bins only.
- Keep the sort on device but let the host rebuild the bit set: the kernel
  returns `(forward, partition_count)` and the host re-sorts that node's
  feature slice once per *applied* categorical split, which is rare.

The row partitioner already handles categorical splits
(`goes_left` tests the bit set, `tests/gpu_row_partition.rs` covers it), so only
the evaluator and the bit-set plumbing are missing.

## 2. `multi_strategy=multi_output_tree` (refused)

Refused for every path except CPU `hist`, GPU included. A vector leaf grows one
histogram per target; the device histogram kernel is scalar, so it would need a
target dimension — the same shape the node dimension took in `27ab94d`.

## 3. SYCL (refused, and out of scope)

`Device::Sycl` is rejected outright. CubeCL has no SYCL backend here, so there
is nothing to run it on.

## Not a feature gap, but outstanding

`docs/gpu-benchmarks.md` records that the device path is 5–14× slower than
XGBoost's `gpu_hist`, with the measured evidence for where the time goes.
