#!/usr/bin/env bash
# Head-to-head: this crate against the pinned reference XGBoost, on identical
# data and configuration.
#
#   tools/compare.sh [threads]
#
# Both sides are timed over the same generated dataset (written once by the
# Rust benchmark and read back by the Python harness) and report the final
# train-rmse so the timings can be checked for like-for-like work.
#
# Each side trains exactly once per case. Repeating the fit would let XGBoost
# reuse the binned matrix it caches on the DMatrix while this crate rebuilt it,
# which is not a like-for-like comparison.
set -euo pipefail

cd "$(dirname "$0")/.."
THREADS="${1:-8}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

BIN=target/release/train_bench
[ -x "$BIN" ] || cargo build --release --no-default-features --bin train_bench

# rows features rounds depth max_bin sparsity
CASES=(
  "100000 50 20 6 256 0.0"
  "500000 50 20 6 256 0.0"
  "100000 200 20 6 256 0.0"
  "1000000 20 20 6 256 0.0"
  "100000 50 20 10 256 0.0"
  "100000 50 20 6 64 0.0"
  "200000 50 20 6 256 0.3"
)

# Best of N whole-process runs: each run still pays full setup (sketch and
# binning), so neither side benefits from a cache the other cannot use, while
# the minimum filters out scheduler noise.
RUNS=3

best_of() { # command... -> "seconds rmse"
  local best="" rmse=""
  for _ in $(seq "$RUNS"); do
    local out t r
    out="$("$@")"
    t=$(echo "$out" | awk '/^train:/{print $2}' | tr -d 's')
    r=$(echo "$out" | awk '/^train:/{print $NF}')
    if [ -z "$best" ] || awk -v a="$t" -v b="$best" 'BEGIN{exit !(a<b)}'; then best="$t"; rmse="$r"; fi
  done
  echo "$best $rmse"
}

printf '%-34s %10s %10s %8s   %s\n' "case (rows/feat/rounds/depth/bin)" "xgboost_rs" "xgboost" "speedup" "rmse match"
for c in "${CASES[@]}"; do
  read -r rows feat rounds depth bin sparsity <<<"$c"
  data="$TMP/d.bin"

  read -r rs_t rs_r <<<"$(best_of "$BIN" --rows "$rows" --features "$feat" --rounds "$rounds" \
        --depth "$depth" --max-bin "$bin" --sparsity "$sparsity" --threads "$THREADS" \
        --repeats 1 --dump "$data")"

  read -r xg_t xg_r <<<"$(best_of .venv-oracle/bin/python tools/bench_xgb.py "$data" \
        --rows "$rows" --features "$feat" --rounds "$rounds" --depth "$depth" \
        --max-bin "$bin" --threads "$THREADS" --repeats 1)"

  speed=$(awk -v a="$xg_t" -v b="$rs_t" 'BEGIN{printf "%.2fx", a/b}')
  match=$(awk -v a="$rs_r" -v b="$xg_r" 'BEGIN{d=a-b; if(d<0)d=-d; print (d <= 1e-5*(b<0?-b:b)+1e-9) ? "yes" : "NO ("a" vs "b")"}')
  printf '%-34s %10s %10s %8s   %s\n' "${rows}/${feat}/${rounds}/${depth}/${bin}$([ "$sparsity" != "0.0" ] && echo " sp$sparsity")" \
    "$rs_t" "$xg_t" "$speed" "$match"
done
