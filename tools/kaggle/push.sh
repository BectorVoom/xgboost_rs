#!/usr/bin/env bash
# Package the crate, upload it as a Kaggle dataset, and push the GPU kernel.
#
#   tools/kaggle/push.sh            # push and return
#   tools/kaggle/push.sh --wait     # push, wait for completion, fetch output
#
# Requires an authenticated `kaggle` CLI (~/.kaggle/). The kernel requests a
# T4 via `machine_shape` — see kernel-metadata.json for why that matters.
set -euo pipefail

cd "$(dirname "$0")/../.."
SLUG="yensen2/xgboost-rs-gpu-oracle"
DATASET="yensen2/xgboost-rs-gpu-src"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

mkdir -p "$STAGE/data" "$STAGE/kernel"

# The crate, minus build output and the committed fixtures (the generator
# rebuilds its datasets from scratch, so they are dead weight in the upload).
tar czf "$STAGE/data/xgboost_rs_kaggle.tar.gz" \
  --exclude=target --exclude=.git --exclude='tests/fixtures' \
  src tests tools Cargo.toml Cargo.lock KAGGLE.md
cat > "$STAGE/data/dataset-metadata.json" <<EOF
{"title": "xgboost-rs gpu src", "id": "$DATASET", "licenses": [{"name": "Apache 2.0"}]}
EOF

cp tools/kaggle/run.py tools/kaggle/kernel-metadata.json "$STAGE/kernel/"

echo "==> uploading source dataset"
if kaggle datasets status "$DATASET" >/dev/null 2>&1; then
  kaggle datasets version -p "$STAGE/data" -m "$(git rev-parse --short HEAD)" -q -d
else
  kaggle datasets create -p "$STAGE/data" -q
fi

# Dataset ingest is asynchronous. Pushing the kernel before it finishes gives a
# session whose /kaggle/input is empty, which is a confusing way to fail.
echo "==> waiting for the dataset to be ready"
for _ in $(seq 1 60); do
  [ "$(kaggle datasets status "$DATASET" 2>&1)" = "ready" ] && break
  sleep 10
done

echo "==> pushing kernel"
kaggle kernels push -p "$STAGE/kernel"

if [ "${1:-}" != "--wait" ]; then
  exit 0
fi

echo "==> waiting for the run"
until kaggle kernels status "$SLUG" 2>&1 | grep -qE "COMPLETE|ERROR|CANCEL"; do
  sleep 60
done
kaggle kernels status "$SLUG"

OUT="kaggle-output"
mkdir -p "$OUT"
kaggle kernels output "$SLUG" -p "$OUT"
echo "==> output in $OUT/"
