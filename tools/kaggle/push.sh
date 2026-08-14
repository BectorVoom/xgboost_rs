#!/usr/bin/env bash
# Package the crate, upload it as a Kaggle dataset, and push a GPU kernel.
#
#   tools/kaggle/push.sh                                  # the oracle kernel
#   tools/kaggle/push.sh --wait                           # ...and wait for it
#   tools/kaggle/push.sh --wait param-bench-metadata.json # a different kernel
#
# The kernel is named by its metadata file, which carries both the slug and the
# script to run, so adding a kernel means adding a metadata file and nothing
# else. Requires an authenticated `kaggle` CLI (~/.kaggle/). The kernels request
# a T4 via `machine_shape` — see the metadata comment for why that matters.
set -euo pipefail

cd "$(dirname "$0")/../.."
WAIT=""
META="kernel-metadata.json"
for arg in "$@"; do
  case "$arg" in
    --wait) WAIT=1 ;;
    *.json) META="$arg" ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

META_PATH="tools/kaggle/$META"
[ -f "$META_PATH" ] || { echo "no such metadata: $META_PATH" >&2; exit 2; }

# The owner comes from whoever the CLI is authenticated as, never from a name
# written down here. A hardcoded one is wrong the moment the credentials change
# and fails in a way that reads like a broken config: `datasets create` reports
# "dataset slugs and hashlink are all null", then the kernel pushes *without*
# its source and dies on an empty /kaggle/input. The committed metadata carries
# `KAGGLE_OWNER` and it is substituted here.
OWNER="$(kaggle config view 2>/dev/null | awk '/^- username:/{print $3}')"
[ -n "$OWNER" ] || { echo "cannot read the Kaggle username — is the CLI authenticated?" >&2; exit 2; }

SLUG="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["id"])' "$META_PATH")"
SLUG="${SLUG/KAGGLE_OWNER/$OWNER}"
CODE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["code_file"])' "$META_PATH")"
DATASET="$OWNER/xgboost-rs-gpu-src"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

echo "==> owner $OWNER, kernel $SLUG (from $CODE)"
mkdir -p "$STAGE/data" "$STAGE/kernel"

# The crate, minus build output and the committed fixtures (the generator
# rebuilds its datasets from scratch, so they are dead weight in the upload).
tar czf "$STAGE/data/xgboost_rs_kaggle.tar.gz" \
  --exclude=target --exclude=.git --exclude='tests/fixtures' \
  src tests tools Cargo.toml Cargo.lock KAGGLE.md
cat > "$STAGE/data/dataset-metadata.json" <<EOF
{"title": "xgboost-rs gpu src", "id": "$DATASET", "licenses": [{"name": "Apache 2.0"}]}
EOF

# Kaggle wants the kernel's metadata under its own fixed name.
cp "tools/kaggle/$CODE" "$STAGE/kernel/"
sed "s|KAGGLE_OWNER|$OWNER|g" "$META_PATH" > "$STAGE/kernel/kernel-metadata.json"

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

[ -n "$WAIT" ] || exit 0

echo "==> waiting for the run"
until kaggle kernels status "$SLUG" 2>&1 | grep -qE "COMPLETE|ERROR|CANCEL"; do
  sleep 60
done
kaggle kernels status "$SLUG"

OUT="kaggle-output/$(basename "$SLUG")"
mkdir -p "$OUT"
# `output` fetches the files the run wrote; `logs` fetches the console, which is
# the only place a failure before the first file shows up.
kaggle kernels output "$SLUG" -p "$OUT" || true
kaggle kernels logs "$SLUG" > "$OUT/console.log" 2>&1 || true
echo "==> output in $OUT/"
