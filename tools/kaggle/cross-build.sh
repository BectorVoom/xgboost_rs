#!/usr/bin/env bash
# Cross-compile the CUDA binaries for a Linux x86_64 GPU box from this machine.
#
#   tools/kaggle/cross-build.sh            # -> prebuilt/{train_bench,bench,kernels,gpu_training}
#
# A Kaggle session rebuilds the crate from source in about three and a half
# minutes; this turns that into a ninety-second build here and a 30 MB
# upload, which is what makes a measure-change-measure loop on a remote GPU
# bearable. `push.sh` ships `prebuilt/` when it exists and `run_gpu_compare.py`
# uses it in preference to building.
#
# Needs the `x86_64-unknown-linux-gnu` target (`rustup target add ...`), the
# `cargo-zigbuild` linker driver (`cargo install cargo-zigbuild`) and a `zig`
# on PATH (`pip install ziglang` puts one at `python3 -m ziglang`; the
# `ZIG` variable names it). `cudarc` is told the CUDA version to bind rather
# than asked to find a toolkit here: 13.0 is what Kaggle's driver reports.
set -euo pipefail
cd "$(dirname "$0")/../.."

TARGET="${TARGET:-x86_64-unknown-linux-gnu.2.31}"
CUDA="${CUDARC_CUDA_VERSION:-13000}"
OUT=prebuilt
TDIR=target/x86

if [ -n "${ZIG:-}" ]; then
  export PATH="$(dirname "$ZIG"):$PATH"
fi
command -v zig >/dev/null || { echo "zig not on PATH (set ZIG=/path/to/zig)" >&2; exit 2; }
command -v cargo-zigbuild >/dev/null || { echo "cargo-zigbuild not installed" >&2; exit 2; }

# Only the CUDA backend: the default `cpu` feature would drag the MLIR JIT
# into a binary that never uses it, and it is most of the build. The
# executables are taken from cargo's own artifact messages — a test binary
# carries a hash, and the newest file in `deps/` is not reliably the one just
# built.
mkdir -p "$OUT"
CUDARC_CUDA_VERSION="$CUDA" CARGO_TARGET_DIR="$TDIR" cargo zigbuild --release \
  --target "$TARGET" --no-default-features --features cuda \
  --bin train_bench --bin bench --test kernels --test gpu_training \
  --message-format=json-render-diagnostics \
| python3 -c '
import json, shutil, sys
want = {"train_bench", "bench", "kernels", "gpu_training"}
for line in sys.stdin:
    try:
        m = json.loads(line)
    except ValueError:
        continue
    if m.get("reason") != "compiler-artifact" or not m.get("executable"):
        continue
    name = m["target"]["name"]
    if name in want:
        shutil.copy2(m["executable"], sys.argv[1] + "/" + name)
        print("prebuilt/" + name, "<-", m["executable"])
' "$OUT"
echo "$(git rev-parse --short HEAD)$(git diff --quiet || echo '-dirty')" > "$OUT/GIT_REV"
# A build stamp the run prints back, so a session that got an older dataset
# version says so.
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$OUT/BUILD_STAMP"
ls -la "$OUT"
