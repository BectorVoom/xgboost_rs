#!/usr/bin/env bash
# Wait for the weekly Kaggle GPU allowance to reset, then run the T4 kernel once.
#
#   tools/kaggle/await-quota.sh [check-interval-seconds]   # default 1800 (30 min)
#
# The weekly allowance is 30 GPU-hours and `kernels push` refuses outright once
# it is spent, so there is nothing to do but wait for the reset. This polls the
# quota and, the first time a session could start, runs push.sh --wait end to
# end (package, upload, push, poll, fetch output) and then exits.
#
# Safe to leave running: it pushes at most once, and exits non-zero only if the
# push itself fails.
set -uo pipefail

cd "$(dirname "$0")/../.."
INTERVAL="${1:-1800}"
LOG="kaggle-await.log"

# uv-installed CLIs carry their interpreter in the shebang; that interpreter is
# the one with kagglesdk importable.
KAGGLE_BIN="$(command -v kaggle || true)"
if [ -z "$KAGGLE_BIN" ]; then
  echo "kaggle CLI not on PATH" >&2
  exit 2
fi
KPY="$(head -1 "$KAGGLE_BIN" | sed 's|^#!||')"
[ -x "$KPY" ] || KPY="$(command -v python3)"

say() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" | tee -a "$LOG"; }

say "waiting for GPU quota (checking every ${INTERVAL}s); interpreter $KPY"
while true; do
  if out="$("$KPY" tools/kaggle/quota.py --need-minutes 30 2>&1)"; then
    say "$out"
    say "quota available - launching the T4 run"
    if bash tools/kaggle/push.sh --wait 2>&1 | tee -a "$LOG"; then
      say "T4 run finished; output in kaggle-output/"
      exit 0
    fi
    say "push failed - see $LOG"
    exit 1
  fi
  say "$out"
  sleep "$INTERVAL"
done
