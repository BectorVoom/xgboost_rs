#!/usr/bin/env bash
# Wait for the weekly Kaggle GPU allowance to reset, then run the T4 kernel once.
#
#   tools/kaggle/await-quota.sh [check-interval-seconds]   # default 1800 (30 min)
#
# The wait can be days, so detach it from your shell rather than leaving it in
# a terminal (or in an agent session, which will eventually end):
#
#   setsid nohup tools/kaggle/await-quota.sh 1800 >/dev/null 2>&1 &
#
# Progress goes to kaggle-await.log either way. To stop it: pkill -f await-quota
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
prev_used=""
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

  # Usage climbing between checks means a session of yours is running right
  # now and drawing the allowance down further — worth naming, because
  # otherwise the log just looks like the quota is refusing to reset.
  used="$("$KPY" tools/kaggle/quota.py --porcelain 2>/dev/null | awk '{print $1}')"
  if [ -n "$prev_used" ] && [ -n "$used" ]; then
    if awk -v a="$used" -v b="$prev_used" 'BEGIN{exit !(a>b+0.001)}'; then
      say "  note: usage rose $(awk -v a="$used" -v b="$prev_used" 'BEGIN{printf "%.2f", a-b}')h since the last check - a GPU session is still running:"
      kaggle kernels list --mine --sort-by dateRun -p 1 2>/dev/null | awk 'NR>2 && NR<6 {print "    " $1}' | tee -a "$LOG"
    fi
  fi
  prev_used="$used"
  sleep "$INTERVAL"
done
