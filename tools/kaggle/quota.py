#!/usr/bin/env python3
"""Report the Kaggle GPU quota, and exit 0 only when a session could start.

`kaggle kernels push` refuses outright once the weekly allowance is spent, with
"Maximum weekly GPU quota of 30.00 hours reached" — a message that reads like a
broken kernel config rather than an account limit. Checking first turns that
into something a script can wait on.

    python3 tools/kaggle/quota.py [--need-minutes N]

Exit status: 0 if at least `--need-minutes` remain, 1 otherwise (or on error).
"""

from __future__ import annotations

import argparse
import sys

from kagglesdk import KaggleClient
from kagglesdk.kernels.types.kernels_api_service import (
    ApiGetAcceleratorQuotaStatisticsRequest,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    # Kaggle reserves session time up front, so leave headroom rather than
    # waiting for the very last minute of the allowance.
    ap.add_argument("--need-minutes", type=float, default=30.0)
    ap.add_argument(
        "--porcelain",
        action="store_true",
        help="print `used allowed reserved remaining` in hours, for scripts",
    )
    args = ap.parse_args()

    try:
        with KaggleClient() as client:
            q = client.kernels.kernels_api_client.get_accelerator_quota_statistics(
                ApiGetAcceleratorQuotaStatisticsRequest()
            ).gpu_quota
    except Exception as exc:  # noqa: BLE001 - report and treat as "cannot run"
        print(f"quota check failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    used_h = q.time_used.total_seconds() / 3600.0
    allowed_h = q.total_time_allowed.total_seconds() / 3600.0
    reserved_h = q.time_reserved.total_seconds() / 3600.0
    remaining_h = allowed_h - used_h - reserved_h

    if args.porcelain:
        print(f"{used_h:.4f} {allowed_h:.4f} {reserved_h:.4f} {remaining_h:.4f}")
    else:
        print(
            f"gpu quota: {used_h:.2f}h used of {allowed_h:.2f}h allowed"
            f"{f', {reserved_h:.2f}h reserved' if reserved_h else ''}"
            f"  ->  {remaining_h:.2f}h remaining"
        )
    return 0 if remaining_h * 60.0 >= args.need_minutes else 1


if __name__ == "__main__":
    sys.exit(main())
