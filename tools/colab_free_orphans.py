#!/usr/bin/env python3
"""Release Colab GPU assignments that the CLI has lost track of.

A `colab new` that fails partway can leave an assignment allocated with no
local session record. It then holds the one-GPU quota — every later `colab new`
fails with `TooManyAssignmentsError` — and `colab stop` cannot address it,
because `stop` looks the session up by name in local state:

    $ colab sessions
    [?] gpu-t4-s-... | Hardware: T4 | Variant: GPU
    $ colab stop -s gpu-t4-s-...
    [colab] Session 'gpu-t4-s-...' not found.

The `[?]` marks exactly this case. This script calls the same unassign endpoint
the stop path would have used, straight through the CLI's own client.

Run it with the CLI's interpreter, which has `colab_cli` importable:

    ~/.local/share/uv/tools/google-colab-cli/bin/python tools/colab_free_orphans.py

It releases **every** assignment on the account, so do not run it while a
session you care about is doing work — check `colab sessions` first.
"""

from __future__ import annotations

import sys

from colab_cli.auth import get_credentials
from colab_cli.client import Client, Prod


def main() -> int:
    creds = get_credentials(None, provider="adc")
    client = Client(Prod(), creds)

    assignments = client.list_assignments()
    if not assignments:
        print("no assignments to release")
        return 0

    print(f"{len(assignments)} assignment(s):")
    for a in assignments:
        print(f"  endpoint={a.endpoint} accelerator={a.accelerator} variant={a.variant}")

    for a in assignments:
        print(f"unassigning {a.endpoint} ...")
        client.unassign(a.endpoint)
        print("  released")

    remaining = len(client.list_assignments())
    print(f"remaining: {remaining}")
    return 0 if remaining == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
