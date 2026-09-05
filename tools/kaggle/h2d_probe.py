#!/usr/bin/env python3
"""Host-to-device bandwidth that owes nothing to CubeCL: PyTorch copies,
pageable and pinned, at a per-round size and a whole-matrix size.

`bench` measures CubeCL's uploads at ~0.37 GB/s on a Kaggle T4 VM. This says
whether that is the VM or the runtime.
"""
import time

try:
    import torch
except ImportError:
    raise SystemExit("no torch")
if not torch.cuda.is_available():
    raise SystemExit("torch sees no cuda")
for mb in (4, 100):
    n = mb << 20
    for kind in ("pageable", "pinned"):
        x = torch.ones(n, dtype=torch.uint8)
        if kind == "pinned":
            x = x.pin_memory()
        torch.cuda.synchronize()
        best = 1e9
        for _ in range(3):
            t = time.perf_counter()
            y = x.cuda(non_blocking=True)
            torch.cuda.synchronize()
            best = min(best, time.perf_counter() - t)
        t = time.perf_counter()
        y.cpu()
        torch.cuda.synchronize()
        back = time.perf_counter() - t
        print(f"torch H2D {kind:8} {mb:>4} MB  {best * 1e3:8.2f} ms  {n / best / 1e9:6.2f} GB/s"
              f"   D2H {back * 1e3:8.2f} ms  {n / back / 1e9:6.2f} GB/s", flush=True)
