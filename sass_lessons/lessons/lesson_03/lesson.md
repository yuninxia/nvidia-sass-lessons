# 03_shared_mem – Shared memory staging

**What to look for in SASS:**


- `LDS` / `STS` pairs for shared memory moves
- A barrier in SASS corresponding to `__syncthreads` (e.g., `BAR.SYNC`)


**Build**
```bash
SASS_ARCH=sm_80 bash ../../scripts/build.sh
# then open: disasm/03_shared_mem.sm_80.O0.sass
```

**Notes**
- SASS is target‑dependent. Your listing may differ across `sm_*` targets and CUDA versions.
