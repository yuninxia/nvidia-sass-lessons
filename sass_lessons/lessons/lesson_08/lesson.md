# 08_barrier – Block synchronization

**What to look for in SASS:**


- Barrier opcode for `__syncthreads`
- Shared memory loads from neighboring indices


**Build**
```bash
SASS_ARCH=sm_80 bash ../../scripts/build.sh
# then open: disasm/08_barrier.sm_80.O0.sass
```

**Notes**
- SASS is target‑dependent. Your listing may differ across `sm_*` targets and CUDA versions.
