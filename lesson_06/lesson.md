# 06_shuffle – Warp shuffles

**What to look for in SASS:**


- Inspect generated shuffle instructions (SASS form varies by arch)
- See how predicates guard the add when `lane < offset`


**Build**
```bash
SASS_ARCH=sm_80 bash ../../scripts/build.sh
# then open: disasm/06_shuffle.sm_80.O0.sass
```

**Notes**
- SASS is target‑dependent. Your listing may differ across `sm_*` targets and CUDA versions.
