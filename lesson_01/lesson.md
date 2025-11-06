# 01_add – Vector add

**What to look for in SASS:**


- `LDG.E`/`LDG` and `STG` patterns for global loads/stores
- A single‐precision add (`FADD`, `FFMA` if fused)  
- Predicated store guarded by a bounds check predicate (e.g., `@P0`)


**Build**
```bash
SASS_ARCH=sm_80 bash ../../scripts/build.sh
# then open: disasm/01_add.sm_80.O0.sass
```

**Notes**
- SASS is target‑dependent. Your listing may differ across `sm_*` targets and CUDA versions.
