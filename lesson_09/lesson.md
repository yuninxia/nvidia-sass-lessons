# 09_special_regs – Special registers

**What to look for in SASS:**


- Inline PTX `mov.u32 %laneid` and how it appears in SASS
- Observe how special registers flow into integer ALU ops


**Build**
```bash
SASS_ARCH=sm_80 bash ../../scripts/build.sh
# then open: disasm/09_special_regs.sm_80.O0.sass
```

**Notes**
- SASS is target‑dependent. Your listing may differ across `sm_*` targets and CUDA versions.
