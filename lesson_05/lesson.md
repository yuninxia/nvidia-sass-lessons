# 05_control_flow – Loops and branches

**What to look for in SASS:**


- Structured control flow markers (e.g., `SSY`, `BRA`, `SYNC`)
- Induction variable math (`IADD3`, `ISCADD`) and loop back‐edges


**Build**
```bash
SASS_ARCH=sm_80 bash ../../scripts/build.sh
# then open: disasm/05_control_flow.sm_80.O0.sass
```

**Notes**
- SASS is target‑dependent. Your listing may differ across `sm_*` targets and CUDA versions.
