# 07_atomics – Global atomics

**What to look for in SASS:**


- Look for global atomic update forms (mnemonics differ by arch)
- Note serialization effects around the atomic instruction


**Build**
```bash
SASS_ARCH=sm_80 bash ../../scripts/build.sh
# then open: disasm/07_atomics.sm_80.O0.sass
```

**Notes**
- SASS is target‑dependent. Your listing may differ across `sm_*` targets and CUDA versions.
