# 04_predication – Predicated execution

**What to look for in SASS:**


- Look for predicate registers (`P0`, `P1`, …) and guarded ops (`@P0`)
- Compare `-O0` vs `-O3` to see whether branches become predicates


**Build**
```bash
SASS_ARCH=sm_80 bash ../../scripts/build.sh
# then open: disasm/04_predication.sm_80.O0.sass
```

**Notes**
- SASS is target‑dependent. Your listing may differ across `sm_*` targets and CUDA versions.
