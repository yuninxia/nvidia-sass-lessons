# 02_memory – Coalescing vs stride

**What to look for in SASS:**


- Observe how strided indexing affects the sequence of `LDG`/`STG` instructions.
- With `-O3`, some address arithmetic may fold into `IADD3` / `LEA` forms.


**Build**
```bash
SASS_ARCH=sm_80 bash ../../scripts/build.sh
# then open: disasm/02_memory.sm_80.O0.sass
```

**Notes**
- SASS is target‑dependent. Your listing may differ across `sm_*` targets and CUDA versions.
