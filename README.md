# NVIDIA SASS Lessons

**Goal:** A high‑quality, hands‑on set of lessons that teach NVIDIA GPU *SASS* (machine code) concepts by compiling small CUDA kernels, then inspecting their disassembly with `nvdisasm` (and/or `cuobjdump`).

> **What you’ll actually do**: Write a tiny CUDA kernel → compile to a `.cubin` for a target architecture (e.g. `sm_80`) → disassemble to SASS → read and reason about the instruction stream (predicates, control flow, memory ops, etc.).

---

## Quick start

### Option 1: Local CUDA Toolkit (no GPU required to compile)
1. Install the CUDA Toolkit (12.x or newer) so that `nvcc`, `ptxas`, `nvdisasm`, and `cuobjdump` are on your `PATH`.
2. Clone this repo.
3. Build and disassemble all lessons (default arch `sm_80`):
   ```bash
   SASS_ARCH=sm_80 bash scripts/build.sh
   ```
4. Open the generated files in `disasm/` and compare `O0` vs `O3` outputs.

### Option 2: Docker (fully reproducible on any host)
```bash
bash scripts/docker-build.sh               # uses nvidia/cuda:devel image
# artifacts appear in ./disasm
```

> **Note**: No GPU is needed to *compile* kernels and produce SASS. A GPU is only needed to run kernels.

---

## Lessons overview

Each `lesson_XX/` folder contains a minimal `kernel.cu` and a short `lesson.md` to guide what to look for in the SASS listing.

1. **01_add** – Warm‑up vector add; see basic `LDG/STG` and arithmetic.
2. **02_memory** – Coalesced vs strided accesses; observe load/store patterns.
3. **03_shared_mem** – Shared memory tiling; note `LDS/STS`, barriers.
4. **04_predication** – `if`/`else` turns into predicate registers and guarded ops.
5. **05_control_flow** – Loops and branches (`SSY`, `BRA`, divergence).
6. **06_shuffle** – Warp shuffles (`__shfl_sync`) and register moves.
7. **07_atomics** – Global atomics (`ATOMG`/`RED` forms depending on arch).
8. **08_barrier** – `__syncthreads` and memory ordering at block scope.
9. **09_special_regs** – Read `%laneid` via inline PTX; map to SASS usage.

> Heads‑up: SASS is **architecture‑specific**. Use `SASS_ARCH=sm_75|sm_80|sm_86|sm_89|sm_90…`. Outputs will differ by arch and by optimization level.

---

## Build scripts

- `scripts/build.sh` compiles every lesson twice (with `-O0` and `-O3`) for `${SASS_ARCH:-sm_80}`, emitting:
  - `build/<lesson>.<arch>.O*.cubin`
  - `disasm/<lesson>.<arch>.O*.sass`

- `scripts/clean.sh` removes build artifacts.
- `scripts/docker-build.sh` runs the build inside an NVIDIA CUDA *devel* container (no GPU required).

The GitHub Actions workflow mirrors this setup and uploads the `disasm/` directory as an artifact on every push/PR.

---

## Tools you’ll use

- **nvdisasm** – disassemble `.cubin` files to SASS and print line info.
- **cuobjdump** – extract PTX and SASS from host binaries (optional).
- **nvcc/ptxas** – compile CUDA C++ to cubin.

> We also link to community **unofficial SASS assemblers** in `docs/links.md` if you want to go beyond disassembly. Not required for these lessons.

---

## Contributing

PRs adding new bite‑sized lessons are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for style and review guidelines, and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
