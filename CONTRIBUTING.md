# Contributing Guide

Thanks for your interest in contributing! This project has three pillars:

1. **Tiny, focused lessons** – one concept per folder; minimal code; great comments.
2. **Reproducibility** – every lesson should compile to a `.cubin` and produce a readable SASS listing.
3. **Teach by diff** – where possible, show how `-O0` vs `-O3` and/or different access patterns affect SASS.

## How to add a lesson

1. Create `lessons/NN_<slug>/kernel.cu` and `lessons/NN_<slug>/lesson.md`.
2. Keep kernels small. Prefer single‑purpose kernels with clear data hazards and a few loads/stores.
3. Add any lesson‑specific build notes to the `lesson.md`.
4. Run `SASS_ARCH=sm_80 bash scripts/build.sh` locally (or `scripts/docker-build.sh`).
5. Open a PR with **screenshots** or short snippets that highlight the important SASS patterns.
6. Avoid copyrighted content and do not copy from proprietary documentation.

## Coding style

- C++17, `.cu` files only.
- Avoid complicated host scaffolding – use simple host stubs if needed, but most lessons don't run kernels.
- Keep comments short, accurate, and architecture‑agnostic when possible.
