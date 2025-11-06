#!/usr/bin/env bash
set -euo pipefail

ARCH="${SASS_ARCH:-sm_80}"
IMAGE="${CUDA_IMAGE:-nvidia/cuda:12.4.1-devel-ubuntu22.04}"

echo "[docker] Using image ${IMAGE} ARCH=${ARCH}"

docker run --rm -v "$PWD":/workspace -w /workspace "${IMAGE}" bash -lc "
  set -e
  echo '[toolchain]'
  nvcc --version || { echo 'nvcc not found in image'; exit 1; }
  which nvdisasm || { echo 'nvdisasm not found in image'; exit 1; }
  SASS_ARCH=${ARCH} bash scripts/build.sh
"
