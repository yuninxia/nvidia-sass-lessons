#!/usr/bin/env bash
set -euo pipefail

ARCH="${SASS_ARCH:-sm_80}"
OPT_LEVELS="${SASS_OPT_LEVELS:-O0 O3}"

echo "[build] Using ARCH=${ARCH} OPTS=${OPT_LEVELS}"

command -v nvcc >/dev/null 2>&1 || { echo "nvcc not found on PATH"; exit 1; }
command -v nvdisasm >/dev/null 2>&1 || { echo "nvdisasm not found on PATH"; exit 1; }

mkdir -p build disasm

for lesson in lesson_* lessons/*; do
  [ -d "$lesson" ] || continue
  base="$(basename "$lesson")"
  src="$lesson/kernel.cu"

  if [ ! -f "$src" ]; then
    echo "[skip] $base has no kernel.cu"
    continue
  fi

  for opt in $OPT_LEVELS; do
    cubin="build/${base}.${ARCH}.${opt}.cubin"
    sass="disasm/${base}.${ARCH}.${opt}.sass"
    echo "[nvcc] $src -> $cubin ($opt)"
    nvcc -std=c++17 -lineinfo -arch="${ARCH}" -Xptxas -${opt} -cubin "$src" -o "$cubin"
    echo "[nvdisasm] $cubin -> $sass"
    nvdisasm --print-line-info "$cubin" > "$sass"
  done
done

echo "[done] See disasm/ for SASS listings."
