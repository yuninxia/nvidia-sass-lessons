# Minimal image that has nvcc + nvdisasm without requiring a host GPU
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Nothing else needed; nvcc/nvdisasm are in PATH
WORKDIR /workspace
