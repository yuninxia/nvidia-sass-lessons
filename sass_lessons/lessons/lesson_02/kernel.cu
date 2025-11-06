#include <cuda_runtime.h>

extern "C" __global__
void stride_copy(const float* __restrict__ src,
                 float* __restrict__ dst,
                 int n, int stride)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = i * stride;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}
