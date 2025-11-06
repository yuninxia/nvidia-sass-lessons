#include <cuda_runtime.h>

extern "C" __global__
void tile_copy(const float* __restrict__ src,
               float* __restrict__ dst,
               int n)
{
    extern __shared__ float tile[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    if (i < n) {
        tile[tid] = src[i];
        __syncthreads();
        dst[i] = tile[tid];
    }
}
