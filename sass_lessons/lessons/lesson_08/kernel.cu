#include <cuda_runtime.h>

extern "C" __global__
void neighbor_exchange(float* __restrict__ x, int n)
{
    extern __shared__ float tile[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    if (i >= n) return;

    tile[tid] = x[i];
    __syncthreads();

    float left = (tid > 0) ? tile[tid - 1] : tile[tid];
    x[i] = 0.5f * (left + tile[tid]);
}
