#include <cuda_runtime.h>

extern "C" __global__
void sum_loop(const float* __restrict__ x, float* __restrict__ out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0.0f;
    for (int k = i; k < n; k += blockDim.x * gridDim.x) {
        acc += x[k];
    }
    if (i < n) out[i] = acc;
}
