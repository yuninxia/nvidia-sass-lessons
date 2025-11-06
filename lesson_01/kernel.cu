#include <cuda_runtime.h>
#include <cstdint>

extern "C" __global__
void vec_add(const float* __restrict__ a,
             const float* __restrict__ b,
             float* __restrict__ c,
             int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
