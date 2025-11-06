#include <cuda_runtime.h>

extern "C" __global__
void threshold(float* __restrict__ x, int n, float t)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        if (v < t) {
            x[i] = 0.0f;
        } else {
            x[i] = v;
        }
    }
}
