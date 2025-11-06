#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__
void atomic_histogram(const uint8_t* __restrict__ input,
                      unsigned int* __restrict__ hist,
                      int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned int b = input[i];
        atomicAdd(&hist[b], 1u);
    }
}
