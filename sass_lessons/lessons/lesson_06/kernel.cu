#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__
void warp_prefix_sum(int *x)
{
    int lane = threadIdx.x & 31;
    unsigned mask = __activemask();
    int v = x[threadIdx.x];

    // Inclusive prefix sum via shuffles
    for (int offset = 1; offset < 32; offset <<= 1) {
        int n = __shfl_up_sync(mask, v, offset);
        if (lane >= offset) v += n;
    }
    x[threadIdx.x] = v;
}
