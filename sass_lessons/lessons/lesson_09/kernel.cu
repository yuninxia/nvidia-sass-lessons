#include <cuda_runtime.h>

__device__ __forceinline__ unsigned lane_id() {
    unsigned id;
    asm volatile ("mov.u32 %0, %%laneid;" : "=r"(id));
    return id;
}

extern "C" __global__
void write_laneid(unsigned *out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = lane_id();
}
