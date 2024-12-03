#include <cuda_runtime.h>
// Simple kernel to initialize data
__global__ void initData(float *data, int rank, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        data[idx] = rank + 0.1f * idx;
    }
}
