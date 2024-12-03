#pragma once

#include <cuda_runtime.h>

__global__ void initData(float *data, int rank, int size);
