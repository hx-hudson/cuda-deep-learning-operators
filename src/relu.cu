
#include <cuda_runtime.h>

#include <cstdint>

__global__ void relu(const float* x, float* y, int64_t N){

    int64_t idx = 
        static_cast<int64_t>(blockDim.x) * blockIdx.x + threadIdx.x;

    if(idx < N){
        y[idx] = fmaxf(x[idx], 0.0f);
    }
}

