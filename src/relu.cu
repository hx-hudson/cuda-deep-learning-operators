#include <stdio.h>
#include <cuda_runtime.h>

__global__ void relu(float* x, float* y, int N){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(idx < N){
        y[idx] = fmaxf(x[idx], 0.0f);
    }
}