#include <stdio.h>
#include <cuda_runtime.h>
#include "cuda_utils.cuh"
#include "matmul_kernels.cuh"

__global__ void matmul_naive(
    float* a, float* b, float* c, int N, int M, int K
){
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if(row < N && col < M){
        float sum = 0;

        for(int k = 0; k < K; k++){
            sum += a[row*K+k] * b[k*M+col];
        }

        c[row*M + col] = sum;
    }
    
}
