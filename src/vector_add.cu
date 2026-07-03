#include <stdio.h>
#include <cuda_runtime.h>
#include "cuda_utils.cuh"

__global__ void vector_add(int* a, int* b, int* c, int N){
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if(id < N)
        c[id] = a[id] + b[id];
}

int main(){
    int N = 100000;
    size_t bytes = N * sizeof(int);

    // cpu
    int* h_a = (int*)malloc(bytes);
    int* h_b = (int*)malloc(bytes);
    int* h_c = (int*)malloc(bytes);

    for(int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // gpu
    int* d_a = NULL, *d_b = NULL, *d_c = NULL;
    CHECK_CUDA(cudaMalloc((void**)&d_a, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_b, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_c, bytes));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + 256 - 1) / 256;

    CHECK_CUDA(cudaEventRecord(start));
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CHECK_CUDA(cudaGetLastError());
    
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    double bandwidth = (3.0 * N * sizeof(int)) / (ms / 1000.0) / 1e9;
    printf("N = %d, bandwidth = %lf\n", N, bandwidth);

    CHECK_CUDA(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            printf("Wrong at %d\n", i);
            return 1;
        }
    }

    printf("PASSED\n");

    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    
    return 0;
}