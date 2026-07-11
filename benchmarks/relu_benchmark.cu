#include <stdio.h>
#include <cuda_runtime.h>

#include "other_kernels.cuh"
#include "cuda_utils.cuh"

void correctness_check(int N){

    float* h_a = (float*)malloc(N*sizeof(float));
    float* h_b = (float*)malloc(N*sizeof(float));
    
    init_matrix(h_a, N, 1);

    float* d_a = NULL, *d_b = NULL;
    cudaMalloc((void**)&d_a, N*sizeof(float));
    cudaMalloc((void**)&d_b, N*sizeof(float));

    cudaMemcpy(d_a, h_a, N*sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize = 256;
    dim3 gridSize = (N + blockSize.x - 1) / blockSize.x;
    relu<<<gridSize, blockSize>>>(d_a, d_b, N);

    CHECK_CUDA(cudaGetLastError());
    cudaDeviceSynchronize();

    cudaMemcpy(h_b, d_b, N*sizeof(float), cudaMemcpyDeviceToHost);

    float* h_ref = (float*)malloc(N*sizeof(float));
    for(int i = 0; i < N; i++){
        h_ref[i] = std::max<float>(0.0f, h_a[i]);
    }

    if (check_result(h_ref, h_b, N)) {
        printf("Correctness check passed!\n");
    } else {
        printf("Correctness check failed!\n");
    }
}

void benchmark(int N){

    float* h_a = (float*)malloc(N*sizeof(float));
    float* h_b = (float*)malloc(N*sizeof(float));
    
    init_matrix(h_a, N, 1);

    float* d_a = NULL, *d_b = NULL;
    cudaMalloc((void**)&d_a, N*sizeof(float));
    cudaMalloc((void**)&d_b, N*sizeof(float));

    cudaMemcpy(d_a, h_a, N*sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize = 256;
    dim3 gridSize = (N + blockSize.x - 1) / blockSize.x;
    
    // warm up
    for(int i=0; i<10; i++)
        relu<<<gridSize, blockSize>>>(d_a, d_b, N);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    float time_cost = 0;

    cudaEventRecord(start);

    for(int i=0; i<500; i++)
        relu<<<gridSize, blockSize>>>(d_a, d_b, N);

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time_cost,start,end);

    printf("relu time cost: %f\n", time_cost/500);

    cudaEventDestroy(start); cudaEventDestroy(end);

    cudaFree(d_a); cudaFree(d_b);

    free(h_a); free(h_b);
}

int main(){

    correctness_check(1000);

    for (int N : {
        1 << 20,
        1 << 22,
        1 << 24,
        1 << 26
    }) {
        benchmark(N);
    }

    return 0;
}