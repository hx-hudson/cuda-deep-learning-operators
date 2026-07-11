#include <stdio.h>
#include <cuda_runtime.h>

#include "other_kernels.cuh"
#include "cuda_utils.cuh"

void correctness_check(int rows, int cols){

    float* h_x = (float*)malloc(rows*cols*sizeof(float));
    float* h_y = (float*)malloc(rows*cols*sizeof(float));
    
    init_matrix(h_x, rows*cols, 1);

    float* d_x = NULL, *d_y = NULL;
    cudaMalloc((void**)&d_x, rows*cols*sizeof(float));
    cudaMalloc((void**)&d_y, rows*cols*sizeof(float));

    cudaMemcpy(d_x, h_x, rows*cols*sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize = 128;
    dim3 gridSize = rows;
    softmax<<<gridSize, blockSize>>>(d_x, d_y, rows, cols);

    CHECK_CUDA(cudaGetLastError());
    cudaDeviceSynchronize();

    cudaMemcpy(h_y, d_y, rows*cols*sizeof(float), cudaMemcpyDeviceToHost);

    float* h_ref = (float*)malloc(rows*cols*sizeof(float));
    softmax_cpu(h_x, h_ref, rows, cols);

    if (check_result(h_ref, h_y, rows*cols)) {
        printf("Correctness check passed!\n");
    } else {
        printf("Correctness check failed!\n");
    }
}

void correctness_check_warp(int rows, int cols){

    float* h_x = (float*)malloc(rows*cols*sizeof(float));
    float* h_y = (float*)malloc(rows*cols*sizeof(float));
    
    init_matrix(h_x, rows*cols, 1);

    float* d_x = NULL, *d_y = NULL;
    cudaMalloc((void**)&d_x, rows*cols*sizeof(float));
    cudaMalloc((void**)&d_y, rows*cols*sizeof(float));

    cudaMemcpy(d_x, h_x, rows*cols*sizeof(float), cudaMemcpyHostToDevice);

    dim3 gridSize = rows;
    softmax_warp_shuffle<128>
        <<<gridSize, 128>>>(d_x, d_y, rows, cols);

    CHECK_CUDA(cudaGetLastError());
    cudaDeviceSynchronize();

    cudaMemcpy(h_y, d_y, rows*cols*sizeof(float), cudaMemcpyDeviceToHost);

    float* h_ref = (float*)malloc(rows*cols*sizeof(float));
    softmax_cpu(h_x, h_ref, rows, cols);

    if (check_result(h_ref, h_y, rows*cols)) {
        printf("Correctness check passed!\n");
    } else {
        printf("Correctness check failed!\n");
    }
}

void benchmark(int rows, int cols){

    float* h_a = (float*)malloc(rows*cols*sizeof(float));
    float* h_b = (float*)malloc(rows*cols*sizeof(float));
    
    init_matrix(h_a, rows*cols, 1);

    float* d_a = NULL, *d_b = NULL;
    cudaMalloc((void**)&d_a, rows*cols*sizeof(float));
    cudaMalloc((void**)&d_b, rows*cols*sizeof(float));

    cudaMemcpy(d_a, h_a, rows*cols*sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize = 128;
    dim3 gridSize = rows;
    
    // warm up
    for(int i=0; i<10; i++)
        softmax<<<gridSize, blockSize>>>(d_a, d_b, rows, cols);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    float time_cost = 0;

    cudaEventRecord(start);

    for(int i=0; i<500; i++)
        softmax<<<gridSize, blockSize>>>(d_a, d_b, rows, cols);

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time_cost,start,end);

    printf("Rows = %d, Cols = %d, softmax time cost: %f\n",
        rows, cols, time_cost/500);

    cudaEventDestroy(start); cudaEventDestroy(end);

    cudaFree(d_a); cudaFree(d_b);

    free(h_a); free(h_b);
}

template <int BLOCK_SIZE>
void benchmark_warp(int rows, int cols){

    float* h_a = (float*)malloc(rows*cols*sizeof(float));
    float* h_b = (float*)malloc(rows*cols*sizeof(float));
    
    init_matrix(h_a, rows*cols, 1);

    float* d_a = NULL, *d_b = NULL;
    cudaMalloc((void**)&d_a, rows*cols*sizeof(float));
    cudaMalloc((void**)&d_b, rows*cols*sizeof(float));

    cudaMemcpy(d_a, h_a, rows*cols*sizeof(float), cudaMemcpyHostToDevice);

    dim3 gridSize = rows;
    
    // warm up
    for(int i=0; i<10; i++)
        softmax_warp_shuffle<BLOCK_SIZE>
            <<<gridSize, BLOCK_SIZE>>>(d_a, d_b, rows, cols);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    float time_cost = 0;

    cudaEventRecord(start);

    for(int i=0; i<500; i++)
        softmax_warp_shuffle<BLOCK_SIZE>
            <<<gridSize, BLOCK_SIZE>>>(d_a, d_b, rows, cols);

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time_cost,start,end);

    printf("Rows = %d, Cols = %d, softmax_warp time cost: %f\n",
        rows, cols, time_cost/500);

    cudaEventDestroy(start); cudaEventDestroy(end);

    cudaFree(d_a); cudaFree(d_b);

    free(h_a); free(h_b);
}

int main(){

    correctness_check(10, 100);
    correctness_check_warp(10, 100);

    int benchmark_cols[] = {
        32,
        64,
        100,
        128,
        256,
        512,
        1024,
        2048,
        4096
    };

    for(auto cols: benchmark_cols){
        benchmark(4096, cols);
        benchmark_warp<128>(4096, cols);
    }

    return 0;
}