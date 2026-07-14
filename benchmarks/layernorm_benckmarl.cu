#include <stdio.h>
#include <cuda_runtime.h>

#include "cuda_utils.cuh"
#include "other_kernels.cuh"

void correctness_check(int rows, int cols){

    float* h_x = (float*)malloc(rows*cols*sizeof(float));
    float* h_y = (float*)malloc(rows*cols*sizeof(float));
    float* h_gamma = (float*)malloc(cols*sizeof(float));
    float* h_beta = (float*)malloc(cols*sizeof(float));
    
    init_matrix(h_x, rows*cols, 1);
    init_matrix(h_gamma, cols, 1);
    init_matrix(h_beta, cols, 1);

    float* d_x = NULL, *d_y = NULL, *d_gamma = NULL, *d_beta = NULL;
    cudaMalloc((void**)&d_x, rows*cols*sizeof(float));
    cudaMalloc((void**)&d_y, rows*cols*sizeof(float));
    cudaMalloc((void**)&d_gamma, cols*sizeof(float));
    cudaMalloc((void**)&d_beta, cols*sizeof(float));

    cudaMemcpy(d_x, h_x, rows*cols*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, cols*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, cols*sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize = 128;
    dim3 gridSize = rows;
    layernorm_kernel<128><<<gridSize, blockSize>>>
            (d_x, d_gamma, d_beta, d_y, rows, cols);

    CHECK_CUDA(cudaGetLastError());
    cudaDeviceSynchronize();

    cudaMemcpy(h_y, d_y, rows*cols*sizeof(float), cudaMemcpyDeviceToHost);

    float* h_ref = (float*)malloc(rows*cols*sizeof(float));
    layer_norm_cpu(h_x, h_gamma, h_beta, h_ref, rows, cols);

    if (check_result(h_ref, h_y, rows*cols)) {
        printf("Correctness check passed!\n");
    } else {
        printf("Correctness check failed!\n");
    }

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_gamma); cudaFree(d_beta);
    free(h_x); free(h_y); free(h_gamma); free(h_beta); free(h_ref);
}

template<int BLOCK_SIZE>
void benchmark(int rows, int cols){

    float* h_x = (float*)malloc(rows*cols*sizeof(float));
    float* h_y = (float*)malloc(rows*cols*sizeof(float));
    float* h_gamma = (float*)malloc(cols*sizeof(float));
    float* h_beta = (float*)malloc(cols*sizeof(float));
    
    init_matrix(h_x, rows*cols, 1);
    init_matrix(h_gamma, cols, 1);
    init_matrix(h_beta, cols, 1);

    float* d_x = NULL, *d_y = NULL, *d_gamma = NULL, *d_beta = NULL;
    cudaMalloc((void**)&d_x, rows*cols*sizeof(float));
    cudaMalloc((void**)&d_y, rows*cols*sizeof(float));
    cudaMalloc((void**)&d_gamma, cols*sizeof(float));
    cudaMalloc((void**)&d_beta, cols*sizeof(float));

    cudaMemcpy(d_x, h_x, rows*cols*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, cols*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, cols*sizeof(float), cudaMemcpyHostToDevice);

    dim3 gridSize = rows;
    for(int i=0; i<10; i++)
        layernorm<BLOCK_SIZE><<<gridSize, BLOCK_SIZE>>>
                (d_x, d_gamma, d_beta, d_y, rows, cols);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    float time_cost = 0;

    cudaEventRecord(start);
    for(int i=0; i<500; i++)
        layernorm<BLOCK_SIZE><<<gridSize, BLOCK_SIZE>>>
                (d_x, d_gamma, d_beta, d_y, rows, cols);

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time_cost,start,end);

    printf("Rows = %d, Cols = %d, layernorm time cost: %f\n",
        rows, cols, time_cost/500);

    cudaEventDestroy(start); cudaEventDestroy(end);
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_gamma); cudaFree(d_beta);
    free(h_x); free(h_y); free(h_gamma); free(h_beta);
}

int main(){

    correctness_check(10,100);

    int benchmark_cols[] = {
        128,
        256,
        512,
        1024,
        2048,
        4096
    };

    for(auto col: benchmark_cols){
        benchmark<128>(4096, col);
    }

    return 0;
}