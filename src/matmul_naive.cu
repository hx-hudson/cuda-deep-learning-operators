#include <stdio.h>
#include <cuda_runtime.h>
#include "cuda_utils.cuh"
#include "matmul_kernels.cuh"

__global__ void matmul_naive(
    int* a, int* b, int* c, int N, int M, int K
){
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if(row < N && col < M){
        int sum = 0;

        for(int k = 0; k < K; k++){
            sum += a[row*K+k] * b[k*M+col];
        }

        c[row*M + col] = sum;
    }
    
}

// int main(){

//     int N = 1024, M = 1024, K = 1024;
//     int* h_a = (int*)malloc(N * K * sizeof(int));
//     int* h_b = (int*)malloc(K * M * sizeof(int));
//     int* h_c = (int*)malloc(N * M * sizeof(int));

//     init_matrix(h_a, N, K);
//     init_matrix(h_b, K, M);

//     int* d_a = NULL, *d_b = NULL, *d_c = NULL;
//     CHECK_CUDA(cudaMalloc((void**)&d_a, N*K*4));
//     CHECK_CUDA(cudaMalloc((void**)&d_b, K*M*4));
//     CHECK_CUDA(cudaMalloc((void**)&d_c, N*M*4));

//     CHECK_CUDA(cudaMemcpy(d_a, h_a, N*K*4, cudaMemcpyHostToDevice));
//     CHECK_CUDA(cudaMemcpy(d_b, h_b, K*M*4, cudaMemcpyHostToDevice));

//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);

//     dim3 blocksize(16,16);
//     dim3 gridsize(
//         (M+blocksize.x-1)/blocksize.x,(N+blocksize.y-1)/blocksize.y
//     );

//     cudaEventRecord(start);

//     matmul_naive<<<gridsize, blocksize>>>(d_a,d_b,d_c,N,M,K);
//     CHECK_CUDA(cudaGetLastError());

//     cudaEventRecord(stop);
//     cudaEventSynchronize(stop);

//     float ms = 0.0f;
//     CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
//     double gflops = (2.0 * M * N * K) / (ms / 1000.0) / 1e9;

//     printf("Matrix size: %d x %d x %d\n", N, M, K);
//     printf("Kernel time: %.6f ms\n", ms);
//     printf("Performance: %.3f GOPS\n", gflops);

//     CHECK_CUDA(cudaMemcpy(h_c, d_c, N*M*4, cudaMemcpyDeviceToHost));

//     // check correctness
//     int* h_ref = (int*)malloc(N*M*4);
//     matmul_cpu(h_a, h_b, h_ref, N, M, K);

//     if (check_result(h_ref, h_c, N * M)) {
//         printf("Correctness check passed!\n");
//     } else {
//         printf("Correctness check failed!\n");
//     }

//     free(h_a);free(h_b);free(h_c);free(h_ref);
//     cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

//     return 0;
// }