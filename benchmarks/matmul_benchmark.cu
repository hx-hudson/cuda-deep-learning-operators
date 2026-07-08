#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "matmul_kernels.cuh"
#include "cuda_utils.cuh"

int tile_size = 16;

void calculate_benchmark(
    float naive_cost,
    float tiled_cost,
    float tiled_const_cost,
    float cublas_cost,
    int N, int M, int K
){
    float naive_ms = naive_cost / 500.0f;
    float tiled_ms = tiled_cost / 500.0f;
    float tiled_const_ms = tiled_const_cost / 500.0f;
    float cublas_ms = cublas_cost / 500.0f;

    // Matrix multiplication performs approximately 2 * N * M * K FLOPs
    double operations = 2.0 * N * M * K;

    double naive_gflops =
        operations / (naive_ms * 1e6);

    double tiled_gflops =
        operations / (tiled_ms * 1e6);

    double tiled_const_gflops =
        operations / (tiled_const_ms * 1e6);

    double cublas_gflops =
        operations / (cublas_ms * 1e6);

    printf("\nMatrix size: %d x %d x %d\n\n", N, M, K);

    printf("%-20s %-12s %-12s %-10s\n",
        "Kernel", "Time(ms)", "GFLOPS", "Speedup");

    printf("%-20s %-12.6f %-12.2f %-10.2fx\n",
        "Naive",
        naive_ms,
        naive_gflops,
        1.0);

    printf("%-20s %-12.6f %-12.2f %-10.2fx\n",
        "Tiled",
        tiled_ms,
        tiled_gflops,
        naive_ms / tiled_ms);

    printf("%-20s %-12.6f %-12.2f %-10.2fx\n",
        "Tiled Constant",
        tiled_const_ms,
        tiled_const_gflops,
        naive_ms / tiled_const_ms);

    printf("%-20s %-12.6f %-12.2f %-10.2fx\n",
        "cuBLAS",
        cublas_ms,
        cublas_gflops,
        naive_ms / cublas_ms);
}

void correctness_check(int N, int M, int K){
    float* h_a = (float*)malloc(N*K*sizeof(float));
    float* h_b = (float*)malloc(K*M*sizeof(float));
    float* h_c = (float*)malloc(N*M*sizeof(float));

    init_matrix(h_a, N, K);
    init_matrix(h_b, K, M);

    float* h_ref = (float*)malloc(N*M*sizeof(float));
    matmul_cpu(h_a, h_b, h_ref, N, M, K);

    float* d_a = NULL, *d_b = NULL, *d_c = NULL;
    cudaMalloc((void**)&d_a, N*K*sizeof(float));
    cudaMalloc((void**)&d_b, K*M*sizeof(float));
    cudaMalloc((void**)&d_c, N*M*sizeof(float));

    cudaMemcpy(d_a,h_a,N*K*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,K*M*sizeof(float),cudaMemcpyHostToDevice);

    dim3 blocksize(16,16);
    dim3 gridsize(
        (M+blocksize.x-1)/blocksize.x,(N+blocksize.y-1)/blocksize.y
    );
    matmul_naive<<<gridsize, blocksize>>>(d_a,d_b,d_c,N,M,K);

    cudaMemcpy(h_c, d_c, N*M*sizeof(float), cudaMemcpyDeviceToHost);
    if (check_result(h_ref, h_c, N * M)) {
        printf("Correctness check passed!\n");
    } else {
        printf("Correctness check failed!\n");
    }

    dim3 tile_blocksize(tile_size,tile_size);
    dim3 tile_gridsize(
        (M+tile_size-1)/tile_size,(N+tile_size-1)/tile_size
    );
    size_t shared_mem_size = 2*tile_size*tile_size*sizeof(float);
    matmul_tiled<<<tile_gridsize,tile_blocksize,shared_mem_size>>>
                (d_a,d_b,d_c,N,M,K,tile_size);
    
    cudaMemcpy(h_c, d_c, N*M*sizeof(float), cudaMemcpyDeviceToHost);
    if (check_result(h_ref, h_c, N * M)) {
        printf("Correctness check passed!\n");
    } else {
        printf("Correctness check failed!\n");
    }

    matmul_tiled_constant<16><<<tile_gridsize,tile_blocksize>>>
                                (d_a,d_b,d_c,N,M,K);
        
    cudaMemcpy(h_c, d_c, N*M*sizeof(float), cudaMemcpyDeviceToHost);
    if (check_result(h_ref, h_c, N * M)) {
        printf("Correctness check passed!\n");
    } else {
        printf("Correctness check failed!\n");
    }

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f, beta = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
            M, N, K, &alpha, d_b, M, d_a, K, &beta, d_c, M);

    cudaMemcpy(h_c, d_c, N*M*sizeof(float), cudaMemcpyDeviceToHost);
    if (check_result(h_ref, h_c, N * M)) {
        printf("Correctness check passed!\n");
    } else {
        printf("Correctness check failed!\n");
    }

    free(h_a); free(h_b); free(h_c); free(h_ref);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cublasDestroy(handle);
}

void benchmark(int N, int M, int K){

    float* h_a = (float*)malloc(N*K*sizeof(float));
    float* h_b = (float*)malloc(K*M*sizeof(float));

    init_matrix(h_a, N, K);
    init_matrix(h_b, K, M);

    float* d_a = NULL, *d_b = NULL, *d_c = NULL;
    cudaMalloc((void**)&d_a, N*K*sizeof(float));
    cudaMalloc((void**)&d_b, K*M*sizeof(float));
    cudaMalloc((void**)&d_c, N*M*sizeof(float));

    cudaMemcpy(d_a,h_a,N*K*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,K*M*sizeof(float),cudaMemcpyHostToDevice);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    float naive_cost = 0, tiled_cost = 0,
        tiled_const_cost = 0, cublas_cost = 0;

    dim3 blocksize(16,16);
    dim3 gridsize(
        (M+blocksize.x-1)/blocksize.x,(N+blocksize.y-1)/blocksize.y
    );

    // warm up
    for(int i=0;i<10;i++) 
        matmul_naive<<<gridsize, blocksize>>>(d_a,d_b,d_c,N,M,K);

    CHECK_CUDA(cudaGetLastError());


    cudaEventRecord(start);

    for(int i=0;i<500;i++)
        matmul_naive<<<gridsize, blocksize>>>(d_a,d_b,d_c,N,M,K);

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&naive_cost,start,end);

    dim3 tile_blocksize(tile_size,tile_size);
    dim3 tile_gridsize(
        (M+tile_size-1)/tile_size,(N+tile_size-1)/tile_size
    );
    size_t shared_mem_size = 2 * tile_size * tile_size * sizeof(float);
    
    // warm up
    for(int i=0;i<10;i++)
        matmul_tiled<<<tile_gridsize,tile_blocksize,shared_mem_size>>>
        (d_a,d_b,d_c,N,M,K,tile_size);

    CHECK_CUDA(cudaGetLastError());


    cudaEventRecord(start);

    for(int i=0;i<500;i++)
        matmul_tiled<<<tile_gridsize,tile_blocksize,shared_mem_size>>>
        (d_a,d_b,d_c,N,M,K,tile_size);

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&tiled_cost,start,end);


    for(int i=0;i<10;i++)
        matmul_tiled_constant<16><<<tile_gridsize,tile_blocksize>>>
        (d_a,d_b,d_c,N,M,K);

    
    cudaEventRecord(start);
    for(int i=0;i<500;i++)
        matmul_tiled_constant<16><<<tile_gridsize,tile_blocksize>>>
        (d_a,d_b,d_c,N,M,K);

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&tiled_const_cost,start,end);

    // cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f, beta = 0.0f;

    // warm up
    for(int i = 0; i < 10; i++)
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
            M, N, K, &alpha, d_b, M, d_a, K, &beta, d_c, M);

    cudaEventRecord(start);

    for(int i = 0; i < 500; i++)
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K, &alpha, d_b, M, d_a, K, &beta, d_c, M);

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&cublas_cost,start,end);

    calculate_benchmark(
        naive_cost, tiled_cost, tiled_const_cost, cublas_cost,
        N, M, K
    );

    free(h_a); free(h_b);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cublasDestroy(handle);

}

int main(){

    // check correctness
    correctness_check(256, 256, 256);

    int matrix_size[] = {512, 1024, 2048, 4096};

    for(int size: matrix_size){
        benchmark(size, size, size);
    }
    return 0;
}