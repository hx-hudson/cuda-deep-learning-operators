#include <stdio.h>
#include <cuda_runtime.h>
#include "matmul_kernels.cuh"
#include "cuda_utils.cuh"

int main(){

    int N = 2048, M = 2048, K = 2048, tile_size = 16;
    int* h_a = (int*)malloc(N*K*sizeof(int));
    int* h_b = (int*)malloc(K*M*sizeof(int));
    int* h_c = (int*)malloc(N*M*sizeof(int));

    init_matrix(h_a, N, K);
    init_matrix(h_b, K, M);
    int* h_ref = (int*)malloc(N*M*4);
    matmul_cpu(h_a, h_b, h_ref, N, M, K);

    int* d_a = NULL, *d_b = NULL, *d_c = NULL;
    cudaMalloc((void**)&d_a, N*K*sizeof(int));
    cudaMalloc((void**)&d_b, K*M*sizeof(int));
    cudaMalloc((void**)&d_c, N*M*sizeof(int));

    cudaMemcpy(d_a,h_a,N*K*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,K*M*sizeof(int),cudaMemcpyHostToDevice);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    float naive_cost = 0, tiled_cost = 0, tiled_cost_const=0;

    dim3 blocksize(16,16);
    dim3 gridsize(
        (M+blocksize.x-1)/blocksize.x,(N+blocksize.y-1)/blocksize.y
    );

    // warm up
    for(int i=0;i<10;i++) 
        matmul_naive<<<gridsize, blocksize>>>(d_a,d_b,d_c,N,M,K);

    CHECK_CUDA(cudaGetLastError());

    cudaMemcpy(h_c, d_c, N*M*sizeof(int), cudaMemcpyDeviceToHost);
    if (check_result(h_ref, h_c, N * M)) {
        printf("Correctness check passed!\n");
    } else {
        printf("Correctness check failed!\n");
    }

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
    size_t shared_mem_size = 2 * tile_size * tile_size * sizeof(int);
    
    // warm up
    for(int i=0;i<10;i++)
        matmul_tiled<<<tile_gridsize,tile_blocksize,shared_mem_size>>>
        (d_a,d_b,d_c,N,M,K,tile_size);

    CHECK_CUDA(cudaGetLastError());

    cudaMemcpy(h_c, d_c, N*M*sizeof(int), cudaMemcpyDeviceToHost);
    if (check_result(h_ref, h_c, N * M)) {
        printf("Correctness check passed!\n");
    } else {
        printf("Correctness check failed!\n");
    }

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

    cudaMemcpy(h_c, d_c, N*M*sizeof(int), cudaMemcpyDeviceToHost);
    if (check_result(h_ref, h_c, N * M)) {
        printf("Correctness check passed!\n");
    } else {
        printf("Correctness check failed!\n");
    }
    
    cudaEventRecord(start);
    for(int i=0;i<500;i++)
        matmul_tiled_constant<16><<<tile_gridsize,tile_blocksize>>>
        (d_a,d_b,d_c,N,M,K);

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&tiled_cost_const,start,end);

    printf("naive cost:%f\n",naive_cost/500);
    printf("tiled cost:%f\n",tiled_cost/500);
    printf("tiled const cost:%f\n",tiled_cost_const/500);

}