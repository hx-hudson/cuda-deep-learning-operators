#include <stdio.h>
#include <cuda_runtime.h>
#include "cuda_utils.cuh"
#include "matmul_kernels.cuh"

__global__ void matmul_tiled
(int* a, int* b, int* c, int N, int M, int K, int tile_size){

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockDim.x*blockIdx.x + tx;
    int row = blockDim.y*blockIdx.y + ty;

    extern __shared__ int shared_mem[];
    int* a_s = shared_mem;
    int* b_s = shared_mem + tile_size*tile_size;

    int sum = 0;
    int num_tiles = (K + tile_size - 1)/ tile_size;
    for(int i = 0; i < num_tiles; i++){
        int global_a_x = i*tile_size + tx;
        int global_b_y = i*tile_size + ty;

        if(global_a_x < K && row < N){
            a_s[ty*tile_size+tx] = a[row*K + global_a_x];
        }
        else{
            a_s[ty*tile_size+tx] = 0;
        }
        if(col < M && global_b_y < K)
            b_s[ty*tile_size+tx] = b[col + (global_b_y)*M];
        else
            b_s[ty*tile_size+tx] = 0;

        __syncthreads();

        for(int k=0; k<tile_size; k++)
            sum += a_s[ty*tile_size+k] * b_s[k*tile_size+tx];

        __syncthreads();
    }
    if(row < N && col < M)
        c[row*M + col] = sum;
}

// int main(){
    
//     int N = 1000, M = 900, K = 700, tile_size = 16;
//     int* h_a = (int*)malloc(N * K * sizeof(int));
//     int* h_b = (int*)malloc(K * M * sizeof(int));
//     int* h_c = (int*)malloc(N * M * sizeof(int));

//     init_matrix(h_a, N, K);
//     init_matrix(h_b, K, M);

//     int* d_a = NULL, *d_b = NULL, *d_c = NULL;
//     CHECK_CUDA(cudaMalloc((void**)&d_a, N * K * sizeof(int)));
//     CHECK_CUDA(cudaMalloc((void**)&d_b, K * M * sizeof(int)));
//     CHECK_CUDA(cudaMalloc((void**)&d_c, N * M * sizeof(int)));

//     CHECK_CUDA(cudaMemcpy(
//         d_a ,h_a, N*K*sizeof(int), cudaMemcpyHostToDevice)
//     );
//     CHECK_CUDA(cudaMemcpy(
//         d_b ,h_b, K*M*sizeof(int), cudaMemcpyHostToDevice)
//     );

//     dim3 blockSize(tile_size, tile_size);
//     dim3 gridSize(
//         (M+blockSize.x-1)/blockSize.x,(N+blockSize.y-1)/blockSize.y
//     );
//     int shared_mem_bytes = tile_size*tile_size*sizeof(int)*2;
//     matmul_tiled<<<gridSize,blockSize,shared_mem_bytes>>>
//     (d_a, d_b, d_c, N, M, K, tile_size);

//     CHECK_CUDA(cudaDeviceSynchronize());

//     CHECK_CUDA(cudaMemcpy(
//         h_c, d_c, N*M*sizeof(int), cudaMemcpyDeviceToHost
//     ));

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