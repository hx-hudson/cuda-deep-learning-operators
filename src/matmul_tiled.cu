
#include <cuda_runtime.h>

__global__ void matmul_tiled
(float* a, float* b, float* c, int N, int M, int K, int tile_size){

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockDim.x*blockIdx.x + tx;
    int row = blockDim.y*blockIdx.y + ty;

    extern __shared__ float shared_mem[];
    float* a_s = shared_mem;
    float* b_s = shared_mem + tile_size*tile_size;

    float sum = 0;
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
