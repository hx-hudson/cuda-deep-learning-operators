#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>

template<int TILE>
__global__ void matmul_tiled_constant
(float* a, float* b, float* c, int N, int M, int K){

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockDim.x*blockIdx.x + tx;
    int row = blockDim.y*blockIdx.y + ty;

    __shared__ float a_s[TILE*TILE];
    __shared__ float b_s[TILE*TILE];

    float sum = 0;
    int tile_size = TILE;
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

torch::Tensor matmul_cuda(torch::Tensor a, torch::Tensor b){

    TORCH_CHECK(a.dim() == 2, "a must be 2D");
    TORCH_CHECK(b.dim() == 2, "b must be 2D");
    TORCH_CHECK(a.size(1) == b.size(0),
            "incompatible matrix dimensions");
    TORCH_CHECK(a.device() == b.device(),
            "a and b must be on the same device");
    TORCH_CHECK(a.is_contiguous(), "a must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "b must be contiguous");

    int N = a.size(0), M = b.size(1), K = a.size(1);

    auto c = torch::empty({N, M}, a.options());

    if(N == 0 || M == 0) return c;

    c10::cuda::CUDAGuard device_guard(a.device());
    cudaStream_t stream = 
            at::cuda::getCurrentCUDAStream(a.get_device());

    dim3 blockSize(16, 16);
    dim3 gridSize((M + 15) / 16, (N + 15) / 16);
    matmul_tiled_constant<16><<<gridSize, blockSize, 0, stream>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), 
        c.data_ptr<float>(), N, M, K
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return c;
}