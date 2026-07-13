#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

__device__ __forceinline__
float warp_reduce_sum(float value){

    for(int s = 16; s > 0; s /= 2){
        value += __shfl_down_sync(0xffffffff, value, s);
    }

    return value;
}

template<int BLOCK_SIZE>
__device__ __forceinline__
float reduce_sum(float* shared, float local_sum){

    constexpr int NUM_WARPS = BLOCK_SIZE / 32;

    float warp_sum = warp_reduce_sum(local_sum);

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    if(lane_id == 0) shared[warp_id] = warp_sum;
    __syncthreads();

    if(warp_id == 0){

        float block_sum = lane_id < NUM_WARPS? shared[lane_id]: 0.0f;

        block_sum = warp_reduce_sum(block_sum);

        if(lane_id == 0) shared[0] = block_sum;
    }
    __syncthreads();

    float value = shared[0];
    __syncthreads();

    return value;
}

template<int BLOCK_SIZE>
__global__ void layernorm_kernel
(const float* x, const float* gamma, const float* beta,
    float* y, int rows, int cols, float eps = 1e-5f){
    
    static_assert(
        BLOCK_SIZE % 32 == 0,
        "BLOCK_SIZE must be a multiple of 32"
    );

    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    __shared__ float shared[NUM_WARPS];

    int idx = threadIdx.x;
    int row = blockIdx.x;

    if(row >= rows) return;

    const float* row_x = x + row * cols;
    float* row_y = y + row * cols;

    float local_sum = 0.0f;
    for(int col = idx; col < cols; col += BLOCK_SIZE){
        local_sum += row_x[col];
    }

    float mean = reduce_sum<BLOCK_SIZE>(shared, local_sum)/cols;

    local_sum = 0.0f;
    for(int col = idx; col < cols; col += BLOCK_SIZE){
        float temp = row_x[col] - mean;
        local_sum += temp * temp;
    }

    float variance = reduce_sum<BLOCK_SIZE>(shared, local_sum)/cols;

    float temp = rsqrtf(variance + eps);
    for(int col = idx; col < cols; col += BLOCK_SIZE){
        row_y[col] = (row_x[col] - mean) * 
                    temp * gamma[col] + beta[col];
    }
}

torch::Tensor layernorm_cuda(
    torch::Tensor input, torch::Tensor gamma,
    torch::Tensor beta, float eps=1e-5
){

    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dim() == 2, "input must be a 2D matrix");
    TORCH_CHECK(
        input.scalar_type() == torch::kFloat32,
        "input must have dtype torch.float32"
    );

    int rows = input.size(0);
    int cols = input.size(1);

    auto output = torch::empty_like(input);

    c10::cuda::CUDAGuard device_guard(input.device());

    cudaStream_t stream = 
        at::cuda::getCurrentCUDAStream(input.get_device());

    int threads = 256;
    int blocks = rows;

    layernorm_kernel<256><<<blocks, threads, 0, stream>>>(
        input.data_ptr<float>(), gamma.data_ptr<float>(),
        beta.data_ptr<float>(), output.data_ptr<float>(),
        rows, cols, eps
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}