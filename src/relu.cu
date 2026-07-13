#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

__global__ void relu(const float* x, float* y, int64_t N){

    int64_t idx = 
        static_cast<int64_t>(blockDim.x) * blockIdx.x + threadIdx.x;

    if(idx < N){
        y[idx] = fmaxf(x[idx], 0.0f);
    }
}

torch::Tensor relu_cuda(torch::Tensor input){

    TORCH_CHECK(
        input.is_cuda(), "input must be a CUDA tnesor"
    );

    TORCH_CHECK(
        input.scalar_type() == at::kFloat,
        "input must have dtype torch.float32"
    );

    TORCH_CHECK(
        input.is_contiguous(), "input must be contiguous"
    );

    auto output = torch::empty_like(input);

    int64_t N = input.numel();

    if(N == 0) return output;

    constexpr int threads = 256;
    int blocks = static_cast<int>((N + threads - 1) / threads);

    cudaStream_t stream = 
        at::cuda::getCurrentCUDAStream(input.get_device());

    relu<<<blocks, threads, 0, stream>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), N
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}