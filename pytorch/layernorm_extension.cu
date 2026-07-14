#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "other_kernels.cuh"

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