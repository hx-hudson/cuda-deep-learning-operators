#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "other_kernels.cuh"

#include <cstdint>
#include <cfloat>

torch::Tensor softmax_cuda(torch::Tensor input){

    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dim() == 2, "input must be a 2D matrix");
    TORCH_CHECK(
        input.scalar_type() == torch::kFloat32,
        "input must have dtype torch.float32"
    );

    int cols = input.size(1);
    int rows = input.size(0);

    auto output = torch::empty_like(input);

    c10::cuda::CUDAGuard device_guard(input.device());

    int threads = 256;
    int blocks = rows;

    cudaStream_t stream = 
        at::cuda::getCurrentCUDAStream(input.get_device());

    softmax_warp_shuffle<256><<<blocks, threads, 0, stream>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), rows, cols
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}