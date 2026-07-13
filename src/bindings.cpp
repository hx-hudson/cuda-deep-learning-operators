#include <torch/extension.h>

torch::Tensor relu_cuda(torch::Tensor input);
torch::Tensor softmax_cuda(torch::Tensor input);
torch::Tensor layernorm_cuda(
    torch::Tensor input, torch::Tensor gamma,
    torch::Tensor beta, float eps=1e-5
);
torch::Tensor matmul_cuda(torch::Tensor a, torch::Tensor b);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){

    m.def(
        "relu_cuda",
        &relu_cuda,
        "RELU forward using a custom CUDA kernel"
    );

    m.def(
        "softmax_cuda",
        &softmax_cuda,
        "softmax forward using a custom CUDA kernel"
    );

    m.def(
        "layernorm_cuda",
        &layernorm_cuda,
        "layernorm forward using a custom CUDA kernel"
    );

    m.def(
        "matmul_cuda",
        &matmul_cuda,
        "Matrix Multiplication"
    );
    
}