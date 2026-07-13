import torch
from . import _C

def relu(x):
    """
    Apply relu using the custom CUDA kernel
    """
    return _C.relu_cuda(x)

def softmax(x):
    """
    Apply softmax using the custom CUDA kernel
    """
    return _C.softmax_cuda(x)

def layernorm(x, gamma, beta, eps):
    """
    Apply layernrom using the custom CUDA kernel
    """
    return _C.layernorm_cuda(x, gamma, beta, eps)

def matmul(a, b):
    """
    Apply matrix multiplication
    """
    return _C.matmul_cuda(a, b)

__all__ = ["relu", "softmax", "layernorm", "matmul"]