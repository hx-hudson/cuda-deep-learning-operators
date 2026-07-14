from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="custom_ops",
    version="0.1.0",

    packages=find_packages(),

    ext_modules=[
        CUDAExtension(
            name="custom_ops._C",
            sources=[
                "pytorch/bindings.cpp",
                "pytorch/relu_extension.cu",
                "pytorch/softmax_extension.cu",
                "pytorch/layernorm_extension.cu",
                "pytorch/matmul_extension.cu",
                "src/relu.cu"
            ],
            include_dirs=[
                "include",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3"]
            }
        )
    ],
    
    cmdclass={"build_ext": BuildExtension},

    zip_safe=False
)