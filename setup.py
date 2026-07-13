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
                "src/bindings.cpp",
                "src/relu.cu",
                "src/softmax_warp_shuffle.cu",
                "src/layernorm.cu",
                "src/matmul_tiled.cu"
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