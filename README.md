# CUDA-Accelerated Deep Learning Operators

A CUDA learning project focused on implementing and optimizing common deep learning operators from scratch.

The current stage focuses on **FP32 matrix multiplication (GEMM)** and compares hand-written CUDA kernels against NVIDIA cuBLAS.

## Implemented Matrix Multiplication Kernels

### 1. Naive CUDA MatMul

Each CUDA thread computes one output element of the result matrix directly from global memory.

This version serves as the baseline implementation.

### 2. Tiled Shared-Memory MatMul

The input matrices are divided into tiles. Threads in the same block cooperatively load tiles of `A` and `B` into dynamic shared memory and reuse them during computation.

This implementation uses:

- `16 x 16` thread blocks
- dynamic shared memory
- block-level tiling
- coalesced global-memory loads

### 3. Compile-Time Tiled MatMul

This version uses the same shared-memory tiling strategy, but the tile size is fixed at compile time:

```cpp
matmul_tiled_constant<16>
```

Making the tile size a compile-time constant allows the compiler to optimize indexing and loop structure more aggressively.

### 4. cuBLAS SGEMM

`cublasSgemm` is used as the optimized NVIDIA library baseline.

This comparison shows the remaining gap between a basic hand-written CUDA kernel and a production-level GEMM implementation.

---

## Benchmark Methodology

All matrix multiplication benchmarks use FP32 data.

- GPU: NVIDIA GeForce RTX 5080 Laptop GPU
- Tile size: `16 x 16`
- Warm-up iterations: `10`
- Timed iterations: `500`
- Timing method: CUDA Events
- Correctness test: compared against a CPU reference implementation
- Correctness test size: `256 x 256 x 256`
- Performance metric:

```text
GFLOPS = 2 × N × M × K / execution_time
```

All four implementations passed the correctness check.

---

## Performance Results

### 512 x 512 x 512

| Kernel | Time (ms) | GFLOPS | Speedup vs. Naive |
|---|---:|---:|---:|
| Naive CUDA | 0.130372 | 2059.00 | 1.00x |
| Tiled Shared Memory | 0.144321 | 1859.99 | 0.90x |
| Tiled Compile-Time | 0.097749 | 2746.16 | **1.33x** |
| cuBLAS | 0.030295 | 8860.86 | **4.30x** |

### 1024 x 1024 x 1024

| Kernel | Time (ms) | GFLOPS | Speedup vs. Naive |
|---|---:|---:|---:|
| Naive CUDA | 0.953866 | 2251.35 | 1.00x |
| Tiled Shared Memory | 0.987538 | 2174.58 | 0.97x |
| Tiled Compile-Time | 0.649937 | 3304.14 | **1.47x** |
| cuBLAS | 0.121371 | 17693.52 | **7.86x** |

### 2048 x 2048 x 2048

| Kernel | Time (ms) | GFLOPS | Speedup vs. Naive |
|---|---:|---:|---:|
| Naive CUDA | 7.251064 | 2369.29 | 1.00x |
| Tiled Shared Memory | 7.667174 | 2240.70 | 0.95x |
| Tiled Compile-Time | 5.229117 | 3285.42 | **1.39x** |
| cuBLAS | 0.772498 | 22239.38 | **9.39x** |

### 4096 x 4096 x 4096

| Kernel | Time (ms) | GFLOPS | Speedup vs. Naive |
|---|---:|---:|---:|
| Naive CUDA | 68.344002 | 2010.99 | 1.00x |
| Tiled Shared Memory | 71.486374 | 1922.59 | 0.96x |
| Tiled Compile-Time | 50.667389 | 2712.57 | **1.35x** |
| cuBLAS | 6.628629 | 20734.15 | **10.31x** |

---

## Performance Summary

The compile-time tiled kernel is the fastest hand-written implementation.

Across the tested matrix sizes, it achieves approximately:

- **1.33x to 1.47x speedup** over the naive CUDA kernel
- up to **3.30 TFLOPS** on the tested workload

The dynamic shared-memory tiled implementation does not outperform the naive kernel. Its performance ranges from approximately `0.90x` to `0.97x` of the naive implementation.

cuBLAS remains substantially faster:

- `4.30x` faster than naive at `512 x 512`
- `7.86x` faster at `1024 x 1024`
- `9.39x` faster at `2048 x 2048`
- `10.31x` faster at `4096 x 4096`

Compared with the best hand-written kernel, cuBLAS is approximately `3.23x` to `7.64x` faster.

---

## Key Findings

### Shared memory does not automatically improve performance

The dynamic tiled implementation is slightly slower than the naive kernel even though it reduces repeated global-memory accesses.

This shows that adding shared memory alone is not sufficient. Performance also depends on:

- synchronization overhead
- indexing overhead
- compiler optimization opportunities
- register usage
- occupancy
- tile size

### Compile-time tile size significantly improves performance

Changing the tile size from a runtime value to a compile-time template parameter produces a clear speedup.

The best custom kernel reaches:

```text
3304.14 GFLOPS
```

for a `1024 x 1024 x 1024` matrix multiplication.

This is approximately `1.47x` faster than the naive implementation.

### The gap to cuBLAS grows with matrix size

The cuBLAS speedup over the naive kernel increases from `4.30x` at size `512` to `10.31x` at size `4096`.

This indicates that cuBLAS is much more effective at utilizing the GPU on large GEMM workloads.

---

## Build and Run

Example build command:

```bash
nvcc \
    -Iinclude \
    src/matmul_naive.cu \
    src/matmul_tiled.cu \
    src/matmul_tiled_constant.cu \
    benchmarks/matmul_benchmark.cu \
    -lcublas \
    -o benchmark
```

Run:

```bash
./benchmark
```

## Main Result

The main result of the matrix multiplication optimization stage is:

```text
Naive CUDA
    ↓
Compile-Time Shared-Memory Tiling
    ↓
1.33x–1.47x speedup
```

The comparison against cuBLAS also demonstrates that basic shared-memory tiling is only the first step toward high-performance GEMM.
