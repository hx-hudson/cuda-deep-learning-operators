#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

inline bool check_result(int* ref, int* out, int size) {
    for (int i = 0; i < size; i++) {
        if (ref[i] != out[i]) {
            printf("Mismatch at %d: CPU %d, GPU %d\n",
                   i, ref[i], out[i]);
            return false;
        }
    }
    return true;
}

inline void matmul_cpu(int* a, int* b, int* c, int N, int M, int K) {
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < M; col++) {
            int sum = 0;

            for (int k = 0; k < K; k++) {
                sum += a[row*K+k] * b[k*M+col];
            }

            c[row*M+col] = sum;
        }
    }
}