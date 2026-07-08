#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include <cmath>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

inline bool check_result(float* ref, float* out, int size) {
    const float atol = 1e-3f;
    const float rtol = 1e-3f;

    for (int i = 0; i < size; i++) {
        float diff = std::fabs(ref[i] - out[i]);
        float tolerance = atol + rtol * std::fabs(ref[i]);

        if (diff > tolerance) {
            printf(
                "Mismatch at %d: CPU %f, GPU %f, diff %f, tolerance %f\n",
                i, ref[i], out[i], diff, tolerance
            );
            return false;
        }
    }

    return true;
}

inline void matmul_cpu(float* a, float* b, float* c, int N, int M, int K) {
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < M; col++) {
            float sum = 0;

            for (int k = 0; k < K; k++) {
                sum += a[row*K+k] * b[k*M+col];
            }

            c[row*M+col] = sum;
        }
    }
}

inline void init_matrix(float* a, int N, int M){
    for(int i = 0; i < N; i++)
        for(int j = 0; j < M; j++)
            a[i*M+j] = rand() / 100;
}