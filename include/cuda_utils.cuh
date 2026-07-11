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

inline void softmax_cpu(const float* x, float* y, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        int row_offset = r * cols;

        float max_val = x[row_offset];
        for (int c = 1; c < cols; ++c) {
            max_val = std::max(max_val, x[row_offset + c]);
        }

        float sum_exp = 0.0f;
        for (int c = 0; c < cols; ++c) {
            sum_exp += std::exp(x[row_offset + c] - max_val);
        }

        for (int c = 0; c < cols; ++c) {
            y[row_offset + c] = std::exp(x[row_offset + c] - max_val) / sum_exp;
        }
    }
}

void layer_norm_cpu
(const float* x, const float* gamma, const float* beta, 
float* y, int batch_size, int hidden_size, float eps = 1e-5f){

    for (int b = 0; b < batch_size; ++b){
        int offset = b * hidden_size;

        float mean = 0.0f;
        for (int i = 0; i < hidden_size; ++i) {
            mean += x[offset + i];
        }
        mean /= hidden_size;

        float var = 0.0f;
        for (int i = 0; i < hidden_size; ++i) {
            float diff = x[offset + i] - mean;
            var += diff * diff;
        }
        var /= hidden_size;

        float rstd = 1.0f / std::sqrt(var + eps);

        for (int i = 0; i < hidden_size; ++i) {
            float normalized = (x[offset + i] - mean) * rstd;
            y[offset + i] = normalized * gamma[i] + beta[i];
        }
    }

}