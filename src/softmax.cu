#include <cuda_runtime.h>
#include <cfloat>

__global__ void softmax(float* x, float* y, int rows, int cols){

    __shared__ float x_s[128];

    int row = blockIdx.x;
    int idx = threadIdx.x;

    if (row >= rows) {
        return;
    }

    // local max
    float max = -FLT_MAX;
    for(int col = idx; col < cols; col += 128){
        max = fmaxf(x[row*cols + col], max);
    }

    // load to shared memory
    x_s[idx] = max;

    __syncthreads();

    // find the max value
    for(int s = 64 ; s > 0 ; s /= 2){

        if(idx < s){
            x_s[idx] = fmaxf(x_s[idx], x_s[idx+s]);
        }

        __syncthreads();
    }

    // local sum
    max = x_s[0];
    float sum = 0;
    for(int col = idx; col < cols; col += 128){
        float exp_x = expf(x[row*cols + col] - max);
        sum += exp_x;
        y[row*cols + col] = exp_x;
    }

    x_s[idx] = sum;

    __syncthreads();

    // calculate the sum
    for(int s = 64 ; s > 0 ; s /= 2){

        if(idx < s){
            x_s[idx] = x_s[idx] + x_s[idx+s];
        }

        __syncthreads();
    }

    float total_sum = x_s[0];
    for(int col = idx; col < cols; col += 128){
        y[row*cols + col] /= total_sum;
    }
}