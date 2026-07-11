#include <cuda_runtime.h>
#include <cfloat>


__device__ __forceinline__
float warp_reduce_max(float value){

    for (int s = 16; s > 0; s /= 2) {
        value = fmaxf(
            value,
            __shfl_down_sync(0xffffffff, value, s)
        );
    }

    return value;
}

template<int BLOCK_SIZE>
__device__ __forceinline__
float reduce_max(float* shared, float local_max){

    constexpr int NUM_WARPS = BLOCK_SIZE / 32;

    float warp_max = warp_reduce_max(local_max);

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    if(lane_id == 0) shared[warp_id] = warp_max;
    __syncthreads();
    
    if(warp_id == 0){

        float thread_max = 
            lane_id < NUM_WARPS? shared[lane_id]: -FLT_MAX;

        thread_max = warp_reduce_max(thread_max);

        if(lane_id == 0) shared[0] = thread_max;
    }
    __syncthreads();

    float result = shared[0];
    __syncthreads();

    return result;
}

__device__  __forceinline__
float warp_reduce_sum(float sum){

    for(int s = 16; s > 0; s /= 2){
        sum += __shfl_down_sync(0xffffffff, sum, s);
    }

    return sum;
}

template<int BLOCK_SIZE>
__device__ __forceinline__
float reduce_sum(float* shared, float local_sum){

    constexpr int NUM_WARPS = BLOCK_SIZE / 32;

    float warp_sum = warp_reduce_sum(local_sum);

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    if(lane_id == 0) shared[warp_id] = warp_sum;
    __syncthreads();

    if(warp_id == 0){

        float thread_sum = 
            lane_id < NUM_WARPS? shared[lane_id]: 0.0f;

        thread_sum = warp_reduce_sum(thread_sum);

        if(lane_id == 0) shared[0] = thread_sum;
    }
    __syncthreads();

    return shared[0];

}

template<int BLOCK_SIZE>
__global__ void softmax_warp_shuffle(
    const float* x,
    float* y,
    int rows,
    int cols
){
    static_assert(
        BLOCK_SIZE % 32 == 0,
        "BLOCK_SIZE must be a multiple of 32"
    );

    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    __shared__ float shared[NUM_WARPS];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= rows) return;

    const float* row_x = x + row * cols;
    float* row_y = y + row * cols;

    // find local max
    float local_max = -FLT_MAX;
    for(int col = tid; col < cols; col += BLOCK_SIZE){
        local_max = fmaxf(local_max, row_x[col]);
    }

    // find max value
    float max = reduce_max<BLOCK_SIZE>(shared, local_max);



    // local sum
    float local_sum = 0;
    for(int col = tid; col < cols; col += BLOCK_SIZE){
        float exp_x = expf(row_x[col] - max);
        local_sum += exp_x;
        row_y[col] = exp_x;
    }

    // sum
    float sum = reduce_sum<BLOCK_SIZE>(shared, local_sum);

    // store to y
    for(int col = tid; col < cols; col += BLOCK_SIZE){
            row_y[col] /= sum;
    }
}