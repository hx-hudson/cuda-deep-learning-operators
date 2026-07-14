#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>

#include "matmul_kernels.cuh"
#include "cuda_utils.cuh"

namespace{

constexpr int TileSize = 16;
constexpr int BenchmarkWarmup = 10;
constexpr int BenchmarkRepeat = 500;

struct ProblemSize{
    int m;
    int n;
    int k;
};

struct BenchmarkResult{
    std::string kernel;
    int block_x;
    int block_y;
    float time_ms;
    double gflops;
    double speedup_vs_naive;
};

const char* cublas_status_string(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
        default:
            return "CUBLAS_STATUS_UNKNOWN";
    }
}

#define CHECK_CUBLAS(call)                                                    \
    do {                                                                      \
        const cublasStatus_t status_ = (call);                                \
        if (status_ != CUBLAS_STATUS_SUCCESS) {                               \
            std::fprintf(stderr,                                              \
                         "cuBLAS error at %s:%d: %s (error code: %d)\n",      \
                         __FILE__,                                            \
                         __LINE__,                                            \
                         cublas_status_string(status_),                       \
                         status_);                                            \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

template<typename LaunchFn, typename CheckFn>
bool check_func(
    const char* kernel_name,
    LaunchFn launch_func,
    CheckFn check_func,
    float* d_c,
    float* h_c,
    float* h_ref,
    std::size_t c_elements
){
    CHECK_CUDA(cudaMemset(d_c, 0, c_elements * sizeof(float)));

    launch_func();
    check_func();
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(
        h_c, d_c,
        c_elements * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    const bool passed = 
        check_result(h_ref, h_c, c_elements);
    std::printf(
        "  %-18s %s\n", kernel_name, passed ? "PASSED" : "FAILED"
    );
    return passed;
}

bool correctness_check(int M, int N, int K){

    const std::size_t a_elements = static_cast<std::size_t>(M) * K;
    const std::size_t b_elements = static_cast<std::size_t>(K) * N;
    const std::size_t c_elements = static_cast<std::size_t>(M) * N;

    std::vector<float> h_a(a_elements);
    std::vector<float> h_b(b_elements);
    std::vector<float> h_c(c_elements);
    std::vector<float> h_ref(c_elements);

    init_matrix(h_a.data(), M, K);
    init_matrix(h_b.data(), K, N);
    matmul_cpu(h_a.data(), h_b.data(), h_ref.data(), M, N, K);

    float* d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&d_a), a_elements * sizeof(float)
    ));
    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&d_b), b_elements * sizeof(float)
    ));
    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&d_c), c_elements * sizeof(float)
    ));

    CHECK_CUDA(cudaMemcpy(
        d_a, h_a.data(),
        a_elements * sizeof(float),
        cudaMemcpyHostToDevice
    ));
    CHECK_CUDA(cudaMemcpy(
        d_b, h_b.data(),
        b_elements * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    const dim3 block_naive(16, 16);
    const dim3 grid_naive(
        (N + block_naive.x - 1) / block_naive.x,
        (M + block_naive.y - 1) / block_naive.y
    );

    const dim3 block_tiled(TileSize, TileSize);
    const dim3 grid_tiled(
        (N + TileSize - 1) / TileSize,
        (M + TileSize - 1) / TileSize
    );
    const std::size_t dynamic_sharedmem = 
        2ULL * TileSize * TileSize * sizeof(float);

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    const float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t last_cublas_status = CUBLAS_STATUS_SUCCESS;

    const auto check_cuda_launch = []() {
        CHECK_CUDA(cudaGetLastError());
    };
    const auto check_cublas_launch = [&]() {
        CHECK_CUBLAS(last_cublas_status);
    };

    std::printf(
        "Correctness check: A[%d x %d] * B[%d x %d]\n", M, K, K, N
    );

    bool all_passed = true;

    all_passed &= check_func(
        "Naive",
        [&](){
            matmul_naive<<<grid_naive, block_naive>>>(
                d_a, d_b, d_c, M, N, K
            );
        },
        check_cuda_launch,
        d_c, h_c.data(), h_ref.data(), c_elements
    );

    all_passed &= check_func(
        "Tile",
        [&](){
            matmul_tiled<<<
                grid_naive, block_naive, dynamic_sharedmem
            >>>(
                d_a, d_b, d_c, M, N, K, TileSize
            );
        },
        check_cuda_launch,
        d_c, h_c.data(), h_ref.data(), c_elements
    );

    all_passed &= check_func(
        "Tiled constant",
        [&](){
            matmul_tiled_constant<TileSize><<<
                grid_tiled, block_tiled
            >>>(
                d_a, d_b, d_c, M, N, K
            );
        },
        check_cuda_launch,
        d_c, h_c.data(), h_ref.data(), c_elements
    );

    all_passed &= check_func(
        "cuBLAS",
        [&](){
            last_cublas_status = cublasSgemm(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                N, M, K,
                &alpha,
                d_b, N,
                d_a, K,
                &beta,
                d_c, N
            );
        },
        check_cublas_launch,
        d_c, h_c.data(), h_ref.data(), c_elements
    );

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    return all_passed;
}

double calculate_gflops(int M, int N, int K, float time){
    const double operations 
        = 2.0 * static_cast<double>(M) * N * K;

    return operations / (time * 1.0e6);
}

template<typename LaunchFn, typename CheckFn>
float measure(LaunchFn launch_func, CheckFn check_func){

    for(int i = 0; i < BenchmarkWarmup; i++) launch_func();

    check_func();

    cudaEvent_t start = nullptr, end = nullptr;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&end));

    CHECK_CUDA(cudaEventRecord(start));

    for(int i = 0; i < BenchmarkRepeat; i++) launch_func();

    check_func();
    CHECK_CUDA(cudaEventRecord(end));
    CHECK_CUDA(cudaEventSynchronize(end));

    float time = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&time, start, end));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(end));

    return time / BenchmarkRepeat;
}

std::vector<BenchmarkResult> benchmark(int M, int N, int K){

    const std::size_t a_elements = static_cast<std::size_t>(M) * K;
    const std::size_t b_elements = static_cast<std::size_t>(K) * N;
    const std::size_t c_elements = static_cast<std::size_t>(M) * N;

    std::vector<float> h_a(a_elements), h_b(b_elements);
    init_matrix(h_a.data(), M, K);
    init_matrix(h_b.data(), K, N);

    float* d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    cudaMalloc(
        reinterpret_cast<void**>(&d_a), a_elements * sizeof(float)
    );
    cudaMalloc(
        reinterpret_cast<void**>(&d_b), b_elements * sizeof(float)
    );
    cudaMalloc(
        reinterpret_cast<void**>(&d_c), c_elements * sizeof(float)
    );

    CHECK_CUDA(cudaMemcpy(
        d_a, h_a.data(),
        a_elements * sizeof(float),
        cudaMemcpyHostToDevice
    ));
    CHECK_CUDA(cudaMemcpy(
        d_b, h_b.data(),
        b_elements * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    const dim3 block_naive(16, 16);
    const dim3 grid_naive(
        (N + block_naive.x - 1) / block_naive.x,
        (M + block_naive.y - 1) / block_naive.y
    );

    const dim3 block_tiled(TileSize, TileSize);
    const dim3 grid_tiled(
        (N + TileSize - 1) / TileSize,
        (M + TileSize - 1) / TileSize
    );
    const std::size_t dynamic_sharedmem = 
        2ULL * TileSize * TileSize * sizeof(float);

    cublasHandle_t handle = nullptr;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    const float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t last_cublas_status = CUBLAS_STATUS_SUCCESS;

    const auto check_cuda_launch = []() {
        CHECK_CUDA(cudaGetLastError());
    };
    const auto check_cublas_launch = [&]() {
        CHECK_CUBLAS(last_cublas_status);
    };

    std::vector<BenchmarkResult> results;
    results.reserve(4);

    float time_naive = measure(
        [&](){
            matmul_naive<<<grid_naive, block_naive>>>(
                d_a, d_b, d_c,
                M, N, K
            );
        },
        check_cuda_launch
    );
    results.push_back({
        "Naive", 16, 16, time_naive, 
        calculate_gflops(M, N, K, time_naive),
        1.0
    });

    float time_tiled = measure(
        [&](){
            matmul_tiled<<<
                grid_tiled, block_tiled, dynamic_sharedmem
            >>>(
                d_a, d_b, d_c,
                M, N, K, TileSize
            );
        },
        check_cuda_launch
    );
    results.push_back({
        "Tiled Runtime", TileSize, TileSize, time_tiled, 
        calculate_gflops(M, N, K, time_tiled),
        time_naive / time_tiled
    });

    float time_tiled_const = measure(
        [&](){
            matmul_tiled_constant<TileSize><<<
                grid_tiled, block_tiled
            >>>(
                d_a, d_b, d_c,
                M, N, K
            );
        },
        check_cuda_launch
    );
    results.push_back({
        "Tiled Constant", TileSize, TileSize, time_tiled_const, 
        calculate_gflops(M, N, K, time_tiled_const),
        time_naive / time_tiled_const
    });

    float time_cuBLAS = measure(
        [&](){
            last_cublas_status = cublasSgemm(
                handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, 
                &alpha,
                d_b, N,
                d_a, K,
                &beta,
                d_c, N
            );
        },
        check_cublas_launch
    );
    results.push_back({
        "cuBLAS", 0, 0, time_cuBLAS, 
        calculate_gflops(M, N, K, time_cuBLAS),
        time_naive / time_cuBLAS
    });

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    return results;
}

void print_results(
    const ProblemSize& problem,
    const std::vector<BenchmarkResult>& results
) {
    std::printf(
        "\nMatrix multiplication: A[%d x %d] * B[%d x %d]\n\n",
        problem.m,
        problem.k,
        problem.k,
        problem.n
    );

    std::printf(
        "%-20s %-10s %-12s %-12s %-12s\n",
        "Kernel",
        "Block",
        "Time(ms)",
        "GFLOPS",
        "vs Naive"
    );
    std::printf(
        "-----------------------------------------------------------------------\n"
    );

    for (const BenchmarkResult& result : results) {
        char block_text[32];
        if (result.block_x == 0) {
            std::snprintf(block_text, sizeof(block_text), "-");
        } else {
            std::snprintf(
                block_text,
                sizeof(block_text),
                "%dx%d",
                result.block_x,
                result.block_y
            );
        }

        std::printf(
            "%-20s %-10s %-12.6f %-12.2f %-12.2fx\n",
            result.kernel.c_str(),
            block_text,
            result.time_ms,
            result.gflops,
            result.speedup_vs_naive
        );
    }
}

void write_csv_header(std::ofstream& csv) {
    csv << "M,N,K,kernel,block_x,block_y,warmup_iterations,"
           "benchmark_iterations,time_ms,gflops,speedup_vs_naive\n";
}

void append_csv_results(
    std::ofstream& csv,
    const ProblemSize& problem,
    const std::vector<BenchmarkResult>& results
) {
    csv << std::fixed << std::setprecision(6);

    for (const BenchmarkResult& result : results) {
        csv << problem.m << ','
            << problem.n << ','
            << problem.k << ','
            << result.kernel << ','
            << result.block_x << ','
            << result.block_y << ','
            << BenchmarkWarmup << ','
            << BenchmarkRepeat << ','
            << result.time_ms << ','
            << result.gflops << ','
            << result.speedup_vs_naive << '\n';
    }
}

bool parse_positive_int(const char* text, int* value) {
    try {
        const int parsed = std::stoi(text);
        if (parsed <= 0) {
            return false;
        }
        *value = parsed;
        return true;
    } catch (...) {
        return false;
    }
}

}

int main(int argc, char** argv) {
    std::vector<ProblemSize> problems;
    std::string csv_path = "matmul_results.csv";

    if (argc == 1) {
        problems = {
            {512, 512, 512},
            {1024, 1024, 1024},
            {2048, 2048, 2048},
            {4096, 4096, 4096}
        };
    } else if (argc == 4 || argc == 5) {
        ProblemSize problem{};
        if (!parse_positive_int(argv[1], &problem.m) ||
            !parse_positive_int(argv[2], &problem.n) ||
            !parse_positive_int(argv[3], &problem.k)) {
            std::fprintf(stderr, "M, N, and K must be positive integers.\n");
            return EXIT_FAILURE;
        }
        problems.push_back(problem);

        if (argc == 5) {
            csv_path = argv[4];
        }
    } else {
        std::fprintf(
            stderr,
            "Usage:\n"
            "  %s                         # run default square sizes\n"
            "  %s M N K [output.csv]      # run one custom size\n",
            argv[0],
            argv[0]
        );
        return EXIT_FAILURE;
    }

    if (!correctness_check(256, 256, 256) ||
        !correctness_check(257, 129, 513)) {
        std::fprintf(
            stderr, 
            "At least one correctness check failed.\n"
        );
        return EXIT_FAILURE;
    }

    std::ofstream csv(csv_path, std::ios::out | std::ios::trunc);
    if (!csv.is_open()) {
        std::fprintf(
            stderr, 
            "Failed to open CSV file: %s\n", 
            csv_path.c_str()
        );
        return EXIT_FAILURE;
    }
    write_csv_header(csv);

    for (const ProblemSize& problem : problems) {
        const std::vector<BenchmarkResult> results =
            benchmark(problem.m, problem.n, problem.k);
        print_results(problem, results);
        append_csv_results(csv, problem, results);
    }

    std::printf("\nCSV results written to: %s\n", csv_path.c_str());
    return EXIT_SUCCESS;
}