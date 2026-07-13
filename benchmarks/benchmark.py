import torch
import custom_ops

@torch.inference_mode()
def benchmark_func(func, warmup=20, repeat=500):
    
    for _ in range(warmup):
        func()

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(repeat):
        func()

    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / repeat


def print_results(results):
    print(
        f"{'Cols':<10}"
        f"{'Custom(ms)':<15}"
        f"{'PyTorch(ms)':<15}"
        f"{'Custom/PyTorch':<15}"
    )

    print("-" * 55)

    for result in results:
        col = result["col"]
        custom_time = result["custom_time"]
        torch_time = result["torch_time"]

        ratio = custom_time / torch_time

        print(
            f"{col:<10}"
            f"{custom_time:<15.6f}"
            f"{torch_time:<15.6f}"
            f"{ratio:<15.2f}"
        )

    print()


def relu_benchmark():

    print("relu benchmark")
    print("Rows = 4096")

    row =4096
    cols = [128, 256, 512, 1024]

    results = []
    for col in cols:
        x = torch.randn(
            row, col, device="cuda", dtype=torch.float32
            )

        custom_time = benchmark_func(lambda: custom_ops.relu(x))
        torch_time = benchmark_func(lambda: torch.relu(x))

        results.append({
            "col": col,
            "custom_time": custom_time,
            "torch_time": torch_time
        })

    print_results(results)


def softmax_benchmark():

    print("softmax benchmark")
    print("Rows = 4096")

    row =4096
    cols = [128, 256, 512, 1024]

    results = []
    for col in cols:
        x = torch.randn(
            row, col, device="cuda", dtype=torch.float32
            )

        custom_time = benchmark_func(lambda: custom_ops.softmax(x))
        torch_time = benchmark_func(lambda: torch.softmax(x,-1))

        results.append({
            "col": col,
            "custom_time": custom_time,
            "torch_time": torch_time
        })

    print_results(results)

def layernrom_benchmark():

    print("layernorm benchmark")
    print("Rows = 4096")

    row =4096
    cols = [128, 256, 512, 1024]

    results = []
    for col in cols:

        x = torch.randn(
            row, col, device="cuda", dtype=torch.float32
            )
        gamma = torch.randn(col, device="cuda", dtype=torch.float32)
        beta = torch.randn(col, device="cuda", dtype=torch.float32)

        custom_time = benchmark_func(
            lambda: custom_ops.layernorm(x, gamma, beta, eps=1e-5)
            )
        torch_time = benchmark_func(
            lambda: torch.layer_norm(
                x, (col, ), gamma, beta, eps=1e-5)
            )

        results.append({
            "col": col,
            "custom_time": custom_time,
            "torch_time": torch_time
        })

    print_results(results)

def matmul_benchmark():

    print("matmul benchmark")
    print("N * N matrix")

    N = [512, 1024, 2048, 4096]

    results = []
    for col in N:
        x = torch.randn(
            col, col, device="cuda", dtype=torch.float32
            )

        custom_time = benchmark_func(
            lambda: custom_ops.matmul(x, x)
            )
        torch_time = benchmark_func(
            lambda: torch.matmul(x, x)
            )

        results.append({
            "col": col,
            "custom_time": custom_time,
            "torch_time": torch_time
        })

    print_results(results)

if __name__ == "__main__":

    relu_benchmark()
    softmax_benchmark()
    layernrom_benchmark()
    matmul_benchmark()
    