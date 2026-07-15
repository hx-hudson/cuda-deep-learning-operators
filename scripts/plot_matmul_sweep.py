from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


CSV_PATH = Path("results/matmul_block_sweep.csv")
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def configuration_name(row: pd.Series) -> str:
    if row["kernel"] == "cuBLAS":
        return "cuBLAS"
    return f'{row["kernel"]} {int(row["block_x"])}x{int(row["block_y"])}'


def ordered_configurations(data: pd.DataFrame) -> list[str]:
    order = []
    for kernel in ["Naive", "Tiled Runtime", "Tiled Constant"]:
        for block_size in [8, 16, 32]:
            name = f"{kernel} {block_size}x{block_size}"
            if name in set(data["configuration"]):
                order.append(name)

    if "cuBLAS" in set(data["configuration"]):
        order.append("cuBLAS")

    return order


def plot_gflops(data: pd.DataFrame, configurations: list[str]) -> None:
    plt.figure(figsize=(11, 7))

    for configuration in configurations:
        subset = (
            data[data["configuration"] == configuration]
            .sort_values("M")
        )
        plt.plot(
            subset["M"],
            subset["gflops"],
            marker="o",
            label=configuration,
        )

    plt.xlabel("Square matrix size (M = N = K)")
    plt.ylabel("GFLOPS")
    plt.title("CUDA MatMul Block/Tile Size Sweep")
    plt.xticks(sorted(data["M"].unique()))
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "matmul_gflops.png", dpi=200)
    plt.close()


def plot_relative_to_cublas(
    data: pd.DataFrame,
    configurations: list[str],
) -> None:
    plt.figure(figsize=(11, 7))

    for configuration in configurations:
        subset = (
            data[data["configuration"] == configuration]
            .sort_values("M")
        )
        plt.plot(
            subset["M"],
            subset["relative_to_cublas"] * 100.0,
            marker="o",
            label=configuration,
        )

    plt.xlabel("Square matrix size (M = N = K)")
    plt.ylabel("Performance relative to cuBLAS (%)")
    plt.title("MatMul Performance Relative to cuBLAS")
    plt.xticks(sorted(data["M"].unique()))
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "matmul_relative_to_cublas.png",
        dpi=200,
    )
    plt.close()


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"{CSV_PATH} was not found. Run the CUDA benchmark first."
        )

    data = pd.read_csv(CSV_PATH)

    required_columns = {
        "M",
        "N",
        "K",
        "kernel",
        "block_x",
        "block_y",
        "gflops",
        "relative_to_cublas",
    }
    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {sorted(missing)}"
        )

    data["configuration"] = data.apply(
        configuration_name,
        axis=1,
    )
    configurations = ordered_configurations(data)

    plot_gflops(data, configurations)
    plot_relative_to_cublas(data, configurations)

    print("Generated:")
    print(OUTPUT_DIR / "matmul_gflops.png")
    print(OUTPUT_DIR / "matmul_relative_to_cublas.png")


if __name__ == "__main__":
    main()