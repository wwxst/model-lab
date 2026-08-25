"""Print the local Python and PyTorch runtime information."""

import warnings

warnings.filterwarnings(
    "ignore",
    message="Failed to initialize NumPy:.*",
    category=UserWarning,
    module=r"torch\._subclasses\.functional_tensor",
)


def main() -> None:
    from model_lab.environment import get_environment_info

    info = get_environment_info()
    print(f"Python: {info.python_version}")
    print(f"PyTorch: {info.pytorch_version}")
    print(f"Device: {info.device}")
    print(f"CUDA Available: {info.cuda_available}")
    print(f"GPU Name: {info.gpu_name or 'N/A'}")


if __name__ == "__main__":
    main()
