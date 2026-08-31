"""Print the local Python and PyTorch runtime information."""

from model_lab.environment import get_environment_info


def main() -> None:
    info = get_environment_info()
    print(f"Python: {info.python_version}")
    print(f"PyTorch: {info.pytorch_version}")
    print(f"Device: {info.device}")
    print(f"CUDA Available: {info.cuda_available}")
    print(f"GPU Name: {info.gpu_name or 'N/A'}")


if __name__ == "__main__":
    main()
