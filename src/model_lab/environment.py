"""Runtime information for the local Python and PyTorch installation."""

import platform
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class EnvironmentInfo:
    """The small set of runtime facts needed by the environment check."""

    python_version: str
    pytorch_version: str
    cuda_available: bool
    device: str
    gpu_name: str | None


def get_environment_info() -> EnvironmentInfo:
    """Return local Python, PyTorch, and CUDA availability information."""

    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name() if cuda_available else None
    return EnvironmentInfo(
        python_version=platform.python_version(),
        pytorch_version=torch.__version__.split("+")[0],
        cuda_available=cuda_available,
        device="cuda" if cuda_available else "cpu",
        gpu_name=gpu_name,
    )
