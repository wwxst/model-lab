import subprocess
import sys
from pathlib import Path

import pytest
import torch

from model_lab.environment import EnvironmentInfo, get_environment_info


def test_model_lab_package_imports() -> None:
    import model_lab

    assert model_lab.__name__ == "model_lab"


def test_torch_creates_and_adds_cpu_tensors() -> None:
    left = torch.tensor([1.0, 2.0], device="cpu")
    right = torch.tensor([3.0, 4.0], device="cpu")

    result = left + right

    assert result.device.type == "cpu"
    assert result.tolist() == [4.0, 6.0]


def test_environment_info_has_expected_runtime_fields() -> None:
    info = get_environment_info()

    assert isinstance(info, EnvironmentInfo)
    assert info.python_version == sys.version.split()[0]
    assert info.pytorch_version == torch.__version__.split("+")[0]
    assert isinstance(info.cuda_available, bool)
    assert info.device in {"cpu", "cuda"}
    if info.cuda_available:
        assert info.device == "cuda"
        assert info.gpu_name
    else:
        assert info.device == "cpu"
        assert info.gpu_name is None


@pytest.mark.skipif(torch.cuda.is_available(), reason="covers CPU-only fallback")
def test_cpu_only_environment_is_valid() -> None:
    info = get_environment_info()

    assert info.cuda_available is False
    assert info.device == "cpu"
    assert info.gpu_name is None


def test_environment_check_script_has_clean_output() -> None:
    script = Path(__file__).parents[1] / "scripts" / "check_environment.py"

    result = subprocess.run(
        [sys.executable, str(script)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout.splitlines() == [
        f"Python: {sys.version.split()[0]}",
        f"PyTorch: {torch.__version__.split('+')[0]}",
        f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}",
        f"CUDA Available: {torch.cuda.is_available()}",
        "GPU Name: "
        f"{torch.cuda.get_device_name() if torch.cuda.is_available() else 'N/A'}",
    ]
