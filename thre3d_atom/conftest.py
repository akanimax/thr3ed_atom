from pathlib import Path

import numpy as np
import pytest
import torch

from thre3d_atom.utils.constants import SEED

project_root_path = Path(__file__).parent.parent.absolute()
test_scene = "hotdog"


@pytest.fixture
def data_path() -> Path:
    return project_root_path / "data/3d_scenes/nerf_synthetic" / test_scene


@pytest.fixture(autouse=True)
def execute_before_every_test():
    torch.manual_seed(SEED)
    np.random.seed(SEED)


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def batch_size() -> int:
    return 32


@pytest.fixture
def num_samples() -> int:
    return 64
