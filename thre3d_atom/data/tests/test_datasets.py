from pathlib import Path

import numpy as np
import torch
import torch.utils.data as torch_data

from thre3d_atom.data.datasets import PosedImagesDataset
from thre3d_atom.utils.constants import NUM_COLOUR_CHANNELS, SEED


def test_posed_images_dataset(data_path: Path) -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    # GIVEN: images directory and corresponding camera parameters.
    # Note that we use the `train` dir and `train_camera_params.json` for this test
    images_dir = data_path / "train"
    camera_params_json = data_path / "train_camera_params.json"
    batch_size = 3

    # WHEN: creating a PosedImageDataset and extracting a random batch of samples
    posed_images_dataset = PosedImagesDataset(images_dir, camera_params_json)
    data_loader = torch_data.DataLoader(
        posed_images_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    image_batch, pose_batch = next(iter(data_loader))

    # THEN:
    # 1. shape of loaded images must match camera intrinsics
    # 2. loaded images must be in the proper numeric range
    # 3. shape of pose matrices must be [3, 4]
    # 4. rotation part of the pose matrices must have a determinant of 1.0 ideally, but numerically around 0.9
    # 5. the hemispherical radius should have the known value of 4.031128406524658, for the scene which we are
    # testing.
    assert image_batch.shape == (
        batch_size,
        NUM_COLOUR_CHANNELS,
        posed_images_dataset.camera_intrinsics.height,
        posed_images_dataset.camera_intrinsics.width,
    )
    assert -1 <= image_batch.min() <= image_batch.max() <= 1
    assert pose_batch.shape == (batch_size, 3, 4)  # shape of pose matrix
    for pose in pose_batch:
        np.testing.assert_almost_equal(np.linalg.det(pose[:, :-1]), 0.9, decimal=1)
    np.testing.assert_almost_equal(
        posed_images_dataset.get_hemispherical_radius_estimate(),
        4.031128406524658,
        decimal=3,
    )
