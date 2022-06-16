from pathlib import Path

import torch

from thre3d_atom.data.datasets import PosedImagesDataset
from thre3d_atom.modules.trainers import train_sh_vox_grid_vol_mod_with_posed_images
from thre3d_atom.modules.volumetric_model import VolumetricModel
from thre3d_atom.thre3d_reprs.renderers import (
    render_sh_voxel_grid,
    SHVoxGridRenderConfig,
)
from thre3d_atom.thre3d_reprs.voxels import VoxelGrid, VoxelSize

# -------------------------------------------------------------------------------------
#  Tweakable parameters                                                               |
# -------------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_path = Path("data/3d_scenes/nerf_synthetic/hotdog")
output_path = Path("logs/hotdog")

grid_world_size = 3.0
grid_size, num_samples_per_ray = 128, 256
white_bkgd = True
# -------------------------------------------------------------------------------------

# create a dataset:
train_dataset = PosedImagesDataset(
    images_dir=data_path / "train",
    camera_params_json=data_path / "train_camera_params.json",
    downsample_factor=2,
    rgba_white_bkgd=True,
)


# construct the VoxelGrid Repr:
# fmt: off
densities = torch.empty((grid_size, grid_size, grid_size, 1), device=device)
torch.nn.init.uniform_(densities, -1.0, 1.0)
features = torch.empty((grid_size, grid_size, grid_size, 27), device=device)
torch.nn.init.uniform_(features, -1.0, 1.0)
voxel_grid = VoxelGrid(
    densities=densities,
    features=features,
    voxel_size=VoxelSize(grid_world_size / grid_size, grid_world_size / grid_size, grid_world_size / grid_size),
    density_preactivation=torch.nn.Identity(),
    density_postactivation=torch.nn.ReLU(),
    tunable=True,
)
# fmt: on

# set up a volumetricModel using this voxel-grid
# noinspection PyTypeChecker
vox_grid_vol_mod = VolumetricModel(
    thre3d_repr=voxel_grid,
    render_procedure=render_sh_voxel_grid,
    render_config=SHVoxGridRenderConfig(
        num_samples_per_ray=num_samples_per_ray,
        camera_bounds=train_dataset.camera_bounds,
        white_bkgd=white_bkgd,
    ),
    device=device,
)


# train the model:
train_sh_vox_grid_vol_mod_with_posed_images(
    vol_mod=vox_grid_vol_mod,
    train_dataset=train_dataset,
    output_dir=output_path,
    num_workers=0,
)
