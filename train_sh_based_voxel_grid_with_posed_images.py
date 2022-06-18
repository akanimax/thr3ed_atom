from typing import NamedTuple

import click
import torch
from easydict import EasyDict
from torch.backends import cudnn

from thre3d_atom.data.datasets import PosedImagesDataset
from thre3d_atom.modules.trainers import train_sh_vox_grid_vol_mod_with_posed_images
from thre3d_atom.modules.volumetric_model import VolumetricModel
from thre3d_atom.thre3d_reprs.renderers import (
    render_sh_voxel_grid,
    SHVoxGridRenderConfig,
)
from thre3d_atom.thre3d_reprs.voxels import VoxelGrid, VoxelSize
# Age-old custom code for fast training :)
from thre3d_atom.utils.logging import log
from thre3d_atom.utils.misc import log_config_to_disk

cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GridDims(NamedTuple):
    x_dims: int
    y_dims: int
    z_dims: int


# -------------------------------------------------------------------------------------
#  Command line configuration for the script                                          |
# -------------------------------------------------------------------------------------
# fmt: off
@click.command()

# Required arguments:
@click.option("-d", "--data_path", type=click.Path(file_okay=False, dir_okay=True),
              required=True, help="path to the input dataset")
@click.option("-o", "--output_path", type=click.Path(file_okay=False, dir_okay=True),
              required=True, help="path for training output")

# Input dataset related arguments:
@click.option("--data_downsample_factor", type=click.FloatRange(min=1.0),
              default=1.0, help="downscale factor for the input images if needed", show_default=True)

# Voxel-grid related arguments:
@click.option("--grid_dims", type=click.INT, nargs=3, default=(128, 128, 128),
              help="dimensions (#voxels) of the grid along x, y and z axes", show_default=True)
@click.option("--grid_location", type=click.FLOAT, nargs=3, default=(0.0, 0.0, 0.0),
              help="dimensions (#voxels) of the grid along x, y and z axes", show_default=True)

# fmt: on
# -------------------------------------------------------------------------------------
def main(**kwargs) -> None:
    config = EasyDict(kwargs)

    log.info("logging configuration file ...")
    log_config_to_disk(config, config.output_path)

    exit()

    grid_world_size = 3.0
    grid_size, num_samples_per_ray = 128, 256
    white_bkgd = True

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
        ray_batch_size=16384,
        num_iterations_per_stage=200,
        num_workers=0,
    )


if __name__ == "__main__":
    main()
