from pathlib import Path

import imageio
import numpy as np
import torch
from matplotlib import pyplot as plt

from thre3d_atom.modules.volumetric_model import VolumetricModel
from thre3d_atom.thre3d_reprs.renderers import (
    render_sh_voxel_grid,
    SHVoxGridRenderConfig,
)
from thre3d_atom.thre3d_reprs.voxels import VoxelGrid, VoxelSize
from thre3d_atom.utils.constants import EXTRA_ACCUMULATED_WEIGHTS
from thre3d_atom.utils.imaging_utils import (
    CameraIntrinsics,
    CameraBounds,
    pose_spherical,
    postprocess_depth_map,
    get_thre360_animation_poses,
)
from thre3d_atom.visualizations.animations import (
    render_camera_path_for_volumetric_model,
)


def get_default_volumetric_model(
    camera_bounds: CameraBounds, device: torch.device
) -> VolumetricModel:
    # GIVEN: The following configuration:
    grid_size, num_samples_per_ray = 16, 256
    num_samples_per_ray = 256
    white_bkgd = True

    # construct the VoxelGrid Repr:
    # fmt: off
    densities = torch.empty((grid_size, grid_size, grid_size, 1), device=device)
    densities = torch.nn.init.uniform_(densities, -10.0, 10.0)
    features = torch.empty((grid_size, grid_size, grid_size, 27), device=device)
    features = torch.nn.init.uniform_(features, -10.0, 10.0)
    voxel_grid = VoxelGrid(
        densities=densities,
        features=features,
        voxel_size=VoxelSize(2.0 / grid_size, 2.0 / grid_size, 2.0 / grid_size),
        density_preactivation=torch.nn.Identity(),
        density_postactivation=torch.nn.ReLU()
    )
    # fmt: on

    # set up a volumetricModel using this voxel-grid
    # noinspection PyTypeChecker
    vox_grid_vol_mod = VolumetricModel(
        thre3d_repr=voxel_grid,
        render_procedure=render_sh_voxel_grid,
        render_config=SHVoxGridRenderConfig(
            num_samples_per_ray=num_samples_per_ray,
            camera_bounds=camera_bounds,
            white_bkgd=white_bkgd,
        ),
        device=device,
    )

    return vox_grid_vol_mod


def test_volumetric_model_render(device: torch.device) -> None:
    camera_bounds = CameraBounds(0.5, 8.0)
    vox_grid_vol_mod = get_default_volumetric_model(camera_bounds, device)

    camera_intrinsics = CameraIntrinsics(400, 400, 512.0)
    # construct a random camera pose:
    yaw, pitch = np.random.uniform(0.0, 360.0), np.random.uniform(0.0, 180.0)
    radius = np.random.uniform(4.0, 5.0)
    camera_pose = pose_spherical(yaw=yaw, pitch=pitch, radius=radius)

    # render the output:
    rendered_output = vox_grid_vol_mod.render(
        camera_pose, camera_intrinsics, verbose=True
    )

    # plot the final render for visual inspection :D
    # noinspection DuplicatedCode
    height, width, _ = camera_intrinsics
    # noinspection DuplicatedCode
    colour_render = (
        rendered_output.colour.reshape(height, width, 3).detach().cpu().numpy()
    )
    depth_render = (
        rendered_output.depth.reshape(height, width, 1).detach().cpu().numpy()
    )
    depth_render = postprocess_depth_map(depth_render, camera_bounds)
    acc_render = rendered_output.extra[EXTRA_ACCUMULATED_WEIGHTS]
    acc_render = acc_render.reshape(height, width, 1).detach().cpu().numpy()
    # show the rendered output:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.set_title("colour render")
    ax1.imshow(colour_render)
    ax2.set_title("depth render")
    ax2.imshow(depth_render)
    ax3.set_title("acc render")
    ax3.imshow(acc_render, cmap="gray")
    plt.show()


def test_volumetric_model_render_animation(device: torch.device) -> None:
    camera_bounds = CameraBounds(0.5, 8.0)
    vox_grid_vol_mod = get_default_volumetric_model(camera_bounds, device)
    hemispherical_radius, camera_pitch, num_poses = 4.0, 60.0, 42
    camera_intrinsics = CameraIntrinsics(400, 400, 512.0)

    # get render poses
    animation_poses = get_thre360_animation_poses(
        hemispherical_radius=hemispherical_radius,
        camera_pitch=camera_pitch,
        num_poses=num_poses,
    )

    animation = render_camera_path_for_volumetric_model(
        vox_grid_vol_mod, animation_poses, camera_intrinsics=camera_intrinsics
    )

    imageio.mimwrite(Path("~/test_animation.mp4"), animation)
