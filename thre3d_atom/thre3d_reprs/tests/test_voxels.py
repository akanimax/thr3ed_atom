import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from thre3d_atom.rendering.volumetric.utils.misc import cast_rays, flatten_rays
from thre3d_atom.thre3d_reprs.renderers import (
    render_sh_voxel_grid,
    SHVoxGridRenderConfig,
)
from thre3d_atom.thre3d_reprs.voxels import (
    VoxelGrid,
    VoxelSize,
)
from thre3d_atom.utils.constants import EXTRA_ACCUMULATED_WEIGHTS
from thre3d_atom.utils.imaging_utils import (
    pose_spherical,
    CameraIntrinsics,
    CameraBounds,
    postprocess_depth_map,
)


def _plot_all_cube_sides(
    voxel_grid: VoxelGrid,
    camera_intrinsics: CameraIntrinsics,  # shouldn't be too high
    num_samples_per_ray: int,
    camera_bounds: CameraBounds,
    radius: float,
    device: torch.device,
) -> float:
    height, width, _ = camera_intrinsics

    # render all 6 sides of the cube:
    render_times = []
    for side, (yaw, pitch) in enumerate(
        ((0, 0), (90, 0), (180, 0), (270, 0), (0, -90), (0, 90)), 1
    ):
        camera_pose = pose_spherical(yaw=yaw, pitch=pitch, radius=radius)
        rays = cast_rays(camera_intrinsics, camera_pose, device=device)

        # render the voxel grid:

        start_time = time.perf_counter()
        with torch.no_grad():
            rendered_output = render_sh_voxel_grid(
                voxel_grid=voxel_grid,
                rays=flatten_rays(rays),
                render_config=SHVoxGridRenderConfig(
                    num_samples_per_ray=num_samples_per_ray,
                    camera_bounds=camera_bounds,
                    white_bkgd=True,
                ),
            )
        end_time = time.perf_counter()
        render_time = (end_time - start_time) * 1000  # ms
        render_times.append(render_time)

        # process the rendered output:
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
        fig.suptitle(f"side {side}")
        ax1.set_title("colour render")
        ax1.imshow(colour_render)
        ax2.set_title("depth render")
        ax2.imshow(depth_render)
        ax3.set_title("acc render")
        ax3.imshow(acc_render, cmap="gray")

    plt.show()
    render_time = np.mean(render_times).item()
    return render_time


def test_trilinear_interpolation_single_cube(device: torch.device) -> None:
    # fmt: off
    voxel_grid = VoxelGrid(
        densities=torch.tensor(
            [
                np.random.uniform(-10.0, 10.0, 1).item(),
                np.random.uniform(-10.0, 10.0, 1).item(),
                np.random.uniform(-10.0, 10.0, 1).item(),
                np.random.uniform(-10.0, 10.0, 1).item(),
                np.random.uniform(-10.0, 10.0, 1).item(),
                np.random.uniform(-10.0, 10.0, 1).item(),
                np.random.uniform(-10.0, 10.0, 1).item(),
                np.random.uniform(-10.0, 10.0, 1).item(),
            ],
            device=device,
            dtype=torch.float32,
        ).reshape(2, 2, 2, 1),
        features=torch.tensor(
            [
                10.0, -10.0, -10.0,
                -10.0, 10.0, -10.0,
                -10.0, -10.0, 10.0,
                10.0, 10.0, -10.0,
                -10.0, 10.0, 10.0,
                10.0, -10.0, 10.0,
                10.0, 10.0, 10.0,
                -10.0, -10.0, -10.0,
            ],
            device=device,
            dtype=torch.float32,
        ).reshape(2, 2, 2, 3),
        voxel_size=VoxelSize(2, 2, 2),
        density_preactivation=torch.nn.Identity(),
        density_postactivation=torch.nn.ReLU()
    )
    # fmt: on

    print(voxel_grid)

    _plot_all_cube_sides(
        voxel_grid,
        CameraIntrinsics(200, 200, 240),
        num_samples_per_ray=512,
        camera_bounds=CameraBounds(5.0, 18.0),
        radius=10.0,
        device=device,
    )


def test_render_speed(device: torch.device) -> None:
    # GIVEN: The following configuration:
    grid_size, num_samples_per_ray = 128, 256
    camera_intrinsics = CameraIntrinsics(400, 400, 512.0)
    num_samples_per_ray = 256
    camera_bounds = CameraBounds(0.5, 8.0)
    n_times = 100  # number of runs over which time is averaged

    # fmt: off
    densities = torch.empty((grid_size, grid_size, grid_size, 1), device=device)
    densities = torch.nn.init.uniform_(densities, -10.0, 10.0)
    features = torch.empty((grid_size, grid_size, grid_size, 3), device=device)
    features = torch.nn.init.uniform_(features, -10.0, 10.0)
    voxel_grid = VoxelGrid(
        densities=densities,
        features=features,
        voxel_size=VoxelSize(2.0 / 128, 2.0 / 128, 2.0 / 128),
    )
    # fmt: on

    print(voxel_grid)

    # render the voxel grid:
    print(f"rendering images {n_times} times ...")
    render_times = []
    for _ in tqdm(range(n_times)):
        # sample a random pose:
        yaw, pitch = np.random.uniform(0.0, 360.0), np.random.uniform(0.0, 180.0)
        radius = np.random.uniform(4.0, 5.0)
        camera_pose = pose_spherical(yaw=yaw, pitch=pitch, radius=radius)
        rays = cast_rays(camera_intrinsics, camera_pose, device=device)
        flat_rays = flatten_rays(rays)

        start_time = time.perf_counter()
        with torch.no_grad():
            rendered_output = render_sh_voxel_grid(
                voxel_grid=voxel_grid,
                rays=flat_rays,
                render_config=SHVoxGridRenderConfig(
                    num_samples_per_ray=num_samples_per_ray,
                    camera_bounds=camera_bounds,
                    white_bkgd=True,
                ),
                parallel_points_chunk_size=None,
            )
        end_time = time.perf_counter()
        render_time = (end_time - start_time) * 1000  # ms
        render_times.append(render_time)

    # plot the final render for visual inspection :D
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

    avg_render_time = np.mean(render_times).item()
    print(f"total time taken for rendering: {avg_render_time} ms")
