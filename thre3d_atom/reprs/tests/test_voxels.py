import torch
import numpy as np
import matplotlib.pyplot as plt
from thre3d_atom.rendering.volumetric.utils.misc import cast_rays, flatten_rays
from thre3d_atom.reprs.renderers import (
    render_sh_voxel_grid,
    ProbingConfig,
    AccumulationConfig,
)
from thre3d_atom.reprs.voxels import (
    VoxelGrid,
    VoxelSize,
)
from thre3d_atom.utils.imaging_utils import (
    pose_spherical,
    CameraIntrinsics,
    CameraBounds,
)


def _plot_all_cube_sides(
    voxel_grid: VoxelGrid,
    camera_intrinsics: CameraIntrinsics,  # shouldn't be too high
    num_samples_per_ray: int,
    camera_bounds: CameraBounds,
    radius: float,
    device: torch.device,
) -> None:
    height, width, _ = camera_intrinsics

    # render all 6 sides of the cube:
    for side, (yaw, pitch) in enumerate(
        ((0, 0), (90, 0), (180, 0), (270, 0), (0, -90), (0, 90)), 1
    ):
        camera_pose = pose_spherical(yaw=yaw, pitch=pitch, radius=radius)
        rays = cast_rays(camera_intrinsics, camera_pose, device=device)

        # render the voxel grid:
        rendered_output = render_sh_voxel_grid(
            voxel_grid=voxel_grid,
            rays=flatten_rays(rays),
            probing_config=ProbingConfig(
                num_samples_per_ray=num_samples_per_ray, camera_bounds=camera_bounds
            ),
            accumulation_confing=AccumulationConfig(),
        )

        # show the rendered_colour:
        plt.figure(f"side {side}")
        plt.imshow(
            rendered_output.colour.reshape(height, width, 3).detach().cpu().numpy()
        )

    plt.show()


def test_trilinear_interpolation_single_cube(device: torch.device) -> None:
    # fmt: off
    voxel_grid = VoxelGrid(
        densities=torch.tensor(
            [
                np.random.uniform(-1.0, 1.0, 1).item(),
                np.random.uniform(-1.0, 1.0, 1).item(),
                np.random.uniform(-1.0, 1.0, 1).item(),
                np.random.uniform(-1.0, 1.0, 1).item(),
                np.random.uniform(-1.0, 1.0, 1).item(),
                np.random.uniform(-1.0, 1.0, 1).item(),
                np.random.uniform(-1.0, 1.0, 1).item(),
                np.random.uniform(-1.0, 1.0, 1).item(),
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
    )
    # fmt: on

    print(voxel_grid)

    _plot_all_cube_sides(
        voxel_grid,
        CameraIntrinsics(100, 100, 120),
        num_samples_per_ray=128,
        camera_bounds=CameraBounds(5.0, 18.0),
        radius=10.0,
        device=device,
    )
