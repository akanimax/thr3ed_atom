import torch
from torch import Tensor

from thre3d_atom.rendering.volumetric.render_interface import Rays
from thre3d_atom.utils.constants import NUM_COORD_DIMENSIONS
from thre3d_atom.utils.imaging_utils import CameraIntrinsics, CameraPose


def cast_rays(
    camera_intrinsics: CameraIntrinsics,
    pose: CameraPose,
    device: torch.device = torch.device("cpu"),
) -> Rays:
    # convert the camera pose into tensors if they are numpy arrays
    if not (isinstance(pose.rotation, Tensor) and isinstance(pose.translation, Tensor)):
        rot = torch.from_numpy(pose.rotation)
        trans = torch.from_numpy(pose.translation)
        pose = CameraPose(rot, trans)
    # bring the pose on the requested device
    if not (pose.rotation.device == device and pose.translation.device == device):
        pose = CameraPose(pose.rotation.to(device), pose.translation.to(device))

    # cast the rays for the given CameraPose
    height, width, focal = camera_intrinsics
    # note the specific use of torch.float32. Which means, even if the poses have higher
    # precision (float64), the casted rays will have 32-bit precision only.
    x_coords, y_coords = torch.meshgrid(
        torch.linspace(0.5, width - 0.5, width, dtype=torch.float32, device=device),
        torch.linspace(0.5, height - 0.5, height, dtype=torch.float32, device=device),
    )
    # not that this transpose is needed because torch's meshgrid is in ij format
    # instead of numpy's xy format
    x_coords, y_coords = x_coords.T, y_coords.T

    dirs = torch.stack(
        [
            (x_coords - width * 0.5) / focal,
            -(y_coords - height * 0.5) / focal,
            -torch.ones_like(x_coords, device=device),
        ],
        -1,
    )

    rays_d = (pose.rotation @ dirs[..., None])[..., 0]
    rays_o = torch.broadcast_to(pose.translation.squeeze(), rays_d.shape)
    return Rays(rays_o, rays_d)


def flatten_rays(rays: Rays) -> Rays:
    return Rays(
        origins=rays.origins.reshape(-1, NUM_COORD_DIMENSIONS),
        directions=rays.directions.reshape(-1, NUM_COORD_DIMENSIONS),
    )
