import math
from typing import NamedTuple, Tuple, Union, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from thre3d_atom.utils.constants import NUM_COLOUR_CHANNELS


# ----------------------------------------------------------------------------------
# Custom NamedTuples (types for easy-use)
# ----------------------------------------------------------------------------------


class CameraIntrinsics(NamedTuple):
    height: int
    width: int
    focal: float


class CameraPose(NamedTuple):
    rotation: np.array  # shape [3 x 3]
    translation: np.array  # shape [3 x 1]


class CameraBounds(NamedTuple):
    near: float
    far: float


# ----------------------------------------------------------------------------------
# Miscellaneous utility functions
# ----------------------------------------------------------------------------------


def mse2psnr(x):
    return -10.0 * math.log(x) / math.log(10.0) if x != 0.0 else math.inf


def to8b(x: np.array) -> np.array:
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)


def adjust_dynamic_range(
    data: Union[np.array, Tensor],
    drange_in: Tuple[float, float],
    drange_out: Tuple[float, float],
    slack: bool = False,
) -> np.array:
    """
    converts the data from the range `drange_in` into `drange_out`
    Args:
        data: input data array
        drange_in: data range [total_min_val, total_max_val]
        drange_out: output data range [min_val, max_val]
        slack: whether to cut some slack in range adjustment :D
    Returns: range_adjusted_data
    """
    if drange_in != drange_out:
        if slack:
            scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (
                np.float32(drange_in[1]) - np.float32(drange_in[0])
            )
            bias = np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale
            data = data * scale + bias
        else:
            old_min, old_max = np.float32(drange_in[0]), np.float32(drange_in[1])
            new_min, new_max = np.float32(drange_out[0]), np.float32(drange_out[1])
            data = (
                (data - old_min) / (old_max - old_min) * (new_max - new_min)
            ) + new_min
            data = data.clip(drange_out[0], drange_out[1])
    return data


def get_2d_coordinates(
    height: int, width: int, drange: Tuple[float, float] = (-1.0, 1.0)
) -> Tensor:
    range_a, range_b = drange
    return torch.stack(
        torch.meshgrid(
            torch.linspace(range_a, range_b, height, dtype=torch.float32),
            torch.linspace(range_a, range_b, width, dtype=torch.float32),
        ),
        dim=-1,
    )


# ----------------------------------------------------------------------------------
# visualization related utility functions
# ----------------------------------------------------------------------------------


def postprocess_depth_map(
    depth_map: np.array, camera_bounds: Optional[CameraBounds] = None
) -> np.array:
    if camera_bounds is None:
        depth_min, depth_max = depth_map.min(), depth_map.max()
    else:
        depth_min, depth_max = 0.0, camera_bounds.far

    # squeeze the depth_map's last dimension if it exists
    if len(depth_map.shape) == 3 and depth_map.shape[-1] == 1:
        depth_map = np.squeeze(depth_map, axis=-1)

    depth_map = adjust_dynamic_range(
        depth_map, drange_in=(depth_min, depth_max), drange_out=(0, 1), slack=False
    )
    colour_map = plt.get_cmap("turbo", lut=1024)
    return to8b(colour_map(depth_map))[..., :NUM_COLOUR_CHANNELS]


# ----------------------------------------------------------------------------------
# Camera intrinsics utility functions
# ----------------------------------------------------------------------------------


def scale_camera_intrinsics(
    camera_intrinsics: CameraIntrinsics, scale_factor: float = 1.0
) -> CameraIntrinsics:
    # note that height and width are integers while focal length is a float
    return CameraIntrinsics(
        height=int(np.ceil(camera_intrinsics.height * scale_factor)),
        width=int(np.ceil(camera_intrinsics.width * scale_factor)),
        focal=camera_intrinsics.focal * scale_factor,
    )


# ----------------------------------------------------------------------------------
# Camera extrinsics (Transform) utility functions
# ----------------------------------------------------------------------------------


def _translate_z(z: float, device=torch.device("cpu")) -> Tensor:
    return torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, z],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )


def _rotate_pitch(pitch: float, device=torch.device("cpu")) -> Tensor:
    return torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, np.cos(pitch), -np.sin(pitch), 0.0],
            [0.0, np.sin(pitch), np.cos(pitch), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )


def _rotate_yaw(yaw: float, device=torch.device("cpu")) -> Tensor:
    return torch.tensor(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0, 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )


def pose_spherical(
    yaw: float, pitch: float, radius: float, device=torch.device("cpu")
) -> CameraPose:
    c2w = _translate_z(radius, device)
    c2w = _rotate_pitch(pitch / 180.0 * np.pi, device) @ c2w
    c2w = _rotate_yaw(yaw / 180.0 * np.pi, device) @ c2w
    return CameraPose(rotation=c2w[:3, :3], translation=c2w[:3, 3:])


# ----------------------------------------------------------------------------------
