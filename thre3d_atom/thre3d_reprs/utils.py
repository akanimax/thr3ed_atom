import torch

from torch import Tensor
from typing import NamedTuple, Tuple

from thre3d_atom.utils.imaging_utils import adjust_dynamic_range


class Thre3dObjectLocation(NamedTuple):
    """indicates where the Voxel-Grid is located in World Coordinate System
    i.e. indicates where the centre of the grid is located in the World
    The Grid is always assumed to be axis aligned"""

    x_coord: float = 0.0
    y_coord: float = 0.0
    z_coord: float = 0.0


class AxisAlignedBoundingBox(NamedTuple):
    """defines a tight axis-aligned cage around a 3d-object"""

    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    z_range: Tuple[float, float]


def normalize_points(aabb: AxisAlignedBoundingBox, points: Tensor) -> Tensor:
    normalized_points = torch.empty_like(points, device=points.device)
    for coordinate_index, coordinate_range in enumerate(aabb):
        normalized_points[:, coordinate_index] = adjust_dynamic_range(
            points[:, coordinate_index],
            drange_in=coordinate_range,
            drange_out=(-1.0, 1.0),
            slack=True,
        )
    return normalized_points


def test_inside_volume(aabb: AxisAlignedBoundingBox, points: Tensor) -> Tensor:
    """
    tests whether the points are inside the AABB or not
    Args:
        points: Tensor of shape [N x 3 (NUM_COORD_DIMENSIONS)]
    Returns: Tensor of shape [N x 1]  (boolean)
    """
    return torch.logical_and(
        torch.logical_and(
            torch.logical_and(
                points[..., 0:1] > aabb.x_range[0],
                points[..., 0:1] < aabb.x_range[1],
            ),
            torch.logical_and(
                points[..., 1:2] > aabb.y_range[0],
                points[..., 1:2] < aabb.y_range[1],
            ),
        ),
        torch.logical_and(
            points[..., 2:] > aabb.z_range[0],
            points[..., 2:] < aabb.z_range[1],
        ),
    )
