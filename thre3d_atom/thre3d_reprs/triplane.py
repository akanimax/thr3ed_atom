import torch

from torch import Tensor
from torch.nn import Module
from torch.nn.functional import grid_sample
from thre3d_atom.thre3d_reprs.utils import (
    Thre3dObjectLocation,
    AxisAlignedBoundingBox,
    normalize_points,
)
from typing import Callable, Dict, Any

from thre3d_atom.utils.constants import NUM_COORD_DIMENSIONS


class TriplaneStruct(Module):
    def __init__(
        self,
        features: Tensor,
        size: float = 1.0,
        location: Thre3dObjectLocation = Thre3dObjectLocation(),
        feature_preactivation: Callable[[Tensor], Tensor] = torch.nn.Identity(),
        tunable: bool = False,
    ) -> None:
        """
        Defines a Voxel-Grid denoting a 3D-volume. To obtain features of a particular point inside
        the volume, we obtain continuous features by doing trilinear interpolation.
        Args:
            features: Tensor of shape [3 (NUM_COORD_DIMENSIONS) x H x W x F]
                      corresponds to the features on the thre3 planar grids used to represent the Tri-plane
            size: Size of side length of the AABB.
            location: Location of the triplane object in the 3D space
            feature_preactivation: the activation to be applied to the features before interpolating.
            tunable: Whether to treat the densities and features Tensors as tunable (trainable) parameters
        """
        # assertions about the inputs:
        assert (
            len(features.shape) == 4
        ), f"features should be of shape [{NUM_COORD_DIMENSIONS} x H x W x F] as opposed to ({features.shape})"
        assert (
            features.shape[1] == features.shape[2]
        ), f"Note that the feature planes should be square. I.e. H == W"

        super().__init__()

        # initialize the state of the object
        self._features = features
        self._size = size
        self._location = location
        self._feature_preactivation = feature_preactivation
        self._tunable = tunable

        if tunable:
            self._features = torch.nn.Parameter(self._features)
        self._device = self._features.device

        # setup the bounding box planes:
        self._aabb = self._setup_bounding_box_planes()

    def _setup_bounding_box_planes(self) -> AxisAlignedBoundingBox:
        # compute half grid dimensions
        half_width = half_depth = half_height = self._size / 2.0

        # compute the AABB around the tri-plane structure:
        # fmt: off
        return AxisAlignedBoundingBox(
            x_range=(self._location.x_coord - half_width, self._location.x_coord + half_width),
            y_range=(self._location.y_coord - half_depth, self._location.y_coord + half_depth),
            z_range=(self._location.z_coord - half_height, self._location.z_coord + half_height),
        )
        # fmt: on

    def get_save_config_dict(self) -> Dict[str, Any]:
        return {
            "size": self._size,
            "location": self._location,
            "feature_preactivation": self._feature_preactivation,
            "tunable": self._tunable,
        }

    @property
    def aabb(self) -> AxisAlignedBoundingBox:
        return self._aabb

    @property
    def features(self) -> Tensor:
        return self._features

    def forward(self, points: Tensor) -> Tensor:
        """
        computes the features at the requested 3D points by interpolating along the tri-planes
        Args:
            points: Tensor of shape [N x 3 (NUM_COORD_DIMENSIONS)]
        Returns: Tensor of shape [N x 3 * <features> (features concatenated from the three planes)]
        """

        # normalize the points based on the aabb so that most of them lie in the
        # range [-1, 1]
        normalized_points = normalize_points(self._aabb, points)

        xy_feature_plane = self._features.permute(0, 3, 1, 2)[0:1, ...]
        yz_feature_plane = self._features.permute(0, 3, 1, 2)[1:2, ...]
        xz_feature_plane = self._features.permute(0, 3, 1, 2)[2:, ...]

        xy_norm_points = normalized_points[..., [0, 1]]
        yz_norm_points = normalized_points[..., [1, 2]]
        xz_norm_points = normalized_points[..., [0, 2]]

        xy_features = (
            grid_sample(
                xy_feature_plane,
                xy_norm_points[None, None, ...],
                padding_mode="zeros",
                align_corners=False,
            )
            .permute(3, 1, 0, 2)
            .squeeze()
        )
        yz_features = (
            grid_sample(
                yz_feature_plane,
                yz_norm_points[None, None, ...],
                padding_mode="zeros",
                align_corners=False,
            )
            .permute(3, 1, 0, 2)
            .squeeze()
        )
        xz_features = (
            grid_sample(
                xz_feature_plane,
                xz_norm_points[None, None, ...],
                padding_mode="zeros",
                align_corners=False,
            )
            .permute(3, 1, 0, 2)
            .squeeze()
        )

        output_features = torch.cat([xy_features, yz_features, xz_features], dim=-1)

        return output_features
