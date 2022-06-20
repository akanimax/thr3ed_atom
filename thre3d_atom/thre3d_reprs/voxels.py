""" manually written sort-of-low-level implementation for voxel-based 3D volumetric representations """
from typing import Tuple, NamedTuple, Optional, Callable, Dict, Any

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import grid_sample, interpolate

from thre3d_atom.thre3d_reprs.constants import (
    THRE3D_REPR,
    STATE_DICT,
    u_DENSITIES,
    u_FEATURES,
    CONFIG_DICT,
)
from thre3d_atom.utils.imaging_utils import adjust_dynamic_range


class VoxelSize(NamedTuple):
    """lengths of a single voxel's edges in the x, y and z dimensions
    allows for the possibility of anisotropic voxels"""

    x_size: float = 1.0
    y_size: float = 1.0
    z_size: float = 1.0


class VoxelGridLocation(NamedTuple):
    """indicates where the Voxel-Grid is located in World Coordinate System
    i.e. indicates where the centre of the grid is located in the World
    The Grid is always assumed to be axis aligned"""

    x_coord: float = 0.0
    y_coord: float = 0.0
    z_coord: float = 0.0


class AxisAlignedBoundingBox(NamedTuple):
    """defines an axis-aligned voxel grid's spatial extent"""

    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    z_range: Tuple[float, float]


class VoxelGrid(Module):
    def __init__(
        self,
        # grid values:
        densities: Tensor,
        features: Tensor,
        # grid coordinate-space properties:
        voxel_size: VoxelSize,
        grid_location: Optional[VoxelGridLocation] = VoxelGridLocation(),
        # density activations:
        density_preactivation: Callable[[Tensor], Tensor] = torch.abs,
        density_postactivation: Callable[[Tensor], Tensor] = torch.nn.Identity(),
        # feature activations:
        feature_preactivation: Callable[[Tensor], Tensor] = torch.nn.Identity(),
        feature_postactivation: Callable[[Tensor], Tensor] = torch.nn.Identity(),
        # radiance function / transfer function:
        radiance_transfer_function: Callable[[Tensor, Tensor], Tensor] = None,
        expected_density_scale: float = 1.0,
        tunable: bool = False,
    ):
        """
        Defines a Voxel-Grid denoting a 3D-volume. To obtain features of a particular point inside
        the volume, we obtain continuous features by doing trilinear interpolation.
        Args:
            densities: Tensor of shape [W x D x H x 1] corresponds to the volumetric density in the scene
            features: Tensor of shape [W x D x H x F] corresponds to the features on the grid-vertices
            voxel_size: Size of each voxel. (could be different in different axis (x, y, z))
            grid_location: Location of the center of the grid
            density_preactivation: the activation to be applied to the raw density values before interpolating.
            density_postactivation: the activation to be applied to the raw density values after interpolating.
            feature_preactivation: the activation to be applied to the features before interpolating.
            feature_postactivation: the activation to be applied to the features after interpolating.
            radiance_transfer_function: the function that maps (can map)
                                        the interpolated features to RGB (radiance) values
            expected_density_scale: expected scale of the raw-density values. Defaults to a nice constant=100.0
            tunable: whether to treat the densities and features Tensors as tunable (trainable) parameters
        """
        # as usual start with assertions about the inputs:
        assert (
            len(densities.shape) == 4 and densities.shape[-1] == 1
        ), f"features should be of shape [W x D x H x 1] as opposed to ({features.shape})"
        assert (
            len(features.shape) == 4
        ), f"features should be of shape [W x D x H x F] as opposed to ({features.shape})"
        assert (
            densities.device == features.device
        ), f"densities and features are not on the same device :("

        super().__init__()

        # initialize the state of the object
        self._densities = densities
        self._features = features
        self._density_preactivation = density_preactivation
        self._density_postactivation = density_postactivation
        self._feature_preactivation = feature_preactivation
        self._feature_postactivation = feature_postactivation
        self._radiance_transfer_function = radiance_transfer_function
        self._grid_location = grid_location
        self._voxel_size = voxel_size
        self._expected_density_scale = expected_density_scale
        self._tunable = tunable

        if tunable:
            self._densities = torch.nn.Parameter(self._densities)
            self._features = torch.nn.Parameter(self._features)

        # either densities or features can be used:
        self._device = features.device

        # note the x, y and z conventions for the width (+ve right), depth (+ve inwards) and height (+ve up)
        self.width_x, self.depth_y, self.height_z = (
            self._features.shape[0],
            self._features.shape[1],
            self._features.shape[2],
        )

        # setup the bounding box planes
        self._aabb = self._setup_bounding_box_planes()

    @property
    def densities(self) -> Tensor:
        return self._densities

    @property
    def features(self) -> Tensor:
        return self._features

    @features.setter
    def features(self, features: Tensor) -> None:
        assert (
            features.shape == self._features.shape
        ), f"new features don't match original feature tensor's dimensions"
        if self._tunable and not isinstance(features, torch.nn.Parameter):
            self._features = torch.nn.Parameter(features)
        else:
            self._features = features

    @densities.setter
    def densities(self, densities: Tensor) -> None:
        assert (
            densities.shape == self._densities.shape
        ), f"new densities don't match original densities tensor's dimensions"
        if self._tunable and not isinstance(densities, torch.nn.Parameter):
            self._densities = torch.nn.Parameter(densities)
        else:
            self._densities = densities

    @property
    def aabb(self) -> AxisAlignedBoundingBox:
        return self._aabb

    @property
    def grid_dims(self) -> Tuple[int, int, int]:
        return self.width_x, self.depth_y, self.height_z

    @property
    def voxel_size(self) -> VoxelSize:
        return self._voxel_size

    @voxel_size.setter
    def voxel_size(self, voxel_size: VoxelSize) -> None:
        self._voxel_size = voxel_size

    def get_save_config_dict(self) -> Dict[str, Any]:
        save_config_dict = self.get_config_dict()
        save_config_dict.update({"voxel_size": self._voxel_size})
        return save_config_dict

    def get_config_dict(self) -> Dict[str, Any]:
        return {
            "grid_location": self._grid_location,
            "density_preactivation": self._density_preactivation,
            "density_postactivation": self._density_postactivation,
            "feature_preactivation": self._feature_preactivation,
            "feature_postactivation": self._feature_postactivation,
            "radiance_transfer_function": self._radiance_transfer_function,
            "expected_density_scale": self._expected_density_scale,
            "tunable": self._tunable,
        }

    def _setup_bounding_box_planes(self) -> AxisAlignedBoundingBox:
        # compute half grid dimensions
        half_width = (self.width_x * self._voxel_size.x_size) / 2
        half_depth = (self.depth_y * self._voxel_size.y_size) / 2
        half_height = (self.height_z * self._voxel_size.z_size) / 2

        # compute the AABB (bounding_box_planes)
        width_x_range = (
            self._grid_location.x_coord - half_width,
            self._grid_location.x_coord + half_width,
        )
        depth_y_range = (
            self._grid_location.y_coord - half_depth,
            self._grid_location.y_coord + half_depth,
        )
        height_z_range = (
            self._grid_location.z_coord - half_height,
            self._grid_location.z_coord + half_height,
        )

        # return the computed planes in the packed AABB datastructure:
        return AxisAlignedBoundingBox(
            x_range=width_x_range,
            y_range=depth_y_range,
            z_range=height_z_range,
        )

    def _normalize_points(self, points: Tensor) -> Tensor:
        normalized_points = torch.empty_like(points, device=points.device)
        for coordinate_index, coordinate_range in enumerate(self._aabb):
            normalized_points[:, coordinate_index] = adjust_dynamic_range(
                points[:, coordinate_index],
                drange_in=coordinate_range,
                drange_out=(-1.0, 1.0),
                slack=True,
            )
        return normalized_points

    def extra_repr(self) -> str:
        return (
            f"grid_dims: {(self.width_x, self.depth_y, self.height_z)}, "
            f"feature_dims: {self._features.shape[-1]}, "
            f"voxel_size: {self._voxel_size}, "
            f"grid_location: {self._grid_location}, "
            f"tunable: {self._tunable}"
        )

    def get_bounding_volume_vertices(self) -> Tensor:
        x_min, x_max = self._aabb.x_range
        y_min, y_max = self._aabb.y_range
        z_min, z_max = self._aabb.z_range
        return torch.tensor(
            [
                [x_min, y_min, z_min],
                [x_min, y_min, z_max],
                [x_min, y_max, z_min],
                [x_min, y_max, z_max],
                [x_max, y_min, z_min],
                [x_max, y_min, z_max],
                [x_max, y_max, z_min],
                [x_max, y_max, z_max],
            ],
            dtype=torch.float32,
        )

    def test_inside_volume(self, points: Tensor) -> Tensor:
        """
        tests whether the points are inside the AABB or not
        Args:
            points: Tensor of shape [N x 3 (NUM_COORD_DIMENSIONS)]
        Returns: Tensor of shape [N x 1]  (boolean)
        """
        return torch.logical_and(
            torch.logical_and(
                torch.logical_and(
                    points[..., 0:1] > self._aabb.x_range[0],
                    points[..., 0:1] < self._aabb.x_range[1],
                ),
                torch.logical_and(
                    points[..., 1:2] > self._aabb.y_range[0],
                    points[..., 1:2] < self._aabb.y_range[1],
                ),
            ),
            torch.logical_and(
                points[..., 2:] > self._aabb.z_range[0],
                points[..., 2:] < self._aabb.z_range[1],
            ),
        )

    def forward(self, points: Tensor, viewdirs: Optional[Tensor] = None) -> Tensor:
        """
        computes the features/radiance at the requested 3D points
        Args:
            points: Tensor of shape [N x 3 (NUM_COORD_DIMENSIONS)]
            viewdirs: Tensor of shape [N x 3 (NUM_COORD_DIMENSIONS)]
                      this tensor represents viewing directions in world-coordinate-system
        Returns: either Tensor of shape [N x <3 + 1> (NUM_COLOUR_CHANNELS + density)]
                 or of shape [N x <features + 1> (number of features + density)], depending upon
                 whether the `self._radiance_transfer_function` is None.
        """
        # obtain the range-normalized points for interpolation
        normalized_points = self._normalize_points(points)

        # interpolate and compute densities
        # Note the pre- and post-activations :)
        preactivated_densities = self._density_preactivation(
            self._densities * self._expected_density_scale
        )  # note the use of the expected density scale
        interpolated_densities = (
            grid_sample(
                # note the weird z, y, x convention of PyTorch's grid_sample.
                # reference ->
                # https://discuss.pytorch.org/t/surprising-convention-for-grid-sample-coordinates/79997/3
                preactivated_densities[None, ...].permute(0, 4, 3, 2, 1),
                normalized_points[None, None, None, ...],
                align_corners=False,
            )
            .permute(0, 2, 3, 4, 1)
            .squeeze()
        )[
            ..., None
        ]  # note this None is required because of the squeeze operation :sweat_smile:
        interpolated_densities = self._density_postactivation(interpolated_densities)

        # interpolate and compute features
        preactivated_features = self._feature_preactivation(self._features)
        interpolated_features = (
            grid_sample(
                preactivated_features[None, ...].permute(0, 4, 3, 2, 1),
                normalized_points[None, None, None, ...],
                align_corners=False,
            )
            .permute(0, 2, 3, 4, 1)
            .squeeze()
        )
        interpolated_features = self._feature_postactivation(interpolated_features)

        # apply the radiance transfer function if it is not None and if view-directions are available
        if self._radiance_transfer_function is not None and viewdirs is not None:
            interpolated_features = self._radiance_transfer_function(
                interpolated_features, viewdirs
            )

        # return a unified tensor containing interpolated features and densities
        return torch.cat([interpolated_features, interpolated_densities], dim=-1)


def scale_voxel_grid_with_required_output_size(
    voxel_grid: VoxelGrid, output_size: Tuple[int, int, int], mode: str = "trilinear"
) -> VoxelGrid:

    # extract relevant information from the original input voxel_grid:
    og_unified_feature_tensor = torch.cat(
        [voxel_grid.features, voxel_grid.densities], dim=-1
    )
    og_voxel_size = voxel_grid.voxel_size

    # compute the new features using pytorch's interpolate function
    new_features = interpolate(
        og_unified_feature_tensor.permute(3, 0, 1, 2)[None, ...],
        size=output_size,
        mode=mode,
        align_corners=False,  # never use align_corners=True :D
        recompute_scale_factor=False,  # this needs to be set for some reason, I can't remember :D
    )[0]
    new_features = new_features.permute(1, 2, 3, 0)

    # a paranoid check that the interpolated features have the exact same output_size as required
    assert new_features.shape[:-1] == output_size

    # new voxel size is also similarly scaled
    new_voxel_size = VoxelSize(
        (og_voxel_size.x_size * voxel_grid.width_x) / output_size[0],
        (og_voxel_size.y_size * voxel_grid.depth_y) / output_size[1],
        (og_voxel_size.z_size * voxel_grid.height_z) / output_size[2],
    )

    # create a new voxel_grid by cloning the input voxel_grid and update the newly scaled properties
    new_voxel_grid = VoxelGrid(
        densities=new_features[..., -1:],
        features=new_features[..., :-1],
        voxel_size=new_voxel_size,
        **voxel_grid.get_config_dict(),
    )

    # noinspection PyProtectedMember
    return new_voxel_grid


def create_voxel_grid_from_saved_info_dict(saved_info: Dict[str, Any]) -> VoxelGrid:
    densities = torch.empty_like(saved_info[THRE3D_REPR][STATE_DICT][u_DENSITIES])
    features = torch.empty_like(saved_info[THRE3D_REPR][STATE_DICT][u_FEATURES])
    voxel_grid = VoxelGrid(
        densities=densities, features=features, **saved_info[THRE3D_REPR][CONFIG_DICT]
    )
    voxel_grid.load_state_dict(saved_info[THRE3D_REPR][STATE_DICT])
    return voxel_grid
