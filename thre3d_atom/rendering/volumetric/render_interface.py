import dataclasses
from typing import NamedTuple, Any, Dict, Callable, Optional

import torch
from torch import Tensor

from thre3d_atom.utils.constants import NUM_COORD_DIMENSIONS, NUM_COLOUR_CHANNELS
from thre3d_atom.utils.imaging_utils import CameraBounds

ExtraInfo = Dict[str, Any]  # Type for carrying/passing around extra information


@dataclasses.dataclass
class Rays:
    origins: Tensor  # shape [... x NUM_COORD_DIMENSIONS]
    directions: Tensor  # shape [... x NUM_COORD_DIMENSIONS]
    # Please note that these can be arranged in any form. Mostly, arranging the rays (origins + directions)
    # as image tensors is common :)

    def __post_init__(self):
        # make the required assertions for creating this data-type:
        assert (
            self.origins.shape == self.directions.shape
        ), f"ray-origins and ray-directions are incompatible :("
        assert (
            self.origins.shape[-1] == self.directions.shape[-1] == NUM_COORD_DIMENSIONS
        ), (
            f"Sorry, we only support 3D coordinate-spaces at the moment. "
            f"Please cast your rays in 3 dimensions only :D"
        )

    def __getitem__(self, item) -> Any:
        """This is overridden to allow indexing and slicing of rays"""
        return Rays(
            origins=self.origins[item, :],
            directions=self.directions[item, :],
        )

    def __len__(self) -> int:
        return len(self.origins)

    def to(self, device: torch.device) -> Any:
        """shorthand to move a bunch of rays around GPU and CPU"""
        return Rays(self.origins.to(device), self.directions.to(device))


@dataclasses.dataclass
class RenderOut:
    colour: Tensor  # shape [... x NUM_COLOUR_CHANNELS]
    depth: Tensor  # shape [... x 1]
    extra: Optional[ExtraInfo] = None  # extra information

    def __post_init__(self):
        # make the required assertions for creating this data-type:
        assert (
            self.colour.shape[:-1] == self.depth.shape[:-1]
        ), f"rendered colour maps and depth maps are shape-incompatible"
        assert (
            self.colour.shape[-1] == NUM_COLOUR_CHANNELS
        ), f"Sorry, spectral rendering is not supported atm. Only RGB colours are possible"
        assert (
            self.depth.shape[-1] == 1
        ), f"Sorry, depth map should only have 1 dimensional data channel"

        # convert extra into an empty dict if it's defaulted to None:
        if self.extra is None:
            self.extra = {}

    def detach(self) -> Any:
        """shorthand to detach all the rendered output tensors from pytorch-diff-graph"""
        return RenderOut(
            colour=self.colour.detach(),
            depth=self.depth.detach(),
            extra={key: value.detach() for key, value in self.extra.items()},
        )

    def to(self, device: torch.device) -> Any:
        """shorthand to move the rendered output around GPU and CPU"""
        return RenderOut(
            colour=self.colour.to(device),
            depth=self.depth.to(device),
            extra={key: value.to(device) for key, value in self.extra.items()},
        )


class SampledPointsOnRays(NamedTuple):
    # Note the dimensions as follows. These are expected as it is
    # by the following render procedure interface
    points: Tensor  # shape [N x num_samples x NUM_COORD_DIMENSIONS]
    depths: Tensor  # shape [N x num_samples]


# The dataType is exactly the same, but this renaming improves readability
ProcessedPointsOnRays = SampledPointsOnRays

# Functional-components of the render-interface are defined as below:
# the int below is the number of sampled points
RaySamplerFunction = Callable[[Rays, CameraBounds, int], SampledPointsOnRays]
PointProcessorFunction = Callable[[SampledPointsOnRays, Rays], ProcessedPointsOnRays]
AccumulatorFunction = Callable[[ProcessedPointsOnRays, Rays], RenderOut]


def render(
    rays: Rays,
    camera_bounds: CameraBounds,
    num_samples: int,
    sampler_fn: RaySamplerFunction,
    point_processor_fn: PointProcessorFunction,
    accumulator_fn: AccumulatorFunction,
) -> RenderOut:
    """
    Defines the overall flow of execution of the differentiable
    volumetric rendering process. Please note that this interface has been
    designed to allow enough flexibility in the rendering process.
    Note that this render interface strongly assumes ``FLAT RAYS''.
    This is done to make the interface consistent and debugging easier
    Args:
        rays: virtual casted rays (origins and directions). Aka, ray-marching probes
        camera_bounds: SceneBounds (near and far) of the scene being rendered
        num_samples: number of points sampled on the rays
        sampler_fn: function that maps from casted rays to sampled points on the rays.
        point_processor_fn: function to process the points on the rays.
        accumulator_fn: function that accumulates the processed points into rendered
                        output.
    Returns: rendered output (rgb, depth and extra information)
    """
    assert (
        len(rays.origins.shape) == len(rays.directions.shape) == 2
    ), f"Please note that the RENDER interface only works with FLAT RAYS!"

    sampled_points = sampler_fn(rays, camera_bounds, num_samples)
    processed_points = point_processor_fn(sampled_points, rays)
    rendered_output = accumulator_fn(processed_points, rays)
    return rendered_output
