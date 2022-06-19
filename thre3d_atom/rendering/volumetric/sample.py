from typing import Union, Tuple

import torch
from torch import Tensor

from thre3d_atom.rendering.volumetric.render_interface import (
    Rays,
    SampledPointsOnRays,
)
from thre3d_atom.thre3d_reprs.voxels import AxisAlignedBoundingBox
from thre3d_atom.utils.constants import ZERO_PLUS
from thre3d_atom.utils.imaging_utils import CameraBounds


def sample_uniform_points_on_rays(
    rays: Rays,
    bounds: Union[CameraBounds, Tensor],
    num_samples: int,
    perturb: bool = True,
    linear_disparity_sampling: bool = False,
) -> SampledPointsOnRays:
    """Most basic version of sampling uniform (possibly jittered) points on rays.
    Proper credit: A lot of the code in this function is reminiscent of the
    point sampling logic from OG NeRF https://github.com/bmild/nerf/blob/master/run_nerf.py#L48"""
    # use the device used by the input rays Tensor
    dtype, device = rays.origins.dtype, rays.origins.device

    # unpack origins and directions from the Rays
    rays_o, rays_d = rays.origins, rays.directions

    # flatten them if required:
    if len(rays_o.shape) > 2 or len(rays_d) > 2:
        rays_o = rays_o.reshape(-1, rays_o.shape[-1])
        rays_d = rays_d.reshape(-1, rays_d.shape[-1])
    num_rays = rays_o.shape[0]

    # handle cases where bounds are either CameraBounds or have per ray different values in a tensor
    if isinstance(bounds, CameraBounds):
        near = torch.tensor([bounds.near], dtype=dtype, device=device)
        far = torch.tensor([bounds.far], dtype=dtype, device=device)
        near, far = near.repeat([num_rays, 1]), far.repeat([num_rays, 1])
    else:
        near, far = bounds[:, :1], bounds[:, 1:]

    # very simple ray sampling logic :)
    t_vals = torch.linspace(0.0, 1.0, num_samples, dtype=dtype, device=device)
    t_vals = t_vals[None, ...]  # extra dim for compatibility with near and far tensors
    if linear_disparity_sampling:
        # sample in the inverse depth space. I.e. more samples towards the near bound
        # and exponentially fewer samples towards the far bound
        z_vals = 1.0 / (1.0 / (near + ZERO_PLUS) * (1.0 - t_vals) + 1.0 / far * t_vals)
    else:
        # sample in normal depth space uniformly
        z_vals = near * (1.0 - t_vals) + far * t_vals

    # perturb sampled points along each ray
    if perturb:
        # get intervals between samples
        mid_points = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper_points = torch.cat([mid_points, z_vals[..., -1:]], -1)
        lower_points = torch.cat([z_vals[..., :1], mid_points], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(*z_vals.shape, dtype=dtype, device=device)
        z_vals = lower_points + (upper_points - lower_points) * t_rand

    # geometrically compute points using the z_vals (ray-depths essentially)
    sampled_points = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    return SampledPointsOnRays(sampled_points, z_vals)


def _ray_aabb_intersection(
    rays: Rays, bounds: CameraBounds, aabb: AxisAlignedBoundingBox
) -> Tuple[Tensor, Tensor]:
    """Please refer this blog for the implementation ->
    https://www.scratchapixel.com/lessons/3d-basic-rendering/->
    ->minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
    --------------------------------------------------------------------------------------
    compute the near and far bounds for each ray based on its intersection with the aabb
    --------------------------------------------------------------------------------------"""
    # TODO: !!!Test!!! and refactor this function
    #  Turns out there is a much more elegant form of this function in Ray-Tracing-Gems II :)
    #  Here -> https://link.springer.com/content/pdf/10.1007/978-1-4842-7185-8.pdf
    #  Page 88 / 884. The physical hardcover copy is not expensive btw :)

    # preamble :D
    dtype, device = rays.origins.dtype, rays.origins.device
    origins, directions = rays.origins, rays.directions
    num_rays = origins.shape[0]
    orig_ray_bounds = (
        torch.tensor([bounds.near, bounds.far], dtype=dtype, device=device)
        .reshape(1, -1)
        .repeat(num_rays, 1)
    )
    intersecting = torch.tensor([[True]], dtype=dtype, device=device).repeat(
        [num_rays, 1]
    )
    non_intersecting = torch.tensor([[False]], dtype=dtype, device=device).repeat(
        [num_rays, 1]
    )

    # compute intersections with the X-planes:
    x_min = (aabb.x_range[0] - origins[:, 0]) / (directions[:, 0] + ZERO_PLUS)
    x_max = (aabb.x_range[1] - origins[:, 0]) / (directions[:, 0] + ZERO_PLUS)
    x_ray_bounds = torch.stack([x_min, x_max], dim=-1)
    # noinspection PyTypeChecker
    x_ray_bounds = torch.where(
        x_ray_bounds[:, :1] > x_ray_bounds[:, 1:], x_ray_bounds[:, [1, 0]], x_ray_bounds
    )
    final_ray_bounds = x_ray_bounds

    # compute intersections with the Y-planes:
    y_min = (aabb.y_range[0] - origins[:, 1]) / (directions[:, 1] + ZERO_PLUS)
    y_max = (aabb.y_range[1] - origins[:, 1]) / (directions[:, 1] + ZERO_PLUS)
    # noinspection DuplicatedCode
    y_ray_bounds = torch.stack([y_min, y_max], dim=-1)

    # noinspection PyTypeChecker
    y_ray_bounds = torch.where(
        y_ray_bounds[:, :1] > y_ray_bounds[:, 1:], y_ray_bounds[:, [1, 0]], y_ray_bounds
    )

    intersecting = torch.where(
        torch.logical_or(
            final_ray_bounds[:, :1] > y_ray_bounds[:, 1:],
            y_ray_bounds[:, :1] > final_ray_bounds[:, 1:],
        ),
        non_intersecting,
        intersecting,
    )

    final_ray_bounds[:, 0] = torch.where(
        y_ray_bounds[:, 0] > final_ray_bounds[:, 0],
        y_ray_bounds[:, 0],
        final_ray_bounds[:, 0],
    )

    final_ray_bounds[:, 1] = torch.where(
        y_ray_bounds[:, 1] < final_ray_bounds[:, 1],
        y_ray_bounds[:, 1],
        final_ray_bounds[:, 1],
    )

    # compute intersections with the Z-planes:
    z_min = (aabb.z_range[0] - origins[:, 2]) / (directions[:, 2] + ZERO_PLUS)
    z_max = (aabb.z_range[1] - origins[:, 2]) / (directions[:, 2] + ZERO_PLUS)
    # noinspection DuplicatedCode
    z_ray_bounds = torch.stack([z_min, z_max], dim=-1)
    # noinspection PyTypeChecker
    z_ray_bounds = torch.where(
        z_ray_bounds[:, :1] > z_ray_bounds[:, 1:], z_ray_bounds[:, [1, 0]], z_ray_bounds
    )

    intersecting = torch.where(
        torch.logical_or(
            final_ray_bounds[:, :1] > z_ray_bounds[:, 1:],
            z_ray_bounds[:, :1] > final_ray_bounds[:, 1:],
        ),
        non_intersecting,
        intersecting,
    )

    final_ray_bounds[:, 0] = torch.where(
        z_ray_bounds[:, 0] > final_ray_bounds[:, 0],
        z_ray_bounds[:, 0],
        final_ray_bounds[:, 0],
    )

    final_ray_bounds[:, 1] = torch.where(
        z_ray_bounds[:, 1] < final_ray_bounds[:, 1],
        z_ray_bounds[:, 1],
        final_ray_bounds[:, 1],
    )

    # finally revert the non_intersecting rays to the original scene_bounds:
    final_ray_bounds = torch.where(
        torch.logical_not(intersecting), orig_ray_bounds, final_ray_bounds
    )

    # We don't consider the intersections behind the camera
    final_ray_bounds = torch.clip(final_ray_bounds, min=0.0)

    # return the computed intersections (final_ray_bounds) and the boolean Tensor intersecting
    # denoting whether the ray intersected the aabb or not.
    return final_ray_bounds, intersecting


def sample_aabb_bound_uniform_points_on_rays(
    rays: Rays,
    bounds: CameraBounds,
    num_samples: int,
    aabb: AxisAlignedBoundingBox,
    perturb: bool = True,
) -> SampledPointsOnRays:
    aabb_intersected_ray_bounds, _ = _ray_aabb_intersection(rays, bounds, aabb)

    # return uniform points sampled on the rays using the new CameraBounds per ray (in a tensor):
    return sample_uniform_points_on_rays(
        rays,
        bounds=aabb_intersected_ray_bounds,
        num_samples=num_samples,
        perturb=perturb,
    )
