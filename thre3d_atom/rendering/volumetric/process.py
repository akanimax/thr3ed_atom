from functools import partial
from typing import Optional

import numpy as np
import torch

from thre3d_atom.rendering.volumetric.render_interface import (
    SampledPointsOnRays,
    Rays,
    ProcessedPointsOnRays,
)
from thre3d_atom.rendering.volumetric.utils.spherical_harmonics import (
    evaluate_spherical_harmonics,
)
from thre3d_atom.thre3d_reprs.voxels import VoxelGrid
from thre3d_atom.utils.constants import NUM_COLOUR_CHANNELS, INFINITY
from thre3d_atom.utils.misc import batchify


def process_points_with_sh_voxel_grid(
    sampled_points: SampledPointsOnRays,
    rays: Rays,
    voxel_grid: VoxelGrid,
    render_diffuse: bool = False,
    parallel_points_chunk_size: Optional[int] = None,
) -> ProcessedPointsOnRays:
    dtype, device = sampled_points.points.dtype, sampled_points.points.device

    # extract shape information
    num_rays, num_samples_per_ray, num_coords = sampled_points.points.shape

    # obtain interpolated features from the voxel_grid:
    flat_sampled_points = sampled_points.points.reshape(-1, num_coords)

    # account for point/sample-based parallelization if requested
    if parallel_points_chunk_size is None:
        interpolated_features = voxel_grid(flat_sampled_points)
    else:
        interpolated_features = batchify(
            voxel_grid,
            collate_fn=partial(torch.cat, dim=0),
            chunk_size=parallel_points_chunk_size,
        )(flat_sampled_points)

    # unpack sh_coeffs and density features:
    sh_coeffs, raw_densities = (
        interpolated_features[..., :-1],
        interpolated_features[..., -1:],
    )

    # evaluate the spherical harmonics using the interpolated coeffs and for the given view-dirs
    # compute view_dirs
    viewdirs = rays.directions / rays.directions.norm(dim=-1, keepdim=True)
    viewdirs_tiled = (
        viewdirs[:, None, :].repeat(1, num_samples_per_ray, 1).reshape(-1, num_coords)
    )

    # compute the SH-degree based on the available features:
    if render_diffuse:
        # if rendering the diffuse variant, then we only use the degree 0 features
        sh_coeffs = sh_coeffs.reshape(sh_coeffs.shape[0], NUM_COLOUR_CHANNELS, -1)
        sh_coeffs = sh_coeffs[..., :1]
        sh_degree = 0
    else:
        # otherwise use all the features
        sh_coeffs = sh_coeffs.reshape(sh_coeffs.shape[0], NUM_COLOUR_CHANNELS, -1)
        sh_degree = int(np.sqrt(sh_coeffs.shape[-1])) - 1

    # evaluate the spherical harmonics with the viewdirs
    # TODO: parallel chunking for evaluating the SH based components. Think of a solution sometime
    #  to handle the case of multiple sequence inputs. Or just change the evaluate_spherical_harmonics function
    raw_radiance = evaluate_spherical_harmonics(
        degree=sh_degree,
        sh_coeffs=sh_coeffs,
        viewdirs=viewdirs_tiled,
    )

    # filter out radiance and density values outside the AABB of the voxel grid
    # fmt: off
    inside_points_mask = voxel_grid.test_inside_volume(flat_sampled_points)
    minus_infinity_radiance = torch.full(raw_radiance.shape, -INFINITY, dtype=dtype, device=device)
    filtered_raw_radiance = torch.where(inside_points_mask, raw_radiance, minus_infinity_radiance)
    zero_densities = torch.zeros_like(raw_densities, dtype=dtype, device=device)
    filtered_raw_densities = torch.where(inside_points_mask, raw_densities, zero_densities)
    # fmt: on

    # construct and reshape processed_points
    processed_points = torch.cat(
        [filtered_raw_radiance, filtered_raw_densities], dim=-1
    )
    processed_points = processed_points.reshape(num_rays, num_samples_per_ray, -1)

    return ProcessedPointsOnRays(
        processed_points,
        sampled_points.depths,
    )
