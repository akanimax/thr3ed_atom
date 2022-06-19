import dataclasses
from functools import partial
from typing import Callable, Optional, Any

import torch
from torch import Tensor
from torch.nn import Module

from thre3d_atom.rendering.volumetric.accumulate import (
    density2occupancy_pb,
    accumulate_radiance_density_on_rays,
)
from thre3d_atom.rendering.volumetric.process import process_points_with_sh_voxel_grid
from thre3d_atom.rendering.volumetric.render_interface import RenderOut, Rays, render
from thre3d_atom.rendering.volumetric.sample import (
    sample_aabb_bound_uniform_points_on_rays,
    sample_uniform_points_on_rays,
)
from thre3d_atom.thre3d_reprs.voxels import VoxelGrid
from thre3d_atom.utils.imaging_utils import CameraBounds

# All the rendering procedures below follow this functional type
# The Optional[int] is for possible sample-based-parallel processing
RenderConfig = Any  # This is an updatable dataclass which will hold the configuration of a particular render procedure
RenderProcedure = Callable[[Module, Rays, RenderConfig, Optional[int]], RenderOut]


@dataclasses.dataclass
class SHVoxGridRenderConfig:
    # ProbingConfig:
    num_samples_per_ray: int
    camera_bounds: CameraBounds
    perturb_sampled_points: bool = True
    optimized_sampling: bool = False

    # AccumulationConfig
    density2occupancy: Callable[[Tensor, Tensor], Tensor] = density2occupancy_pb
    radiance_hdr_tone_map: Callable[[Tensor], Tensor] = torch.sigmoid
    stochastic_density_noise_std: float = 0.0  # used by NeRF not by us :)
    white_bkgd: bool = False

    # Misc Render mode config
    render_diffuse: bool = False
    render_num_samples_per_ray: int = 512
    parallel_rays_chunk_size: int = 32768


def render_sh_voxel_grid(
    voxel_grid: VoxelGrid,
    rays: Rays,
    render_config: SHVoxGridRenderConfig,
    parallel_points_chunk_size: Optional[int] = None,
) -> RenderOut:
    """
    renders an SH-based voxel grid
    Args:
        voxel_grid: the VoxelGrid being rendered
        rays: the rays (aka. probes) used for rendering
        render_config: configuration used by this render_procedure
        parallel_points_chunk_size: size of each chunk, in case sample/point based parallel processing is required
    Returns: rendered output per ray (RenderOut) :)
    """
    # select the sampler function based on whether optimized sampling is requested:
    if render_config.optimized_sampling:
        sampler_function = partial(
            sample_aabb_bound_uniform_points_on_rays,
            aabb=voxel_grid.aabb,
            perturb=render_config.perturb_sampled_points,
        )
    else:
        sampler_function = partial(
            sample_uniform_points_on_rays,
            perturb=render_config.perturb_sampled_points,
        )
    # prepare the processor_function
    point_processor_function = partial(
        process_points_with_sh_voxel_grid,
        voxel_grid=voxel_grid,
        render_diffuse=render_config.render_diffuse,
        parallel_points_chunk_size=parallel_points_chunk_size,
    )
    # finally, prepare the accumulator_function
    accumulator_function = partial(
        accumulate_radiance_density_on_rays,
        stochastic_density_noise_std=render_config.stochastic_density_noise_std,
        density2occupancy=render_config.density2occupancy,
        radiance_hdr_tone_map=render_config.radiance_hdr_tone_map,
        white_bkgd=render_config.white_bkgd,
        extra_debug_info=False,
    )

    # render the output using the render-interface
    # the following suppression is used in order to use partials :)
    # noinspection PyTypeChecker
    return render(
        rays,
        camera_bounds=render_config.camera_bounds,
        num_samples=render_config.num_samples_per_ray,
        sampler_fn=sampler_function,
        point_processor_fn=point_processor_function,
        accumulator_fn=accumulator_function,
    )
