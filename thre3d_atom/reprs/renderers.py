from functools import partial
from typing import NamedTuple, Callable, Optional

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
from thre3d_atom.reprs.voxels import VoxelGrid
from thre3d_atom.utils.imaging_utils import CameraBounds

# All the rendering procedures below follow this functional type
# The Optional[int] is for possible sample-based-parallel processing
RenderProcedure = Callable[[Module, Rays, Optional[int]], RenderOut]


class ProbingConfig(NamedTuple):
    num_samples_per_ray: int
    camera_bounds: CameraBounds
    perturb_sampled_points: bool = True
    optimized_sampling: bool = False


class AccumulationConfig(NamedTuple):
    density2occupancy: Callable[[Tensor, Tensor], Tensor] = density2occupancy_pb
    radiance_hdr_tone_map: Callable[[Tensor], Tensor] = torch.sigmoid
    stochastic_density_noise_std: float = 0.0
    white_bkgd: bool = False


def render_sh_voxel_grid(
    voxel_grid: VoxelGrid,
    rays: Rays,
    probing_config: ProbingConfig,
    accumulation_confing: AccumulationConfig,
    render_diffuse: bool = False,
    parallel_chunk_size: Optional[int] = None,
) -> RenderOut:
    """
    renders an SH-based voxel grid
    Args:
        voxel_grid: the VoxelGrid being rendered
        rays: the rays (aka. probes) used for rendering
        probing_config: configuration required for probing the 3D volume
        accumulation_confing: configuration required for accumulating the probed radiance and density
        render_diffuse: whether to render the diffuse version of the SH-based voxel grid
        parallel_chunk_size: size of each chunk, in case sample/point based parallel processing is required
    Returns: rendered output per ray (RenderOut) :)
    """
    # select the sampler function based on whether optimized sampling is requested:
    if probing_config.optimized_sampling:
        sampler_function = partial(
            sample_aabb_bound_uniform_points_on_rays,
            aabb=voxel_grid.aabb,
            perturb=probing_config.perturb_sampled_points,
        )
    else:
        sampler_function = partial(
            sample_uniform_points_on_rays,
            perturb=probing_config.perturb_sampled_points,
        )
    # prepare the processor_function
    point_processor_function = partial(
        process_points_with_sh_voxel_grid,
        voxel_grid=voxel_grid,
        render_diffuse=render_diffuse,
        parallel_chunk_size=parallel_chunk_size,
    )
    # finally, prepare the accumulator_function
    accumulator_function = partial(
        accumulate_radiance_density_on_rays,
        stochastic_density_noise_std=accumulation_confing.stochastic_density_noise_std,
        density2occupancy=accumulation_confing.density2occupancy,
        radiance_hdr_tone_map=accumulation_confing.radiance_hdr_tone_map,
        white_bkgd=accumulation_confing.white_bkgd,
        extra_debug_info=False,
    )

    # render the output using the render-interface
    # the following suppression is used in order to use partials :)
    # noinspection PyTypeChecker
    return render(
        rays,
        camera_bounds=probing_config.camera_bounds,
        num_samples=probing_config.num_samples_per_ray,
        sampler_fn=sampler_function,
        point_processor_fn=point_processor_function,
        accumulator_fn=accumulator_function,
    )
