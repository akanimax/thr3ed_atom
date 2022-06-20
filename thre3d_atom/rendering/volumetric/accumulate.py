from typing import Callable

import torch
from torch import Tensor

from thre3d_atom.rendering.volumetric.render_interface import (
    Rays,
    ProcessedPointsOnRays,
    RenderOut,
)
from thre3d_atom.utils.constants import (
    ZERO_PLUS,
    INFINITY,
    EXTRA_ACCUMULATED_WEIGHTS,
    EXTRA_POINT_WEIGHTS,
    EXTRA_POINT_DEPTHS,
    EXTRA_POINT_OCCUPANCIES,
    EXTRA_POINT_DENSITIES,
    EXTRA_SAMPLE_INTERVALS,
    EXTRA_DISPARITY,
)


def density2occupancy_pb(densities: Tensor, deltas: Tensor) -> Tensor:
    """computes occupancy values from density values in the range [0, inf).
    The occupancy (being a probability) is always strictly between [0, 1].
    This function is physically based, and can be derived from Lambert's law"""
    return 1.0 - torch.exp(-(densities * deltas))


def accumulate_radiance_density_on_rays(
    processed_points: ProcessedPointsOnRays,
    rays: Rays,
    stochastic_density_noise_std: float = 1.0,
    density2occupancy: Callable[[Tensor, Tensor], Tensor] = density2occupancy_pb,
    radiance_hdr_tone_map: Callable[[Tensor], Tensor] = torch.sigmoid,
    white_bkgd: bool = True,
    extra_debug_info: bool = False,
) -> RenderOut:
    dtype, device = processed_points.points.dtype, processed_points.points.device

    # unpack the radiance and density from the processed points
    raw_radiance, raw_density = (
        processed_points.points[..., :-1],
        processed_points.points[..., -1],
    )

    # compute sample-intervals for ray-time integration
    deltas = processed_points.depths[..., 1:] - processed_points.depths[..., :-1]
    inf_delta = torch.full(
        size=(*deltas.shape[:-1], 1), fill_value=INFINITY, dtype=dtype, device=device
    )
    deltas = torch.cat([deltas, inf_delta], dim=-1)  # [N_rays, N_samples]
    # need proper deltas in coordinate domain space, hence normalizing ray-dir norms
    deltas = deltas * rays.directions[..., None, :].norm(dim=-1)

    # add stochastic noise to density if requested :).
    # can be used to regularize repr during training (kind-of-sort-of prevents floater artifacts).
    density_noise = (
        torch.randn(raw_density.shape, dtype=dtype, device=device)
        * stochastic_density_noise_std
    )
    alpha = density2occupancy(raw_density + density_noise, deltas)

    # compute the radiance weights for accumulation along the ray
    ones = torch.ones((alpha.shape[0], 1), dtype=dtype, device=device)
    weights = alpha * torch.cumprod(torch.cat([ones, 1.0 - alpha], -1), -1)[:, :-1]

    # accumulate the predicted radiance values of the samples using the computed alphas
    colour = radiance_hdr_tone_map(raw_radiance)
    # dims below: [N, NUM_COLOUR_CHANNELS]
    colour_render = torch.sum(colour * weights[..., None], dim=-2)

    # sum of weights along each ray, should ideally be 1.0 everywhere
    acc_render = torch.sum(weights, dim=-1, keepdim=True)

    if white_bkgd:
        # add a white background if requested. Mathematically, note that we assume
        # the background to be emitting a solid bright white colour RGB=(1.0, 1.0, 1.0)
        # hence the simplification of the alpha-composition formula :)
        colour_render = colour_render + (1 - acc_render)

    # compute depth_render and disparity_render (inverse depth)
    depth_render = (processed_points.depths * weights).sum(dim=-1, keepdims=True)
    disparity_render = 1.0 / torch.maximum(
        torch.full(acc_render.shape, ZERO_PLUS, device=depth_render.device),
        depth_render / acc_render,
    )

    # additional renders
    extra_dict = {
        EXTRA_DISPARITY: disparity_render,
        EXTRA_ACCUMULATED_WEIGHTS: acc_render,
    }

    if extra_debug_info:
        # adds very big tensor buffers
        # num samples per ray (or ray-chunk). Please only use for debugging purposes
        extra_dict.update(
            {
                EXTRA_POINT_DENSITIES: raw_density,
                EXTRA_POINT_OCCUPANCIES: alpha,
                EXTRA_POINT_WEIGHTS: weights,
                EXTRA_POINT_DEPTHS: processed_points.depths,
                EXTRA_SAMPLE_INTERVALS: deltas,
            }
        )

    return RenderOut(
        colour=colour_render,
        depth=depth_render,
        extra=extra_dict,
    )
