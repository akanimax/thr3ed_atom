from functools import partial
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module, LeakyReLU, Identity
from torch.nn.functional import softplus

from thre3d_atom.neural_networks.layers.embedders import PositionalEmbeddingsEncoder
from thre3d_atom.neural_networks.skip_mlp import SkipMLP
from thre3d_atom.rendering.volumetric.render_interface import (
    SampledPointsOnRays,
    Rays,
    ProcessedPointsOnRays,
)
from thre3d_atom.rendering.volumetric.utils.spherical_harmonics import (
    evaluate_spherical_harmonics,
)
from thre3d_atom.thre3d_reprs.triplane import TriplaneStruct
from thre3d_atom.thre3d_reprs.utils import test_inside_volume
from thre3d_atom.thre3d_reprs.voxels import VoxelGrid
from thre3d_atom.utils.constants import (
    NUM_COLOUR_CHANNELS,
    INFINITY,
    NUM_COORD_DIMENSIONS,
)
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
    inside_points_mask = test_inside_volume(voxel_grid.aabb, flat_sampled_points)
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


class RenderMLP(Module):
    def __init__(
        self,
        input_dims: int,
        feat_emb_dims: int = 0,
        dir_emb_dims: int = 4,
        dnet_layer_depths: Tuple[int] = (128, 128, 128),
        dnet_skips: Tuple[bool] = (False, True, False),
        rnet_layer_depths: Tuple[int] = (128,),
        rnet_skips: Tuple[bool] = (False,),
        activation_fn: Module = LeakyReLU(),
    ) -> None:
        super().__init__()

        # state of the object:
        self._input_dims = input_dims
        self._feat_emb_dims = feat_emb_dims
        self._dir_emb_dims = dir_emb_dims
        self._dnet_layer_depths = dnet_layer_depths
        self._dnet_skips = dnet_skips
        self._rnet_layer_depths = rnet_layer_depths
        self._rnet_skips = rnet_skips
        self._activation_fn = activation_fn

        self._feats_encoder = PositionalEmbeddingsEncoder(
            input_dims=input_dims, emb_dims=feat_emb_dims
        )
        self._dir_encoder = PositionalEmbeddingsEncoder(
            input_dims=NUM_COORD_DIMENSIONS, emb_dims=dir_emb_dims
        )
        self._density_net = SkipMLP(
            input_dims=self._feats_encoder.output_size,
            output_dims=dnet_layer_depths[-1] + 1,
            layer_depths=dnet_layer_depths,
            skips=dnet_skips,
            dropout_prob=0.0,
            activation_fn=activation_fn,
            out_activation_fn=Identity(),
        )
        self._radiance_net = SkipMLP(
            input_dims=dnet_layer_depths[-1] + self._dir_encoder.output_size,
            output_dims=NUM_COLOUR_CHANNELS,
            layer_depths=rnet_layer_depths,
            skips=rnet_skips,
            dropout_prob=0.0,
            activation_fn=activation_fn,
            out_activation_fn=Identity(),
        )

    def get_save_config_dict(self) -> Dict[str, Any]:
        return {
            "input_dims": self._input_dims,
            "feat_emb_dims": self._feat_emb_dims,
            "dir_emb_dims": self._dir_emb_dims,
            "dnet_layer_depths": self._dnet_layer_depths,
            "dnet_skips": self._dnet_skips,
            "rnet_layer_depths": self._rnet_layer_depths,
            "rnet_skips": self._rnet_skips,
            "activation_fn": self._activation_fn,
        }

    def forward(self, features: Tensor) -> Tensor:
        features, view_dirs = (
            features[:, :-NUM_COORD_DIMENSIONS],
            features[:, -NUM_COORD_DIMENSIONS:],
        )

        # density network
        pe_features = self._feats_encoder(features)
        out = self._density_net(pe_features)
        mlp_feats, densities = out[:, :-1], out[:, -1:]
        densities = softplus(densities)

        # radiance network
        pe_viewdirs = self._dir_encoder(view_dirs)
        radiance = self._radiance_net(torch.cat([mlp_feats, pe_viewdirs], dim=-1))
        return torch.cat([radiance, densities], dim=-1)


def process_points_with_triplane_and_mlp(
    sampled_points: SampledPointsOnRays,
    rays: Rays,
    triplane: TriplaneStruct,
    render_mlp: RenderMLP,
    parallel_points_chunk_size: Optional[int] = None,
) -> ProcessedPointsOnRays:
    dtype, device = sampled_points.points.dtype, sampled_points.points.device
    num_rays, num_samples_per_ray, num_coords = sampled_points.points.shape
    flat_sampled_points = sampled_points.points.reshape(-1, num_coords)

    # account for point/sample-based parallelization if requested
    interpolated_features = triplane(flat_sampled_points)
    viewdirs = rays.directions / rays.directions.norm(dim=-1, keepdim=True)
    viewdirs_tiled = (
        viewdirs[:, None, :].repeat(1, num_samples_per_ray, 1).reshape(-1, num_coords)
    )
    input_features = torch.cat([interpolated_features, viewdirs_tiled], dim=-1)
    if parallel_points_chunk_size is None:
        processed_points = render_mlp(input_features)
    else:
        processed_points = batchify(
            render_mlp,
            collate_fn=partial(torch.cat, dim=0),
            chunk_size=parallel_points_chunk_size,
        )(input_features)

    raw_radiance, raw_densities = processed_points[:, :-1], processed_points[:, -1:]

    # fmt: off
    inside_points_mask = test_inside_volume(triplane.aabb, flat_sampled_points)
    minus_infinity_radiance = torch.full(raw_radiance.shape, -INFINITY, dtype=dtype, device=device)
    filtered_raw_radiance = torch.where(inside_points_mask, raw_radiance, minus_infinity_radiance)
    zero_densities = torch.zeros_like(raw_densities, dtype=dtype, device=device)
    filtered_raw_densities = torch.where(inside_points_mask, raw_densities, zero_densities)
    # fmt: on

    processed_points = torch.cat(
        [filtered_raw_radiance, filtered_raw_densities], dim=-1
    )
    processed_points = processed_points.reshape(num_rays, num_samples_per_ray, -1)
    return ProcessedPointsOnRays(
        processed_points,
        sampled_points.depths,
    )
