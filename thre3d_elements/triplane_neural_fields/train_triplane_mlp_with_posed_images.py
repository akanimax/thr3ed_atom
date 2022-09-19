import imageio
import torch

from functools import partial
from pathlib import Path

from thre3d_atom.data.datasets import PosedImagesDataset
from thre3d_atom.modules.trainers import train_triplane_mlp_vol_mod_with_posed_images
from thre3d_atom.modules.volumetric_model import VolumetricModel
from thre3d_atom.rendering.volumetric.process import RenderMLP
from thre3d_atom.thre3d_reprs.renderers import (
    render_triplane_mlp,
    TriplaneMLPRenderConfig,
)
from thre3d_atom.thre3d_reprs.triplane import TriplaneStruct
from thre3d_atom.utils.constants import NUM_COORD_DIMENSIONS
from thre3d_atom.utils.imaging_utils import get_thre360_animation_poses
from thre3d_atom.utils.logging import log
from thre3d_atom.visualizations.animations import (
    render_camera_path_for_volumetric_model,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

all_scenes = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]

# ---------------------------------------------------------------------------------------------------------------------
# Script parameters:
# ---------------------------------------------------------------------------------------------------------------------
scene_id = 3
data_path = Path(f"../../data/3d_scenes/nerf_synthetic/{all_scenes[scene_id]}")
logs_path = Path(f"../../logs/triplane_mlp/set_1/{all_scenes[scene_id]}")
grid_size, num_samples_per_ray = 128, 256
grid_world_size = 3.0
ray_batch_size = 4096
feature_size = 8
white_bkgd = True

# visualization related:
parallel_points_chunk_size = 32768
num_frames, camera_pitch = 180, 60.0
# ---------------------------------------------------------------------------------------------------------------------


# dataset:
train_dataset, test_dataset = (
    PosedImagesDataset(
        images_dir=data_path / mode,
        camera_params_json=data_path / f"{mode}_camera_params.json",
        normalize_scene_scale=False,
        downsample_factor=2.0,
        rgba_white_bkgd=white_bkgd,
    )
    for mode in ("train", "test")
)


# fmt: off
features = torch.empty((NUM_COORD_DIMENSIONS, grid_size, grid_size, feature_size), device=device)
features = torch.nn.init.xavier_uniform_(features)
triplane = TriplaneStruct(
    features=features,
    size=grid_world_size,
    feature_preactivation=torch.nn.Tanh(),
    tunable=True,
)
render_mlp = RenderMLP(input_dims=3*feature_size).to(device)
# fmt: on

# set up a volumetricModel using this voxel-grid
# noinspection PyTypeChecker
triplane_vol_mod = VolumetricModel(
    thre3d_repr=triplane,
    render_procedure=partial(render_triplane_mlp, render_mlp=render_mlp),
    render_config=TriplaneMLPRenderConfig(
        num_samples_per_ray=num_samples_per_ray,
        camera_bounds=train_dataset.camera_bounds,
        white_bkgd=white_bkgd,
    ),
    device=device,
)

train_triplane_mlp_vol_mod_with_posed_images(
    triplane_vol_mod,
    render_mlp,
    train_dataset,
    logs_path,
    test_dataset=test_dataset,
    ray_batch_size=ray_batch_size,
)

log.info("Creating rotating animation for the final trained model ...")
animation_poses = get_thre360_animation_poses(
    hemispherical_radius=train_dataset.get_hemispherical_radius_estimate(),
    camera_pitch=camera_pitch,
    num_poses=num_frames,
)
animation = render_camera_path_for_volumetric_model(
    triplane_vol_mod,
    animation_poses,
    camera_intrinsics=train_dataset.camera_intrinsics,
    use_optimized_sampling=False,
    parallel_points_chunk_size=parallel_points_chunk_size,
)

imageio.mimwrite(logs_path / "final_rendered_animation.mp4", animation)
