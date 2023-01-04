from datetime import timedelta
from pathlib import Path
from typing import Optional

import imageio
import numpy as np
from PIL import ImageDraw, Image
from matplotlib import pyplot as plt

from thre3d_atom.data.datasets import PosedImagesDataset
from thre3d_atom.modules.volumetric_model import VolumetricModel
from thre3d_atom.rendering.volumetric.render_interface import RenderOut
from thre3d_atom.rendering.volumetric.utils.misc import (
    cast_rays,
    ndcize_rays,
    flatten_rays,
)
from thre3d_atom.utils.constants import EXTRA_ACCUMULATED_WEIGHTS, NUM_COLOUR_CHANNELS
from thre3d_atom.utils.imaging_utils import (
    CameraPose,
    CameraIntrinsics,
    CameraBounds,
    to8b,
    postprocess_depth_map,
)
from thre3d_atom.utils.logging import log


def visualize_camera_rays(
    dataset: PosedImagesDataset,
    output_dir: Path,
    num_rays_per_image: int = 30,
    do_ndcize_rays: bool = False,
) -> None:
    all_poses = [
        dataset.extract_pose(camera_param)
        for camera_param in dataset.camera_parameters.values()
    ]
    all_camera_locations = []

    fig = plt.figure()
    fig.suptitle("Camera rays visualization")
    ax = fig.add_subplot(111, projection="3d")
    for pose in all_poses:
        rays = flatten_rays(cast_rays(dataset.camera_intrinsics, pose))
        if do_ndcize_rays:
            rays = ndcize_rays(rays, dataset.camera_intrinsics)

        # randomly select only num_rays_per_image rays for visualization
        combined_rays = np.concatenate([rays.origins, rays.directions], axis=-1)
        np.random.shuffle(combined_rays)
        selected_rays = combined_rays[:num_rays_per_image]
        selected_ray_origins, selected_ray_directions = (
            selected_rays[:, :3],
            selected_rays[:, 3:],
        )
        # add the ray origin to camera locations
        all_camera_locations.append(selected_ray_origins[0])

        far_plane = dataset.camera_bounds.far if not do_ndcize_rays else 1.0
        points_a = selected_ray_origins
        points_b = selected_ray_origins + (selected_ray_directions * far_plane)
        # plot all the rays (from point_a to point_b) sequentially
        for (point_a, point_b) in zip(points_a, points_b):
            combined = np.stack([point_a, point_b])
            ax.plot(combined[:, 0], combined[:, 1], combined[:, 2], color="b")
    # scatter all the start points in different colour:
    all_camera_locations = np.stack(all_camera_locations, axis=0)
    ax.scatter(
        all_camera_locations[:, 0],
        all_camera_locations[:, 1],
        all_camera_locations[:, 2],
        color="m",
    )

    # save the figure
    plt.tight_layout()
    plt.savefig(output_dir / "casted_camera_rays.png", dpi=600)
    plt.close(fig)


def _process_rendered_output_for_feedback_log(
    rendered_output: RenderOut,
    training_time: Optional[float] = None,
) -> np.array:
    # obtain the colour, acc and depth maps and concatenate them side-by-side.
    colour_map = to8b(rendered_output.colour.cpu().numpy())
    depth_map = postprocess_depth_map(
        rendered_output.depth.cpu().squeeze().numpy(),
        acc_map=rendered_output.extra[EXTRA_ACCUMULATED_WEIGHTS].cpu().numpy(),
    )
    # invert the acc_map for better looking visualization
    acc_map = to8b(1.0 - rendered_output.extra[EXTRA_ACCUMULATED_WEIGHTS].cpu().numpy())
    acc_map = np.tile(acc_map, (1, 1, NUM_COLOUR_CHANNELS))

    text_colour = (0, 0, 0)  # use black ink by default
    feedback_image = np.concatenate(
        [colour_map, depth_map, acc_map],
        axis=1,
    )

    # add the training timestamp to the rendered feedback images if it is available:
    if training_time is not None:
        formatted_time_string = str(timedelta(seconds=training_time))
        pil_feedback_image = Image.fromarray(feedback_image)
        drawable_image = ImageDraw.Draw(pil_feedback_image)
        drawable_image.text((10, 10), formatted_time_string, text_colour)
        # noinspection PyTypeChecker
        feedback_image = np.array(pil_feedback_image)

    return feedback_image


def visualize_sh_vox_grid_vol_mod_rendered_feedback(
    vol_mod: VolumetricModel,
    render_feedback_pose: CameraPose,
    camera_intrinsics: CameraIntrinsics,
    global_step: int,
    feedback_logs_dir: Path,
    parallel_rays_chunk_size: int = 32768,
    training_time: Optional[float] = None,
    log_diffuse_rendered_version: bool = True,
    use_optimized_sampling_mode: bool = False,
    overridden_num_samples_per_ray: Optional[int] = None,
    verbose_rendering: bool = True,
) -> None:
    # Bump up the num_samples_per_ray to a high-value for reducing MC noise
    if overridden_num_samples_per_ray is None:
        overridden_num_samples_per_ray_for_beautiful_renders = 1024  # :)
    else:
        overridden_num_samples_per_ray_for_beautiful_renders = (
            overridden_num_samples_per_ray
        )

    # render images
    log.info(f"rendering intermediate output for feedback")

    specular_rendered_output = vol_mod.render(
        camera_pose=render_feedback_pose,
        camera_intrinsics=camera_intrinsics,
        parallel_rays_chunk_size=parallel_rays_chunk_size,
        gpu_render=True,
        verbose=verbose_rendering,
        optimized_sampling=use_optimized_sampling_mode,
        num_samples_per_ray=overridden_num_samples_per_ray_for_beautiful_renders,
    )
    specular_feedback_image = _process_rendered_output_for_feedback_log(
        specular_rendered_output, training_time
    )
    imageio.imwrite(
        feedback_logs_dir / f"specular_{global_step}.png",
        specular_feedback_image,
    )

    if log_diffuse_rendered_version:
        diffuse_rendered_output = vol_mod.render(
            camera_pose=render_feedback_pose,
            camera_intrinsics=camera_intrinsics,
            parallel_rays_chunk_size=parallel_rays_chunk_size,
            gpu_render=True,
            verbose=verbose_rendering,
            optimized_sampling=use_optimized_sampling_mode,
            render_diffuse=True,
            num_samples_per_ray=overridden_num_samples_per_ray_for_beautiful_renders,
        )
        diffuse_feedback_image = _process_rendered_output_for_feedback_log(
            diffuse_rendered_output, training_time
        )
        imageio.imwrite(
            feedback_logs_dir / f"diffuse_{global_step}.png",
            diffuse_feedback_image,
        )
