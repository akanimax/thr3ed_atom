from typing import Optional

import lpips as lpips
import numpy as np
import torch
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from thre3d_atom.modules.volumetric_model import VolumetricModel
from thre3d_atom.utils.imaging_utils import CameraPose
from thre3d_atom.utils.logging import log
from thre3d_atom.utils.metric_utils import mse2psnr


def test_sh_vox_grid_vol_mod_with_posed_images(
    vol_mod: VolumetricModel,
    test_dl: DataLoader,
    parallel_rays_chunk_size: int,
    tensorboard_writer: Optional[SummaryWriter] = None,
    global_step: Optional[int] = None,
) -> None:
    log.info(f"Testing the model on {len(test_dl)} heldout images")
    all_psnrs, all_lpips = [], []
    vgg_lpips_computer = lpips.LPIPS(net="vgg").to(vol_mod.device)
    for (image, pose) in tqdm(test_dl):
        image, pose = image[0], pose[0]  # testing batching is always 1
        # noinspection PyUnresolvedReferences
        rendered_output = vol_mod.render(
            camera_pose=CameraPose(rotation=pose[:, :3], translation=pose[:, 3:]),
            camera_intrinsics=test_dl.dataset.camera_intrinsics,
            parallel_rays_chunk_size=parallel_rays_chunk_size,
            gpu_render=True,
            optimized_sampling=False,
            num_samples_per_ray=vol_mod.render_config.render_num_samples_per_ray
        )
        rendered_colour = rendered_output.colour.permute(2, 0, 1)

        with torch.no_grad():
            # compute the PSNR metric:
            psnr = mse2psnr(mse_loss(rendered_colour, image).item())

            # compute the LPIPS metric
            vgg_lpips = vgg_lpips_computer(
                rendered_colour[None, ...],
                image[None, ...],
                normalize=True,
            ).item()

        all_psnrs.append(psnr)
        all_lpips.append(vgg_lpips)

    # compute average of the computed per sample metrics and log them to console and
    # possibly to the Tensorboard
    mean_psnr, mean_lpips = [
        np.mean(metric_scores) for metric_scores in (all_psnrs, all_lpips)
    ]
    log.info(f"Mean PSNR on holdout set: {mean_psnr}")
    log.info(f"Mean LPIPS on holdout set: {mean_lpips}")
    if tensorboard_writer is not None and global_step is not None:
        for metric_tag, metric_value in [
            ("TEST_SET_PSNR", mean_psnr),
            ("TEST_SET_LPIPS", mean_lpips),
        ]:
            tensorboard_writer.add_scalar(
                metric_tag, metric_value, global_step=global_step
            )

    # delete the vgg_lpips computer from memory for saving up memory:
    del vgg_lpips_computer
