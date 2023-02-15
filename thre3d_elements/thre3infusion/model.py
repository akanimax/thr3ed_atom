import gc
import imageio
import numpy as np
import torch

from gaussian_diffusion import GaussianDiffusion
from thre3d_atom.utils.constants import CAMERA_INTRINSICS, HEMISPHERICAL_RADIUS
from thre3d_elements.thre3infusion.timestep_sampler import UniformSampler
from unet import UNetModel
from thre3d_atom.visualizations.animations import (
    render_camera_path_for_volumetric_model,
)
from thre3d_atom.thre3d_reprs.renderers import (
    SHVoxGridRenderConfig,
    render_sh_voxel_grid,
)
from thre3d_atom.modules.volumetric_model import (
    VolumetricModel,
    create_volumetric_model_from_saved_model,
)
from thre3d_atom.rendering.volumetric.utils.misc import (
    compute_expected_density_scale_for_relu_field_grid,
)
from thre3d_atom.thre3d_reprs.voxels import (
    VoxelGrid,
    VoxelSize,
    create_voxel_grid_from_saved_info_dict,
)

from thre3d_atom.utils.imaging_utils import (
    CameraBounds,
    CameraIntrinsics,
    get_thre360_animation_poses,
)

from pathlib import Path
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm.auto import tqdm
from typing import Tuple


class RandomCrop3D:
    def __init__(self, img_sz, crop_sz):
        h, w, d = img_sz
        assert (h, w, d) > crop_sz
        self.img_sz = tuple((h, w, d))
        self.crop_sz = tuple(crop_sz)

    def __call__(self, x):
        slice_hwd = [self._get_slice(i, k) for i, k in zip(self.img_sz, self.crop_sz)]
        return self._crop(x, *slice_hwd)

    @staticmethod
    def _get_slice(sz, crop_sz):
        try:
            lower_bound = torch.randint(sz - crop_sz, (1,)).item()
            return lower_bound, lower_bound + crop_sz
        except:
            return (None, None)

    @staticmethod
    def _crop(x, slice_h, slice_w, slice_d):
        return x[
            :,
            :,
            slice_h[0] : slice_h[1],
            slice_w[0] : slice_w[1],
            slice_d[0] : slice_d[1],
        ]


class Thre3inFusionModel(nn.Module):
    def __init__(self, unet: UNetModel, diffusion: GaussianDiffusion):
        super().__init__()

        self.unet = unet
        self.diffusion = diffusion
        self.device = next(self.unet.parameters()).device

        self.density_scale_range: Tuple[float, float] = (1.0, 1.0)
        self.features_scale_range: Tuple[float, float] = (1.0, 1.0)

        # ----------------------------------------------------------------------------------
        # initialize these to default values:
        # ----------------------------------------------------------------------------------
        self.render_procedure = render_sh_voxel_grid
        self.render_config = SHVoxGridRenderConfig(
            num_samples_per_ray=256,
            camera_bounds=CameraBounds(near=4.5, far=19.5),
            white_bkgd=True,
        )
        self.voxel_size = VoxelSize(0.078125, 0.078125, 0.078125)
        self.hemispherical_radius = 12.0
        self.camera_pitch = 45.0
        self.camera_intrinsics = CameraIntrinsics(
            height=512,
            width=512,
            focal=512.0,
        )

        # kept constant for now:
        self.vox_grid_density_activations_dict = {
            "density_preactivation": torch.nn.Identity(),
            "density_postactivation": torch.nn.Softplus(),
            # note this expected density value :)
            "expected_density_scale": compute_expected_density_scale_for_relu_field_grid(
                (10.0, 10.0, 5.0)
            ),
        }
        self.camera_pitch = 45.0
        # ----------------------------------------------------------------------------------

    def sample(self, shape: Tuple[int, int, int], num_samples: int = 1) -> torch.Tensor:
        shape = (num_samples, 4) + shape
        print(f"sampling {num_samples} samples from the model ...")
        return self.diffusion.p_sample_loop(
            model=self.unet,
            shape=shape,
            clip_denoised=True,
            device=self.device,
            progress=True,
            return_all_samples=False,
        )

    @staticmethod
    def serialize_VolMod_to_tensor_grid(vol_mod: VolumetricModel) -> torch.Tensor:
        voxel_grid = vol_mod.thre3d_repr
        densities = voxel_grid.densities
        features = voxel_grid.features
        grid = torch.cat([densities, features], dim=-1)
        grid = grid.permute(3, 0, 1, 2)[None, ...]
        return grid

    def deserialize_tensor_grid_to_VolMod(self, grid: torch.Tensor) -> VolumetricModel:
        serialized_grid = grid.permute(0, 2, 3, 4, 1)[0]
        densities, features = serialized_grid[:, :, :, :1], serialized_grid[:, :, :, 1:]

        voxel_grid = VoxelGrid(
            densities=densities,
            features=features,
            voxel_size=self.voxel_size,
            **self.vox_grid_density_activations_dict,
            tunable=True,
        )
        vox_grid_vol_mod = VolumetricModel(
            thre3d_repr=voxel_grid,
            render_procedure=self.render_procedure,
            render_config=self.render_config,
            device=self.device,
        )
        return vox_grid_vol_mod

    def scale_tensor_grids(self, grids: torch.Tensor) -> torch.Tensor:
        grids = (grids * 0.5) + 0.5
        densities, features = grids[:, :1, :, :, :], grids[:, 1:, :, :, :]
        densities = (
            densities * (self.density_scale_range[1] - self.density_scale_range[0])
        ) + self.density_scale_range[0]
        features = (
            features * (self.features_scale_range[1] - self.features_scale_range[0])
        ) + self.features_scale_range[0]
        grids = torch.cat([densities, features], dim=1)
        return grids

    def visualize_samples_mosaic(
        self,
        shape: Tuple[int, int, int],
        num_samples: int,
        save_path: Path,
        num_frames: int = 120,
        fps: float = 60.0,
    ) -> None:
        generated_samples = self.sample(shape, num_samples)
        generated_samples = self.scale_tensor_grids(generated_samples)

        # render the rotating frames for all the generated samples:
        vis_list = []
        print("Rendering videos for each sample ...")
        for generated_sample in tqdm(generated_samples):
            sample = generated_sample[None, ...]

            vol_mod = self.deserialize_tensor_grid_to_VolMod(sample)
            camera_path = get_thre360_animation_poses(
                hemispherical_radius=self.hemispherical_radius,
                camera_pitch=self.camera_pitch,
                num_poses=num_frames,
            )
            rendered_frames = render_camera_path_for_volumetric_model(
                vol_mod, camera_path, self.camera_intrinsics, verbose=False
            )
            rf_torch = torch.from_numpy(rendered_frames)
            vis_list.append(rf_torch)
        vis_list = torch.stack(vis_list)

        # make a single mosaic video for them:
        vis_frames = vis_list.permute(1, 0, 2, 3, 4)
        ncols = int(np.ceil(np.sqrt(num_samples)))
        vis = torch.stack(
            [
                make_grid(frame.permute(0, 3, 1, 2), nrow=ncols, padding=0).permute(
                    1, 2, 0
                )
                for frame in vis_frames
            ],
            dim=0,
        ).numpy()

        # save the video:
        imageio.mimwrite(
            save_path,
            vis,
            fps=fps,
        )

    def forward(self, shape: Tuple[int, int], num_samples: int = 1) -> torch.Tensor:
        return self.sample(shape, num_samples)

    def save_model(self, save_path: Path) -> None:
        save_info = {
            "unet": self.unet,
            "diffusion": self.diffusion
        }
        torch.save(save_info, save_path)

    def train(
        self,
        volume_model_path: Path,
        output_path: Path,
        crop_ratio: float = 0.95,
        num_iters: int = 100_000,
        batch_size: int = 8,
        learning_rate: float = 3e-4,
        loss_feedback_frequency: int = 50,
        sample_frequency: int = 500,
        save_frequency: int = 1000,
    ):
        # load the volumetric model:
        vol_mod, extra_info = create_volumetric_model_from_saved_model(
            model_path=volume_model_path,
            thre3d_repr_creator=create_voxel_grid_from_saved_info_dict,
            device=self.device,
        )

        # update the current state:
        self.render_config = vol_mod.render_config
        self.render_procedure = vol_mod.render_procedure
        self.voxel_size = vol_mod.thre3d_repr.voxel_size
        self.hemispherical_radius = extra_info[HEMISPHERICAL_RADIUS]
        self.camera_intrinsics = extra_info[CAMERA_INTRINSICS]

        # pre-process the volumetric model for training
        training_grid = self.serialize_VolMod_to_tensor_grid(vol_mod)
        self.density_scale_range = (
            training_grid[:, :1, ...].min().item(),
            training_grid[:, :1, ...].max().item(),
        )
        self.features_scale_range = (
            training_grid[:, 1:, ...].min().item(),
            training_grid[:, 1:, ...].max().item(),
        )
        training_grid[:, :1, ...] = (
            training_grid[:, :1, ...] - self.density_scale_range[0]
        ) / (self.density_scale_range[1] - self.density_scale_range[0])
        training_grid[:, 1:, ...] = (
            training_grid[:, 1:, ...] - self.features_scale_range[0]
        ) / (self.features_scale_range[1] - self.features_scale_range[0])
        training_grid = (training_grid * 2.0) - 1.0
        training_grid = training_grid.detach().to(self.device)

        # compute the training_crop_size:
        full_grid_size = training_grid.shape[2:]
        print("Full grid size: ", full_grid_size)
        training_grid_voxels = np.prod(full_grid_size)
        crop_voxels = training_grid_voxels * crop_ratio
        crop_size = int(np.ceil(crop_voxels ** (1.0 / 3.0)))
        training_crop_size = (crop_size, crop_size, crop_size)
        print("Training crop size: ", training_crop_size)

        # create the 3D random cropper:
        random_cropper = RandomCrop3D(full_grid_size, training_crop_size)

        # create optmizer:
        optimizer = torch.optim.Adam(self.unet.parameters(), lr=learning_rate)

        # create timestep sampler:
        timestep_sampler = UniformSampler(self.diffusion)

        # make output directories:
        model_dir = output_path / "saved_models"
        log_dir = output_path / "generated_samples"
        tensorboard_dir  = output_path / "tensorboard_logs"
        for directory in (model_dir, log_dir, tensorboard_dir):
            directory.mkdir(parents=True, exist_ok=True)

        # create tensorboard writer:
        tensorboard_writer = SummaryWriter(str(tensorboard_dir))

        # train the model:
        print("Training the model ...")
        for step in range(1, num_iters + 1):
            # -----------------------------------------------------------------------------------
            # Main training code
            # -----------------------------------------------------------------------------------
            # create batch of crops from the training grid:
            batch = torch.cat(
                [random_cropper(training_grid) for _ in range(batch_size)]
            )

            # sample timesteps: (weights are ignored because they are ones)
            timesteps, _ = timestep_sampler.sample(batch_size, self.device)

            # compute the loss:
            loss = self.diffusion.training_losses(
                model=self.unet,
                x_start=batch,
                t=timesteps,
                clip_denoised=True,
            )["loss"].mean()

            # optimization steps:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # -----------------------------------------------------------------------------------

            # remaining are logging, visualization etc.
            if step % loss_feedback_frequency == 0:
                tensorboard_writer.add_scalar("loss", loss.item(), step)
                print(f"Step {step}/{num_iters}: loss = {loss.item():.4f}")

            if step % sample_frequency == 0:
                print("Creating  intermediate samples for visualization ...")
                # clear cuda cache:
                torch.cuda.empty_cache()
                gc.collect()

                self.visualize_samples_mosaic(
                    shape=full_grid_size,
                    num_samples=9,
                    save_path=log_dir / f"samples_{step}.mp4",
                    num_frames=180,
                    fps=60,
                )

                # clear cuda cache again:
                torch.cuda.empty_cache()
                gc.collect()
            
            if step % save_frequency == 0:
                print("Saving the model ...")
                self.save_model(model_dir / f"model_{step}.pt")
