from pathlib import Path
import torch
from gaussian_diffusion import (
    GaussianDiffusion,
    LossType,
    ModelMeanType,
    ModelVarType,
    get_named_beta_schedule,
)
from model import Thre3inFusionModel

from unet import UNetModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    raise NotImplementedError("Not implemented yet")


if __name__ == "__main__":
    training_model_path = Path(
        "/home/karnewar/projects/thre3infusion/base_scenes/white_bkgd/cute_forest/saved_models/model_stage_4_iter_2000.pth"
    )
    output_path = Path("/home/karnewar/projects/thre3infusion/diffusion/cute_forest_epsilon_small_resolution_500_steps")

    unet_model = UNetModel(
        image_size=256,
        in_channels=4,
        model_channels=32,
        out_channels=4,
        num_res_blocks=1,
        attention_resolutions=[],
        use_bottleneck_attn=True,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=3,
        num_classes=None,
        use_checkpoint=True,
        num_heads=4,
        num_head_channels=-1,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_new_attention_order=True,
    ).to(device)
    print("Model has been created ...")

    gaussian_diffusion = GaussianDiffusion(
        betas=get_named_beta_schedule(
            schedule_name="cosine",
            num_diffusion_timesteps=500,
            beta_start_unscaled=0.0001,
            beta_end_unscaled=0.02,
        ),
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
        rescale_timesteps=False,
    )

    model = Thre3inFusionModel(
        unet=unet_model,
        diffusion=gaussian_diffusion,
    )

    model.train(
        volume_model_path=training_model_path,
        output_path=output_path,
        num_iters=100_000,
        learning_rate=8e-5, 
        crop_ratio=0.8,
        batch_size=32,
        loss_feedback_frequency=50,
        sample_frequency=5000,
        save_frequency=5000,
    )
