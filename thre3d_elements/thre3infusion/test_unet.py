import torch

from unet import UNetModel


if __name__ == "__main__":

    """
        Test the UNetModel class.  Please uncomment the print statements in the 
        UNetModel class to see the output of the forward pass.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet_model = UNetModel(
        image_size=512,
        in_channels=4,
        model_channels=16,
        out_channels=4,
        num_res_blocks=1,
        attention_resolutions=[],
        use_bottleneck_attn=True,
        channel_mult=(1, 1, 2, 2),
        conv_resample=True,
        dims=3,
        num_classes=None,
        use_checkpoint=True,
        num_heads=1,
        num_head_channels=-1,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_new_attention_order=True,
    ).to(device)
    print("Model has been created ...")

    print(f"currently using device: {next(unet_model.parameters()).device}")

    batch_size = 6
    random_input = torch.randn(
        batch_size, 4, 128, 128, 64, dtype=torch.float32, device=device
    )
    random_timesteps = torch.randint(
        low=0, high=100, size=(batch_size,), dtype=torch.int32, device=device
    )
    output = unet_model(random_input, random_timesteps)

    # testing backward pass:
    output.sum().backward()
    print("Backward pass has been completed ...")
