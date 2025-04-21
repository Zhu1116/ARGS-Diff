from diffusion.unet import UNetModel
from diffusion.fcn import FCN
from diffusion.diffusion_subspace import GaussianDiffusion
import torch

def create_models(args):
    spa = UNetModel(
        in_channels=args['rank'],
        model_channels=64,
        out_channels=args['rank'],
        num_res_blocks=2,
        channel_mult=(1, 2, 3, 4),
        use_scale_shift_norm=True
    )
    spa.load_state_dict(torch.load(args['ckpt_spa']))
    spa = spa.to(args['device'])

    if 'pavia' in args['data_type']:
        spe = FCN(
            ch_in=103, ch_out=103,
            num_hidden=[256, 512, 256],
            time_embedding_dim=60,
            num_embeddings=1000
        )
    elif 'chikusei' in args['data_type']:
        spe = FCN(
            ch_in=128, ch_out=128,
            num_hidden=[256, 512, 256],
            time_embedding_dim=70,
            num_embeddings=1000
        )
    elif 'ksc' in args['data_type']:
        spe = FCN(
            ch_in=176, ch_out=176,
            num_hidden=[400, 800, 400],
            time_embedding_dim=100,
            num_embeddings=1000
        )
    elif 'houston' in args['data_type']:
        spe = FCN(
            ch_in=48, ch_out=48,
            num_hidden=[128, 256, 128],
            time_embedding_dim=30,
            num_embeddings=1000
        )
    else:
        spe = None
    spe.load_state_dict(torch.load(args['ckpt_spe']))
    spe = spe.to(args['device'])

    diffusion = GaussianDiffusion(
        [spa, spe],
        timesteps=args['num_steps'],
        sampling_timesteps=args['sampling_steps'],
        beta_schedule=args['beta_schedule'],
        args=args
    ).to(args['device'])

    return diffusion
