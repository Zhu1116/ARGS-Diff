import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm.auto import tqdm


# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def to_rgb(img, data_type):
    if 'pavia' in data_type:
        return np.concatenate((img[:, :, 61:62], img[:, :, 36:37], img[:, :, 10:11]), axis=2)
    elif 'chikusei' in data_type:
        return np.concatenate((img[:, :, 56:57], img[:, :, 23:24], img[:, :, 13:14]), axis=2)
    elif 'ksc' in data_type:
        return np.concatenate((img[:, :, 43:44], img[:, :, 20:21], img[:, :, 12:13]), axis=2)
    elif 'houston' in data_type:
        return np.concatenate((img[:, :, 17:18], img[:, :, 12:13], img[:, :, 10:11]), axis=2)
    return None

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def exp_beta_schedule(timesteps):
    k = 10
    alphas_cumprod = np.exp(-k * np.arange(timesteps + 1) / timesteps)
    alphas_cumprod = np.flip(1 - alphas_cumprod)
    alphas_cumprod = (alphas_cumprod - alphas_cumprod.min()) / (alphas_cumprod.max() - alphas_cumprod.min()) * (
                1 - 1e-3) + 1e-3
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = 1 - alphas
    return torch.tensor(betas)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_noise',
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        offset_noise_strength = 0.,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5,
        args = None
    ):
        super().__init__()

        self.args = args
        self.model = model
        self.srf = None
        self.down_fn = None
        self.lambd1 = 1
        self.lambd2 = 1

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        elif beta_schedule == 'exp':
            beta_schedule_fn = exp_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        self.alphas = alphas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # offset noise strength - in blogpost, they claimed 0.1 was ideal

        self.offset_noise_strength = offset_noise_strength

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    @property
    def device(self):
        return self.betas.device

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance_subspace(self, x, t, x_self_cond = None, clip_denoised = True, rederive_pred_noise = False):
        spa_img = x[0]
        spe_img = x[1]

        spa_pred_noise = self.model[0](spa_img, t, x_self_cond)
        spe_pred_noise = self.model[1](spe_img.squeeze(0), t).unsqueeze(0)

        spa_x_start = self.predict_start_from_noise(spa_img, t, spa_pred_noise)
        spe_x_start = self.predict_start_from_noise(spe_img, t, spe_pred_noise)

        if clip_denoised:
            spa_x_start.clamp_(-1., 1.)
            spe_x_start.clamp_(-1., 1.)

        if rederive_pred_noise:
            spa_pred_noise = self.predict_noise_from_start(spa_img, t, spa_x_start)
            spe_pred_noise = self.predict_noise_from_start(spe_img, t, spe_x_start)

        spa_mean, spa_variance, spa_log_variance = self.q_posterior(x_start = spa_x_start, x_t = spa_img, t = t)
        spe_mean, spe_variance, spe_log_variance = self.q_posterior(x_start = spe_x_start, x_t = spe_img, t = t)
        return {
            'spa_mean': spa_mean,
            'spa_variance': spa_variance,
            'spa_log_variance': spa_log_variance,
            'spa_x_start': spa_x_start,
            'spa_pred_noise': spa_pred_noise,
            'spe_mean': spe_mean,
            'spe_variance': spe_variance,
            'spe_log_variance': spe_log_variance,
            'spe_x_start': spe_x_start,
            'spe_pred_noise': spe_pred_noise
        }

    def p_sample_loop_subspace(self, x, y):
        out_img = None
        distances = []

        times = torch.linspace(-1, self.num_timesteps - 1, steps = self.sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        device = self.device
        spa_img, spe_img = x[0], x[1]
        b = spa_img.shape[0]

        rho1 = 0.05  # 空间步长
        rho2 = 0.05  # 光谱步长
        beta1 = 0.9  # 一阶矩估计的衰减率
        beta2 = 0.999  # 二阶矩估计的衰减率
        epsilon = 1e-8  # 防止除零的小数
        adam_t = 0  # Adam时间步

        # 初始化一阶和二阶矩估计
        m1 = torch.zeros_like(spa_img)  # 一阶矩
        v1 = torch.zeros_like(spa_img)  # 二阶矩
        m2 = torch.zeros_like(spe_img)  # 一阶矩
        v2 = torch.zeros_like(spe_img)  # 二阶矩

        pbar = tqdm(time_pairs)
        for time, time_next in pbar:
            spa_img = spa_img.requires_grad_()
            spe_img = spe_img.requires_grad_()

            batched_times = torch.full((b,), time, device = device, dtype = torch.long)
            out = self.p_mean_variance_subspace(x = [spa_img, spe_img], t = batched_times, x_self_cond = None,
                                                clip_denoised = True, rederive_pred_noise = True)

            if time_next < 0:
                spa_img = out['spa_x_start']
                spe_img = out['spe_x_start']
                spa_img = spa_img.detach()
                spe_img = spe_img.detach()
                out_img = self.record_img(spa_img, spe_img, time)
                break

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = self.ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            spa_noise = torch.randn_like(spa_img) if time > 0 else 0.
            spe_noise = torch.randn_like(spe_img) if time > 0 else 0.
            spa_pred_img = out['spa_x_start'] * alpha_next.sqrt() + c * out['spa_pred_noise'] + sigma * spa_noise
            spe_pred_img = out['spe_x_start'] * alpha_next.sqrt() + c * out['spe_pred_noise'] + sigma * spe_noise

            # do dps
            spa_x_start = self.unnormalize(out['spa_x_start'])
            spe_x_start = self.unnormalize(out['spe_x_start'])
            x_start_hat = torch.einsum("ijkl, ijp->ipkl", spa_x_start, spe_x_start)

            LRHSI_hat = self.spa_deg_fn(x_start_hat)
            HRMSI_hat = self.spe_deg_fn(x_start_hat)

            norm1 = torch.norm(y['LRHSI'] - LRHSI_hat)
            norm2 = torch.norm(y['HRMSI'] - HRMSI_hat)

            likelihood = norm1 + norm2 * self.lambd1

            spa_norm_grad, spe_norm_grad = torch.autograd.grad(outputs=likelihood, inputs=[spa_img, spe_img])

            # Adam
            adam_t += 1
            m1 = beta1 * m1 + (1 - beta1) * spa_norm_grad
            m2 = beta1 * m2 + (1 - beta1) * spe_norm_grad
            v1 = beta2 * v1 + (1 - beta2) * (spa_norm_grad ** 2)
            v2 = beta2 * v2 + (1 - beta2) * (spe_norm_grad ** 2)
            m_hat1 = m1 / (1 - beta1 ** adam_t)
            m_hat2 = m2 / (1 - beta1 ** adam_t)
            v_hat1 = v1 / (1 - beta2 ** adam_t)
            v_hat2 = v2 / (1 - beta2 ** adam_t)
            spa_pred_img = spa_pred_img - rho1 * m_hat1 / (torch.sqrt(v_hat1) + epsilon)
            spe_pred_img = spe_pred_img - rho2 * m_hat2 / (torch.sqrt(v_hat2) + epsilon)

            spa_img = spa_pred_img.detach()
            spe_img = spe_pred_img.detach()

            # Adaptive Residual Guided Module (ARGM)
            for i in range(1):
                spa_img, spe_img = self.argm(spa_img, spe_img, y, rho1, rho2)

            # record distance
            pbar.set_postfix({'distance': likelihood.item()}, refresh=False)
            distances.append(likelihood.item())

            # recode img
            if time in [999, 899, 799, 699, 599, 499, 399, 299, 199, 99, 1]:
                out_img = self.record_img(spa_img, spe_img, time)


        plt.plot(distances[200:])
        plt.savefig(self.args['result_dir'] + '/distances.png')

        # data range is -1~1
        out_spa_img = np.transpose(spa_img.cpu().detach().squeeze(0).numpy(), (1, 2, 0))
        out_spe_img = np.transpose(spe_img.cpu().detach().squeeze(0).numpy())

        return {
            'img': out_img,
            'spa': out_spa_img,
            'spe': out_spe_img,
            'distance': likelihood.item()
        }

    def sample_subspace(self, x, y):
        return self.p_sample_loop_subspace(x, y)

    def argm(self, spa_img, spe_img, y, rho1, rho2, ratio=10.0):
        spa_img = spa_img.requires_grad_()
        spe_img = spe_img.requires_grad_()

        spa_hat = self.unnormalize(spa_img)
        spe_hat = self.unnormalize(spe_img)
        x_hat = torch.einsum("ijkl, ijp->ipkl", spa_hat, spe_hat)

        LRHSI_hat = self.spa_deg_fn(x_hat)
        HRMSI_hat = self.spe_deg_fn(x_hat)

        norm1 = torch.norm(y['LRHSI'] - LRHSI_hat)
        norm2 = torch.norm(y['HRMSI'] - HRMSI_hat)

        likelihood = norm1 + norm2 * self.lambd2
        spa_norm_grad, spe_norm_grad = torch.autograd.grad(outputs=likelihood, inputs=[spa_img, spe_img])

        spa_img = spa_img - spa_norm_grad * (rho1 / ratio)
        spe_img = spe_img - spe_norm_grad * (rho2 / ratio)

        spa_img = spa_img.detach()
        spe_img = spe_img.detach()
        return spa_img, spe_img

    def spa_deg_fn(self, x):
        return self.down_fn(x)

    def spe_deg_fn(self, x):
        srf = np.transpose(self.srf)
        srf = torch.from_numpy(srf).float().to(self.device)
        hr_msi = torch.squeeze(x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)) @ srf
        hr_msi = hr_msi.permute(1, 0).reshape(1, srf.shape[1], x.shape[2], x.shape[3])
        return hr_msi

    def record_img(self, spa_img, spe_img, idx):
        b, c, H, W = spa_img.shape
        _, _, C = spe_img.shape
        out_shape = (b, C, H, W)

        out_img = (torch.bmm(self.unnormalize(spe_img).permute([0, 2, 1]),
                     self.unnormalize(spa_img).reshape(b, c, -1)).reshape(*out_shape))
        out_img = out_img.cpu().squeeze().numpy()
        out_img = np.transpose(out_img, (1, 2, 0))
        out_img = np.clip(out_img, 0, 1)

        file_path = self.args['result_dir'] + f'/x/x_{str(idx).zfill(4)}.png'
        plt.imsave(file_path, to_rgb(out_img, self.args['data_type']))

        return out_img
