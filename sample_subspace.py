import argparse
import os
import numpy as np
import torch as th
import scipy.io as sio
import torch.cuda
from functools import partial
import torch.nn.functional as F

from diffusion.estR import estR
from diffusion.estKR import estKR
from utils.evaluation import MetricsCal
from utils.resizer import Resizer
from diffusion.create_model import create_models
import random

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def set_seed(seed):
    # https://github.com/pytorch/pytorch/issues/7068
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_data(arg):
    data_path = os.path.join(arg['data_dir'], data_type + '.mat')
    mat = sio.loadmat(data_path)

    HRMSI = np.float32(mat['HRMSI'])
    LRHSI = np.float32(mat['LRHSI'])
    if 'HRHSI' in mat:
        HRHSI = np.float32(mat['HRHSI'])
    else:
        HRHSI = np.ones((HRMSI.shape[0], HRMSI.shape[1], LRHSI.shape[2]), dtype=np.float32)

    HRMSI_tensor = th.from_numpy(HRMSI).permute(2, 0, 1).unsqueeze(0)
    LRHSI_tensor = th.from_numpy(LRHSI).permute(2, 0, 1).unsqueeze(0)

    return HRHSI, {'LRHSI': LRHSI_tensor.to(arg['device']), 'HRMSI': HRMSI_tensor.to(arg['device'])}


if __name__ == "__main__":
    data_type = 'pavia'  # pavia, chikusei, ksc, houston

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default=data_type)
    parser.add_argument('--data_dir', type=str, default='data/'+data_type)
    parser.add_argument('--result_dir', type=str, default='result/'+data_type+'/output')
    parser.add_argument('--ckpt_spa', type=str, default='ckpt/'+data_type+'/spa.pt')
    parser.add_argument('--ckpt_spe', type=str, default='ckpt/'+data_type+'/spe.pt')
    parser.add_argument('--mode', type=str, default='semi', choices=['semi', 'blind'])

    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--sampling_steps', type=int, default=500)
    parser.add_argument('--beta_schedule', type=str, default='exp')

    args = parser.parse_args()
    args = vars(args)

    set_seed(47)

    os.makedirs(args['result_dir'], exist_ok=True)
    if not os.path.exists(os.path.join(args['result_dir'], 'x')):
        os.mkdir(os.path.join(args['result_dir'], 'x'))

    args['device'] = 'cpu' if int(args['gpu']) < 0 else 'cuda:'+args['gpu']

    # initialize model
    diffusion = create_models(args)

    # load data
    HRHSI, y = load_data(args)
    H, W, C = HRHSI.shape

    if args['mode'] == 'semi':
        # est srf
        diffusion.down_fn = Resizer((1, C, H, W), 1 / args['scale']).to(args['device'])
        srf_path = args['data_dir'] + '/R.mat'
        if not os.path.exists(srf_path):
            hr_msi = y['HRMSI']
            lr_msi1 = diffusion.spa_deg_fn(hr_msi)
            lr_hsi = y['LRHSI']
            est = estR(device=args['device'])
            srf = est.start_est(lr_msi1, lr_hsi)

            diffusion.srf = srf
            sio.savemat(srf_path, {'R': srf})
        else:
            diffusion.srf = sio.loadmat(srf_path)['R']
    else:
        # est srf and psf
        kr_path = args['data_dir'] + '/KR.mat'
        ks = 2 * args['scale'] - 1
        if not os.path.exists(kr_path):
            hr_msi = y['HRMSI']
            lr_hsi = y['LRHSI']
            est = estKR(lr_hsi, hr_msi, ks, args['device'])
            psf, srf = est.start_est()

            sio.savemat(kr_path, {'K': psf, 'R': srf})
        else:
            kr = sio.loadmat(kr_path)
            srf = kr['R']
            psf = kr['K']

        diffusion.srf = srf
        psf = torch.tensor(psf).repeat(C, 1, 1, 1).to(args['device'])
        diffusion.down_fn = partial(F.conv2d, weight=psf, stride=args['scale'], padding=(ks - 1) // 2, groups=C)

    # initialize input(noise)
    spa_img = torch.randn([1, args['rank'], H, W], device=args['device'])
    spe_img = torch.randn([1, args['rank'], C], device=args['device'])

    # sample
    out = diffusion.sample_subspace(x=[spa_img, spe_img], y=y)
    recon = out['img']

    # evaluate
    rmse, psnr, sam, ergas, ssim, uiqi = MetricsCal(HRHSI, recon, 4)
    print('rmse: ', rmse)
    print('psnr: ', psnr)
    print('sam:  ', sam)
    print('ergas:', ergas)
    print('ssim: ', ssim)
    print('uiqi: ', uiqi)

    # save
    sio.savemat(args['result_dir']+'/recon_info.mat',
                {
                    'rmse': rmse,
                    'psnr': psnr,
                    'sam': sam,
                    'ergas': ergas,
                    'ssim': ssim,
                    'uiqi': uiqi,
                    'distance': out['distance'],
                    'HRHSI': HRHSI,
                    'recon': recon,
                    'A': out['spa'],
                    'E': out['spe']
                })
