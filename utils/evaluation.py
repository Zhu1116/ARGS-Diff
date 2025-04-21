import scipy.io as sio
import numpy as np


def compute_ergas(img1, img2, scale):
    d = img1 - img2
    ergasroot = 0
    for i in range(d.shape[2]):
        ergasroot = ergasroot + np.mean(d[:, :, i] ** 2) / np.mean(img1[:, :, i]) ** 2

    ergas = 100 / scale * np.sqrt(ergasroot / d.shape[2])
    return ergas


def compute_psnr(img1, img2):
    assert img1.ndim == 3 and img2.ndim == 3

    img_c, img_w, img_h = img1.shape
    ref = img1.reshape(img_c, -1)
    tar = img2.reshape(img_c, -1)
    msr = np.mean((ref - tar) ** 2, 1)
    # max1 = np.max(ref, 1)
    max1 = np.max(ref, 1)

    psnrall = 10 * np.log10(max1**2 / msr)
    out_mean = np.mean(psnrall)
    return psnrall, out_mean, max1, np.mean(msr) ** 0.5


def compute_sam(x_true, x_pred):
    assert x_true.ndim == 3 and x_true.shape == x_pred.shape

    w, h, c = x_true.shape
    x_true = x_true.reshape(-1, c)
    x_pred = x_pred.reshape(-1, c)

    x_pred[np.where((np.linalg.norm(x_pred, 2, 1)) == 0),] += 0.0001

    sam = (x_true * x_pred).sum(axis=1) / (
        np.linalg.norm(x_true, 2, 1) * np.linalg.norm(x_pred, 2, 1)
    )

    sam = np.arccos(sam) * 180 / np.pi
    mSAM = sam.mean()
    var_sam = np.var(sam)
    return mSAM, var_sam


def MetricsCal(GT, P, scale):  # c,w,h
    GT = GT.transpose(2, 0, 1)
    P = P.transpose(2, 0, 1)

    psnrall, m1, GTmax, rmse = compute_psnr(GT, P) # bandwise mean psnr

    GT = GT.transpose(1, 2, 0)
    P = P.transpose(1, 2, 0)

    m2, _ = compute_sam(GT, P) # sam

    m3 = compute_ergas(GT, P, scale)

    from skimage.metrics import structural_similarity as ssim

    ssims = []
    for i in range(GT.shape[2]):
        ssimi = ssim(
            GT[:, :, i], P[:, :, i], data_range=P[:, :, i].max() - P[:, :, i].min()
        )
        ssims.append(ssimi)
    m4 = np.mean(ssims)

    from sewar.full_ref import uqi

    m5 = uqi(GT, P)

    return np.float64(rmse), np.float64(m1), np.float64(m2), m3, m4, m5
