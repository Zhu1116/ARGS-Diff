import torch as torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim

class Para(nn.Module):
    def __init__(self, ks, C, c):
        super(Para, self).__init__()
        psf = torch.ones([1, 1, ks, ks], dtype=torch.float32) * (1.0 / (ks * ks))
        self.psf = nn.Parameter(psf)
        srf = torch.ones([c, C, 1, 1], dtype=torch.float32) * (1.0 / C)
        self.srf = nn.Parameter(srf)
        self.ks = ks
        self.C = C
        self.c = c

    def forward(self,):
        return self.psf,  self.srf

class estKR:
    def __init__(self, lr_hsi, hr_msi, ks, device):
        self.device = device

        self.hr_msi = hr_msi.to(self.device)
        self.lr_hsi = lr_hsi.to(self.device)
        _,c,_,_ = self.hr_msi.shape
        _,C,w,h = self.lr_hsi.shape

        self.ks = ks
        self.w = w
        self.h = h
        self.C = C
        self.c = c

    def start_est(self, scale_factor=1/4, ite=3000):
        para = Para(self.ks, self.C, self.c).to(self.device)
        scale = int(1/scale_factor)

        para.train()
        optimizer = optim.Adam(para.parameters(), lr=5e-5)
        for i in range(ite):
            psf, srf = para()
            
            pd = int((self.ks - 1)/2)
            lr_msi1 = F.conv2d(self.hr_msi, psf.repeat(self.c, 1, 1, 1), None, (1, 1), (pd, pd), groups=self.c)
            lr_msi1 = lr_msi1[:, :, ::scale, :: scale]
            lr_msi1 = torch.clamp(lr_msi1, 0.0, 1.0)

            srf_div = torch.sum(srf, dim=1, keepdim=True)
            srf_div = torch.div(1.0, srf_div)
            srf_div = torch.transpose(srf_div, 0, 1)
            lr_msi2 = F.conv2d(self.lr_hsi, srf, None)
            lr_msi2 = torch.mul(lr_msi2, srf_div)
            lr_msi2 = torch.clamp(lr_msi2, 0.0, 1.0)

            res = lr_msi1 - lr_msi2
            loss = torch.sum(torch.abs(res))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            para.apply(self.check_weight)
            
            if (i+1) % 500 == 0:
                print(f"Iter: [{i + 1:4d}] -- Loss: {loss.item():.6f}")

        srf = np.squeeze(srf.detach().cpu().numpy())
        psf = np.squeeze(psf.detach().cpu().numpy())
        return psf, srf

    @staticmethod
    def check_weight(model):
        if hasattr(model, 'psf'):
            w = model.psf.data
            w.clamp_(0.0, 1.0)
            psf_div = torch.sum(w)
            psf_div = torch.div(1.0, psf_div)
            w.mul_(psf_div)
        if hasattr(model, 'srf'):
            w = model.srf.data
            w.clamp_(0.0, 10.0)
            srf_div = torch.sum(w, dim=1, keepdim=True)
            srf_div = torch.div(1.0, srf_div)
            w.mul_(srf_div)
