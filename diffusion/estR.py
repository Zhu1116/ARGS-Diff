import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Para(nn.Module):
    def __init__(self, c, C):
        super().__init__()
        srf = torch.ones([c, C], dtype=torch.float32) * (1.0 / C)
        self.srf = nn.Parameter(srf)

    def forward(self):
        return F.softmax(self.srf)


class estR:
    def __init__(self, device):
        self.device = device

    def start_est(self, lr_msi1, lr_hsi, ite=3000):
        _, c, _, _ = lr_msi1.shape
        b, C, w, h = lr_hsi.shape
        para = Para(c, C).to(self.device)
        para.train()
        optimizer = optim.Adam(para.parameters(), lr=5e-2)

        for i in range(ite):
            srf = para()
            lr_hsi = lr_hsi.reshape(1, C, -1)
            lr_msi2 = torch.matmul(srf, lr_hsi.squeeze(0)).reshape(srf.shape[0], w, h).unsqueeze(0)

            res = lr_msi1 - lr_msi2
            loss = torch.mean(torch.abs(res))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 1000 == 0:
                print(f"Iter: [{i + 1:4d}] -- Loss: {loss.item():.8f}")
        return srf.detach().cpu().numpy()