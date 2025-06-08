# [CVPR 2025] ARGS-Diff

<a href="https://arxiv.org/abs/2505.11800"><img src="https://img.shields.io/badge/ariXv-2505.11800-A42C25.svg" alt="arXiv"></a>

> **Self-Learning Hyperspectral and Multispectral Image Fusion via Adaptive Residual Guided Subspace Diffusion Model**
> <br>
> Jian Zhu, [He Wang](https://scholar.google.com.hk/citations?user=J5bNDdYAAAAJ), [Yang Xu](https://scholar.google.com.hk/citations?user=c8j941EAAAAJ), [Zebin Wu](https://scholar.google.com.hk/citations?user=y_FtCsYAAAAJ), and Zhihui Wei
> <br>
> Nanjing University of Science and Technology

## Framework

<img src='./assets/framework.png' width='100%' />

## Requirements 

1. Environment setup

```shell
conda create -n args python=3.9
conda activate args
```

2. Requirements installation

```shell
pip install -r requirements
```

## Quick Start (using the Pavia dataset as an example)

### Train

refer to [ARGS-Diff-train](https://github.com/Zhu1116/ARGS-Diff-train) to train the spatial and spectral networks

### Sample

1. Place the `pavia.mat` file into the `data` folder. This file should contain the following keys: `LR-HSI`, `HR-MSI`, and optionally `HR-HSI`.

2. Copy the pretrained model file `ema_0.9999_030000.pt` from the training project [ARGS-Diff-train](https://github.com/Zhu1116/ARGS-Diff-train) `spatial_train_result/pavia/` to the `ckpt/pavia/` directory of the current project, and rename it to `spa.pt`.

3. Modify line 46 in `sample_subspace.py` to use `"pavia"`, then run:

   ```bash
   python sample_subspace.py --mode 'semi'
   ```

## Acknowledge

Some of the codes are built upon [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) and [MIAE](https://github.com/liuofficial/MIAE).

