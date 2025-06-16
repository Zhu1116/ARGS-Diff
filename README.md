# [CVPR 2025] ARGS-Diff

<a href="https://arxiv.org/abs/2505.11800"><img src="https://img.shields.io/badge/ariXv-2505.11800-A42C25.svg" alt="arXiv"></a> [![arXiv](https://img.shields.io/badge/paper-cvpr2025-cyan)](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhu_Self-Learning_Hyperspectral_and_Multispectral_Image_Fusion_via_Adaptive_Residual_Guided_CVPR_2025_paper.pdf)

> **Self-Learning Hyperspectral and Multispectral Image Fusion via Adaptive Residual Guided Subspace Diffusion Model**
>
> Jian Zhu, [He Wang](https://scholar.google.com.hk/citations?user=J5bNDdYAAAAJ), [Yang Xu](https://scholar.google.com.hk/citations?user=c8j941EAAAAJ), [Zebin Wu](https://scholar.google.com.hk/citations?user=y_FtCsYAAAAJ), and Zhihui Wei
>
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
pip install -r requirements.txt
```

## Quick Start 

```bash
python sample_subspace.py --mode 'semi'
```

## Sample on Your Own Data

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

