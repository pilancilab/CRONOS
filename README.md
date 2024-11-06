# CRONOS: Convex Neural Networks via Operator Splitting

Welcome to the official implementation for the **CRONOS project**! Check out the [paper](https://arxiv.org/abs/2411.01088) for more details.

## Overview

We introduce the **CRONOS** algorithm for convex optimization of two-layer neural networks. This repo contains the official JAX implementation of the CRONOS paper, and allows installation has a handy pip package for all your binary classification needs.

## CRONOS and CRONOS-AM

- **CRONOS**: Uses convex optimization to train two-layer neural networks efficiently at scale. Experiments include fullsize ImageNet, downsampled ImageNet, IMDb, Food, FMNIST, CIFAR-10, MNIST, and synthetic datasets.
- **CRONOS-AM**: CRONOS with Alternating Minimization. This extension allows training of multi-layer networks with arbitrary architectures (MLP, CNN, GPT, etc.).

## Key Features

- **Scalability**: CRONOS can handle high-dimensional datasets.
- **Convergence**: Our theoretical analysis demonstrates that CRONOS converges to the global minimum of the convex reformulation under mild assumptions.
- **Performance**: Large-scale numerical experiments with GPU acceleration in JAX. Optimized to be VRAM friendly without sacrificing speed. 

## Results

---

## Installation

```bash
pip install CRONOS -- user sets dataset, exp, model, optimizer
```
---

## Citation

```bash
@inproceedings{feng2024,
    title={{CRONOS: Convex Neural Networks via Operator Splitting}},
    author={Miria Feng, Zachary Frangella and Mert Pilanci},
    booktitle={Advances in Neural Information Processing Systems},
    year={2024}
}```
---

# TODO: 
- add in jupyter demo
- hydra + omegaconf (user sets dataset, add new dataset, template loader)
- add in instructions for vision and GPT2, especially GPT2 (3 step run process)
- add a requirements.txt? RTX4090 minimum, JAX, NVIDIA, CUDA, NVIDIA driver versions
- add in sharding here, or in separate codebase? 
- consolidate 3 step run process for gpt, consolidate 2 runners
- populate tests for all modules
- populate requirements.txt
