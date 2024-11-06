# CRONOS: Convex Neural Networks via Operator Splitting

Welcome to the official implementation for the **CRONOS project**! 

## Overview

We introduce the **CRONOS** algorithm for convex optimization of two-layer neural networks. 

## CRONOS and CRONOS-AM

- **CRONOS**: Our primary algorithm, leveraging convex optimization to train two-layer neural networks efficiently at scale. Experiments include fullsize ImageNet, downsampled ImageNet, IMDb, Food, FMNIST, CIFAR-10, MNIST, and synthetic datasets.
- **CRONOS-AM**: Building on CRONOS, we develop CRONOS-AM which combines CRONOS with alternating minimization. This extension allows training of multi-layer networks with arbitrary architectures.

## Key Features

- **Scalability**: CRONOS can handle high-dimensional datasets.
- **Convergence**: Our theoretical analysis demonstrates that CRONOS converges to the global minimum of the convex reformulation under mild assumptions.
- **Performance**: Large-scale numerical experiments with GPU acceleration in JAX. Optimized to be VRAM friendly without sacrificing speed. 

## Results

---

## Installation

```bash
pip install CRONOS -- user sets dataset, exp, model, optimzier
```
---

https://arxiv.org/abs/2411.01088 


# TODO: 
- add in jupyter demo
- hydra + omegaconf (user sets dataset, add new dataset, template loader)
- add in instructions for vision and GPT2, especially GPT2 (3 step run process)
- add a requirements.txt? RTX4090 minimum, JAX, NVIDIA, CUDA, NVIDIA driver versions
- add in sharding here, or in separate codebase? 
- consolidate 3 step run process for gpt, consolidate 2 runners
- populate tests for all modules
- populate requirements.txt