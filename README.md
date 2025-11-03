# A First-Principles Replication of Sharpness-Aware Minimization (SAM)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Framework-EE4C2C.svg)](https://pytorch.org/)
[![wandb](https://img.shields.io/badge/Weights_&_Biases-Experiment_Tracking-blue)](https://wandb.ai)

## 1. Objective

This repository contains an independent, from-scratch implementation of the **Sharpness-Aware Minimization (SAM)** optimizer, as introduced in the ICLR 2021 paper by Foret et al.

The primary objective of this project was not simply to use the algorithm, but to **build it from first principles**, using only the original paper as a blueprint. This is an exercise in scientific rigor, deep comprehension, and reproducibility.

**Original Paper:** [Sharpness-Aware Minimization for Efficiently Improving Generalization](https://arxiv.org/abs/2010.01412)

---

## 2. Core Implementation

The heart of this repository is a minimalist, pure Python `SAM` class that inherits from `torch.optim.Optimizer`. It follows the two-step process described in the paper:

1.  **`first_step()` (The Ascent):** The optimizer first performs an "adversarial" step, moving the model's weights to a point of higher loss within a defined `rho` neighborhood. This is achieved by calculating the global gradient norm and ascending along the scaled gradient direction.

2.  **`second_step()` (The Descent):** After the ascent, a second forward and backward pass is performed in the main training loop to get the gradient at this perturbed position. The `second_step()` then reverts the weights to their original state and uses this "sharpness-aware" gradient to perform the actual descent with a standard base optimizer (e.g., SGD).

---

## 3. A Note on Performance: Pure Python vs. Production Implementations

This implementation is **algorithmically correct but not performance-optimized.** During testing on a GTX 1650, it achieves a speed of approximately `~1.9s/it`.

This slowdown is an expected and understood consequence of a pure Python implementation of the `_grad_norm` function. The function iterates through each parameter in Python, launching numerous small computational kernels on the GPU. The overhead of these sequential launches from the CPU is the primary bottleneck.

Production-grade implementations of SAM (like the one in `timm`) overcome this by using optimized C++ or CUDA extensions to compute the gradient norm in a single, massive, low-level operation.

The focus of this project is **algorithmic correctness and clarity**, not production-level performance optimization. It is a demonstration of the ability to translate a complex algorithm from a paper into functional code.

---

## 4. How to Run the Replication

### Prerequisites
- Python 3.10+
- PyTorch
- `wandb`
- `tqdm`
- `numpy`

### Installation

```bash
git clone https://github.com/S-Sairam/sam-optimizer.git
cd sam-optimizer
pip install -r requirements.txt
```

### Training
To launch the full 200-epoch replication run, use the following command with the parameters specified in the paper's appendix:
```bash
python3 train.py --epochs 200 --lr 0.1 --batch_size 128 --rho 0.05
```
## 5. Results

The goal is to reproduce the paper's reported accuracy for a Wide-ResNet-28-10 on CIFAR-10.

| Model / Experiment | Reported Test Accuracy (%) |
| :--- | :---: |
| **SAM (Foret et al., ICLR 2021)** | **~97.3** |
| **This Replication** | **~96.74** |

The full, transparent logs for the official replication run are publicly available on Weights & Bienses.

[**View the Run on Weights & Biases**](https://wandb.ai/pesu-ai-ml/sam-replication-cifar10/runs/mjyz5xy4)

