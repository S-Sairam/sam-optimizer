# A Minimal, Rigorous Replication of Sharpness-Aware Minimization (SAM) for CIFAR-10

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Framework-EE4C2C.svg)](https://pytorch.org/)
[![wandb](https://img.shields.io/badge/Weights_&_Biases-Experiment_Tracking-blue)](https://wandb.ai)

## Objective

This repository contains an independent, minimal, and rigorous replication of the CIFAR-10 results from the landmark paper: **"Sharpness-Aware Minimization for Efficiently Improving Generalization"** by Foret et al. (ICLR 2021).

The goal of this project was not to invent a new method, but to demonstrate a deep, first-principles understanding of a state-of-the-art optimization algorithm by reproducing its published results. This is an exercise in scientific rigor and reproducibility.

**Original Paper:** [https://arxiv.org/abs/2010.01412](https://arxiv.org/abs/2010.01412)

---

## Core Implementation

The core of this repository is a clean-room implementation of the SAM optimizer, written in PyTorch. The implementation is designed to be minimal and easy to understand, directly translating the algorithm presented in the paper.
