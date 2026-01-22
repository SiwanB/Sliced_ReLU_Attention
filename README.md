# Sliced_ReLU_Attention


## Overview

This repository contains a PyTorch implementation of **Sliced ReLU attention** mechanisms. It includes a modular **pre-LN** Transformer implementation, with interchangeable attention modules: standard softmax, sliced ReLU, or sliced ReLU-bump. 

The proposed attention mechanisms rely on 1-D slicing and ReLU-based kernels, enabling **quasi-linear attention computation** and avoiding the quadratic complexity of standard softmax attention.

The repository also contains minimal runnable examples and tests to verify correctness and integration.


## Paper:

The theoretical foundations and detailed analysis of Sliced ReLU attention are presented in the paper: 
**Sliced ReLU attention: Quasi-linear contextual expressivity via sorting**
https://arxiv.org/abs/2512.11411

