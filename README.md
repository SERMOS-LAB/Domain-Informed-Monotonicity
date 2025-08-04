# Domain-Informed-Monotonicity
DIM: Enforcing Domain-Informed Monotonicity in Deep Neural Networks

# Overview
This repository contains the implementation and experimental code for the paper "DIM: Enforcing Domain-Informed Monotonicity in Deep Neural Networks". DIM is a novel regularization method that maintains domain-informed monotonic relationships in deep learning models to improve predictions and reduce overfitting.

# Key Features
Model-Agnostic: Works with any neural network architecture without requiring structural modifications
Domain-Informed: Incorporates expert knowledge about monotonic relationships between features and outputs
Linear Baseline Reference: Establishes objective violation measurement through fitted linear trends
Consistent Performance: Demonstrates MSE improvements of 20-30% across multiple architectures

# Method Overview
DIM addresses the fundamental limitation of existing monotonicity methods by establishing an explicit linear reference trend before measuring violations. For each monotonic feature, the method:
1) Fits a linear baseline to current model predictions
2) Sorts predictions and corresponding feature values
3) Measures deviations from expected monotonic behavior
4) Applies squared penalty for violations
5) Integrates penalty into training loss function

# Repository Structure
├── README.md
├── models/
│   ├── ann_model.py          # Single-layer ANN implementation
│   ├── cnn_model.py          # Conv1D model implementation
│   ├── mlp3_model.py         # 3-layer MLP implementation
│   └── mlp5_model.py         # 5-layer MLP implementation
├── data/
│   └── alldata_downtownTodowntown.csv  # Chicago ridesourcing dataset

# Installation
Python 3.8 or higher
TensorFlow 2.7.0
CUDA-capable GPU (recommended for faster training)
