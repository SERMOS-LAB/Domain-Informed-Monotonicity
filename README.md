# Domain-Informed-Monotonicity
DIM: Enforcing Domain-Informed Monotonicity in Deep Neural Networks

# Overview
This repository contains the implementation and experimental code for the paper "DIM: Enforcing Domain-Informed Monotonicity in Deep Neural Networks". DIM is a new regularization method that maintains domain-informed monotonic relationships in deep learning models to improve predictions and reduce overfitting.

# Key Features
1) **Model-Agnostic**: Works with any neural network architecture without requiring structural modifications
2) **Domain-Informed**: Incorporates expert knowledge about monotonic relationships between features and outputs
3) **Linear Baseline Reference**: Establishes objective violation measurement through fitted linear trends
4) **Consistent Performance**: Demonstrates MSE improvements of 20-30% across multiple architectures

# Method Overview
DIM addresses the fundamental limitation of existing monotonicity methods by establishing an explicit linear reference trend before measuring violations. For each monotonic feature, the method:
1) Fits a linear baseline to current model predictions
2) Sorts predictions and corresponding feature values
3) Measures deviations from expected monotonic behavior
4) Applies squared penalty for violations
5) Integrates penalty into training loss function

# Repository Structure
```
├── README.md
├── models/
│   ├── ann_model.py              # Single-layer ANN implementation
│   ├── cnn_model.py              # Conv1D model implementation
│   ├── mlp3_model.py             # 3-layer MLP implementation
│   └── mlp5_model.py             # 5-layer MLP implementation
├── data/
│   ├── alldata_downtownTodowntown.csv     # Chicago ridesourcing dataset
│   └── synthetic_monotonic_trips.csv      # Synthetic data generation
```

# Installation
Python 3.8 or higher
TensorFlow 2.7.0
CUDA-capable GPU (recommended for faster training)

# Dataset Configuration

By default, all models are configured to use the Chicago ridesourcing dataset. To run experiments on the synthetic dataset, you need to modify the `file_path` variable in each model file.

# Switching to Synthetic Dataset

In each model file (`ann_model.py`, `cnn_model.py`, `mlp3_model.py`, `mlp5_model.py`), change the file path:

```python
# Change this line:
file_path = './alldata_downtownTodowntown.csv'

# To this:
file_path = './synthetic_monotonic_trips.csv'

```
# Dataset-Specific Monotonic Features
## Chicago Dataset:
```python
monotonic_features = ['downtown_downtown', 'EmpDen_Des', 'EmpDen_Ori', 'Commuters_HW', 'Commuters_WH']
```
## Synthetic Dataset:
```python
monotonic_features = ['x1', 'x2', 'x3']  # Adjust based on your synthetic data structure
```
Make sure to update the monotonic_features list accordingly when switching between datasets.

# Dropped Columns
## Chicago Dataset:
```python
X = data.drop(columns=['total_number_trips', 'Unnamed: 0'])
```
## Synthetic Dataset:
```python
X = data.drop(columns=['total_number_trips'])
```
Make sure to update the dropped columns accordingly when switching between datasets.
