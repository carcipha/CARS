# CARS
Competitive Adaptive Reweighted Sampling for feature selection

This repository provides a Python implementation of **Competitive Adaptive Reweighted Sampling (CARS)** for feature selection using **Partial Least Squares Regression (PLS-R)**.

> CARS was originally described in:  
> **Li, H., Liang, Y., Xu, Q., & Cao, D. (2009).**  
> *Key wavelengths screening using competitive adaptive reweighted sampling method for multivariate calibration.*  
> *Analytica Chimica Acta*, 648(1), 77–84.  
> [Read the full paper here](https://www.sciencedirect.com/science/article/pii/S0003267009008332)

## Usage

```python
import cars

# X: numpy array with features
# y: numpy array with the response
selected_features = cars(X, y, n_iterations=100, ncomp=30, V=True)

X = X[:,selected_features] # subset selected features
```

