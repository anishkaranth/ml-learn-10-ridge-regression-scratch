# Week 10 — Ridge (L2) Regression from Scratch (NumPy)

## Learning goal

Build intuition for **L2 regularization** by implementing **ridge regression** with NumPy: closed-form solution and gradient descent, compare OLS vs ridge on synthetic **collinear** features, watch coefficient shrinkage vs lambda, and plot train/test MSE curves across a lambda grid.

## What you'll build

- A collinear 3-feature regression toy set (no downloads, no scikit-learn for the core)
- A `RidgeRegression` class with closed-form and GD solvers (intercept unpenalized)
- OLS baseline, coefficient bar comparison, shrinkage path, train/test MSE vs lambda

## How to run

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Script (VS Code / terminal):**

```bash
python ridge_regression_scratch.py
```

This generates collinear features, fits OLS + ridge (closed-form & GD), sweeps lambda, prints MSE diagnostics, and saves `outputs/ridge_results.png`.

**Notebook (interactive):**

```bash
jupyter notebook notebooks/ridge_regression_scratch.ipynb
```

Both share the same core ideas. The notebook adds markdown intuition and inline plots; the `.py` script is a clean, runnable walkthrough.

## Requirements

- Python 3.9+
- `numpy`, `matplotlib`, `jupyter` (see `requirements.txt`)
- No scikit-learn for the ridge core — the ML logic and toy data are from scratch

## What you'll learn

- Why **multicollinearity** makes OLS coefficients unstable
- How the **closed-form** ridge solution `(X'X + λD)^{-1} X'y` shrinks weights
- That **GD** recovers the same coefficients (approximately) as the closed form
- How train vs test **MSE curves** reveal the bias–variance trade-off over λ
