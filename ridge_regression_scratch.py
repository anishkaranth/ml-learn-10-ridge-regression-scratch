"""
Week 10 — Ridge (L2) Regression from Scratch (NumPy)

Educational script: closed-form ridge and gradient descent, compare OLS vs
ridge on synthetic collinear features, coefficient shrinkage vs lambda, and
train/test MSE curves across a lambda grid.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# 1. Synthetic collinear regression data
# ---------------------------------------------------------------------------

def make_collinear_regression(
    n_samples: int = 120,
    noise: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    y = 3*x0 - 2*x1 + 0*x2 + noise, where x1 ≈ x0 and x2 ≈ x0
    (strong multicollinearity so OLS coefficients become unstable).

    Returns X (n, 3), y (n,), true_coef (3,)
    """
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n_samples)
    x1 = x0 + rng.normal(scale=0.05, size=n_samples)  # nearly collinear with x0
    x2 = x0 + rng.normal(scale=0.08, size=n_samples)  # also nearly collinear
    X = np.column_stack([x0, x1, x2])
    true_coef = np.array([3.0, -2.0, 0.0])
    y = X @ true_coef + rng.normal(scale=noise, size=n_samples)
    return X, y, true_coef


def train_test_split_xy(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.3, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.permutation(n)
    n_test = max(1, int(round(n * test_size)))
    te, tr = idx[:n_test], idx[n_test:]
    return X[tr], X[te], y[tr], y[te]


def add_bias(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X])


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


# ---------------------------------------------------------------------------
# 2. OLS and Ridge (closed-form + GD)
# ---------------------------------------------------------------------------

class RidgeRegression:
    """
    Ridge regression with optional intercept (not penalized).

    Closed form (design matrix Phi = [1 | X]):
      beta = (Phi^T Phi + lambda * D)^{-1} Phi^T y
    where D = diag(0, 1, 1, ..., 1) so the intercept is unpenalized.

    Also supports plain gradient descent for pedagogy.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        method: str = "closed_form",
        lr: float = 0.05,
        n_iters: int = 2000,
        seed: int = 0,
    ):
        self.alpha = float(alpha)
        self.method = method
        self.lr = float(lr)
        self.n_iters = int(n_iters)
        self.seed = int(seed)
        self.coef_: np.ndarray | None = None   # weights for features (no bias)
        self.intercept_: float = 0.0
        self.loss_history_: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegression":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        Phi = add_bias(X)
        n, p = Phi.shape
        # Penalty matrix: do not penalize intercept
        D = np.eye(p)
        D[0, 0] = 0.0

        if self.method == "closed_form":
            A = Phi.T @ Phi + self.alpha * D
            beta = np.linalg.solve(A, Phi.T @ y)
            self.loss_history_ = []
        elif self.method == "gd":
            rng = np.random.default_rng(self.seed)
            beta = rng.normal(scale=0.01, size=p)
            self.loss_history_ = []
            for _ in range(self.n_iters):
                pred = Phi @ beta
                resid = pred - y
                # (1/n) Phi^T resid + (alpha/n) D beta
                grad = (Phi.T @ resid) / n + (self.alpha / n) * (D @ beta)
                beta = beta - self.lr * grad
                loss = 0.5 * float(np.mean(resid**2)) + 0.5 * (self.alpha / n) * float(beta[1:] @ beta[1:])
                self.loss_history_.append(loss)
        else:
            raise ValueError("method must be 'closed_form' or 'gd'")

        self.intercept_ = float(beta[0])
        self.coef_ = beta[1:].copy()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Call fit() first.")
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_


def ols_fit(X: np.ndarray, y: np.ndarray) -> RidgeRegression:
    """OLS is ridge with alpha=0 (closed form)."""
    return RidgeRegression(alpha=0.0, method="closed_form").fit(X, y)


# ---------------------------------------------------------------------------
# 3. Lambda sweep & plots
# ---------------------------------------------------------------------------

def lambda_sweep(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alphas: np.ndarray,
) -> dict[str, np.ndarray]:
    train_mses, test_mses = [], []
    coefs = []
    for a in alphas:
        model = RidgeRegression(alpha=float(a), method="closed_form").fit(X_train, y_train)
        train_mses.append(mse(y_train, model.predict(X_train)))
        test_mses.append(mse(y_test, model.predict(X_test)))
        coefs.append(model.coef_.copy())
    return {
        "alphas": alphas,
        "train_mse": np.array(train_mses),
        "test_mse": np.array(test_mses),
        "coefs": np.vstack(coefs),
    }


def plot_results(
    sweep: dict[str, np.ndarray],
    ols_coef: np.ndarray,
    ridge_coef: np.ndarray,
    true_coef: np.ndarray,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    alphas = sweep["alphas"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    ax = axes[0]
    ax.plot(alphas, sweep["train_mse"], label="train MSE", color="#1f77b4")
    ax.plot(alphas, sweep["test_mse"], label="test MSE", color="#d62728")
    ax.set_xscale("log")
    ax.set_xlabel("lambda (alpha)")
    ax.set_ylabel("MSE")
    ax.set_title("Train/test MSE vs lambda")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    for j in range(sweep["coefs"].shape[1]):
        ax2.plot(alphas, sweep["coefs"][:, j], label=f"w{j}")
    ax2.axhline(0, color="gray", ls="--", lw=0.8)
    ax2.set_xscale("log")
    ax2.set_xlabel("lambda (alpha)")
    ax2.set_ylabel("coefficient")
    ax2.set_title("Coefficient shrinkage path")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    idx = np.arange(len(true_coef))
    width = 0.25
    ax3.bar(idx - width, true_coef, width, label="true", color="#2ca02c")
    ax3.bar(idx, ols_coef, width, label="OLS", color="#ff7f0e")
    ax3.bar(idx + width, ridge_coef, width, label="ridge", color="#1f77b4")
    ax3.set_xticks(idx)
    ax3.set_xticklabels([f"w{i}" for i in idx])
    ax3.set_title("True vs OLS vs Ridge coefs")
    ax3.legend()
    ax3.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "ridge_results.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

def main() -> None:
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Week 10 — Ridge Regression from Scratch (NumPy)")
    print("=" * 60)

    X, y, true_coef = make_collinear_regression(n_samples=150, noise=0.6, seed=42)
    X_train, X_test, y_train, y_test = train_test_split_xy(X, y, test_size=0.3, seed=1)
    print(f"Train: {len(y_train)}, Test: {len(y_test)}, features: {X.shape[1]}")
    print(f"True coefficients: {true_coef}")

    ols = ols_fit(X_train, y_train)
    ridge = RidgeRegression(alpha=10.0, method="closed_form").fit(X_train, y_train)
    ridge_gd = RidgeRegression(alpha=10.0, method="gd", lr=0.1, n_iters=3000).fit(X_train, y_train)

    print(f"\nOLS  coefs: {ols.coef_.round(4)}, intercept={ols.intercept_:.4f}")
    print(f"OLS  train MSE={mse(y_train, ols.predict(X_train)):.4f}, test MSE={mse(y_test, ols.predict(X_test)):.4f}")
    print(f"Ridge(α=10) closed-form coefs: {ridge.coef_.round(4)}")
    print(f"Ridge train MSE={mse(y_train, ridge.predict(X_train)):.4f}, test MSE={mse(y_test, ridge.predict(X_test)):.4f}")
    print(f"Ridge GD coefs: {ridge_gd.coef_.round(4)}")
    print(f"Max |closed - GD| coef diff: {np.max(np.abs(ridge.coef_ - ridge_gd.coef_)):.4f}")

    alphas = np.logspace(-2, 3, 40)
    sweep = lambda_sweep(X_train, y_train, X_test, y_test, alphas)
    best_idx = int(np.argmin(sweep["test_mse"]))
    print(f"\nBest lambda by test MSE: {alphas[best_idx]:.4g} (test MSE={sweep['test_mse'][best_idx]:.4f})")

    plot_results(sweep, ols.coef_, ridge.coef_, true_coef, out_dir)
    print(f"Saved plot → {out_dir}/ridge_results.png")
    print("Done.")


if __name__ == "__main__":
    main()
