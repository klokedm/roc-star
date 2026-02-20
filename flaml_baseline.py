"""
flaml_baseline.py - FLAML AutoML baseline comparison for roc-star.

Uses the IDENTICAL train/val/test split as optuna_search.py.
Skips gracefully if flaml is not installed.

Usage:
    pip install flaml
    python flaml_baseline.py
"""
import sys
import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Attempt to import flaml; fail gracefully
try:
    from flaml import AutoML
    FLAML_AVAILABLE = True
except ImportError:
    FLAML_AVAILABLE = False

from rocstar import epoch_update_gamma, roc_star_loss

# ---------------------------------------------------------------------------
# Reproducibility (must match optuna_search.py)
# ---------------------------------------------------------------------------
BASE_SEED = 42
torch.manual_seed(BASE_SEED)
np.random.seed(BASE_SEED)

# ---------------------------------------------------------------------------
# Identical dataset + splits as optuna_search.py
# ---------------------------------------------------------------------------
N_SAMPLES = 3000
N_FEATURES = 20

X, y = make_classification(
    n_samples=N_SAMPLES,
    n_features=N_FEATURES,
    n_informative=10,
    n_redundant=5,
    random_state=BASE_SEED,
)

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.20, random_state=BASE_SEED, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, random_state=BASE_SEED, stratify=y_trainval
)

# ---------------------------------------------------------------------------
# roc-star MLP baseline (same model as minimal_example.py / optuna_search.py)
# ---------------------------------------------------------------------------
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_rocstar(
    delta: float = 1.0,
    n_epochs: int = 10,
    batch_size: int = 256,
    seed: int = BASE_SEED,
) -> float:
    """Train MLP with roc_star_loss; return test AUC."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t  = torch.tensor(y_train, dtype=torch.float32)
    X_test_t   = torch.tensor(X_test,  dtype=torch.float32)

    model = MLP(N_FEATURES)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCELoss()

    epoch_y_true: torch.Tensor | None = None
    epoch_y_pred: torch.Tensor | None = None
    gamma = torch.tensor(0.2)

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(len(X_train_t))
        X_shuf = X_train_t[perm]
        y_shuf = y_train_t[perm]

        for i in range(0, len(X_shuf), batch_size):
            xb = X_shuf[i : i + batch_size]
            yb = y_shuf[i : i + batch_size]
            optimizer.zero_grad()
            preds = model(xb)
            if epoch == 0 or epoch_y_true is None:
                loss = bce(preds, yb)
            else:
                loss = roc_star_loss(yb, preds, gamma, epoch_y_true, epoch_y_pred)
            if loss.grad_fn is not None:
                loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            full_preds = model(X_train_t)
        epoch_y_true = y_train_t
        epoch_y_pred = full_preds.detach()
        gamma = epoch_update_gamma(epoch_y_true, epoch_y_pred, epoch=epoch, delta=delta)

    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_t).numpy()
    return float(roc_auc_score(y_test, test_preds))


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------
def main():
    print("=" * 55)
    print("Baseline comparison: FLAML vs roc-star MLP")
    print("=" * 55)

    # --- roc-star ---
    print("\n[roc-star] Training MLP with roc_star_loss (10 epochs) ...")
    rocstar_auc = train_rocstar(delta=1.0, n_epochs=10, seed=BASE_SEED)
    print(f"[roc-star] Test AUC: {rocstar_auc:.4f}")

    # --- FLAML ---
    if not FLAML_AVAILABLE:
        print("\n[FLAML] Not installed. Install with: pip install flaml")
        print("        Skipping FLAML baseline.")
        print("\nSummary:")
        print(f"  roc-star MLP test AUC : {rocstar_auc:.4f}")
        print("  FLAML                 : not available")
        return

    print("\n[FLAML] Running AutoML (budget=60s, metric=roc_auc) ...")
    automl = AutoML()
    automl.fit(
        X_train=X_train,
        y_train=y_train,
        task="classification",
        metric="roc_auc",
        time_budget=60,
        seed=BASE_SEED,
        eval_method="holdout",
        split_ratio=0.25,  # use 25% of X_train as internal val
        verbose=0,
    )

    flaml_test_preds = automl.predict_proba(X_test)[:, 1]
    flaml_auc = roc_auc_score(y_test, flaml_test_preds)
    print(f"[FLAML]  Test AUC: {flaml_auc:.4f} "
          f"(best estimator: {automl.best_estimator})")

    print("\n" + "=" * 55)
    print("Summary:")
    print(f"  roc-star MLP test AUC : {rocstar_auc:.4f}")
    print(f"  FLAML    test AUC     : {flaml_auc:.4f}")
    diff = flaml_auc - rocstar_auc
    winner = "FLAML" if diff > 0 else "roc-star"
    print(f"  Delta (FLAML - roc*)  : {diff:+.4f}  -> {winner} wins")
    print("=" * 55)


if __name__ == "__main__":
    main()
