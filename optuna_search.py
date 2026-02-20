"""
optuna_search.py - Optuna-based hyperparameter search for roc-star loss.

Replaces the deprecated hp_search.py (which used trains.automation).

Features:
- 3-split protocol: train / val / test  (val for HP selection, test held-out)
- Stratified splits to avoid all-negative val sets
- SQLite backend for reproducibility across runs
- TPE sampler with fixed seed
- ASHA pruner via trial.report / trial.should_prune
- Searches: delta (log-uniform 0.3-5.0), n_epochs (5-30), batch_size (64-512)

Usage:
    python optuna_search.py [--n-trials 20] [--seed 42]
"""
import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner

from rocstar import epoch_update_gamma, roc_star_loss

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
BASE_SEED = 42
torch.manual_seed(BASE_SEED)
np.random.seed(BASE_SEED)

# ---------------------------------------------------------------------------
# Data: generated once, shared across all trials
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

# 60 / 20 / 20 three-way stratified split
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.20, random_state=BASE_SEED, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, random_state=BASE_SEED, stratify=y_trainval
)
# 0.25 * 0.80 = 0.20 => train=60%, val=20%, test=20%

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t  = torch.tensor(y_train, dtype=torch.float32)
X_val_t    = torch.tensor(X_val,   dtype=torch.float32)
y_val_t    = torch.tensor(y_val,   dtype=torch.float32)
X_test_t   = torch.tensor(X_test,  dtype=torch.float32)
y_test_t   = torch.tensor(y_test,  dtype=torch.float32)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
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


def train_and_eval(
    delta: float,
    n_epochs: int,
    batch_size: int,
    trial: optuna.Trial | None = None,
    seed: int = BASE_SEED,
) -> tuple[float, float]:
    """
    Train MLP with roc_star_loss and return (val_auc, test_auc).

    trial: if provided, used for ASHA pruning via trial.report / should_prune.
    seed:  per-trial seed to avoid global state contamination between trials.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = MLP(N_FEATURES)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCELoss()

    epoch_y_true: torch.Tensor | None = None
    epoch_y_pred: torch.Tensor | None = None
    gamma = torch.tensor(0.2)

    val_auc = 0.0

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

        # Update epoch state (no global contamination: stored in local vars)
        model.eval()
        with torch.no_grad():
            full_preds = model(X_train_t)

        epoch_y_true = y_train_t
        epoch_y_pred = full_preds.detach()
        gamma = epoch_update_gamma(epoch_y_true, epoch_y_pred, epoch=epoch, delta=delta)

        # Val AUC for pruning
        with torch.no_grad():
            val_preds = model(X_val_t).numpy()
        val_auc = roc_auc_score(y_val, val_preds)

        if trial is not None:
            trial.report(val_auc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    # Test AUC (never used for HP selection)
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_t).numpy()
    test_auc = roc_auc_score(y_test, test_preds)

    return val_auc, test_auc


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------
def objective(trial: optuna.Trial) -> float:
    delta      = trial.suggest_float("delta",      0.3, 5.0, log=True)
    n_epochs   = trial.suggest_int(  "n_epochs",   5,   30)
    batch_size = trial.suggest_int(  "batch_size", 64,  512, step=64)

    # Use trial number as seed offset to isolate global state per trial
    val_auc, test_auc = train_and_eval(
        delta=delta,
        n_epochs=n_epochs,
        batch_size=batch_size,
        trial=trial,
        seed=BASE_SEED + trial.number,
    )

    # Store test AUC as a user attribute (not used for optimization)
    trial.set_user_attr("test_auc", test_auc)

    return val_auc  # optimize val AUC only


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Optuna HP search for roc-star")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--seed",     type=int, default=BASE_SEED)
    parser.add_argument("--db",       type=str, default="sqlite:///optuna_study.db")
    args = parser.parse_args()

    sampler = TPESampler(seed=args.seed)
    pruner  = SuccessiveHalvingPruner()

    study = optuna.create_study(
        study_name="roc_star_search",
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=args.db,
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    best = study.best_trial
    print("\n" + "=" * 55)
    print("Best trial (by val AUC):")
    print(f"  Val  AUC : {best.value:.4f}")
    test_auc_attr = best.user_attrs.get('test_auc')
    if test_auc_attr is not None:
        print(f"  Test AUC : {test_auc_attr:.4f}")
    else:
        print("  Test AUC : n/a")
    print(f"  Params   : {best.params}")
    print("=" * 55)

    # Re-evaluate best params on test set with fixed seed for final report
    val_auc, test_auc = train_and_eval(
        delta=best.params["delta"],
        n_epochs=best.params["n_epochs"],
        batch_size=best.params["batch_size"],
        trial=None,
        seed=args.seed,
    )
    print(f"\nFinal evaluation with best params:")
    print(f"  Val  AUC : {val_auc:.4f}")
    print(f"  Test AUC : {test_auc:.4f}")


if __name__ == "__main__":
    main()
