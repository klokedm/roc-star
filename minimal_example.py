"""
minimal_example.py - Runnable training demo for roc-star loss.

No external data or tracking services required.
Dependencies: pip install torch scikit-learn

Usage:
    python minimal_example.py
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from rocstar import epoch_update_gamma, roc_star_loss

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------
X, y = make_classification(
    n_samples=2000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    random_state=SEED,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.float32)

# ---------------------------------------------------------------------------
# Simple 2-layer MLP
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


model = MLP(X_train.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
bce_loss = nn.BCELoss()

N_EPOCHS = 5
BATCH_SIZE = 256

# Epoch-level state for roc_star (updated at end of each epoch)
epoch_y_true: torch.Tensor | None = None
epoch_y_pred: torch.Tensor | None = None
gamma: torch.Tensor = torch.tensor(0.2)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
print(f"Training MLP on synthetic data ({X_train.shape[0]} samples, "
      f"{X_train.shape[1]} features)")
print("-" * 55)

for epoch in range(N_EPOCHS):
    model.train()
    perm = torch.randperm(len(X_train_t))
    X_shuf = X_train_t[perm]
    y_shuf = y_train_t[perm]

    epoch_preds_list = []

    for i in range(0, len(X_shuf), BATCH_SIZE):
        xb = X_shuf[i : i + BATCH_SIZE]
        yb = y_shuf[i : i + BATCH_SIZE]

        optimizer.zero_grad()
        preds = model(xb)

        if epoch == 0 or epoch_y_true is None:
            # Warm-up epoch: use BCE
            loss = bce_loss(preds, yb)
        else:
            loss = roc_star_loss(yb, preds, gamma, epoch_y_true, epoch_y_pred)

        if loss.grad_fn is not None:
            loss.backward()
        optimizer.step()

        epoch_preds_list.append(preds.detach())

    # Collect full-epoch predictions for gamma update
    model.eval()
    with torch.no_grad():
        full_preds = model(X_train_t)

    epoch_y_true = y_train_t
    epoch_y_pred = full_preds
    gamma = epoch_update_gamma(epoch_y_true, epoch_y_pred, epoch=epoch)

    # Compute train AUC
    train_auc = roc_auc_score(y_train, full_preds.numpy())

    # Compute test AUC
    with torch.no_grad():
        test_preds = model(X_test_t).numpy()
    test_auc = roc_auc_score(y_test, test_preds)

    loss_name = "BCE" if epoch == 0 else "roc_star"
    print(f"Epoch {epoch+1:2d} [{loss_name:8s}] | "
          f"gamma={gamma.item():.4f} | "
          f"train AUC={train_auc:.4f} | test AUC={test_auc:.4f}")

print("-" * 55)
print(f"Final test AUC: {test_auc:.4f}")
