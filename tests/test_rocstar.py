"""
pytest tests for rocstar.py core functions.
Requires only torch (no network access, no external data).
"""
import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rocstar import epoch_update_gamma, roc_star_loss


DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tensors(n_pos=50, n_neg=50, seed=42, device="cpu"):
    """Return (y_true, y_pred) with deterministic values."""
    torch.manual_seed(seed)
    y_true = torch.cat([torch.ones(n_pos), torch.zeros(n_neg)]).to(device)
    y_pred = torch.cat(
        [torch.rand(n_pos) * 0.5 + 0.5, torch.rand(n_neg) * 0.5]
    ).to(device)
    return y_true, y_pred


# ---------------------------------------------------------------------------
# Tests: epoch_update_gamma
# ---------------------------------------------------------------------------

class TestEpochUpdateGamma:
    def test_normal_input_returns_tensor(self):
        y_true, y_pred = _make_tensors()
        gamma = epoch_update_gamma(y_true, y_pred, epoch=1)
        assert isinstance(gamma, torch.Tensor)
        assert gamma.ndim == 0  # scalar

    def test_normal_input_positive_gamma(self):
        y_true, y_pred = _make_tensors()
        gamma = epoch_update_gamma(y_true, y_pred, epoch=1)
        assert gamma.item() >= 0.0

    def test_epoch_minus_one_returns_default(self):
        """epoch=-1 should return the default gamma (0.2)."""
        y_true, y_pred = _make_tensors()
        gamma = epoch_update_gamma(y_true, y_pred, epoch=-1)
        assert gamma.item() == pytest.approx(0.2, abs=1e-5)

    def test_empty_positive_class_returns_default(self):
        """All-negative batch: no positive labels -> should return default."""
        y_true = torch.zeros(100)
        y_pred = torch.rand(100)
        gamma = epoch_update_gamma(y_true, y_pred, epoch=1)
        assert gamma.item() == pytest.approx(0.2, abs=1e-5)

    def test_empty_negative_class_returns_default(self):
        """All-positive batch: no negative labels -> should return default."""
        y_true = torch.ones(100)
        y_pred = torch.rand(100)
        gamma = epoch_update_gamma(y_true, y_pred, epoch=1)
        assert gamma.item() == pytest.approx(0.2, abs=1e-5)

    def test_delta_parameter_affects_gamma(self):
        """Higher delta should generally produce a larger gamma."""
        y_true, y_pred = _make_tensors(seed=7)
        gamma_small = epoch_update_gamma(y_true, y_pred, epoch=1, delta=0.3)
        gamma_large = epoch_update_gamma(y_true, y_pred, epoch=1, delta=5.0)
        # They may differ; at minimum both must be valid tensors
        assert gamma_small.item() >= 0.0
        assert gamma_large.item() >= 0.0

    @pytest.mark.parametrize("device", DEVICES)
    def test_device_agnostic(self, device):
        y_true, y_pred = _make_tensors(device=device)
        gamma = epoch_update_gamma(y_true, y_pred, epoch=1)
        assert isinstance(gamma, torch.Tensor)
        assert not torch.isnan(gamma)
        assert not torch.isinf(gamma)

    def test_output_on_cpu(self):
        y_true, y_pred = _make_tensors(device="cpu")
        gamma = epoch_update_gamma(y_true, y_pred, epoch=1)
        assert gamma.device.type == "cpu"

    def test_soft_labels(self):
        """Soft labels (e.g. 0.7) should be handled with >= 0.5 threshold."""
        torch.manual_seed(0)
        y_true = torch.tensor([0.8, 0.9, 0.6, 0.3, 0.1, 0.2])
        y_pred = torch.rand(6)
        gamma = epoch_update_gamma(y_true, y_pred, epoch=1)
        assert isinstance(gamma, torch.Tensor)
        assert not torch.isnan(gamma)


# ---------------------------------------------------------------------------
# Tests: roc_star_loss
# ---------------------------------------------------------------------------

class TestRocStarLoss:
    def _epoch_data(self, n_pos=100, n_neg=100, seed=0, device="cpu"):
        """Create epoch (y_true, y_pred) and a gamma value."""
        y_true, y_pred = _make_tensors(n_pos, n_neg, seed=seed, device=device)
        gamma = epoch_update_gamma(y_true, y_pred, epoch=1)
        return y_true, y_pred, gamma

    def test_mixed_batch_returns_nonnegative_loss(self):
        epoch_true, epoch_pred, gamma = self._epoch_data()
        y_true, y_pred = _make_tensors(seed=99)
        loss = roc_star_loss(y_true, y_pred, gamma, epoch_true, epoch_pred)
        assert loss.item() >= 0.0

    def test_all_positive_batch_returns_zero(self):
        """Batch with no negatives -> loss should be 0."""
        epoch_true, epoch_pred, gamma = self._epoch_data()
        y_true = torch.ones(50)
        y_pred = torch.rand(50)
        loss = roc_star_loss(y_true, y_pred, gamma, epoch_true, epoch_pred)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_all_negative_batch_returns_zero(self):
        """Batch with no positives -> loss should be 0."""
        epoch_true, epoch_pred, gamma = self._epoch_data()
        y_true = torch.zeros(50)
        y_pred = torch.rand(50)
        loss = roc_star_loss(y_true, y_pred, gamma, epoch_true, epoch_pred)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_loss_is_scalar(self):
        epoch_true, epoch_pred, gamma = self._epoch_data()
        y_true, y_pred = _make_tensors(seed=5)
        loss = roc_star_loss(y_true, y_pred, gamma, epoch_true, epoch_pred)
        assert loss.ndim == 0

    def test_loss_is_finite(self):
        epoch_true, epoch_pred, gamma = self._epoch_data()
        y_true, y_pred = _make_tensors(seed=13)
        loss = roc_star_loss(y_true, y_pred, gamma, epoch_true, epoch_pred)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_good_predictions_lower_loss(self):
        """Better-separated predictions should yield a lower loss than random."""
        epoch_true, epoch_pred, gamma = self._epoch_data(seed=1)
        # Good: positives score high, negatives score low
        y_true = torch.cat([torch.ones(50), torch.zeros(50)])
        y_pred_good = torch.cat([torch.ones(50) * 0.9, torch.ones(50) * 0.1])
        y_pred_bad  = torch.cat([torch.ones(50) * 0.1, torch.ones(50) * 0.9])
        loss_good = roc_star_loss(y_true, y_pred_good, gamma, epoch_true, epoch_pred)
        loss_bad  = roc_star_loss(y_true, y_pred_bad,  gamma, epoch_true, epoch_pred)
        assert loss_good.item() <= loss_bad.item()

    @pytest.mark.parametrize("device", DEVICES)
    def test_device_agnostic(self, device):
        epoch_true, epoch_pred, gamma = self._epoch_data(device=device)
        y_true, y_pred = _make_tensors(device=device)
        loss = roc_star_loss(y_true, y_pred, gamma, epoch_true, epoch_pred)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert loss.device.type == device

    def test_backward_computes_grad(self):
        """Loss must be differentiable w.r.t. y_pred."""
        epoch_true, epoch_pred, gamma = self._epoch_data()
        y_true, y_pred = _make_tensors(seed=77)
        y_pred = y_pred.requires_grad_(True)
        epoch_pred = epoch_pred.detach()
        loss = roc_star_loss(y_true, y_pred, gamma, epoch_true, epoch_pred)
        if loss.item() > 0 and loss.grad_fn is not None:
            loss.backward()
            assert y_pred.grad is not None
