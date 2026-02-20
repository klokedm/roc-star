# ArchitectureRefactor.md - Architecture Alignment Document

## Purpose
This document defines the target architecture, design principles, and refactoring goals for the roc-star project. It serves as the north star for architectural decisions during audits and improvements.

---

## Audit Closure Snapshot (2026-02-20)

| Workstream | Status | Notes |
|------------|--------|-------|
| P0/P1 correctness fixes | ✅ COMPLETE | Implemented in `rocstar.py` and consumed by `example.py` |
| API compatibility | ✅ COMPLETE | Public function signatures preserved |
| Duplicate loss logic | ✅ COMPLETE | Removed duplicate implementations in `example.py` |
| Input validation layer | ⏳ DEFERRED | Planned for v1.1 (`validation.py`) |
| Deterministic sampling | ⏳ DEFERRED | Planned for v1.1 (opt-in generator/seeding) |
| Package restructure | ⏳ DEFERRED | Planned for v2.0 |
| Test/CI infrastructure | 🚫 BLOCKED | Local environment missing `torch`/`pytest`; no CI config |

---

## Design Philosophy

### Core Principles

1. **Simplicity First**
   - Keep the public API minimal and intuitive
   - Favor explicit over implicit behavior
   - Provide sensible defaults while allowing customization

2. **Correctness Over Performance**
   - Prioritize mathematical correctness and numerical stability
   - Optimize only after correctness is verified
   - Document any performance-correctness tradeoffs

3. **Reproducibility**
   - Support deterministic behavior via random seeds
   - Document all sources of randomness
   - Enable bit-exact reproduction of results

4. **Flexibility**
   - Support both CPU and GPU execution
   - Allow configuration of critical parameters
   - Provide hooks for customization without forking

5. **Auditability**
   - Clear separation between public and private APIs
   - Comprehensive input validation with descriptive errors
   - Logging and debugging support

---

## Target Architecture

### Module Structure

```
roc-star/
├── rocstar/              # Main package
│   ├── __init__.py       # Public API exports
│   ├── loss.py           # Loss function implementations
│   ├── gamma.py          # Gamma calculation utilities
│   ├── sampling.py       # Subsampling strategies
│   ├── validation.py     # Input validation utilities
│   └── config.py         # Configuration and defaults
├── tests/                # Test suite
│   ├── __init__.py
│   ├── test_loss.py      # Loss function tests
│   ├── test_gamma.py     # Gamma calculation tests
│   ├── test_sampling.py  # Sampling tests
│   ├── test_validation.py # Validation tests
│   └── test_integration.py # End-to-end tests
├── examples/             # Usage examples
│   ├── basic_usage.py
│   ├── sentiment_analysis.py
│   └── hyperparameter_tuning.py
├── docs/                 # Documentation
│   ├── api.md
│   ├── theory.md
│   └── migration.md
├── setup.py              # Package setup
├── pyproject.toml        # Modern Python project config
├── requirements.txt      # Dependencies
├── requirements-dev.txt  # Development dependencies
└── README.md             # User-facing documentation
```

### Public API Design

#### Primary Interface

```python
# rocstar/__init__.py
from .loss import roc_star_loss
from .gamma import GammaScheduler
from .config import RocStarConfig

__all__ = ['roc_star_loss', 'GammaScheduler', 'RocStarConfig']
```

#### Configuration Object

```python
# rocstar/config.py
@dataclass
class RocStarConfig:
    """Configuration for ROC-star loss function."""
    
    # Gamma calculation
    delta: float = 1.0  # Proportion of too-close to wrong-ordered pairs
    gamma_subsample_size: int = 2000  # Subsample size for gamma calculation
    default_gamma: float = 0.2  # Fallback gamma value
    
    # Loss calculation
    loss_subsample_size: int = 1000  # Max samples per class in loss
    
    # Reproducibility
    random_seed: Optional[int] = None  # Seed for reproducibility
    
    # Device handling
    device: Optional[torch.device] = None  # Auto-detect if None
    
    # Validation
    validate_inputs: bool = True  # Enable input validation
    
    def __post_init__(self):
        """Validate configuration."""
        if self.delta <= 0:
            raise ValueError("delta must be positive")
        # ... more validation
```

#### Loss Function Signature

```python
def roc_star_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    gamma: float,
    epoch_true: torch.Tensor,
    epoch_pred: torch.Tensor,
    config: Optional[RocStarConfig] = None
) -> torch.Tensor:
    """
    ROC-star loss function for binary classification.
    
    Parameters
    ----------
    y_true : torch.Tensor
        Ground truth labels for current batch (0.0 or 1.0)
    y_pred : torch.Tensor
        Predicted values for current batch
    gamma : float
        Padding parameter from last epoch
    epoch_true : torch.Tensor
        Ground truth labels from last epoch
    epoch_pred : torch.Tensor
        Predictions from last epoch
    config : RocStarConfig, optional
        Configuration object. Uses defaults if None.
    
    Returns
    -------
    torch.Tensor
        Loss value (scalar)
    
    Raises
    ------
    ValueError
        If inputs have incompatible shapes or invalid values
    
    Notes
    -----
    Based on Yan et al. (2003) approximation to Wilcoxon-Mann-Whitney statistic.
    See README.md for mathematical background.
    """
```

#### Gamma Scheduler

```python
class GammaScheduler:
    """Manages gamma calculation across epochs."""
    
    def __init__(self, config: Optional[RocStarConfig] = None):
        self.config = config or RocStarConfig()
        self.history = []
    
    def update(
        self, 
        y_true: torch.Tensor, 
        y_pred: torch.Tensor,
        epoch: int = 0
    ) -> float:
        """Calculate gamma for next epoch."""
        gamma = self._calculate_gamma(y_true, y_pred)
        self.history.append(gamma)
        return gamma
    
    def _calculate_gamma(
        self, 
        y_true: torch.Tensor, 
        y_pred: torch.Tensor
    ) -> float:
        """Internal gamma calculation with subsampling."""
        # Implementation...
```

---

## Design Constraints

### Must Maintain
- Backward compatibility with existing usage patterns
- Performance comparable to BCE loss
- GPU acceleration support
- Core mathematical approach from Yan et al. 2003

### Must Add
- Input validation with clear error messages
- CPU fallback for non-GPU environments
- Deterministic behavior option (seeding)
- Type hints throughout
- Comprehensive test coverage

### Must Document
- Mathematical background and derivation
- Parameter selection guidance (delta, subsample sizes)
- Performance characteristics and memory usage
- Common pitfalls and debugging tips
- Migration guide from old API

---

## Refactoring Priorities

### Phase 1: Foundation (No Breaking Changes)
1. Add comprehensive input validation
2. Add CPU/GPU device handling
3. Add random seed support for reproducibility
4. Add type hints
5. Extract magic numbers to named constants
6. Add docstring improvements

### Phase 2: Quality Infrastructure
1. Set up test framework (pytest)
2. Add unit tests for all functions
3. Add integration tests
4. Set up CI/CD (GitHub Actions)
5. Add linting (flake8, mypy)
6. Add code formatting (black)

### Phase 3: API Evolution (With Deprecation Path)
1. Introduce RocStarConfig class
2. Create GammaScheduler class
3. Reorganize into package structure
4. Add setup.py and pyproject.toml
5. Version 1.0 release

### Phase 4: Advanced Features (Future)
1. Multi-class support (one-vs-all)
2. Focal loss variant
3. Label smoothing option
4. Distributed training support
5. TensorFlow/JAX implementations

---

## Module Boundaries

### Public APIs (Stable)
- `roc_star_loss()` - Core loss function
- `GammaScheduler` - Gamma management
- `RocStarConfig` - Configuration

### Private APIs (Internal, May Change)
- `_subsample()` - Internal subsampling logic
- `_calculate_differences()` - Pairwise difference computation
- `_validate_tensor()` - Input validation helpers

### Extension Points
- Custom subsampling strategies
- Custom gamma calculation strategies
- Logging hooks
- Debugging hooks

---

## Testing Strategy

### Unit Tests
- Each function tested in isolation
- Edge cases: empty batches, single class, extreme values
- Input validation: wrong shapes, wrong types, wrong ranges
- Determinism: seeded runs produce identical results
- Device handling: CPU and GPU (if available)

### Integration Tests
- Full training loop with toy dataset
- Gamma update across epochs
- Comparison with BCE baseline
- Memory usage validation

### Property Tests
- Loss is non-negative
- Loss decreases with correct ordering
- Loss increases with incorrect ordering
- Gradient exists and is non-zero for wrong-ordered pairs

### Numerical Tests
- Compare against reference implementation
- Verify against hand-calculated examples
- Check numerical stability with extreme inputs

---

## Performance Targets

### Throughput
- Within 10% of BCE loss throughput
- Support batch sizes up to 10,000
- Subsample sizes configurable for speed/accuracy tradeoff

### Memory
- O(n) memory for batch size n
- Subsample caps prevent OOM on large batches
- Efficient tensor operations (no unnecessary copies)

### Scalability
- Support training sets up to 10M samples
- Efficient epoch storage (reuse across batches)
- GPU memory management

---

## Security Considerations

### Input Validation
- Check tensor shapes match
- Validate value ranges (labels in [0,1])
- Detect NaN/Inf in inputs
- Validate configuration parameters

### Numerical Stability
- Avoid division by zero
- Handle empty positive/negative classes
- Use stable sorting algorithms
- Check for overflow in large datasets

### Reproducibility
- Seed all random operations
- Document floating-point assumptions
- Warn about non-deterministic GPU operations

---

## Documentation Requirements

### Code Documentation
- All public functions have comprehensive docstrings
- NumPy-style docstring format
- Type hints for all parameters and returns
- Examples in docstrings

### User Documentation
- README.md: Quick start and overview
- docs/theory.md: Mathematical background
- docs/api.md: Complete API reference
- docs/migration.md: Upgrading guide

### Developer Documentation
- CONTRIBUTING.md: How to contribute
- ARCHITECTURE.md: This file
- Design decision rationale

---

## Metrics for Success

### Code Quality
- [ ] 100% type hint coverage
- [ ] 90%+ test coverage
- [ ] Zero linting errors (flake8, mypy)
- [ ] All docstrings present

### Functionality
- [x] All critical edge cases handled (division by zero, empty tensors, NaN/Inf)
- [ ] Deterministic mode works
- [x] CPU and GPU support (core loss/gamma functions)
- [x] Backward compatible

### Documentation
- [ ] API docs complete
- [ ] Theory explained clearly
- [ ] Migration guide available
- [ ] Examples work out-of-box

---

## Classifier Pipeline Architecture (ARCH-ROAD-001)

*Added by Architect/Auditor — ARCH-ROAD-001*

### Objective

Design a clean, extensible classifier pipeline that:
- Wraps multiple ML backends behind a single sklearn-compatible interface
- Integrates leading AutoML frameworks for HP search and model selection
- Uses roc-star loss as the optimization target wherever the backend supports it
- Provides clear extension points for new backends and oracle types

---

### 1. Class Hierarchy (ASCII Diagram)

```
classifier/
│
├── base.py
│   ├── BaseClassifier (ABC)            ← common fit/predict/score contract
│   └── RocStarObjective (ABC)          ← objective adapter contract
│
├── torch_classifier.py
│   ├── TorchClassifier(BaseClassifier) ← PyTorch training loop + roc-star loss
│   │   ├── LSTMClassifier              ← LSTM backbone (refactored from example.py)
│   │   └── MLPClassifier               ← MLP backbone (new)
│   └── TorchRocStarObjective           ← stateful roc-star objective for Optuna/Ray
│
├── sklearn_classifier.py
│   └── SklearnClassifier(BaseClassifier) ← thin wrapper; score() returns AUC
│
├── gbm_classifier.py
│   ├── LightGBMClassifier(BaseClassifier)
│   └── XGBoostClassifier(BaseClassifier)
│
└── automl/
    ├── base_automl.py
    │   └── BaseAutoML(ABC)             ← search / best_estimator contract
    ├── optuna_backend.py
    │   └── OptunaAutoML(BaseAutoML)    ← Optuna study over any BaseClassifier
    ├── flaml_backend.py
    │   └── FLAMLAutoML(BaseAutoML)     ← FLAML AutoML (sklearn/LightGBM/XGB)
    └── ray_backend.py
        └── RayAutoML(BaseAutoML)       ← Ray Tune (distributed, any backend)
```

---

### 2. Common Interface Definition (Python pseudocode)

```python
# classifier/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np


class BaseClassifier(ABC):
    """
    sklearn-compatible interface for all classifier backends.
    Concrete subclasses must implement fit, predict_proba, and get_params.
    score() is provided by default using roc_auc_score.
    """

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "BaseClassifier":
        """Train the classifier. Returns self."""

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return positive-class probability scores, shape (n_samples,)."""

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Binary predictions at given threshold."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Validation AUC (primary metric for AutoML objectives)."""
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y, self.predict_proba(X)))

    @abstractmethod
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Return hyperparameters (enables sklearn grid-search compatibility)."""

    def set_params(self, **params) -> "BaseClassifier":
        """Set hyperparameters in-place."""
        for k, v in params.items():
            setattr(self, k, v)
        return self


class RocStarObjective(ABC):
    """
    Adapter that bridges roc-star's stateful epoch protocol to a
    framework-agnostic objective callable.
    Stateful: holds last_epoch_true, last_epoch_pred, gamma across calls.
    """

    @abstractmethod
    def reset(self) -> None:
        """Clear epoch state between HP trials."""

    @abstractmethod
    def on_epoch_end(
        self,
        epoch: int,
        y_true: "torch.Tensor",
        y_pred: "torch.Tensor",
    ) -> float:
        """Update gamma and cache epoch predictions. Returns current gamma."""

    @abstractmethod
    def loss(
        self,
        y_true: "torch.Tensor",
        y_pred: "torch.Tensor",
        epoch: int,
    ) -> "torch.Tensor":
        """Return roc-star loss tensor for the current batch."""


class BaseAutoML(ABC):
    """
    Contract for AutoML backend wrappers.
    search() runs the HP optimization; best_estimator returns the winner.
    """

    @abstractmethod
    def search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: int = 50,
        timeout_s: Optional[int] = None,
    ) -> None:
        """Run HP search. Populates self.best_estimator_ and self.best_params_."""

    @property
    @abstractmethod
    def best_estimator_(self) -> BaseClassifier:
        """Best classifier found during search."""

    @property
    @abstractmethod
    def best_params_(self) -> Dict[str, Any]:
        """Hyperparameters of the best estimator."""

    @property
    @abstractmethod
    def best_score_(self) -> float:
        """Validation AUC of the best estimator."""
```

---

### 3. Module File Structure Proposal

```
roc-star/
├── rocstar/                        # Core package (existing, Phase 1-3 refactor)
│   ├── __init__.py
│   ├── loss.py                     # roc_star_loss
│   ├── gamma.py                    # GammaScheduler (epoch_update_gamma)
│   ├── sampling.py                 # _subsample helpers
│   ├── validation.py               # Input validation (v1.1)
│   └── config.py                   # RocStarConfig dataclass
│
├── classifier/                     # NEW — classifier pipeline (v2.0)
│   ├── __init__.py                 # Exports BaseClassifier, factory helpers
│   ├── base.py                     # ABCs: BaseClassifier, RocStarObjective, BaseAutoML
│   ├── config.py                   # PipelineConfig (backend toggle, HP space)
│   ├── torch_classifier.py         # TorchClassifier, LSTMClassifier, MLPClassifier
│   ├── sklearn_classifier.py       # SklearnClassifier wrapper
│   ├── gbm_classifier.py           # LightGBMClassifier, XGBoostClassifier
│   └── automl/
│       ├── __init__.py
│       ├── base_automl.py          # BaseAutoML ABC
│       ├── optuna_backend.py       # OptunaAutoML (PRIMARY — see §4)
│       ├── flaml_backend.py        # FLAMLAutoML
│       └── ray_backend.py          # RayAutoML (distributed)
│
├── tests/
│   ├── test_loss.py
│   ├── test_gamma.py
│   ├── test_base_classifier.py     # Smoke tests via TorchClassifier
│   ├── test_automl_optuna.py
│   └── test_integration.py
│
├── examples/
│   ├── basic_usage.py              # roc_star_loss standalone
│   ├── sentiment_analysis.py       # Refactored example.py (LSTMClassifier)
│   └── automl_search.py            # OptunaAutoML end-to-end
│
├── setup.py
├── pyproject.toml
└── requirements/
    ├── base.txt                    # torch, numpy, scikit-learn
    ├── automl.txt                  # optuna, flaml, ray[tune]
    └── dev.txt                     # pytest, black, mypy, flake8
```

---

### 4. Recommended AutoML Frameworks (Ranked)

| Rank | Framework | Rationale |
|------|-----------|-----------|
| 1 | **Optuna** | Native PyTorch support; each trial can instantiate a fresh `TorchClassifier` with full roc-star loss loop; objective function receives the trial object and can prune early; AUC or roc-star loss as objective; MIT license; most widely adopted for research. |
| 2 | **FLAML** | Zero-config fast search over sklearn/LightGBM/XGBoost; uses AUC metric natively; ideal for tabular benchmarks where PyTorch is overkill; low overhead. Limitation: cannot use roc-star loss directly (AUC proxy only). |
| 3 | **Ray Tune** | Best for distributed multi-GPU HP search over TorchClassifier; integrates with Optuna/BOHB samplers; useful when training cost is high. Added complexity: requires Ray cluster setup. |
| 4 | **auto-sklearn** | Sklearn pipelines only; no roc-star integration; useful as a baseline comparison. Heavy dependency (SMAC3). |
| 5 | **H2O AutoML** | JVM dependency; limited Python-native roc-star integration; not recommended unless H2O cluster already in use. |

**Recommendation**: Implement `OptunaAutoML` first (covers both PyTorch + tabular backends); add `FLAMLAutoML` as a fast tabular-only path.

**roc-star loss availability by backend:**

| Backend | roc-star loss usable? | Reason |
|---------|----------------------|--------|
| PyTorch (TorchClassifier) | ✅ Yes | Differentiable; supports `.backward()` |
| LightGBM / XGBoost | ⚠️ Proxy | Custom objective API exists but roc-star requires per-pair gradient; use AUC metric instead |
| sklearn | ❌ No | No gradient support; use `score()` AUC as objective |
| FLAML / auto-sklearn | ❌ No | Delegate to backend rules above |

---

### 5. Risk Register with Mitigations

#### RISK-1 (HIGH): Stateful roc-star epoch protocol conflicts with AutoML trial isolation

**Description:**  
`roc_star_loss` is stateful — it requires `last_epoch_true`, `last_epoch_pred`, and a `gamma` value computed at the end of the previous epoch. AutoML frameworks (Optuna, Ray Tune) launch trials independently and may share worker processes. If epoch state leaks between trials, gamma is wrong, producing an invalid loss and inflated AUC estimates.

**Evidence:** `epoch_gamma` is a bare variable in `example.py` (line 324); it would become a process-global in a parallel search without isolation.

**Mitigation:**
- `RocStarObjective.reset()` must be called at the start of every trial.
- `TorchRocStarObjective` holds state as instance variables, never module-level globals.
- Optuna trials instantiate a fresh `TorchClassifier` (and thus a fresh `RocStarObjective`) per trial.
- Add an assertion: if `epoch == 0` and `gamma != default_gamma`, raise a `RuntimeError`.
- In Ray Tune, pin each trial to an isolated actor to prevent state sharing.

---

#### RISK-2 (HIGH): Memory explosion from epoch tensors held across HP trials

**Description:**  
`last_whole_y_pred` and `last_whole_y_t` (full-epoch predictions) are kept in GPU memory between epochs for roc-star loss computation. In a long HP search with many trials, stale tensors from completed trials remain allocated if not explicitly freed, leading to GPU OOM errors — especially on shared GPUs.

**Evidence:** `example.py` lines 309-310 create epoch-level tensors with no explicit `del`/cleanup between trials.

**Mitigation:**
- `RocStarObjective.reset()` explicitly calls `del self._epoch_true; del self._epoch_pred; torch.cuda.empty_cache()`.
- `TorchClassifier.fit()` wraps training in a `try/finally` block to guarantee cleanup.
- Document: subsample sizes (`SUB_SAMPLE_SIZE=2000`, `max_pos/neg=1000`) are the primary knobs to limit memory; expose them in `RocStarConfig`.
- Add a memory budget check at trial start (warn if GPU free memory < threshold).

---

#### RISK-3 (MEDIUM): AutoML framework API churn breaks integrations

**Description:**  
`hp_search.py` already illustrates this: it imports `from trains import ...` (ClearML was rebranded from Trains). Optuna, Ray Tune, and FLAML all release breaking API changes frequently.

**Mitigation:**
- All framework-specific code lives behind the `BaseAutoML` adapter — changing a backend touches only one file.
- Pin framework versions in `requirements/automl.txt` with comments noting the tested version.
- Add CI matrix tests for each supported framework version.

---

#### RISK-4 (MEDIUM): AUC-proxy vs roc-star loss mismatch in mixed-backend searches

**Description:**  
When FLAML or auto-sklearn is used as a backend, the optimization target is `roc_auc_score` (a post-hoc metric), not roc-star loss (a training-time signal). Comparing scores between a roc-star-trained PyTorch model and an AUC-optimized LightGBM model is valid at evaluation time, but the HP search dynamics differ — leading to inconsistent model selection if the user assumes all backends use the same objective.

**Mitigation:**
- `PipelineConfig.objective` must be documented as either `"roc_star"` (PyTorch only) or `"auc"` (all backends).
- `BaseAutoML.search()` logs which objective was actually used per backend.
- `best_score_` is always reported as validation AUC (not loss) to allow cross-backend comparison.

---

### 6. Extension Points for New Frameworks / Oracles

| Extension Point | How to Add |
|----------------|------------|
| **New classifier backend** | Subclass `BaseClassifier`; implement `fit`, `predict_proba`, `get_params`. Drop file in `classifier/`. |
| **New AutoML backend** | Subclass `BaseAutoML`; implement `search`. Drop file in `classifier/automl/`. Register in `classifier/automl/__init__.py`. |
| **New roc-star objective adapter** | Subclass `RocStarObjective`; implement `reset`, `on_epoch_end`, `loss`. Useful for e.g. JAX/TF backends. |
| **Custom subsampling strategy** | Subclass `rocstar/sampling.py:BaseSampler`; pass to `RocStarConfig.sampler`. |
| **Custom gamma scheduler** | Subclass `rocstar/gamma.py:GammaScheduler`; override `_calculate_gamma`. |
| **Oracle / meta-learner** | Implement `BaseClassifier` with `fit()` delegating to an ensemble of backends. `score()` remains AUC for comparability. |
| **Non-binary targets** | Multi-class support planned (Phase 4): `BaseClassifier.predict_proba` returns shape `(n_samples, n_classes)`; roc-star becomes one-vs-all. |

---

### 7. Configuration / Toggle System

```python
# classifier/config.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional, List

@dataclass
class PipelineConfig:
    # Backend selection (toggle)
    backend: Literal["torch", "sklearn", "lightgbm", "xgboost"] = "torch"

    # AutoML toggle
    automl: Optional[Literal["optuna", "flaml", "ray"]] = None  # None = no AutoML

    # Loss objective (roc_star only valid for torch backend)
    objective: Literal["roc_star", "auc"] = "roc_star"

    # Training
    n_epochs: int = 30
    batch_size: int = 128
    initial_lr: float = 1e-3
    device: str = "auto"  # "auto" | "cpu" | "cuda"

    # HP search
    n_trials: int = 50
    timeout_s: Optional[int] = None
    hp_space: dict = field(default_factory=dict)  # backend-specific HP ranges

    # Reproducibility
    seed: Optional[int] = None

    def __post_init__(self):
        if self.objective == "roc_star" and self.backend != "torch":
            raise ValueError(
                f"objective='roc_star' requires backend='torch'; "
                f"got backend='{self.backend}'. Use objective='auc' for non-torch backends."
            )
```

---

### Implementation Phasing (Updated — ROADMAP-2026-02-20)

| Phase | Deliverable | Prerequisites | Status |
|-------|-------------|---------------|--------|
| v1.1 now | `tests/test_rocstar.py`, `minimal_example.py`, `optuna_search.py`, `flaml_baseline.py` | None | ✅ DONE |
| v1.1 next | Type annotations on `rocstar.py` | None | 🔲 TODO |
| v1.2 | Deterministic subsampling (optional `generator` param) | v1.1 | 🔲 DEFERRED |
| v1.2 | Input validation layer (`validate_inputs` flag) | v1.1 | 🔲 DEFERRED |
| v2.0 | `RocStarCallback` for Lightning (event-based, no ABC) | ≥3 model types | 🔲 DEFERRED |
| v2.0 | `rocstar/` package refactor (`config.py`, `GammaScheduler`) | v1.2 | 🔲 DEFERRED |

**Permanently deferred** (GAME-ROAD-001 veto, evidence-based):
- `BaseClassifier` / `BaseAutoML` ABC hierarchy — over-engineering for a loss-function library
- Stacking ensemble infrastructure — requires k-fold CV; prediction correlation inflates expected AUC gains
- GammaNet meta-learning — bi-level instability; delta HP already in Optuna 1D search space

---

*Section added by Architect/Auditor — ARCH-ROAD-001; updated by Orchestrator — ROADMAP-2026-02-20*  
*Date*: 2026-02-20  
*Review gate*: v1.2 planning gate (determinism + validation)

---

*Document maintained by Architect/Auditor*  
*Last Updated*: 2026-02-20 (ROADMAP-2026-02-20 — classifier pipeline decisions finalized; ABC/stacking/GammaNet permanently deferred)  
*Next Review*: v1.2 planning gate
