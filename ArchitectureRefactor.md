# ArchitectureRefactor.md - Architecture Alignment Document

## Purpose
This document defines the target architecture, design principles, and refactoring goals for the roc-star project. It serves as the north star for architectural decisions during audits and improvements.

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
- [ ] All edge cases handled
- [ ] Deterministic mode works
- [ ] CPU and GPU support
- [ ] Backward compatible

### Documentation
- [ ] API docs complete
- [ ] Theory explained clearly
- [ ] Migration guide available
- [ ] Examples work out-of-box

---

*Document maintained by Architect/Auditor*  
*Last Updated*: 2026-02-18 04:11 UTC  
*Next Review*: After subagent audits complete
