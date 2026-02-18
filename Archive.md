# Archive.md - Detailed Audit Findings Repository

## Purpose
This document serves as the comprehensive record of all audit findings, detailed analyses, and implementation rationales. While Progress.md provides a high-level view, Archive.md contains the full forensic details.

---

## Audit Session: Roc-Star Correctness & Maintainability
**Session ID**: AUDIT-2026-02-18  
**Started**: 2026-02-18  
**Repository**: klokedm/roc-star  
**Branch**: copilot/improve-code-correctness

---

## Table of Contents
1. [Baseline State Assessment](#baseline-state-assessment)
2. [Architecture Findings](#architecture-findings)
3. [Security & Bug Findings](#security--bug-findings)
4. [Algorithm Correctness Findings](#algorithm-correctness-findings)
5. [Data Integrity Findings](#data-integrity-findings)
6. [Evaluation Protocol Findings](#evaluation-protocol-findings)
7. [Proposed Refactorings](#proposed-refactorings)
8. [Creative Contradiction Analysis](#creative-contradiction-analysis)
9. [Implementation History](#implementation-history)

---

## Baseline State Assessment

### Repository Structure
```
roc-star/
├── README.md          (17,673 bytes) - Comprehensive documentation
├── rocstar.py         (4,625 bytes)  - Core loss functions
├── example.py         (20,094 bytes) - Full training example with Twitter sentiment
├── hp_search.py       (5,155 bytes)  - Hyperparameter optimization setup
└── .git/              - Git repository
```

### Initial Observations

**Files Present**:
- Core implementation (rocstar.py): 2 functions (epoch_update_gamma, roc_star_loss)
- Working example (example.py): Complete training pipeline
- Hyperparameter search (hp_search.py): Optimization framework integration

**Files Missing**:
- No test suite (no tests/ directory, no test_*.py files)
- No CI/CD configuration (.github/workflows/)
- No setup.py or pyproject.toml for package management
- No requirements.txt or environment.yml
- No type hints or mypy configuration
- No linting configuration

**Documentation Quality**:
- README.md is extensive and well-written
- Explains mathematical background (Yan et al. 2003)
- Provides usage examples
- Links to external resources

**Code Quality Initial Assessment**:
- Functions are reasonably well-documented with docstrings
- Some edge case handling present
- GPU-specific code (.cuda() calls) without CPU fallback
- Random sampling without seed control
- Magic numbers present (1000, 2000.0, 0.2)

---

## Architecture Findings
**Task ID**: ARCH-001 | **Status**: COMPLETE | **Agent**: Architect/Auditor

### Executive Summary
The current roc-star implementation exists as loose, standalone Python files with **no module structure, minimal input validation, tight coupling, and hardcoded configurations**. Significantly violates ArchitectureRefactor.md design principles.

### Public API Surface
**Identified Functions**:
1. `epoch_update_gamma()` - rocstar.py:1-42, example.py:86-124 (DUPLICATE)
2. `roc_star_loss()` - rocstar.py:46-116, example.py:127-191 (DUPLICATE)
3. `train_model()` - example.py:338-473 (tightly coupled to LSTM)

**Critical Finding**: Duplicate implementations with parameter inconsistencies:
- rocstar.py default: `delta=1`, `SUB_SAMPLE_SIZE=2000.0`
- example.py default: `delta=2`, `sub_sample_size=2000.0`
- Confusing epoch condition (rocstar.py:39-42) inverts expected behavior

### Parameter Validation Gaps
- ❌ No tensor shape validation
- ❌ No value range validation (assumes [0,1])
- ⚠️ Partial NaN detection (line 114 filters output only)
- ❌ No type checking
- ❌ Hardcoded `.cuda()` calls without CPU fallback (lines 32, 92, 105)

### Configuration & Toggles
**Magic Numbers Needing Configuration**:
- `2000.0` - Gamma subsample size
- `delta=1/2` - Gamma quantile parameter
- `1000` - Loss subsample cap
- `0.2` - Default gamma fallback
- `0.50` - Label threshold
- `1e-8` - Stub loss for empty batches

**Missing Toggles** (per ArchitectureRefactor.md):
- random_seed, device, validate_inputs, gamma_subsample_size, loss_subsample_size

### Prioritized Refactoring Tasks
**PHASE 1 (Critical - No Breaking Changes)**:
1. P0: Add device detection & configuration (replace all `.cuda()`)
2. P0: Add comprehensive input validation
3. P0: Fix duplicate implementations (consolidate into rocstar.py)
4. P1: Extract magic numbers to constants
5. P1: Add type hints
6. P2: Improve docstrings (NumPy format)

---

## Security & Bug Findings
**Task ID**: SWE-001 | **Status**: COMPLETE | **Agent**: Senior SWE Auditor (Red Team)

### Critical Issues (P0) - Code Will Crash

#### 1. DIVISION BY ZERO in `epoch_update_gamma()`
- **Location**: rocstar.py:15-16, example.py:99-100
- **Trigger**: Single-class batch (cap_pos=0 or cap_neg=0)
- **Impact**: `ZeroDivisionError` crash
- **Fix**: Guard before division

#### 2. DIVISION BY ZERO in `roc_star_loss()`
- **Location**: rocstar.py:76-77, example.py:156-157
- **Issue**: `max_pos/cap_pos` crashes if cap_pos=0
- **ALSO**: Copy-paste bug using `cap_pos` for both pos and neg!
- **Fix**: Check for zero, use correct divisor (cap_neg not cap_pos)

#### 3. EMPTY TENSOR INDEXING
- **Location**: rocstar.py:34, example.py:117
- **Issue**: `diff_neg[left_wing]` crashes when diff_neg is empty
- **Fix**: Check `diff_neg.shape[0] > left_wing` before indexing

#### 4. SILENT NaN PROPAGATION
- **Location**: rocstar.py:114, example.py:189
- **Issue**: Catches NaN but not INF; division by constants (max_pos/max_neg) not actual sizes
- **Impact**: Loss=0 when should be computed; incorrect loss scaling
- **Fix**: Use len2/len3 not max_pos/max_neg; check for INF

### High Priority Issues (P1)

#### 5. NON-DETERMINISTIC RANDOMNESS
- **Location**: rocstar.py:15-16, 76-77
- **Issue**: `torch.rand_like()` without seeding - non-reproducible results
- **Fix**: Use torch.Generator with manual_seed

#### 6. HARDCODED .cuda() WITHOUT CHECK
- **Location**: rocstar.py:32, 92, 105; example.py:75-78, 115, 171
- **Trigger**: Running on CPU-only machine
- **Impact**: `RuntimeError: CUDA is not available`
- **Fix**: Auto-detect device or infer from input tensors

#### 7. WRONG RETURN ON SINGLE-CLASS BATCHES
- **Location**: rocstar.py:63, example.py:144
- **Issue**: Returns `torch.sum(y_pred)*1e-8` instead of true zero
- **Impact**: Training instability
- **Fix**: Return `torch.tensor(0.0, device=y_pred.device)`

### Total Found
**11 critical/high-priority issues** including 4 crash-causing bugs, edge case failures, non-deterministic behavior, GPU-dependency hazards.

---

## Algorithm Correctness Findings
**Task ID**: ALG-001 | **Status**: COMPLETE | **Agent**: Algorithm Researcher

### Mathematical Correctness Verification
**Formula from README (line 87)**: Loss = Σ(max(0, y_i - x_j + Γ))^p

**Implementation Status**: ✅ CORRECT
- Line 87: `diff2 = neg_expand - pos_expand + gamma` ✓
- Line 88: `l2 = diff2[diff2>0]` (ReLU operation) ✓
- Line 89: `m2 = l2 * l2` (p=2 applied) ✓
- Same pattern for diff3, l3, m3 ✓

### Critical Algorithm Bugs

#### Bug #1: Line 77 - Wrong divisor in subsampling
```python
epoch_neg = epoch_neg[torch.rand_like(epoch_neg) < max_neg/cap_pos]  # WRONG
```
Should be `max_neg/cap_neg`. Causes **undersubsampling of negatives** when `cap_pos ≠ cap_neg`.

#### Bug #2: Line 8 - Incorrect DELTA calculation
```python
DELTA = delta+1  # WRONG
```
Per README line 129, DELTA should be `delta` not `delta+1`. Makes gamma **2x larger than intended**.

#### Bug #3: Lines 39-42 - Epoch condition logic unclear
Current logic appears inverted - needs clarification on intended behavior.

#### Bug #4: Lines 108-109 - Denominator normalization error
```python
res2 = torch.sum(m2)/max_pos + torch.sum(m3)/max_neg
```
Divides by **fixed constants** (1000) instead of actual subsample sizes. Should be:
```python
res2 = torch.sum(m2)/len2 + torch.sum(m3)/len3
```

### Numerical Stability
- No overflow protection (large differences squared can overflow)
- Underflow risk (line 114 handles NaN but not near-zero)
- Unbalanced subsampling (different pos/neg sizes not normalized)

### Parameter Evaluation
- δ (delta): Default 1.0 correct, but Bug #2 distorts it
- p: Fixed at 2 (correct - convex, differentiable)
- Subsample sizes: 2000 (gamma), 1000 (loss) - inconsistent, arbitrary
- default_gamma: 0.2 (arbitrary fallback)

### Proposed Improvements
**Conservative**: Fix bugs #1-4 above
**Performance/Correctness**: Normalize by actual pair counts (len2/len3), add numerical safeguards

---

## Data Integrity Findings
**Task ID**: BIO-001 | **Status**: COMPLETE | **Agent**: Bioinformatics Researcher

### Dataset Splitting
**Status**: ✅ SAFE
- Train/valid pre-split from pickle (example.py:67)
- Validation set NEVER used in loss calculations
- No leakage detected between train and validation

### Last-Epoch Mechanism
**Status**: ⚠️ MODERATE RISK (BY DESIGN)
- Loss function uses 1-epoch lag (compares epoch N vs epoch N-1 data)
- **Intentional design** per README line 224 for performance
- Not data leakage but unconventional temporal inconsistency

### Initialization
**Status**: ⚠️ CAUTION
- Epoch 0 uses BCE guard (safe from crash)
- But no explicit initialization of last_epoch_y_pred per README spec (line 280)

### Random Subsampling Bias
**Status**: ⚠️ REPRODUCIBILITY CONCERN
- torch.rand_like() without batch-level seeding
- Different batches see different subsets of epoch data
- Non-reproducible loss values for identical batches

### Class Imbalance
**Status**: ❓ UNKNOWN
- No class weighting detected
- ROC-AUC metric is threshold-agnostic (good for imbalance)
- Hard threshold at 0.5 (line 399) ignores calibration

### Data Integrity Checklist
| Item | Status | Evidence |
|------|--------|----------|
| Train/Valid Separation | ✅ PASS | Lines 75-78 |
| Last-Epoch Mechanism | ⚠️ DESIGN | Lines 413-414 |
| Initialization Safety | ⚠️ CAUTION | Lines 372-379 |
| Random Seeding | ⚠️ PARTIAL | Line 507 |
| Subsampling Bias | ⚠️ CONCERN | Lines 156-157 |
| Class Imbalance | ❓ UNKNOWN | Data-dependent |
| Label Consistency | ✅ PASS | Line 77 |
| Preprocessing | ✅ PASS | Pre-computed |

---

## Evaluation Protocol Findings
**Task ID**: GAME-001 | **Status**: COMPLETE | **Agent**: Game Theory Researcher

### Metric Gaming Opportunities

#### 1. AUC Threshold Hardcoded to 0.5
**Location**: Lines 140, 399, 431
**Issue**: AUC calculated against discretized labels (binary 0/1) while labels may be continuous
**Gaming Risk**: Model can improve loss without improving true AUC

#### 2. No Early Stopping
**README Claims**: "eliminates entirely the need for Early Stopping" (line 69)
**Code Reality**: Model trains all n_epochs (default 30) even after validation AUC plateaus
**FAILURE SCENARIO #1**: Silent overfitting - validation AUC peaks at epoch 5, degrades epochs 6-30

### Selection Bias

#### 3. Single Validation Set Never Shuffled
**Location**: Lines 418-426, line 332
**Issue**: Validation set deterministic order - if dataset has clustering, AUC inflated
**FAILURE SCENARIO #2**: Distribution shift - validation set not representative of test data

### Metric Mismatch

#### 4. Loss vs. AUC Optimization Gap
**Issue**: Loss optimizes pairwise ranking with gamma, AUC is binary threshold metric
**Gaming Risk**: Loss decreases while AUC plateaus

### Default Promotion Risk

#### 5. No Stability Check for Model Selection
**Location**: Line 448
**Issue**: Single validation AUC spike (noise) triggers model save
**Missing**: Should require AUC improvement to persist 2+ epochs or exceed margin

### Silent Failures
| Failure Mode | Location | Silent? | Detection Method |
|---|---|---|---|
| AUC threshold gaming | 140, 399, 431 | ✅ Yes | Compare soft vs. hard AUC |
| No early stopping | 355-473 | ✅ Yes | Monitor train/val AUC gap |
| Validation set bias | 418, 332 | ✅ Yes | Shuffle/resample AUC variance |
| Loss-AUC mismatch | 127-185 vs. 431 | ✅ Yes | Track loss vs AUC correlation |
| Noise-driven promotion | 448-453 | ✅ Yes | Require 2-epoch persistence |
| Loss saturation | 407-409 | ✅ Yes | Monitor loss variance |

### Contrarian Critique
Even assuming roc_star loss is theoretically sound, **evaluation protocol is fragile**. No empirical proof of "no early stopping needed" claim. Validation AUC alone insufficient without confidence measures, shuffle tests, and divergence monitoring.

---

## Proposed Refactorings

### Bold Refactor Proposals
*To be populated during Creative Contradiction Protocol*

TBD

### Safe Incremental Changes
*To be populated during Findings Triage*

TBD

---

## Creative Contradiction Analysis

### Proposal vs. Critique
*To be populated during Phase 5*

TBD

### Evidence-Based Resolution
TBD

---

## Implementation History

### Changes Applied

#### 2026-02-18: Infrastructure Setup
**Task**: INFRA-001 through INFRA-003  
**Changes**:
- Created AGENTS.md (agent discipline guidelines)
- Created SUBAGENT.md (task card templates)
- Created Progress.md (active tracking board)
- Created Archive.md (this file)

**Rationale**: Establish audit infrastructure before spawning subagents

**Test Coverage**: N/A (documentation only)

**Residual Risk**: None

---

## Appendices

### Appendix A: Source Material
- **Yan et al. 2003**: "Optimizing Classifier Performance via an Approximation to the Wilcoxon-Mann-Whitney Statistic"
- **GitHub Article**: https://github.com/iridiumblue/articles/blob/master/roc_star.md
- **Original TFLearn Issue**: http://tflearn.org/objectives/#roc-auc-score

### Appendix B: Terminology
- **AUC**: Area Under the Curve (ROC curve)
- **BCE**: Binary Cross Entropy
- **WMW**: Wilcoxon-Mann-Whitney statistic
- **Γ (gamma)**: Padding parameter enforcing separation between classes
- **δ (delta)**: Proportion of too-close pairs to wrong-ordered pairs
- **p**: Exponent parameter (fixed at 2 in this implementation)

### Appendix C: Test Command Discovery
*To be populated by BASE-001*

TBD

---

*Document maintained by TABNETICS Orchestrator*  
*Last Updated*: 2026-02-18 04:11 UTC
