# Progress.md — Classifier Pipeline Roadmap

**Session**: ROADMAP-2026-02-20 — Classifier Pipeline & AutoML Integration  
**Orchestrator**: TABNETICS Orchestrator (ROADMAP MODE)  
**Status**: 🟢 **PHASE 0 COMPLETE — PHASE 1 IN PROGRESS**

---

## Roadmap Session: Classifier Pipeline & AutoML Integration

### Session Objectives
Plan and implement a modular classifier pipeline built on the roc-star AUC loss, with Optuna-based HP search replacing the deprecated `hp_search.py`, a runnable minimal example, and a FLAML baseline for comparison.

**Contradiction protocol applied** — see Archive.md § ROADMAP-2026-02-20 for dissenting opinions.

---

## Task Board

### Phase 0 — Foundation (✅ COMPLETE)

| Task ID | Description | Owner | Status | Acceptance Criteria |
|---------|-------------|-------|--------|---------------------|
| T-001 | `tests/test_rocstar.py` — pytest tests for core functions (synthetic tensors only) | SWE-ROAD-001 | ✅ DONE | 17/17 tests pass; edge cases: empty class, all-pos/neg, backward |
| T-002 | Fix BIO-R3: align label threshold in `epoch_update_gamma` to `>= 0.5` (matching `roc_star_loss`) | BIO-ROAD-001 | ✅ DONE | Soft labels handled consistently; covered by `test_soft_labels` |
| T-003 | `minimal_example.py` — self-contained MLP on synthetic data; no external deps | SWE-ROAD-001 | ✅ DONE | Runs with `pip install torch scikit-learn`; final test AUC ≥ 0.85 |
| T-004 | Fix `.gitignore` — exclude `__pycache__`, `.pytest_cache`, `*.db` | SWE-ROAD-001 | ✅ DONE | No build artifacts committed |

**Go/No-Go CP-0**: All Phase 0 tests pass (17/17 ✅). Proceed to Phase 1.

---

### Phase 1 — AutoML HP Search (✅ COMPLETE)

| Task ID | Description | Owner | Status | Acceptance Criteria |
|---------|-------------|-------|--------|---------------------|
| T-005 | `optuna_search.py` — Optuna TPE + ASHA, 3-split protocol, SQLite, seed | SWE-ROAD-001 | ✅ DONE | Reproducible 20-trial study; val AUC reported separately from test AUC |
| T-006 | `flaml_baseline.py` — FLAML AutoML baseline, identical split, graceful skip if not installed | SWE-ROAD-001 | ✅ DONE | Runs if `flaml` installed; reports roc-star vs FLAML AUC; skips cleanly |
| T-007 | Update documentation — Progress.md, Archive.md, ArchitectureRefactor.md | Orchestrator | ✅ DONE | Canonical doc set consistent; old cruft removed |

**Go/No-Go CP-1**: `optuna_search.py` completes 20 trials; val AUC spread ≥ 0.005 across delta values.  
**Go/No-Go CP-2** (conditional): FLAML comparison uses identical split + metric. If not, defer comparison reporting.

---

### Phase 2 — Selective Architecture (🔲 DEFERRED)

| Task ID | Description | Condition | Priority |
|---------|-------------|-----------|----------|
| T-008 | Type annotations on `rocstar.py` public API | Anytime — zero risk | P3-Low |
| T-009 | `RocStarCallback` for Lightning (no ABC) | Only if ≥3 model types in repo | P2-Medium |
| T-010 | Deterministic subsampling (optional `generator` param) | v1.1 with deprecation path | P2-Medium |
| T-011 | Input validation layer (`validate_inputs` flag) | v1.1 | P2-Medium |

**Permanently deferred** (per GAME-ROAD-001 decision):  
- `BaseClassifier` / `BaseAutoML` ABC hierarchy (over-engineering for a loss-function library)  
- Stacking ensemble (requires CV infra not in codebase)  
- GammaNet meta-learning moonshot (bi-level instability; delta HP already in Optuna search space)

---

## Validation Checkpoints

| CP | After | Metric | Pass Threshold | Fail Action |
|----|-------|--------|---------------|-------------|
| CP-0 | T-001 | pytest pass rate | 100% (17/17) | Fix rocstar.py before Phase 1 |
| CP-1 | T-003 | Final test AUC (minimal_example) | ≥ 0.85 | Investigate training loop |
| CP-2 | T-005 | AUC spread across 20 Optuna trials | ≥ 0.005 | Redesign HP search space |
| CP-3 | T-005 | ASHA: ≥1 trial pruned in 20-trial study | ≥1 pruned | Check `trial.report` placement |
| CP-4 | T-006 | FLAML comparison validity | Identical split documented | Defer FLAML reporting |

**Observed Results**:
- CP-0 ✅: 17/17 tests pass
- CP-1 ✅: final test AUC = 0.9357 (>> 0.85 threshold)

---

## Architecture Decision Log

| Decision | Chosen | Rejected | Rationale |
|----------|--------|----------|-----------|
| Classifier abstraction | No ABC; plain functions + scripts | `BaseClassifier` hierarchy | Loss-function library; ABC competes with sklearn/Lightning without adding user value |
| AutoML primary | Optuna TPE | H2O, auto-sklearn | PyTorch-native; AUC proxy works end-to-end; MIT license |
| Multi-fidelity | ASHA in Optuna | Standalone Ray Tune | Same code path as TPE; low incremental effort |
| GammaNet moonshot | DEFERRED | — | Bi-level instability; delta parameter already captured in Optuna HP space |
| Stacking | DEFERRED | — | Needs k-fold CV infra; high prediction correlation inflates expected gains |
| Epoch state isolation | Per-trial local vars + seed offset | Module-global state | Fixes BIO-R2: eliminates state contamination across HP trials |
| Data splits | 60/20/20 stratified 3-way | Single train/val | Fixes BIO-R1: held-out test set not used for HP selection |
| Label threshold | `>= 0.5` everywhere | Exact `==1`/`==0` | Fixes BIO-R3: consistent soft-label handling |

---

## Contradiction Summary (required per AGENTS.md)

**Disagreement #1** — ARCH-ROAD-001 vs GAME-ROAD-001:  
ARCH proposed a `BaseClassifier` / `BaseAutoML` ABC hierarchy. GAME rejected it as premature abstraction for a 134-LOC loss library. **Resolution**: No ABC; Optuna + script-per-framework pattern adopted.

**Disagreement #2** — ALG-ROAD-001 vs GAME-ROAD-001:  
ALG rated ASHA as "Low effort." GAME correctly rated it Medium (requires training loop refactoring with `trial.report`/`should_prune`). **Resolution**: ASHA implemented in `optuna_search.py` with explicit `trial.report(val_auc, epoch)` hook; effort confirmed Medium.

**Moonshot critique** — ALG proposed GammaNet meta-learning (+0.01–0.05 AUC). GAME attacked: bi-level optimization instability, 1D-problem solved by 1D HP search (delta in Optuna). **Resolution**: GammaNet permanently deferred; delta in log-scale Optuna search space.

**BIO critique** — BIO-R1 (val-set overfitting in HP search): addressed by 3-split protocol in `optuna_search.py`. BIO-R5 (cross-framework comparison invalidity): FLAML baseline uses identical train/test split and logs AUC on same held-out test set.

---

## Files Delivered This Session

| File | Status | Purpose |
|------|--------|---------|
| `tests/__init__.py` | ✅ New | Package marker |
| `tests/test_rocstar.py` | ✅ New | 17 pytest tests for core functions |
| `minimal_example.py` | ✅ New | Self-contained MLP demo (no external deps) |
| `optuna_search.py` | ✅ New | Optuna TPE + ASHA HP search (replaces hp_search.py) |
| `flaml_baseline.py` | ✅ New | FLAML AutoML baseline comparison |
| `rocstar.py` | ✅ Modified | BIO-R3 fix: `>= 0.5` threshold in `epoch_update_gamma` |
| `.gitignore` | ✅ Modified | Exclude pycache, .pytest_cache, *.db |
| `Progress.md` | ✅ Rewritten | This file (clean roadmap; old audit history in Archive.md) |
| `Archive.md` | ✅ Updated | Roadmap findings from all 5 sub-agents |
| `ArchitectureRefactor.md` | ✅ Updated | Classifier pipeline architecture decisions |

---

*Last Updated*: 2026-02-20  
*Session*: ROADMAP-2026-02-20 | *Orchestrator*: TABNETICS
