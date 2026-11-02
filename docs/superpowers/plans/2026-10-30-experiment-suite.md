# Experiment Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Build and smoke-test the requested reproducible experiment suite.

**Architecture:** A Python runner executes one experiment configuration and writes a self-contained result directory. A Bash script loops the exact requested parameter grids. Public helpers construct multi-column marginal quotas and recompute PCA/UMAP.

**Tech Stack:** Bash, Python, anndata, NumPy, SciPy, scikit-learn, UMAP, matplotlib.

**Spec:** `docs/superpowers/specs/2026-10-30-experiment-suite-design.md`

## Global Constraints

- Experiment 1 uses 60% per selected celltype and similarity 0.7.
- Experiment 2 uses all celltype and ident categories, each at 60%.
- Experiment 3 uses exact unrestricted totals.
- Each run saves its h5ad and embedding reports.
- Reference existing UMAP is never overwritten.
- The workspace has no usable Git repository; tests replace commit verification.

---

### Task 1: Multi-Column Fraction API

**Files:**
- Modify: `src/scrnarefgenerator/conditions.py`
- Modify: `src/scrnarefgenerator/__init__.py`
- Test: `tests/test_condition_spec_builder.py`

- [x] **Step 1: Write a failing test**

```python
def test_multi_column_fraction_uses_exact_equal_totals():
    spec = condition_spec_from_marginal_fraction(
        reference, ["celltype", "ident"], fraction=0.6
    )
    assert set(spec.columns) == {"celltype", "ident"}
    assert sum(spec.marginal["celltype"].values()) == 12
    assert sum(spec.marginal["ident"].values()) == 12
```

- [x] **Step 2: Verify RED**

Run: `PYTHONPATH=src python -m pytest tests/test_condition_spec_builder.py -q`
Expected: FAIL because the helper is missing.

- [x] **Step 3: Implement floor-based per-column quotas**

Generate all category quotas, validate equal totals, and return a marginal
`ConditionSpec`.

- [x] **Step 4: Verify GREEN**

Run: `PYTHONPATH=src python -m pytest tests/test_condition_spec_builder.py -q`
Expected: PASS.

### Task 2: Generated PCA/UMAP

**Files:**
- Create: `src/scrnarefgenerator/embedding.py`
- Modify: `src/scrnarefgenerator/__init__.py`
- Test: `tests/test_embedding.py`

- [x] **Step 1: Write a failing test**

```python
def test_compute_pca_umap_returns_finite_coordinates(toy_counts):
    result = compute_pca_umap(toy_counts, n_pcs=4, n_neighbors=5, seed=3)
    assert result.pca.shape == (toy_counts.shape[0], 4)
    assert result.umap.shape == (toy_counts.shape[0], 2)
    assert np.isfinite(result.pca).all()
    assert np.isfinite(result.umap).all()
```

- [x] **Step 2: Verify RED**

Run: `PYTHONPATH=src python -m pytest tests/test_embedding.py -q`
Expected: FAIL because `embedding.py` is missing.

- [x] **Step 3: Implement sparse normalization, TruncatedSVD, and UMAP**

Use a writable Numba cache directory before importing UMAP.

- [x] **Step 4: Verify GREEN**

Run: `PYTHONPATH=src python -m pytest tests/test_embedding.py -q`
Expected: PASS.

### Task 3: Shared Plot Simulation Column

**Files:**
- Modify: `src/scrnarefgenerator/plotting.py`
- Test: `tests/test_plotting.py`

- [x] **Step 1: Add a test for different reference/simulation color columns**

Reference uses `celltype`, simulated uses `source_celltype`, and plotting
succeeds with matching category labels.

- [x] **Step 2: Verify RED**

Run: `PYTHONPATH=src python -m pytest tests/test_plotting.py -q`
Expected: FAIL because the parameter is unsupported.

- [x] **Step 3: Add `simulated_condition_col` and align sample categories**

Validate both columns and use one shared category set.

- [x] **Step 4: Verify GREEN**

Run: `PYTHONPATH=src python -m pytest tests/test_plotting.py -q`
Expected: PASS.

### Task 4: Python Experiment Runner

**Files:**
- Create: `scripts/run_experiment.py`

- [x] **Step 1: Implement three modes**

`celltype`, `celltype_ident`, and `unrestricted`. Each mode writes the h5ad,
evaluation JSON, independent embedding plots, existing-reference UMAP, and
shared comparison.

- [x] **Step 2: Smoke-test with tiny settings**

Run one small configuration for each mode and assert all expected files exist.

### Task 5: Shell Loop and Final Verification

**Files:**
- Create: `scripts/run_experiments.sh`
- Modify: `README.md`

- [x] **Step 1: Implement exact requested loops**

Use Bash arrays for Experiment 1, 2, and 3 settings.

- [x] **Step 2: Validate shell syntax**

Run: `bash -n scripts/run_experiments.sh`

- [x] **Step 3: Run smoke experiments**

Run at least one setting from each experiment with reduced quotas/counts.

- [x] **Step 4: Run the full test suite**

Run: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -p no:cacheprovider -q`
Expected: zero failures.
