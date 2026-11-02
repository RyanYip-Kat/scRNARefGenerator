# Condition Metadata and Export API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add verifiable source metadata inheritance, structured marginal allocation, and raw matrix/metadata export.

**Architecture:** Condition planning remains the source of target groups. Assembly copies validated source metadata under explicit names. Export is a separate public I/O API that resolves counts before writing aligned matrix and metadata files.

**Tech Stack:** Python 3.10+, anndata, NumPy, SciPy, pandas, pytest.

**Spec:** `docs/superpowers/specs/2026-10-30-condition-metadata-export-api-design.md`

## Global Constraints

- Exact margins and source capacities cannot be silently changed.
- Existing condition and internal metadata names cannot be overwritten by inheritance.
- Raw exports use integer counts and never normalized data.
- Matrix rows/columns and metadata tables remain explicitly aligned.
- The workspace has no usable Git repository; tests replace commit verification.

---

### Task 1: Metadata Inheritance

**Files:**
- Modify: `src/scrnarefgenerator/config.py`
- Modify: `src/scrnarefgenerator/assembly.py`
- Test: `tests/test_api_contract.py`

- [x] **Step 1: Write a failing test with `celltype` and `ident`**

```python
def test_inherited_ident_matches_source_condition(toy_metadata_reference):
    generated = scRNARefGenerator(
        toy_metadata_reference,
        GeneratorConfig(inherit_obs_cols=["ident"]),
    ).simulate(ConditionSpec(columns=["celltype"], joint={("T",): 3}))
    assert (generated.obs["ident"] == generated.obs["source_ident"]).all()
    assert (generated.obs["celltype"] == generated.obs["source_celltype"]).all()
```

- [x] **Step 2: Verify RED**

Run: `PYTHONPATH=src python -m pytest tests/test_api_contract.py -q`
Expected: FAIL because `inherit_obs_cols` is missing.

- [x] **Step 3: Copy all source metadata and validate inherited columns**

Prefix every source obs column with `source_`; add requested columns under
their original names. Reject missing or conflicting columns.

- [x] **Step 4: Verify GREEN**

Run: `PYTHONPATH=src python -m pytest tests/test_api_contract.py -q`
Expected: PASS.

### Task 2: Reference-Proportional Marginal Allocation

**Files:**
- Modify: `src/scrnarefgenerator/conditions.py`
- Test: `tests/test_conditions.py`

- [x] **Step 1: Write a failing allocation test**

```python
def test_marginal_plan_avoids_concentrating_joint_counts(full_support_reference):
    plan = build_condition_plan(
        full_support_reference,
        ConditionSpec(
            columns=["celltype", "ident"],
            marginal={
                "celltype": {"T": 4, "B": 4},
                "ident": {1: 4, 2: 4},
            },
        ),
        None,
    )
    quota = {group.values: group.n_cells for group in plan.groups}
    assert max(quota.values()) - min(quota.values()) <= 1
    assert sum(quota.values()) == 8
```

- [x] **Step 2: Verify RED**

Run: `PYTHONPATH=src python -m pytest tests/test_conditions.py -q`
Expected: FAIL because the zero-objective solver may choose a concentrated plan.

- [x] **Step 3: Add proportional target and L1 deviation objective**

Keep exact total and marginal constraints. Minimize absolute deviation from the
capacity-proportional joint target and then round the integer solution.

- [x] **Step 4: Verify GREEN**

Run: `PYTHONPATH=src python -m pytest tests/test_conditions.py -q`
Expected: PASS.

### Task 3: Multi-Column API

**Files:**
- Test: `tests/test_api_contract.py`

- [x] **Step 1: Add a `celltype` and `ident` end-to-end test**

Assert exact per-column quotas, output condition columns equal source condition
columns, inherited ident equals source ident, and all barcodes are unique.

- [x] **Step 2: Verify behavior and fix any assembly mismatch**

Run: `PYTHONPATH=src python -m pytest tests/test_api_contract.py -q`
Expected: PASS after Task 1 and Task 2.

### Task 4: Raw Matrix and Metadata Export

**Files:**
- Create: `src/scrnarefgenerator/export.py`
- Modify: `src/scrnarefgenerator/__init__.py`
- Test: `tests/test_export.py`

- [x] **Step 1: Write a failing round-trip export test**

```python
def test_export_raw_counts_preserves_matrix_and_metadata_order(toy_adata, tmp_path):
    artifacts = export_raw_counts(
        toy_adata,
        tmp_path,
        prefix="sim",
        matrix_format="mtx",
    )
    matrix = mmread(artifacts.matrix).tocsc()
    obs = pd.read_csv(artifacts.obs, index_col=0)
    var = pd.read_csv(artifacts.var, index_col=0)
    assert matrix.shape == (toy_adata.n_vars, toy_adata.n_obs)
    assert obs.index.tolist() == toy_adata.obs_names.tolist()
    assert var.index.tolist() == toy_adata.var_names.tolist()
```

- [x] **Step 2: Verify RED**

Run: `PYTHONPATH=src python -m pytest tests/test_export.py -q`
Expected: FAIL because `export_raw_counts` is missing.

- [x] **Step 3: Implement MTX/CSV and NPZ export**

Resolve raw counts, write matrix genes-by-cells for MTX or cells-by-genes NPZ,
and write aligned obs/var CSV files with stable ID columns.

- [x] **Step 4: Verify GREEN**

Run: `PYTHONPATH=src python -m pytest tests/test_export.py -q`
Expected: PASS.

### Task 5: Full Verification

**Files:**
- Modify: `README.md`

- [x] **Step 1: Document inherited metadata, marginal semantics, and export**
- [x] **Step 2: Run the full suite**

Run: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -p no:cacheprovider -q`
Expected: zero failures.
