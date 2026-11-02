# scRNARefGenerator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Implement a tested Python package for exact conditional sparse-count simulation from an scRNA reference.

**Architecture:** A public facade coordinates reference validation, condition planning, sparse NB/ZINB simulation, evaluation, and AnnData assembly. Intermediate objects are small and testable; statistical code never performs file I/O.

**Tech Stack:** Python 3.10+, anndata, NumPy, SciPy, pandas, scikit-learn, pytest.

**Spec:** `docs/superpowers/specs/2026-10-30-scrnarefgenerator-design.md` and `design.md`

## Global Constraints

- Python source lives under `src/scrnarefgenerator/`; tests live under `tests/`.
- Python floor is 3.10.
- Counts remain sparse and are never converted to a dense cell-by-gene matrix.
- Default gene order is exactly the reference `var_names` order.
- Randomness uses `numpy.random.default_rng`; global random state is forbidden.
- Existing files are overwritten only when explicitly configured.
- This workspace has no usable Git repository, so commit steps are replaced by diff and test verification.

---

### Task 1: Package Skeleton and Public Configuration

**Files:**
- Create: `pyproject.toml`
- Create: `src/scrnarefgenerator/__init__.py`
- Create: `src/scrnarefgenerator/config.py`
- Create: `src/scrnarefgenerator/errors.py`
- Test: `tests/test_public_api.py`

**Interfaces:**
- Produces: `ConditionSpec(columns, joint=None, marginal=None)`
- Produces: `GeneratorConfig(...)`
- Produces: `EvaluationConfig(...)`
- Produces: `scRNARefGenerator(ref, config=None)`

- [x] **Step 1: Write the failing public API test**

```python
from scrnarefgenerator import GeneratorConfig, EvaluationConfig, ConditionSpec


def test_public_config_defaults_are_valid():
    config = GeneratorConfig(seed=7)
    spec = ConditionSpec(columns=["celltype"], joint={("T",): 2})
    assert config.seed == 7
    assert config.distribution_model == "hierarchical_nb"
    assert EvaluationConfig().auc_range == (0.55, 0.90)
    assert spec.joint == {("T",): 2}
```

- [x] **Step 2: Run the test and verify the package import fails**

Run: `PYTHONPATH=src python -m pytest tests/test_public_api.py -q`
Expected: FAIL because `scrnarefgenerator` does not exist.

- [x] **Step 3: Create package metadata, errors, and validated dataclasses**

Use setuptools src-layout discovery and the `scrnarefgen` console entry point. Implement post-init validation for probabilities, positive integer quotas, valid collision policies, and mutually exclusive joint/marginal specifications.

- [x] **Step 4: Run the test and verify it passes**

Run: `PYTHONPATH=src python -m pytest tests/test_public_api.py -q`
Expected: `1 passed`.

### Task 2: Counts Resolution and Reference Validation

**Files:**
- Create: `src/scrnarefgenerator/validation.py`
- Test: `tests/test_validation.py`

**Interfaces:**
- Consumes: `GeneratorConfig.counts_layer`
- Produces: `ReferenceData(X, obs, obs_names, var_names, path)`
- Produces: `resolve_reference(ref, config) -> ReferenceData`

- [x] **Step 1: Write failing sparse-count tests**

```python
import anndata as ad
import numpy as np
import scipy.sparse as sp
from scrnarefgenerator.config import GeneratorConfig
from scrnarefgenerator.errors import InvalidCountsError
from scrnarefgenerator.validation import resolve_reference


def test_resolves_integral_counts_layer_without_densifying():
    adata = ad.AnnData(sp.csr_matrix([[0.0, 2.0], [3.0, 0.0]]))
    adata.obs_names = ["a", "b"]
    adata.var_names = ["g1", "g2"]
    adata.layers["counts"] = adata.X.copy()
    got = resolve_reference(adata, GeneratorConfig())
    assert sp.issparse(got.X)
    assert got.X.dtype.kind in "iu"
    assert got.X.toarray().tolist() == [[0, 2], [3, 0]]


def test_rejects_nonintegral_counts():
    adata = ad.AnnData(sp.csr_matrix([[0.5, 1.0]]))
    adata.obs_names = ["a"]
    adata.var_names = ["g1", "g2"]
    try:
        resolve_reference(adata, GeneratorConfig())
    except InvalidCountsError:
        pass
    else:
        raise AssertionError("fractional fallback counts must be rejected")
```

- [x] **Step 2: Verify RED**

Run: `PYTHONPATH=src python -m pytest tests/test_validation.py -q`
Expected: FAIL importing `resolve_reference`.

- [x] **Step 3: Implement resolution precedence and validation**

Check finite, nonnegative, integral values; unique axes; shape alignment; raw gene alignment; and missing-count errors. Return CSR integer storage.

- [x] **Step 4: Verify GREEN**

Run: `PYTHONPATH=src python -m pytest tests/test_validation.py -q`
Expected: all tests pass.

### Task 3: Exact Condition Plans

**Files:**
- Create: `src/scrnarefgenerator/conditions.py`
- Test: `tests/test_conditions.py`

**Interfaces:**
- Consumes: `ReferenceData`
- Produces: `TargetGroup(values, n_cells, source_indices)`
- Produces: `ConditionPlan(columns, groups, total_cells, mode)`
- Produces: `build_condition_plan(reference, spec, n_cells)`

- [x] **Step 1: Write failing joint and marginal tests**

```python
def test_joint_quota_selects_only_matching_sources(toy_reference):
    spec = ConditionSpec(columns=["batch"], joint={("B",): 2})
    plan = build_condition_plan(toy_reference, spec, None)
    assert plan.total_cells == 2
    assert toy_reference.obs.iloc[plan.groups[0].source_indices]["batch"].eq("B").all()


def test_impossible_marginals_raise(toy_reference):
    spec = ConditionSpec(
        columns=["batch", "group"],
        marginal={"batch": {"A": 2}, "group": {"x": 2}},
    )
    with pytest.raises(InfeasibleConditionError):
        build_condition_plan(toy_reference, spec, None)
```

- [x] **Step 2: Verify RED**

Run: `PYTHONPATH=src python -m pytest tests/test_conditions.py -q`
Expected: FAIL because the planner is missing.

- [x] **Step 3: Implement joint validation and integer marginal MILP**

Build variables only for observed condition tuples. Include exact total and per-column marginal constraints, use observed cell counts as upper bounds, and convert every solution value to an integer.

- [x] **Step 4: Verify GREEN**

Run: `PYTHONPATH=src python -m pytest tests/test_conditions.py -q`
Expected: all tests pass.

### Task 4: Statistical Fitting and Sparse Simulation

**Files:**
- Create: `src/scrnarefgenerator/distribution.py`
- Create: `src/scrnarefgenerator/simulate.py`
- Test: `tests/test_simulation_core.py`

**Interfaces:**
- Consumes: `ReferenceData`, `ConditionPlan`, `GeneratorConfig`
- Produces: `fit_distribution(reference, plan, config) -> FittedDistribution`
- Produces: `sample_counts(reference, plan, fitted, config, rng) -> (csr_matrix, obs_table)`

- [x] **Step 1: Write failing stochastic-contract tests**

```python
def test_same_seed_generates_identical_sparse_counts(toy_reference, toy_plan):
    config = GeneratorConfig(seed=11, latent_k=0, strict_evaluation=False)
    a = sample_candidate(toy_reference, toy_plan, config, attempt=0)
    b = sample_candidate(toy_reference, toy_plan, config, attempt=0)
    assert (a.counts != b.counts).nnz == 0
    assert a.counts.dtype.kind in "iu"


def test_target_library_size_is_met_within_sampling_tolerance(toy_reference, toy_plan):
    config = GeneratorConfig(seed=3, sigma_lib=0.0, latent_k=0)
    candidate = sample_candidate(toy_reference, toy_plan, config, attempt=0)
    target = candidate.obs["_target_library_size"].to_numpy()
    observed = np.asarray(candidate.counts.sum(axis=1)).ravel()
    assert np.median(observed / target) > 0.75
    assert np.median(observed / target) < 1.25
```

- [x] **Step 2: Verify RED**

Run: `PYTHONPATH=src python -m pytest tests/test_simulation_core.py -q`
Expected: FAIL because the simulator is missing.

- [x] **Step 3: Implement fit, normalization, NB/ZINB, and optional latent perturbation**

Use sparse weighted sums for gene probabilities and moments, shrink group parameters, renormalize each cell's gene weights before NB sampling, and accumulate only nonzero entries.

- [x] **Step 4: Verify GREEN**

Run: `PYTHONPATH=src python -m pytest tests/test_simulation_core.py -q`
Expected: all tests pass.

### Task 5: Evaluation and Retry Gate

**Files:**
- Create: `src/scrnarefgenerator/evaluation.py`
- Test: `tests/test_evaluation.py`

**Interfaces:**
- Produces: `evaluate_candidate(reference, candidate, config) -> EvaluationResult`
- Produces: `passed: bool`, `metrics: dict`, and `violations: list[str]`

- [x] **Step 1: Write failing metric tests**

```python
def test_identical_reference_sample_is_flagged_as_too_similar(toy_reference, toy_candidate):
    result = evaluate_candidate(toy_reference, toy_candidate, EvaluationConfig())
    assert "mean_spearman" in result.metrics
    assert "library_ks" in result.metrics
    assert 0.0 <= result.metrics["classifier_auc"] <= 1.0


def test_bad_similarity_report_lists_each_violation():
    result = EvaluationResult(
        passed=False,
        metrics={"mean_spearman": 0.1, "library_ks": 0.8},
        violations=["mean_spearman", "library_ks"],
    )
    assert result.violations == ["mean_spearman", "library_ks"]
```

- [x] **Step 2: Verify RED**

Run: `PYTHONPATH=src python -m pytest tests/test_evaluation.py -q`
Expected: FAIL because evaluation is missing.

- [x] **Step 3: Implement matched metrics and gate evaluation**

Use sparse normalized means, sparse detection counts, kNN in a deterministic SVD embedding, and stratified logistic-regression AUC. Record threshold direction for every metric.

- [x] **Step 4: Verify GREEN**

Run: `PYTHONPATH=src python -m pytest tests/test_evaluation.py -q`
Expected: all tests pass.

### Task 6: Assembly, Output, and Public Facade

**Files:**
- Create: `src/scrnarefgenerator/assembly.py`
- Create: `src/scrnarefgenerator/io.py`
- Create: `src/scrnarefgenerator/pipeline.py`
- Create: `src/scrnarefgenerator/api.py`
- Modify: `src/scrnarefgenerator/__init__.py`
- Test: `tests/test_api_contract.py`

- [x] **Step 1: Write failing end-to-end contract tests**

```python
def test_simulate_has_exact_quota_and_reference_gene_order(toy_reference):
    gen = scRNARefGenerator(
        toy_reference,
        GeneratorConfig(seed=5, latent_k=0, strict_evaluation=False),
    )
    result = gen.simulate(
        ConditionSpec(columns=["batch"], joint={("A",): 2, ("B",): 1})
    )
    assert result.n_obs == 3
    assert result.var_names.tolist() == toy_reference.var_names.tolist()
    assert result.obs["batch"].value_counts().to_dict() == {"A": 2, "B": 1}
    assert result.obs_names.str.endswith("_sim").all()
```

- [x] **Step 2: Verify RED**

Run: `PYTHONPATH=src python -m pytest tests/test_api_contract.py -q`
Expected: FAIL because the facade is not implemented.

- [x] **Step 3: Implement assembly, collision policy, JSON-safe manifest, atomic writes, and retries**

Preflight all barcodes before generation. Serialize threshold tuples and category keys into JSON-safe representations. Adjust strength only when strict evaluation is enabled.

- [x] **Step 4: Verify GREEN**

Run: `PYTHONPATH=src python -m pytest tests/test_api_contract.py -q`
Expected: all tests pass.

### Task 7: CLI

**Files:**
- Create: `src/scrnarefgenerator/cli.py`
- Test: `tests/test_cli.py`

- [x] **Step 1: Write a failing subprocess smoke test**

```python
def test_cli_writes_h5ad_and_report(tmp_path, tiny_h5ad):
    out = tmp_path / "sim.h5ad"
    report = tmp_path / "report.json"
    proc = subprocess.run(
        [
            sys.executable, "-m", "scrnarefgenerator.cli", "simulate",
            "--ref", str(tiny_h5ad), "--out", str(out),
            "--condition-json", '{"batch":{"A":2}}',
            "--condition-col", "batch", "--n-cells", "2",
            "--strict-evaluation", "false", "--latent-k", "0",
            "--report", str(report),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")},
    )
    assert proc.returncode == 0, proc.stderr
    assert out.exists()
    assert json.loads(report.read_text())["actual_quota"]
```

- [x] **Step 2: Verify RED**

Run: `PYTHONPATH=src python -m pytest tests/test_cli.py -q`
Expected: FAIL because `python -m scrnarefgenerator.cli` has no implementation.

- [x] **Step 3: Implement argparse parsing and JSON quota normalization**

Support single and repeated `--condition-col`; validate quota keys, seed, output, report, strength, model, and overwrite flags.

- [x] **Step 4: Verify GREEN**

Run: `PYTHONPATH=src python -m pytest tests/test_cli.py -q`
Expected: all tests pass.

### Task 8: Real-Data Integration and Release Verification

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/test_generator_pbmc5k.py`
- Create: `README.md`

- [x] **Step 1: Write the real-data test**

Read `testdata/pbmc5k_querydata.h5ad`, use `celltype` values `CD14` and `NK3`, request one desired cell from each, cap fitting cells, and assert exact quota, sparse integer output, reproducible counts, and a JSON-safe report.

- [x] **Step 2: Verify RED, implement fixture or integration adjustments, then verify GREEN**

Run: `PYTHONPATH=src python -m pytest tests/test_generator_pbmc5k.py -q`
Expected before implementation: missing fixture/import. Expected after implementation: all tests pass.

- [x] **Step 3: Run the full suite**

Run: `PYTHONPATH=src python -m pytest -q`
Expected: zero failures.

- [x] **Step 4: Verify package build and CLI help**

Run: `python -m pip install -e . --no-deps`
Run: `scrnarefgen --help`
Expected: editable installation succeeds and help exits zero.

- [x] **Step 5: Record verification in the final handoff**

Report exact test counts, commands, output files, and any remaining limitations. Do not claim completion without the latest command outputs.
