# Controlled Novelty Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Implement selectable fidelity/balanced/novel generation and source-copy diagnostics.

**Architecture:** Geometry fitting stores a low-dimensional conditional covariance and gene loadings. The sparse sampler applies a latent perturbation before Poisson count sampling. A separate novelty evaluator compares simulated profiles with source and random same-condition reference profiles.

**Tech Stack:** Python 3.10+, NumPy, SciPy, anndata, pytest.

**Spec:** `docs/superpowers/specs/2026-10-30-novelty-control-design.md`

## Global Constraints

- Exact quotas, source barcode uniqueness, gene order, and sparse integer counts remain mandatory.
- `novelty_mode="fidelity"` must preserve the current generation path.
- Reports include the effective novelty strength and copy diagnostics.
- No full dense cell-by-gene output is allowed.
- The workspace has no usable Git repository; tests replace commit verification.

---

### Task 1: Novelty Configuration

**Files:**
- Modify: `src/scrnarefgenerator/config.py`
- Test: `tests/test_public_api.py`

**Interfaces:**
- Produces: `novelty_mode`, `novelty_strength`, `effective_novelty_strength`

- [x] **Step 1: Write failing tests**

```python
def test_balanced_novelty_is_the_default():
    config = GeneratorConfig()
    assert config.novelty_mode == "balanced"
    assert config.effective_novelty_strength == 0.25


def test_fidelity_disables_novelty():
    assert GeneratorConfig(novelty_mode="fidelity").effective_novelty_strength == 0
```

- [x] **Step 2: Verify RED**

Run: `PYTHONPATH=src python -m pytest tests/test_public_api.py -q`
Expected: FAIL because the novelty fields are missing.

- [x] **Step 3: Add validated mode and override fields**

Validate mode names and require an explicit override to be within `[0, 1]`.

- [x] **Step 4: Verify GREEN**

Run: `PYTHONPATH=src python -m pytest tests/test_public_api.py -q`
Expected: PASS.

### Task 2: Geometry Covariance Decoder

**Files:**
- Modify: `src/scrnarefgenerator/geometry.py`
- Test: `tests/test_geometry.py`

**Interfaces:**
- Produces: `GroupGeometry.novelty_factors` with shape `(n_genes, latent_dim)`
- Produces: `perturbation_from_latent(geometry, n_cells, rng, strength)`

- [x] **Step 1: Write failing tests**

```python
def test_geometry_has_gene_space_novelty_factors(toy_group):
    geometry = fit_group_geometry(toy_group, np.arange(18), GeneratorConfig(), seed=3)
    assert geometry.novelty_factors.shape[0] == 6
    perturbation = perturbation_from_latent(geometry, 4, np.random.default_rng(1), 0.2)
    assert perturbation.shape == (4, 6)
```

- [x] **Step 2: Verify RED**

Run: `PYTHONPATH=src python -m pytest tests/test_geometry.py -q`
Expected: FAIL because novelty factors are missing.

- [x] **Step 3: Fit conditional covariance and decoder**

Compute factor loadings from fit-set log-normalized expression and standard PCA scores. Multiply loadings by a Cholesky covariance factor once during fitting.

- [x] **Step 4: Verify GREEN**

Run: `PYTHONPATH=src python -m pytest tests/test_geometry.py -q`
Expected: PASS.

### Task 3: Novelty-Aware Sampling

**Files:**
- Modify: `src/scrnarefgenerator/simulate.py`
- Test: `tests/test_geometry_sampling.py`

**Interfaces:**
- Consumes: effective novelty strength and geometry factors
- Produces: novelty-perturbed local gene probabilities

- [x] **Step 1: Write failing tests**

```python
def test_balanced_novelty_changes_counts_but_fidelity_is_invariant(toy):
    fidelity = sample_candidate(toy.ref, toy.plan, GeneratorConfig(seed=7, novelty_mode="fidelity"))
    repeat = sample_candidate(toy.ref, toy.plan, GeneratorConfig(seed=7, novelty_mode="fidelity"))
    novel = sample_candidate(toy.ref, toy.plan, GeneratorConfig(seed=7, novelty_mode="balanced"))
    assert (fidelity.counts != repeat.counts).nnz == 0
    assert (fidelity.counts != novel.counts).nnz > 0
```

- [x] **Step 2: Verify RED**

Run: `PYTHONPATH=src python -m pytest tests/test_geometry_sampling.py -q`
Expected: FAIL because sampling ignores novelty strength.

- [x] **Step 3: Apply latent perturbation per cell block**

Draw standard Gaussian latent residuals, project through the precomputed gene-space novelty factors, exponentiate with clipping, multiply into local probabilities, and renormalize.

- [x] **Step 4: Verify GREEN**

Run: `PYTHONPATH=src python -m pytest tests/test_geometry_sampling.py tests/test_simulation_core.py -q`
Expected: PASS.

### Task 4: Source-Copy Diagnostics

**Files:**
- Create: `src/scrnarefgenerator/novelty.py`
- Modify: `src/scrnarefgenerator/__init__.py`
- Test: `tests/test_novelty.py`

**Interfaces:**
- Produces: `NoveltyReport`, `evaluate_novelty(...)`

- [x] **Step 1: Write failing tests**

```python
def test_exact_source_copy_is_detected(toy_simulated_copy):
    report = evaluate_novelty(toy_simulated_copy, toy_reference, "celltype")
    assert report.metrics["exact_source_copy_fraction"] > 0
    assert report.metrics["source_jaccard_gain"] > 1
```

- [x] **Step 2: Verify RED**

Run: `PYTHONPATH=src python -m pytest tests/test_novelty.py -q`
Expected: FAIL because `novelty.py` is missing.

- [x] **Step 3: Implement balanced source and random-profile comparisons**

Map `source_barcode` to reference cells, compare sparse nonzeros, and sample random same-condition profiles.

- [x] **Step 4: Verify GREEN**

Run: `PYTHONPATH=src python -m pytest tests/test_novelty.py -q`
Expected: PASS.

### Task 5: Pipeline and Final Validation

**Files:**
- Modify: `scripts/pipeline.py`
- Modify: `src/scrnarefgenerator/pipeline.py`
- Modify: `README.md`
- Test: `tests/test_pipeline_script.py`

- [x] **Step 1: Add CLI contract assertions**

Check `--novelty-mode` and `--novelty-strength`.

- [x] **Step 2: Verify RED**

Run: `PYTHONPATH=src python -m pytest tests/test_pipeline_script.py -q`
Expected: FAIL.

- [x] **Step 3: Integrate modes, reports, and diagnostics**

Record the effective strength in manifest and JSON. Run `evaluate_novelty` against the training reference.

- [x] **Step 4: Run the full suite**

Run: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -p no:cacheprovider -q`
Expected: zero failures.

- [x] **Step 5: Run final PBMC pipeline and inspect figures**

Run default balanced mode and report holdout similarity plus exact-copy and source-Jaccard diagnostics.
