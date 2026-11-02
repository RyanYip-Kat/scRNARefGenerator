# Neighbor-Preserving Simulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add a local-neighborhood geometry model and holdout-based evaluation to the simulator.

**Architecture:** A new geometry module fits per-condition latent neighborhoods. Distribution fitting stores geometry parameters, and the sparse sampler combines local profiles before vectorized NB/ZINB sampling. The pipeline splits reference data into fit and holdout partitions before generation.

**Tech Stack:** Python 3.10+, NumPy, SciPy, scikit-learn, anndata, pytest.

**Spec:** `docs/superpowers/specs/2026-10-30-neighbor-geometry-simulation-design.md`

## Global Constraints

- Exact condition quotas and raw integer sparse counts remain mandatory.
- Reference gene names and order remain unchanged.
- Randomness uses attempt-specific `numpy.random.Generator` streams.
- Full dense cell-by-gene matrices are forbidden.
- The previous geometry model remains selectable through `geometry_model="none"`.
- The workspace has no usable Git repository, so verification replaces commit steps.

---

### Task 1: Geometry Configuration

**Files:**
- Modify: `src/scrnarefgenerator/config.py`
- Test: `tests/test_public_api.py`

**Interfaces:**
- Produces: `GeneratorConfig.geometry_model`
- Produces: `GeneratorConfig.geometry_neighbors`
- Produces: `GeneratorConfig.local_profile_mix`
- Produces: `GeneratorConfig.group_prior_mix`
- Produces: `GeneratorConfig.geometry_latent_k`
- Produces: `GeneratorConfig.geometry_hvg_count`

- [x] **Step 1: Write failing validation tests**

```python
def test_neighbor_geometry_is_the_default():
    config = GeneratorConfig()
    assert config.geometry_model == "neighbor_preserving"
    assert config.geometry_neighbors == 30


def test_invalid_local_profile_mix_is_rejected():
    with pytest.raises(ValueError):
        GeneratorConfig(local_profile_mix=1.1)
```

- [x] **Step 2: Verify RED**

Run: `PYTHONPATH=src python -m pytest tests/test_public_api.py -q`
Expected: FAIL because the geometry fields are missing.

- [x] **Step 3: Add validated configuration fields**

Use the exact defaults and ranges in the design specification.

- [x] **Step 4: Verify GREEN**

Run: `PYTHONPATH=src python -m pytest tests/test_public_api.py -q`
Expected: all tests pass.

### Task 2: Stratified Reference Split

**Files:**
- Create: `src/scrnarefgenerator/splitting.py`
- Modify: `src/scrnarefgenerator/__init__.py`
- Test: `tests/test_splitting.py`

**Interfaces:**
- Produces: `ReferenceSplit(training, holdout, training_indices, holdout_indices)`
- Produces: `stratified_reference_split(reference, condition_col, fraction=0.2, seed=42)`

- [x] **Step 1: Write failing split tests**

```python
def test_split_is_disjoint_and_condition_stratified(toy_reference):
    split = stratified_reference_split(
        toy_reference, "celltype", fraction=0.25, seed=7
    )
    assert set(split.training_indices) & set(split.holdout_indices) == set()
    assert split.training.n_obs + split.holdout.n_obs == toy_reference.n_obs
    assert set(split.holdout.obs["celltype"]) == set(toy_reference.obs["celltype"])
```

- [x] **Step 2: Verify RED**

Run: `PYTHONPATH=src python -m pytest tests/test_splitting.py -q`
Expected: FAIL because `splitting.py` is missing.

- [x] **Step 3: Implement deterministic per-condition splitting**

Draw and sort indices from a seeded generator. Preserve condition membership and validate minimum partition sizes.

- [x] **Step 4: Verify GREEN**

Run: `PYTHONPATH=src python -m pytest tests/test_splitting.py -q`
Expected: all tests pass.

### Task 3: Latent Neighborhood Fit

**Files:**
- Create: `src/scrnarefgenerator/geometry.py`
- Modify: `src/scrnarefgenerator/distribution.py`
- Test: `tests/test_geometry.py`

**Interfaces:**
- Produces: `GroupGeometry(source_indices, scores, fit_indices, neighbor_model)`
- Produces: `fit_group_geometry(X, source_indices, config, seed) -> GroupGeometry`
- Produces: `neighbor_profiles(geometry, selected_positions, group_X, rng) -> (rows, cols, values, neighbor_distances)`

- [x] **Step 1: Write failing geometry tests**

```python
def test_neighbor_geometry_returns_local_neighbors(toy_group):
    geometry = fit_group_geometry(
        toy_group, GeneratorConfig(geometry_neighbors=3)
    )
    neighbors, distances = query_neighbors(geometry, [0, 1], seed=5)
    assert neighbors.shape == (2, 3)
    assert np.all(distances[:, 0] >= 0)
    assert np.all(neighbors != np.array([[0], [1]]))
```

- [x] **Step 2: Verify RED**

Run: `PYTHONPATH=src python -m pytest tests/test_geometry.py -q`
Expected: FAIL because the geometry module is missing.

- [x] **Step 3: Implement group PCA and nearest-neighbor lookup**

Use normalized log counts, top variance genes, TruncatedSVD, standardized scores, and within-group `NearestNeighbors`.

- [x] **Step 4: Verify GREEN**

Run: `PYTHONPATH=src python -m pytest tests/test_geometry.py -q`
Expected: all tests pass.

### Task 4: Local Profile Sampling

**Files:**
- Modify: `src/scrnarefgenerator/simulate.py`
- Test: `tests/test_geometry_sampling.py`

**Interfaces:**
- Consumes: fitted group geometry
- Produces: sparse counts whose condition means combine source, neighbors, and group prior
- Produces: configurable Poisson residual noise, default `count_noise_model="poisson"`

- [x] **Step 1: Write failing geometry-sampling tests**

```python
def test_neighbor_geometry_changes_cell_specific_profiles(toy_reference, toy_plan):
    config = GeneratorConfig(
        seed=11,
        geometry_model="neighbor_preserving",
        local_profile_mix=0.5,
        group_prior_mix=0.1,
    )
    generated = sample_candidate(toy_reference, toy_plan, config)
    assert generated.counts.ndim == 2
    assert generated.counts.dtype.kind in "iu"
    assert (generated.counts != toy_reference.X[np.ix_(toy_plan.groups[0].source_indices[:3], np.arange(toy_reference.n_vars))]).nnz > 0


def test_neighbor_sampling_is_reproducible(toy_reference, toy_plan):
    config = GeneratorConfig(seed=8, geometry_model="neighbor_preserving")
    first = sample_candidate(toy_reference, toy_plan, config)
    second = sample_candidate(toy_reference, toy_plan, config)
    assert (first.counts != second.counts).nnz == 0
```

- [x] **Step 2: Verify RED**

Run: `PYTHONPATH=src python -m pytest tests/test_geometry_sampling.py -q`
Expected: FAIL because the sampler ignores geometry.

- [x] **Step 3: Implement blocked local-profile blending and NB sampling**

Construct source and neighbor weights with sparse selectors, combine with group probabilities, renormalize, and sample in bounded cell blocks.

- [x] **Step 4: Verify GREEN**

Run: `PYTHONPATH=src python -m pytest tests/test_geometry_sampling.py tests/test_simulation_core.py -q`
Expected: all tests pass.

### Task 5: Holdout Pipeline and Reporting

**Files:**
- Modify: `scripts/pipeline.py`
- Modify: `README.md`
- Test: `tests/test_pipeline_script.py`

**Interfaces:**
- Produces CLI options `--holdout-fraction` and geometry parameters
- Produces holdout baseline metrics in `similarity_evaluation.json`

- [x] **Step 1: Write failing CLI test**

```python
def test_pipeline_exposes_geometry_and_holdout_options():
    output = subprocess.run(
        [sys.executable, "scripts/pipeline.py", "--help"],
        capture_output=True,
        text=True,
    ).stdout
    assert "--holdout-fraction" in output
    assert "--geometry-neighbors" in output
    assert "--local-profile-mix" in output
```

- [x] **Step 2: Verify RED**

Run: `PYTHONPATH=src python -m pytest tests/test_pipeline_script.py -q`
Expected: FAIL because the options are absent.

- [x] **Step 3: Split reference, generate from training data, and evaluate holdout**

Keep quotas based on full reference counts. Record training/holdout sizes, geometry parameters, simulation metrics, and reference baseline metrics.

- [x] **Step 4: Verify GREEN**

Run: `PYTHONPATH=src python -m pytest tests/test_pipeline_script.py -q`
Expected: all tests pass.

### Task 6: Real-Data Regression and Figure Review

**Files:**
- Modify: `tests/test_generator_pbmc5k.py`
- Output: `outputs/pipeline/`

- [x] **Step 1: Add a real-data geometry invariant**

Read the PBMC reference, select two categories, generate with neighbor geometry, and assert sparse integer output, exact quotas, and successful holdout evaluation serialization.

- [x] **Step 2: Run the full suite**

Run: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -p no:cacheprovider -q`
Expected: zero failures.

- [x] **Step 3: Run the PBMC pipeline**

Run: `python scripts/pipeline.py --outdir outputs/pipeline --top-n 10 --fraction 0.60 --formats png pdf svg --overwrite`
Expected: 30,009 simulated cells and updated UMAP/library figures.

- [x] **Step 4: Perform visual QA**

Inspect PNG previews for missing text, clipping, legend overlap, and comparable axes. Iterate on plotting code until clean.

- [x] **Step 5: Record metrics**

Compare neighbor geometry against the previous AUC `0.99507`, kNN ratio `3.28554`, and high-variable Spearman `0.84437`. Report improvements and remaining failures.
