# scRNARefGenerator Implementation Specification

## Goal

Build a Python package that conditionally simulates sparse raw-count scRNA-seq
data from a reference `AnnData`, while preserving exact condition quotas,
gene order, deterministic random behavior, and a documented quality report.

## Public Interface

The package exports:

```python
from scrnarefgenerator import (
    ConditionSpec,
    EvaluationConfig,
    GeneratorConfig,
    scRNARefGenerator,
)
```

`scRNARefGenerator(ref, config).simulate(condition_spec, n_cells, ...)`
returns an `AnnData` and optionally writes `.h5ad` and JSON outputs.

## Architecture

The implementation separates the reference contract, condition planning,
statistical fitting, sparse count generation, evaluation, and output
assembly. `pipeline.py` performs retries and owns finalization. Statistical
backends do not read or write files.

```text
public API
  -> validation.ReferenceData
  -> conditions.ConditionPlan
  -> distribution.FittedDistribution
  -> simulate.CandidateBatch
  -> evaluation.EvaluationResult
  -> assembly/output
```

## Reference Contract

Counts are resolved from `layers[counts_layer]`, then `raw.X`, then `X`.
The resolved matrix must be finite, nonnegative, integral within `1e-6`, and
is converted to CSR integer storage without densification.

`obs_names` and `var_names` must be unique. The output uses reference gene
names and order exactly. Input files are never modified.

## Condition Planning

`ConditionSpec` accepts either exact joint quotas or per-column marginal
quotas. The two forms are mutually exclusive. All quotas are positive
integers, and every requested category must exist.

For marginal quotas, the planner solves an integer feasibility problem over
observed condition combinations:

```text
0 <= x_g <= reference_count(g)
sum_g x_g = total
sum_{g: g[col] == category} x_g = marginal[col][category]
```

`scipy.optimize.milp` returns an exact plan. Infeasibility raises
`InfeasibleConditionError`; quotas are never silently changed.

Each plan group contains a deterministic source-index pool. Source barcodes
are used at most once globally.

## Statistical Model

The default two-level NB model shrinks each target group's gene probabilities
toward the global probabilities using a library-size-scaled prior. Gene
dispersions use method-of-moments estimates shrunk toward a fitted
mean-dispersion trend.

Each simulated cell starts from a selected source library size and optional
source latent coordinates. Library and latent perturbations are controlled
by `similarity_strength`. After all gene multipliers are applied, the gene
weights are renormalized so their expected total expression equals the target
library size.

Counts are generated from a negative-binomial distribution in gene chunks.
Only nonzero entries are retained in COO storage. A ZINB backend uses a
shrunk per-gene zero probability, and an optional latent backend uses
TruncatedSVD scores and gene loadings.

## Evaluation

Metrics are calculated against a deterministic, size-matched sample of the
reference:

- gene-mean Spearman correlation;
- zero-proportion Spearman correlation;
- library-size KS statistic;
- reference/simulation kNN distance ratio;
- cross-validated classifier AUC;
- high-variance gene mean correlation.

Thresholds are configurable. In strict mode, failed candidates trigger
strength adjustment and deterministic retry seeds. If no candidate passes,
`EvaluationGateError` reports the final result. In non-strict mode, the
least-violating candidate is returned with warnings.

## Output Contract

The output contains raw integer counts in both `X` and `layers["counts"]`.
Required metadata columns are:

- `source_barcode`
- `sim_batch`
- `sim_seed`
- `similarity_strength`
- `distribution_model`
- every requested condition column

The `uns["scRNARefGenerator"]` manifest contains only JSON-serializable
values. Existing output files are not overwritten unless explicitly enabled.

## Error Contract

The package defines the exceptions listed in `design.md` plus
`OutputExistsError`. User-caused failures use these exceptions rather than
generic assertion errors. Retry counts, actual quotas, failed metrics, and
warnings are included in the report.

## Testing Strategy

Toy sparse matrices verify formulas and edge cases without relying on a
large fixture. Integration tests use `testdata/pbmc5k_querydata.h5ad` with
small quotas and the explicit `celltype` column. The fixture adds temporary
metadata copies for multi-column tests.

Contract tests cover reproducibility, exact quotas, gene order, barcode
uniqueness, integer sparse counts, infeasibility, missing counts, collisions,
CLI output, and evaluation structure.
