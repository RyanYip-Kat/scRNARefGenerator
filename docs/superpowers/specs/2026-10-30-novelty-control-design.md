# Controlled Novelty Design

## Goal

Add explicit control over the fidelity-novelty tradeoff without changing exact
condition quotas, source-barcode uniqueness, or the raw-count contract.

## Modes

`GeneratorConfig.novelty_mode` accepts:

- `fidelity`: no latent novelty perturbation, equivalent to the current model;
- `balanced`: default, latent perturbation inside each condition's fitted
  covariance;
- `novel`: stronger perturbation for exploratory generation.

`GeneratorConfig.novelty_strength: float | None` overrides the mode preset.

Preset strengths:

```text
fidelity = 0.00
balanced = 0.25
novel    = 0.50
```

The effective strength is recorded in reports and AnnData `obs`.

## Geometry Perturbation

For each condition group:

```text
Y = log1p(library-normalized counts)
PCA scores Z = Y_hvg W
conditional covariance Sigma = cov(Z)
latent factor loadings B = Y.T Z / n

z_novel = L Sigma_epsilon, covariance(L L.T) = Sigma
gene perturbation = z_novel B.T
q_final = normalize(q_local * exp(clip(novelty_strength * perturbation)))
```

Perturbing in latent covariance space creates profiles that are not restricted
to convex combinations of source and neighbor profiles. Poisson residual
sampling preserves the target library scale.

## Novelty Diagnostics

Add public `evaluate_novelty(simulated, source_reference, condition_col)` that
samples balanced cells and reports:

- exact source-count copy fraction;
- median source Jaccard similarity;
- median random same-condition Jaccard similarity;
- source Jaccard gain;
- median simulated/source library ratio.

The pipeline includes these metrics alongside holdout similarity metrics and
reports them in `similarity_evaluation.json`.

## Acceptance

- `fidelity` remains reproducible and selectable.
- A nonzero novelty mode changes candidate counts under the same seed.
- `balanced` remains sparse, integer, exact-quota, and reproducible.
- Copy diagnostics detect a source-copied toy cell and report gain greater
  than one for source-dependent data.
- Existing full test suite remains green.
- The final PBMC run reports both similarity and source-dependence metrics.

## Validated Balanced Run

With `novelty_mode="balanced"` (`novelty_strength=0.25`):

```text
overall classifier AUC:       0.629
overall kNN ratio:            1.043
overall library-size KS:      0.063
overall high-variable corr:   0.990
exact source copy fraction:   0.000
source Jaccard gain:          1.840
source library ratio median:  0.999
```

The overall distribution gate passes. Eight of ten celltype gates pass;
Memory BC and Naive BC remain more distinguishable than the default upper AUC
threshold, indicating that those subsets are not over-copied.
