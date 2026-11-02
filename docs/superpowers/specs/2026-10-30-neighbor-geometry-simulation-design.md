# Neighbor-Preserving Simulation Design

## Goal

Improve cell-level geometry similarity between simulated and reference data
without sacrificing the exact condition quotas or sparse raw-count contract.

The current model preserves group-level gene means but gives all cells in a
condition nearly the same gene probabilities. The new default geometry model
uses a condition-specific latent neighborhood and blends each source cell with
its local reference manifold before negative-binomial sampling.

## Public Configuration

`GeneratorConfig` gains:

```text
geometry_model: "none" | "neighbor_preserving" = "neighbor_preserving"
geometry_neighbors: int = 30
local_profile_mix: float = 0.5
group_prior_mix: float = 0.1
geometry_latent_k: int = 20
geometry_hvg_count: int = 2000
geometry_eval_holdout_fraction: float = 0.2
count_noise_model: "poisson" | "nb" = "poisson"
```

Constraints:

- `geometry_neighbors >= 1`
- `0 <= local_profile_mix <= 1`
- `0 <= group_prior_mix <= 1`
- `geometry_latent_k >= 2`
- `geometry_hvg_count >= 2`
- `0 < geometry_eval_holdout_fraction < 1`

`geometry_model="none"` retains the previous group-probability sampler.

## Model

For each condition group `g`, fit a group-specific PCA/TruncatedSVD from
library-size-normalized and `log1p`-transformed reference counts. Build a
within-group nearest-neighbor graph in the latent space.

For each simulated cell:

```text
i = selected reference source
N_i = k nearest reference sources in the same condition
w_il = inverse-distance neighborhood weights

q_source = X_i / library_i
q_neighbor = sum_l w_il X_l / library_l
q_local = (1 - local_profile_mix) q_source + local_profile_mix q_neighbor
q = normalize((1 - group_prior_mix) q_local + group_prior_mix p_g)

library_sim = library_i * exp(N(0, sigma_lib * (1 - similarity_strength)))
mu_ij = library_sim q_j
X_sim_ij ~ Poisson(mu_ij)                   # default residual noise
# or X_sim_ij ~ NB(mu_ij, theta_gj)         # optional count_noise_model="nb"
```

ZINB applies the same zero-probability layer after NB sampling.

The local profiles already encode cell-to-cell biological variability.
Applying the full marginal dispersion estimated from raw counts again causes
double overdispersion, excessive zero inflation, and library-size shrinkage.
The default geometry model therefore uses Poisson residual sampling; callers
can opt back into NB residuals with `count_noise_model="nb"`.

Neighborhood profiles are constructed in bounded cell blocks; full dense
output matrices are never materialized. Gene order and exact quotas remain
unchanged.

## Holdout Evaluation

The pipeline performs a deterministic stratified split by condition before
generation:

- training cells: used to fit the distribution and supply source barcodes;
- holdout cells: never used for fitting or source selection;
- request quotas remain based on full reference category counts;
- simulation is evaluated against holdout cells;
- training-versus-holdout metrics are reported as the natural reference
  baseline.

The splitter guarantees at least two training and two holdout cells for each
category whenever the category has at least four cells. Categories that do
not meet this requirement cause a clear error.

## Evaluation Output

The pipeline JSON report includes:

```text
geometry_model and its parameters
count_noise_model
reference_split sizes
simulated-versus-holdout metrics
training-versus-holdout baseline metrics
per-condition violations
```

Existing public `evaluate_similarity()` behavior remains available for callers
that already have separate reference and simulation objects.

## Testing

Toy data with structured cell states verifies:

- neighbor model is the default and is selected by configuration;
- source and neighbor profile mixing changes cell-specific expression;
- neighbor sampling remains reproducible;
- sparse integer counts and exact quotas are preserved;
- geometry-model `none` retains the old path;
- stratified split has disjoint indices and complete condition coverage;
- public evaluation accepts held-out reference data;
- real PBMC pipeline produces reports and figures with holdout metadata.

## Validated Result

On the fixed PBMC reference with the 10 largest celltypes and a 60% quota:

```text
simulated cells:       30,009
mean Spearman:          0.945
zero Spearman:          0.946
library-size KS:        0.061
kNN distance ratio:     0.942
classifier AUC:         0.653
high-variable Spearman: 0.994
```

The overall gate and all 10 per-celltype gates passed.
