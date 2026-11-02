# Experiment Suite Design

## Goal

Run the requested conditionality, similarity, and sample-size experiments
through `scripts/run_experiments.sh`, retaining each generated h5ad and
embedding comparison.

## Experiments

### Experiment 1

- condition: `celltype`
- category subsets: 5, 8, 10, 12, 15, all
- quota: floor(60% of each selected reference celltype)
- `similarity_strength=0.70`

### Experiment 2

- conditions: `celltype`, `ident`
- all categories in both columns
- each column uses floor(60% of every reference category)
- exact marginal allocation uses the existing capacity/proportional MILP
- `similarity_strength=0.50, 0.60, 0.70, 0.80, 0.90`

### Experiment 3

- unrestricted simulation
- `n_cells=1000, 3000, 5000, 10000, 20000`
- `similarity_strength=0.70`

## Embeddings

Every generated h5ad receives independently recomputed:

- `obsm["X_pca"]`
- `obsm["X_umap"]`

The experiment output also includes:

- reference existing UMAP from the reference `obsm["X_umap"]`;
- generated-only UMAP;
- jointly fitted reference/generated UMAP for direct comparison.

Shared coordinates are used as the primary distribution comparison. Independently
fitted UMAP coordinates are retained as diagnostics but are not numerically
compared.

## Outputs

```text
outputs/experiments/
  experiment1/subset_5/
    simulated.h5ad
    evaluation.json
    reference_existing_umap.png
    generated_umap.png
    shared_umap_comparison.png
  experiment1/subset_all/
  experiment2/strength_0.5/
  experiment3/n_cells_1000/
```

## APIs

Add:

```python
condition_spec_from_marginal_fraction(reference, columns, fraction=0.60)
compute_pca_umap(counts, ...)
```

`plot_similarity_umap()` gains an optional `simulated_condition_col` so
unconditional simulations can be colored by inherited `source_celltype`.

## Smoke Verification

All three modes completed with reduced settings:

- Experiment 1: five celltypes, 31 generated cells;
- Experiment 2: all celltypes and idents, 586 generated cells;
- Experiment 3: 50 unconditional cells.

Each smoke output contained a writable h5ad with `X_pca` and `X_umap`, the
three requested UMAP outputs, and an evaluation summary. The full requested
grid is defined in `scripts/run_experiments.sh`; it is intentionally resumable
unless `FORCE=1` is set.
