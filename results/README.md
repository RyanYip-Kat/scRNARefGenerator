# Experiment Results

This directory contains the lightweight, GitHub-friendly evaluation artifacts
for `scRNARefGenerator`. Large simulated `.h5ad` files are intentionally not
committed and can be regenerated with:

```bash
FORCE=1 bash scripts/run_experiments.sh
bash scripts/evaluate_experiments_biological.sh
```

## Summary tables

- [`biological_summary_all.csv`](biological_summary_all.csv): all 16
  experiment configurations with similarity, marker, identity-transfer,
  nearest-neighbor, heterogeneity, and pass/fail metrics.
- [`biological_summary_all.json`](biological_summary_all.json): JSON version
  of the same results.
- [`experiment_configurations.csv`](experiment_configurations.csv): experiment
  grid, generated cell counts, condition columns, and output paths.

## Representative figures

### Main conditional pipeline

![Shared UMAP](figures/main/umap_comparison.png)

![Library-size ECDF](figures/main/library_size_ecdf.png)

### Celltype subset experiment

![Experiment 1 shared UMAP](figures/experiment1/umap_comparison.png)

![Experiment 1 biological PCA](figures/experiment1/pca_real_vs_sim.png)

![Experiment 1 marker AUROC](figures/experiment1/marker_auroc.png)

### Celltype × ident experiment

![Experiment 2 shared UMAP](figures/experiment2/umap_comparison.png)

![Experiment 2 biological PCA](figures/experiment2/pca_real_vs_sim.png)

### Unconditional experiment

![Experiment 3 UMAP, 1k cells](figures/experiment3/umap_n1000.png)

![Experiment 3 UMAP, 20k cells](figures/experiment3/umap_n20000.png)

![Experiment 3 biological PCA](figures/experiment3/pca_real_vs_sim.png)

Full interpretation is in [`../evaluation.md`](../evaluation.md).
