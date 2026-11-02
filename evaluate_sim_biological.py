#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评价模拟 scRNA-seq 数据是否保留可验证的生物学结构。

修正版默认：
- 从 source_barcode 重建未参与模拟的真实 holdout；
- 在 training/holdout/sim 中按 celltype 平衡抽样；
- marker 从 training reference 学习，在 holdout 和模拟中独立检验；
- 条件实验不让人为 celltype 配额影响生物身份门禁。

核心输出：
- summary.json / summary.txt：关键指标
- celltype_proportions.csv：celltype 比例
- marker_metrics_real.csv / marker_metrics_sim.csv / marker_metrics_sim_shuffled.csv
- classification_confusion_real_to_sim.csv
- heterogeneity.csv
- pca_real_vs_sim_by_celltype.png
- celltype_proportions.png
- marker_auroc_sim.png

依赖：
pip install scanpy anndata scikit-learn scipy pandas numpy matplotlib
"""

import os
import json
import argparse
import warnings
from pathlib import Path

# Numba/Matplotlib need writable cache directories in restricted HPC setups.
os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "scrnarefgenerator-numba"),
)
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "scrnarefgenerator-matplotlib"),
)
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import anndata as ad
import h5py
from scipy import sparse
from scipy.stats import ks_2samp, spearmanr
from scipy.spatial.distance import jensenshannon

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedKFold, cross_val_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

sc = None


def load_scanpy():
    global sc
    if sc is None:
        import scanpy as scanpy_module

        sc = scanpy_module
    return sc


def is_integer_count_matrix(X, tolerance=1e-6):
    """Return True for finite nonnegative integer-valued raw counts."""
    data = X.data if sparse.issparse(X) else np.asarray(X).ravel()
    if data.size == 0:
        return True
    values = np.asarray(data, dtype=float)
    return bool(
        np.isfinite(values).all()
        and (values >= 0).all()
        and np.allclose(values, np.rint(values), atol=tolerance, rtol=0)
    )


def infer_simulation_celltype_key(adata, requested_key="celltype"):
    """Use the generated celltype column, or inherited source_celltype."""
    if requested_key in adata.obs:
        return requested_key
    source_key = f"source_{requested_key}"
    if source_key in adata.obs:
        return source_key
    raise ValueError(
        f"simulated data has neither obs[{requested_key!r}] nor obs[{source_key!r}]"
    )


def exclude_source_cells(real, simulated):
    """Return real cells not used as simulation source barcodes."""
    if "source_barcode" not in simulated.obs:
        return real.copy()
    source_names = set(simulated.obs["source_barcode"].astype(str))
    mask = ~real.obs_names.astype(str).isin(source_names)
    if int(mask.sum()) == 0:
        raise ValueError("no real cells remain after excluding simulation sources")
    return real[mask].copy()


def stratified_subsample_indices(labels, maximum_per_class, seed=42):
    """Sample up to a fixed number of cells from every nonempty class."""
    if maximum_per_class <= 0:
        raise ValueError("maximum_per_class must be positive")
    labels = np.asarray(labels).astype(str)
    rng = np.random.default_rng(seed)
    selected = []
    for category in sorted(set(labels), key=repr):
        indices = np.flatnonzero(labels == category)
        count = min(maximum_per_class, indices.size)
        selected.append(rng.choice(indices, size=count, replace=False))
    return np.sort(np.concatenate(selected))


def stratified_anndata_sample(adata, key, maximum_per_class, seed=42):
    labels = adata.obs[key].astype(str).to_numpy()
    indices = stratified_subsample_indices(labels, maximum_per_class, seed)
    return adata[indices].copy()


def read_sparse_row_subset(path, row_indices, layer="counts"):
    """Read selected rows directly from a dense or CSR HDF5 matrix."""
    return read_sparse_row_subsets(path, [row_indices], layer=layer)[0]


def read_sparse_row_subsets(path, row_subsets, layer="counts"):
    """Read several row subsets after one sequential scatter read."""
    with h5py.File(path, "r") as handle:
        layer_path = f"layers/{layer}" if layer is not None else None
        if layer_path is not None and layer_path in handle:
            node = handle[layer_path]
        elif layer is not None and layer in handle:
            node = handle[layer]
        else:
            node = handle["X"]
        encoding = node.attrs.get("encoding-type", "")
        if isinstance(encoding, bytes):
            encoding = encoding.decode()
        if isinstance(node, h5py.Group) and encoding == "csr_matrix":
            shape = tuple(int(value) for value in node.attrs["shape"])
            indptr = np.asarray(node["indptr"], dtype=np.int64)
            all_data = node["data"][:]
            all_indices = node["indices"][:]
            results = []
            for row_indices in row_subsets:
                rows = np.asarray(row_indices, dtype=np.int64)
                order = np.argsort(rows)
                sorted_rows = rows[order]
                starts = indptr[sorted_rows]
                ends = indptr[sorted_rows + 1]
                lengths = ends - starts
                positions = np.concatenate(
                    [np.arange(start, end) for start, end in zip(starts, ends)]
                ) if lengths.sum() else np.empty(0, dtype=np.int64)
                new_indptr = np.concatenate(
                    [[0], np.cumsum(lengths)]
                ).astype(np.int64)
                matrix = sparse.csr_matrix(
                    (
                        all_data[positions],
                        all_indices[positions],
                        new_indptr,
                    ),
                    shape=(len(sorted_rows), shape[1]),
                )
                inverse = np.argsort(order)
                results.append(matrix[inverse])
            return results
        elif isinstance(node, h5py.Dataset):
            all_data = node[:]
            return [
                sparse.csr_matrix(all_data[np.asarray(row_indices, dtype=np.int64)])
                for row_indices in row_subsets
            ]
        else:
            raise ValueError(
                f"unsupported HDF5 matrix encoding in {path}: {encoding!r}"
            )


def to_dense(X):
    if sparse.issparse(X):
        return X.toarray().astype(np.float32)
    return np.asarray(X, dtype=np.float32)


def maybe_normalize(adata, name, mode="auto"):
    """自动/强制 normalize_total + log1p。"""
    sc = load_scanpy()
    if mode == "never":
        print(f"[{name}] skip normalization")
        return
    if mode == "always":
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        print(f"[{name}] forced normalize_total + log1p")
        return

    if adata.X.size == 0:
        print(f"[{name}] empty X, skip normalization")
        return
    if is_integer_count_matrix(adata.X):
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        print(f"[{name}] auto: detected counts, normalized/log1p")
    else:
        print(f"[{name}] auto: assume already log-normalized, skip")


def cell_qc_stats(adata):
    X = adata.X
    if sparse.issparse(X):
        n_genes = np.asarray(X.getnnz(axis=1)).ravel()
        total = np.asarray(X.sum(axis=1)).ravel()
    else:
        X = np.asarray(X)
        n_genes = (X > 0).sum(axis=1)
        total = X.sum(axis=1)
    return n_genes, total


def gene_stats(adata):
    """返回共同基因上的均值、方差、零比例。"""
    X = adata.X
    if sparse.issparse(X):
        mean = np.asarray(X.mean(axis=0)).ravel()
        mean_sq = np.asarray(X.multiply(X).mean(axis=0)).ravel()
        zero = 1.0 - np.asarray(X.getnnz(axis=0)).ravel() / X.shape[0]
    else:
        X = np.asarray(X)
        mean = X.mean(axis=0)
        mean_sq = (X ** 2).mean(axis=0)
        zero = 1.0 - (X > 0).mean(axis=0)
    var = np.maximum(mean_sq - mean ** 2, 0.0)
    return mean, var, zero


def get_top_markers(adata, ct_key, n=30):
    """用真实数据 Wilcoxon 找每个 celltype 的 top markers。"""
    sc = load_scanpy()
    ad = adata.copy()
    ad.obs[ct_key] = ad.obs[ct_key].astype("category")
    sc.tl.rank_genes_groups(
        ad,
        groupby=ct_key,
        method="wilcoxon",
        n_genes=min(n, ad.n_vars),
        use_raw=False,
    )
    res = ad.uns["rank_genes_groups"]
    markers = {}
    for ct in ad.obs[ct_key].cat.categories:
        names = list(res["names"][ct])[:n]
        markers[str(ct)] = [g for g in names if g in ad.var_names]
    return markers


def compute_marker_metrics(adata, ct_key, markers, classes):
    """
    对每个 celltype：
    - 用真实数据定义的 top markers
    - 在给定 adata 中计算每个 marker 区分 target vs rest 的 AUROC
    - 平均 AUROC 应 >0.5，direction_rate 是 AUROC>0.5 的比例
    """
    rows = []
    y_all = adata.obs[ct_key].astype(str).values

    for ct in classes:
        genes = markers.get(ct, [])
        if len(genes) == 0:
            continue

        y = (y_all == ct).astype(int)
        if y.sum() < 2 or y.sum() == len(y):
            rows.append({
                "celltype": ct,
                "n_cells": int(len(y)),
                "n_target": int(y.sum()),
                "n_markers": len(genes),
                "auc_mean": np.nan,
                "direction_rate": np.nan,
            })
            continue

        X = adata[:, genes].X
        X = X.toarray() if sparse.issparse(X) else np.asarray(X)

        aucs = []
        for j in range(X.shape[1]):
            x = X[:, j]
            if np.all(x == x[0]):
                aucs.append(0.5)
            else:
                try:
                    aucs.append(roc_auc_score(y, x))
                except Exception:
                    aucs.append(0.5)

        aucs = np.asarray(aucs)
        rows.append({
            "celltype": ct,
            "n_cells": int(len(y)),
            "n_target": int(y.sum()),
            "n_markers": len(genes),
            "auc_mean": float(np.mean(aucs)),
            "direction_rate": float(np.mean(aucs > 0.5)),
        })

    return pd.DataFrame(rows)


def classification_metrics(Z_train, y_train, Z_hold, y_hold, Z_sim, y_sim, classes):
    """Train on real sources, then compare holdout and simulated transfer."""
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )
    clf.fit(Z_train, y_train)
    pred_hold = clf.predict(Z_hold)
    pred_sim = clf.predict(Z_sim)

    ba_hold = balanced_accuracy_score(y_hold, pred_hold)
    f1_hold = f1_score(y_hold, pred_hold, average="macro", zero_division=0)
    ba_sim = balanced_accuracy_score(y_sim, pred_sim)
    f1_sim = f1_score(y_sim, pred_sim, average="macro", zero_division=0)
    cm = confusion_matrix(y_sim, pred_sim, labels=classes)

    rng = np.random.RandomState(1)
    y_sim_shuf = rng.permutation(y_sim)
    ba_shuf = balanced_accuracy_score(y_sim_shuf, pred_sim)
    f1_shuf = f1_score(y_sim_shuf, pred_sim, average="macro", zero_division=0)

    return {
        "real_holdout_balanced_acc": float(ba_hold),
        "real_holdout_macro_f1": float(f1_hold),
        "sim_transfer_balanced_acc": float(ba_sim),
        "sim_transfer_macro_f1": float(f1_sim),
        "sim_transfer_retention_vs_holdout": float(ba_sim - ba_hold),
        "shuffled_sim_balanced_acc": float(ba_shuf),
        "shuffled_sim_macro_f1": float(f1_shuf),
    }, cm


def read_experiment_metadata(simulated_path):
    path = Path(simulated_path).resolve().parent / "experiment_summary.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def build_checks(summary, conditional_celltype):
    checks = {
        "gene_mean_similarity": summary["gene_mean_spearman"] >= 0.80,
        "gene_zero_similarity": summary["gene_zero_frac_spearman"] >= 0.70,
        "gene_variance_similarity": summary["gene_log_var_spearman"] >= 0.70,
        "marker_direction": summary["marker_direction_rate_macro_sim"] >= 0.70,
        "marker_signal_vs_shuffle": (
            summary["marker_auroc_macro_sim"]
            - summary["marker_auroc_macro_sim_shuffled"]
            >= 0.05
        ),
        "marker_retention_vs_holdout": (
            summary["marker_auroc_macro_sim"]
            >= summary["marker_auroc_macro_real"] - 0.15
        ),
        "identity_transfer": (
            summary["sim_transfer_balanced_acc"]
            >= max(0.60, summary["real_holdout_balanced_acc"] - 0.20)
        ),
        "nearest_celltype_identity": (
            summary["nearest_real_same_celltype_fraction"]
            >= max(
                0.50,
                summary["nearest_real_holdout_same_celltype_fraction"] - 0.20,
            )
        ),
        "nn_distance_not_extreme": (
            0.50
            <= summary["nn_distance_ratio_sim_over_real"]
            <= 2.00
        ),
        "heterogeneity_not_extreme": (
            0.50 <= summary["heterogeneity_ratio_median"] <= 2.00
        ),
    }
    if not conditional_celltype:
        checks["celltype_proportion_js"] = (
            summary["celltype_js_distance"] <= 0.35
        )
    return {name: bool(value) for name, value in checks.items()}


def main():
    sc = load_scanpy()
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", default="testdata/pbmc5k_querydata.h5ad")
    parser.add_argument(
        "--sim",
        default="outputs/experiments/experiment1/subset_12/simulated.h5ad",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/experiments/experiment1/subset_12/eval_bio_v2",
    )
    parser.add_argument("--celltype-key", default="celltype")
    parser.add_argument("--n-top-markers", type=int, default=30)
    parser.add_argument("--n-hvg", type=int, default=2000)
    parser.add_argument("--max-cells-per-class", type=int, default=500)
    parser.add_argument("--min-cells-per-class", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--normalize",
        choices=["auto", "always", "never"],
        default="auto",
    )
    args = parser.parse_args()
    if args.max_cells_per_class <= 0:
        raise ValueError("--max-cells-per-class must be positive")
    if args.min_cells_per_class < 2:
        raise ValueError("--min-cells-per-class must be at least 2")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    experiment = read_experiment_metadata(args.sim)
    condition_columns = experiment.get("condition_columns") or []

    print("读取 metadata ...")
    real_backed = ad.read_h5ad(args.real, backed="r")
    sim_backed = ad.read_h5ad(args.sim, backed="r")
    if args.celltype_key not in real_backed.obs:
        raise ValueError(f"真实数据缺少 obs['{args.celltype_key}']")
    sim_ct_key = infer_simulation_celltype_key(sim_backed, args.celltype_key)

    sim_gene_set = set(sim_backed.var_names)
    common_genes = pd.Index(
        [g for g in real_backed.var_names if g in sim_gene_set]
    )
    if len(common_genes) == 0:
        raise ValueError("真实和模拟数据没有共同基因")

    real_obs = real_backed.obs.copy()
    sim_obs = sim_backed.obs.copy()
    real_obs["_eval_celltype"] = real_obs[args.celltype_key].astype(str)
    sim_obs["_eval_celltype"] = sim_obs[sim_ct_key].astype(str)
    source_barcodes = (
        set(sim_obs["source_barcode"].astype(str))
        if "source_barcode" in sim_obs
        else set()
    )
    if source_barcodes:
        training_mask = real_obs.index.astype(str).isin(source_barcodes)
        holdout_mask = ~training_mask
        holdout_mode = "excluded_source_barcodes"
    else:
        training_mask = np.ones(len(real_obs), dtype=bool)
        holdout_mask = np.ones(len(real_obs), dtype=bool)
        holdout_mode = "no_source_barcodes_full_reference"

    candidate_classes = sorted(
        set(real_obs.loc[training_mask, "_eval_celltype"])
        & set(real_obs.loc[holdout_mask, "_eval_celltype"])
        & set(sim_obs["_eval_celltype"]),
        key=repr,
    )
    classes = []
    for celltype in candidate_classes:
        counts = (
            int(
                (
                    training_mask
                    & real_obs["_eval_celltype"].eq(celltype).to_numpy()
                ).sum()
            ),
            int(
                (
                    holdout_mask
                    & real_obs["_eval_celltype"].eq(celltype).to_numpy()
                ).sum()
            ),
            int(sim_obs["_eval_celltype"].eq(celltype).sum()),
        )
        if min(counts) >= args.min_cells_per_class:
            classes.append(celltype)
    excluded_classes = sorted(set(candidate_classes) - set(classes), key=repr)
    if not classes:
        raise ValueError(
            "no celltype reaches --min-cells-per-class in all three datasets"
        )
    print(f"共同 celltype={len(classes)}, 开始按类抽样 ...", flush=True)

    def selected_indices(mask, labels, seed):
        candidates = np.flatnonzero(mask)
        candidate_labels = labels[candidates]
        selected = stratified_subsample_indices(
            candidate_labels,
            args.max_cells_per_class,
            seed,
        )
        return candidates[selected]

    training_labels = real_obs["_eval_celltype"].to_numpy()
    sim_labels = sim_obs["_eval_celltype"].to_numpy()
    training_indices = selected_indices(
        training_mask & real_obs["_eval_celltype"].isin(classes).to_numpy(),
        training_labels,
        args.seed,
    )
    holdout_indices = selected_indices(
        holdout_mask & real_obs["_eval_celltype"].isin(classes).to_numpy(),
        training_labels,
        args.seed + 1,
    )
    sim_indices = selected_indices(
        sim_obs["_eval_celltype"].isin(classes).to_numpy(),
        sim_labels,
        args.seed + 2,
    )
    print(
        f"抽样行数: training={len(training_indices)}, "
        f"holdout={len(holdout_indices)}, sim={len(sim_indices)}",
        flush=True,
    )
    real_matrices = read_sparse_row_subsets(
        args.real,
        [training_indices, holdout_indices],
        layer="counts",
    )
    print("参考矩阵子集读取完成", flush=True)
    training_eval = ad.AnnData(
        X=real_matrices[0],
        obs=real_obs.iloc[training_indices].copy(),
        var=real_backed.var.copy(),
    )
    holdout_eval = ad.AnnData(
        X=real_matrices[1],
        obs=real_obs.iloc[holdout_indices].copy(),
        var=real_backed.var.copy(),
    )
    sim_matrix = read_sparse_row_subsets(
        args.sim,
        [sim_indices],
        layer="counts",
    )[0]
    print("模拟矩阵子集读取完成", flush=True)
    sim_eval = ad.AnnData(
        X=sim_matrix,
        obs=sim_obs.iloc[sim_indices].copy(),
        var=sim_backed.var.copy(),
    )
    real_backed.file.close()
    sim_backed.file.close()
    training_eval = training_eval[:, common_genes].copy()
    holdout_eval = holdout_eval[:, common_genes].copy()
    sim_eval = sim_eval[:, common_genes].copy()
    print("基因对齐完成，开始归一化 ...", flush=True)
    maybe_normalize(training_eval, "training_real", args.normalize)
    maybe_normalize(holdout_eval, "holdout_real", args.normalize)
    maybe_normalize(sim_eval, "simulated", args.normalize)
    print("归一化完成", flush=True)
    print(
        "评估细胞数:",
        f"training={training_eval.n_obs}",
        f"holdout={holdout_eval.n_obs}",
        f"sim={sim_eval.n_obs}",
    )

    summary = {
        "n_real_cells_full": int(len(real_obs)),
        "n_sim_cells": int(sim_eval.n_obs),
        "n_training_cells_evaluated": int(training_eval.n_obs),
        "n_holdout_cells_evaluated": int(holdout_eval.n_obs),
        "n_common_genes": int(len(common_genes)),
        "n_common_celltypes": int(len(classes)),
        "common_celltypes": classes,
        "simulation_celltype_key": sim_ct_key,
        "condition_columns": condition_columns,
        "holdout_mode": holdout_mode,
        "max_cells_per_class": args.max_cells_per_class,
        "min_cells_per_class": args.min_cells_per_class,
        "excluded_celltypes": excluded_classes,
    }

    full_holdout_obs = real_obs.loc[holdout_mask & real_obs["_eval_celltype"].isin(classes)]
    full_sim_obs = sim_obs.loc[sim_obs["_eval_celltype"].isin(classes)]
    counts_real = (
        full_holdout_obs["_eval_celltype"]
        .value_counts()
        .reindex(classes, fill_value=0)
    )
    counts_sim = (
        full_sim_obs["_eval_celltype"]
        .value_counts()
        .reindex(classes, fill_value=0)
    )
    prop_real = counts_real / counts_real.sum()
    prop_sim = counts_sim / counts_sim.sum()
    prop_df = pd.DataFrame(
        {
            "celltype": classes,
            "count_real_holdout": counts_real.values,
            "count_sim": counts_sim.values,
            "prop_real_holdout": prop_real.values,
            "prop_sim": prop_sim.values,
        }
    )
    prop_df.to_csv(outdir / "celltype_proportions.csv", index=False)
    summary["celltype_js_distance"] = float(
        jensenshannon(prop_real.values, prop_sim.values, base=2)
    )
    summary["celltype_max_abs_prop_diff"] = float(
        np.max(np.abs(prop_sim.values - prop_real.values))
    )

    n_genes_r, total_r = cell_qc_stats(holdout_eval)
    n_genes_s, total_s = cell_qc_stats(sim_eval)
    summary["n_genes_median_real"] = float(np.median(n_genes_r))
    summary["n_genes_median_sim"] = float(np.median(n_genes_s))
    summary["n_genes_ks_stat"] = float(ks_2samp(n_genes_r, n_genes_s).statistic)
    summary["total_expr_median_real"] = float(np.median(total_r))
    summary["total_expr_median_sim"] = float(np.median(total_s))
    summary["total_expr_ks_stat"] = float(ks_2samp(total_r, total_s).statistic)
    mean_r, var_r, zero_r = gene_stats(holdout_eval)
    mean_s, var_s, zero_s = gene_stats(sim_eval)
    summary["gene_mean_spearman"] = float(
        spearmanr(mean_r, mean_s).correlation
    )
    summary["gene_log_var_spearman"] = float(
        spearmanr(np.log1p(var_r), np.log1p(var_s)).correlation
    )
    summary["gene_zero_frac_spearman"] = float(
        spearmanr(zero_r, zero_s).correlation
    )

    markers = get_top_markers(
        training_eval,
        "_eval_celltype",
        n=args.n_top_markers,
    )
    marker_real_df = compute_marker_metrics(
        holdout_eval,
        "_eval_celltype",
        markers,
        classes,
    )
    marker_sim_df = compute_marker_metrics(
        sim_eval,
        "_eval_celltype",
        markers,
        classes,
    )
    sim_shuf = sim_eval.copy()
    rng = np.random.RandomState(args.seed)
    sim_shuf.obs["_eval_celltype"] = rng.permutation(
        sim_shuf.obs["_eval_celltype"].astype(str).values
    )
    marker_shuf_df = compute_marker_metrics(
        sim_shuf,
        "_eval_celltype",
        markers,
        classes,
    )
    marker_real_df.to_csv(outdir / "marker_metrics_real.csv", index=False)
    marker_sim_df.to_csv(outdir / "marker_metrics_sim.csv", index=False)
    marker_shuf_df.to_csv(
        outdir / "marker_metrics_sim_shuffled.csv",
        index=False,
    )
    summary["marker_auroc_macro_real"] = float(marker_real_df["auc_mean"].mean())
    summary["marker_auroc_macro_sim"] = float(marker_sim_df["auc_mean"].mean())
    summary["marker_direction_rate_macro_sim"] = float(
        marker_sim_df["direction_rate"].mean()
    )
    summary["marker_auroc_macro_sim_shuffled"] = float(
        marker_shuf_df["auc_mean"].mean()
    )
    summary["marker_direction_rate_macro_sim_shuffled"] = float(
        marker_shuf_df["direction_rate"].mean()
    )

    sc.pp.highly_variable_genes(
        training_eval,
        n_top_genes=min(args.n_hvg, training_eval.n_vars),
        flavor="seurat",
    )
    hvg = training_eval.var_names[
        training_eval.var["highly_variable"]
    ].tolist()
    if len(hvg) < 10:
        hvg = training_eval.var_names.tolist()
    X_train = to_dense(training_eval[:, hvg].X)
    X_hold = to_dense(holdout_eval[:, hvg].X)
    X_sim = to_dense(sim_eval[:, hvg].X)
    n_pca = min(50, len(hvg), X_train.shape[0] - 1)
    if n_pca < 2:
        raise ValueError("too few cells or genes for biological PCA")
    pca = PCA(n_components=n_pca, random_state=args.seed)
    Z_train = pca.fit_transform(X_train).astype(np.float32)
    Z_hold = pca.transform(X_hold).astype(np.float32)
    Z_sim = pca.transform(X_sim).astype(np.float32)
    y_train = training_eval.obs["_eval_celltype"].astype(str).values
    y_hold = holdout_eval.obs["_eval_celltype"].astype(str).values
    y_sim = sim_eval.obs["_eval_celltype"].astype(str).values
    cls_metrics, cm = classification_metrics(
        Z_train,
        y_train,
        Z_hold,
        y_hold,
        Z_sim,
        y_sim,
        classes,
    )
    summary.update(cls_metrics)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(
        outdir / "classification_confusion_real_to_sim.csv"
    )

    nn_real = NearestNeighbors(n_neighbors=2).fit(Z_hold)
    d_r, idx_r = nn_real.kneighbors(Z_hold)
    real_nn_dist = d_r[:, 1]
    nearest_holdout_ct = y_hold[idx_r[:, 1]]
    summary["nearest_real_holdout_same_celltype_fraction"] = float(
        np.mean(nearest_holdout_ct == y_hold)
    )
    nn_sim = NearestNeighbors(n_neighbors=1).fit(Z_hold)
    d_s, idx_s = nn_sim.kneighbors(Z_sim)
    sim_nn_dist = d_s[:, 0]
    nearest_real_ct = y_hold[idx_s[:, 0]]
    summary["nearest_real_same_celltype_fraction"] = float(
        np.mean(nearest_real_ct == y_sim)
    )
    summary["nn_sim_to_real_median"] = float(np.median(sim_nn_dist))
    summary["nn_real_to_real_median"] = float(np.median(real_nn_dist))
    summary["nn_distance_ratio_sim_over_real"] = float(
        np.median(sim_nn_dist) / np.median(real_nn_dist)
    )

    n_pc_use = min(10, Z_hold.shape[1])
    het_rows = []
    for celltype in classes:
        real_mask = y_hold == celltype
        sim_mask = y_sim == celltype
        if real_mask.sum() < 2 or sim_mask.sum() < 2:
            continue
        var_real = float(np.var(Z_hold[real_mask, :n_pc_use], axis=0).sum())
        var_sim = float(np.var(Z_sim[sim_mask, :n_pc_use], axis=0).sum())
        het_rows.append(
            {
                "celltype": celltype,
                "n_real": int(real_mask.sum()),
                "n_sim": int(sim_mask.sum()),
                "pca_var_real": var_real,
                "pca_var_sim": var_sim,
                "var_ratio_sim_over_real": var_sim / var_real
                if var_real > 0
                else np.nan,
            }
        )
    het_df = pd.DataFrame(het_rows)
    het_df.to_csv(outdir / "heterogeneity.csv", index=False)
    summary["heterogeneity_ratio_median"] = (
        float(het_df["var_ratio_sim_over_real"].median())
        if len(het_df)
        else np.nan
    )

    conditional_celltype = "celltype" in condition_columns
    summary["celltype_proportion_is_conditioned"] = conditional_celltype
    checks = build_checks(summary, conditional_celltype)
    summary["checks"] = checks
    summary["biological_evaluation_passed"] = bool(all(checks.values()))

    with open(outdir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            summary,
            handle,
            indent=2,
            ensure_ascii=False,
            default=lambda value: value.item()
            if hasattr(value, "item")
            else str(value),
        )
    with open(outdir / "summary.txt", "w", encoding="utf-8") as handle:
        for key, value in summary.items():
            handle.write(f"{key}: {value}\n")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, Z, y, title in (
        (axes[0], Z_hold, y_hold, "Real holdout"),
        (axes[1], Z_sim, y_sim, "Simulated"),
    ):
        for celltype in classes:
            mask = y == celltype
            ax.scatter(Z[mask, 0], Z[mask, 1], s=3, alpha=0.6, label=celltype)
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    axes[0].legend(markerscale=4, fontsize=6, loc="best")
    plt.tight_layout()
    plt.savefig(outdir / "pca_real_vs_sim_by_celltype.png", dpi=150)
    plt.close()

    prop_df.set_index("celltype")[
        ["prop_real_holdout", "prop_sim"]
    ].plot(kind="bar", figsize=(10, 4))
    plt.ylabel("Proportion")
    plt.tight_layout()
    plt.savefig(outdir / "celltype_proportions.png", dpi=150)
    plt.close()

    marker_plot = (
        marker_real_df[["celltype", "auc_mean"]]
        .rename(columns={"auc_mean": "holdout"})
        .merge(
            marker_sim_df[["celltype", "auc_mean"]].rename(
                columns={"auc_mean": "simulated"}
            ),
            on="celltype",
        )
        .merge(
            marker_shuf_df[["celltype", "auc_mean"]].rename(
                columns={"auc_mean": "shuffled"}
            ),
            on="celltype",
        )
        .set_index("celltype")
    )
    marker_plot.plot(kind="bar", figsize=(11, 4))
    plt.axhline(0.5, color="black", linestyle="--", linewidth=1)
    plt.ylabel("Mean AUROC of training markers")
    plt.tight_layout()
    plt.savefig(outdir / "marker_auroc_sim.png", dpi=150)
    plt.close()

    print("\n===== Biological checks =====")
    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    print(
        "\nbiological_evaluation_passed:",
        summary["biological_evaluation_passed"],
    )
    print(f"Saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
