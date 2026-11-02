# scRNARefGenerator 评测报告

评测日期：2026-11-02  
参考数据：`testdata/pbmc5k_querydata.h5ad`  
参考规模：58,677 cells × 33,421 genes  
完整机器可读结果：

- [`outputs/experiments/biological_summary_all.csv`](outputs/experiments/biological_summary_all.csv)
- [`outputs/experiments/biological_summary_all.json`](outputs/experiments/biological_summary_all.json)

## 1. 评测目标

评测不仅检查模拟数据是否“看起来像”参考数据，还检查以下生物学结构：

- marker 基因是否保持正确的细胞类型方向；
- celltype 身份是否能从训练参考泛化到模拟数据；
- 模拟细胞是否落入正确的局部细胞类型邻域；
- celltype 内异质性是否过低或过高；
- 表达均值、方差、零比例和 library size 是否接近真实 holdout；
- 模拟数据是否只是 source cells 或真实参考的逐细胞复制。

最终评价用于干实验筛选，不替代独立 donor、扰动实验和湿实验验证。

## 2. 数据拆分与防止泄漏

每个模拟 h5ad 都记录了 `source_barcode`。评测脚本从完整参考中排除这些
source cells，剩余细胞作为真实 holdout。

```text
training reference = 模拟实际使用过的 source barcodes
real holdout        = 参考中从未作为 source 使用的细胞
simulation          = 模拟数据
```

为了避免实验配额和大类别主导指标，三个数据集均按 celltype 分层抽样，
默认每个 celltype 最多 500 个细胞。低于 3 个可用细胞的类别不参与统计，
并在 `summary.json` 的 `excluded_celltypes` 中明确列出。

## 3. 评测方法

### 3.1 Marker 保留

1. 在 training reference 中，对每个 celltype 使用 Wilcoxon rank-sum test
   选择 top-30 positive markers。
2. 分别在 real holdout 和 simulation 中计算每个 marker 对
   “目标 celltype vs 其他 celltypes”的 AUROC。
3. 对每个 celltype 的 marker AUROC 取均值，并对 celltype 宏观平均。
4. 将模拟 celltype 标签随机打乱，作为 negative control。

主要指标：

- `marker_auroc_macro_real`：marker 在真实 holdout 中的可重复性；
- `marker_auroc_macro_sim`：marker 在模拟数据中的保留程度；
- `marker_auroc_macro_sim_shuffled`：标签打乱后的期望基线；
- `marker_direction_rate_macro_sim`：AUROC > 0.5 的 marker 比例。

### 3.2 Celltype 身份迁移

1. 在 training reference 上拟合 PCA 和 balanced logistic regression；
2. 用同一个模型分别预测 real holdout 和 simulation；
3. 计算 balanced accuracy 和 macro F1。

`real_holdout_balanced_acc` 是真实参考内部的泛化上限；
`sim_transfer_balanced_acc` 是模拟数据达到的身份迁移准确率。

### 3.3 局部流形一致性

- `nearest_real_same_celltype_fraction`：模拟细胞在 PCA 空间中的最近
  real holdout 细胞是否具有相同 celltype；
- `nearest_real_holdout_same_celltype_fraction`：真实 holdout 自身
  leave-one-out 最近邻的同类比例，作为局部流形混淆基线；
- `nn_distance_ratio_sim_over_real`：模拟到 holdout 的最近邻距离中位数
  / holdout 内部的最近邻距离中位数。

NN ratio 接近 1 表示模拟数据与真实数据到参考流形的距离相近；过小可能
过度复制，过大可能偏离。

### 3.4 分布与异质性

- `gene_mean_spearman`；
- `gene_log_var_spearman`；
- `gene_zero_frac_spearman`；
- 检测基因数和 library size 的 KS statistic；
- celltype 内 PCA 方差比 `var_sim / var_real`；
- 无条件实验额外报告 celltype 比例 Jensen-Shannon distance。

条件实验中的 celltype 比例由用户配额决定，因此不作为生物学真实性门禁。

## 4. 门禁

默认门禁如下：

| 检查 | 条件 |
|---|---|
| 表达均值 | Spearman ≥ 0.80 |
| 表达方差 | Spearman ≥ 0.70 |
| 零比例 | Spearman ≥ 0.70 |
| Marker 方向 | direction rate ≥ 0.70 |
| Marker 高于随机标签 | sim AUROC − shuffled AUROC ≥ 0.05 |
| Marker 相对 holdout 保留 | sim AUROC ≥ real − 0.15 |
| Celltype 身份迁移 | BA ≥ max(0.60, real BA − 0.20) |
| 最近邻身份 | sim ≥ max(0.50, real nearest-baseline − 0.20) |
| NN ratio | 0.50–2.00 |
| 异质性比 | 0.50–2.00 |
| 无条件 celltype 比例 | JS distance ≤ 0.35 |

所有检查通过时，`biological_evaluation_passed=True`。

## 5. Experiment 1：Celltype 子集规模

设置：`similarity_strength=0.70`，每个选中 celltype 模拟参考数量的 60%。

| Celltypes | Cells | Marker AUROC | Transfer BA | Nearest same-type | NN ratio | Heterogeneity | Pass |
|---:|---:|---:|---:|---:|---:|---:|---|
| 5 | 20,803 | 0.796 | 0.961 | 0.901 | 1.029 | 0.710 | yes |
| 8 | 26,950 | 0.773 | 0.945 | 0.874 | 1.022 | 0.683 | yes |
| 10 | 30,009 | 0.798 | 0.951 | 0.879 | 1.030 | 0.626 | yes |
| 12 | 31,796 | 0.818 | 0.955 | 0.877 | 1.040 | 0.631 | yes |
| 15 | 33,640 | 0.819 | 0.946 | 0.851 | 1.035 | 0.620 | yes |
| all 21 | 35,201 | 0.827 | 0.920 | 0.783 | 1.050 | 0.551 | yes |

结论：

- 6/6 个设置通过。
- 增加 celltype 数量后，marker AUROC 和身份迁移仍保持稳定。
- 全 celltype 设置最难，最近邻同类比例和异质性比最低，但仍高于门禁。
- 表达均值、方差和零比例 Spearman 在全部设置中约为 0.956–0.972。

## 6. Experiment 2：Similarity Strength

设置：control columns 为 `celltype` 和 `ident`，所有类别各取参考数量的 60%。

| Similarity | Cells | Marker AUROC | Transfer BA | Nearest same-type | NN ratio | Heterogeneity | Pass |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.5 | 35,206 | 0.823 | 0.912 | 0.768 | 1.030 | 0.804 | yes |
| 0.6 | 35,206 | 0.823 | 0.901 | 0.739 | 1.033 | 0.815 | yes |
| 0.7 | 35,206 | 0.824 | 0.909 | 0.780 | 1.033 | 0.818 | yes |
| 0.8 | 35,206 | 0.823 | 0.895 | 0.750 | 1.033 | 0.843 | yes |
| 0.9 | 35,206 | 0.823 | 0.914 | 0.768 | 1.026 | 0.806 | yes |

结论：

- 5/5 个设置通过。
- `similarity_strength` 从 0.5 到 0.9 没有造成明显单调退化。
- marker AUROC、身份迁移和 NN ratio 对该参数较稳定；异质性比由约 0.80
  增至约 0.84，仍接近 1。
- `celltype × ident` 联合约束没有破坏 celltype marker 和身份结构。

## 7. Experiment 3：无条件模拟数量

设置：无 condition，`similarity_strength=0.70`。模拟数据的 celltype
身份使用继承的 `source_celltype` 进行评价。

| Requested cells | Marker AUROC | Transfer BA | Nearest same-type | NN ratio | Heterogeneity | Pass |
|---:|---:|---:|---:|---:|---:|---|
| 1,000 | 0.760 | 0.635 | 0.547 | 1.361 | 1.060 | no |
| 3,000 | 0.763 | 0.588 | 0.495 | 1.364 | 1.013 | no |
| 5,000 | 0.760 | 0.546 | 0.444 | 1.361 | 1.085 | no |
| 10,000 | 0.759 | 0.591 | 0.475 | 1.351 | 1.021 | no |
| 20,000 | 0.763 | 0.570 | 0.468 | 1.359 | 1.072 | no |

这些设置中：

- marker AUROC 约 0.76，明显高于 shuffled baseline 约 0.50；
- real holdout 身份迁移准确率约为 0.76–0.86；
- 模拟身份迁移仅约 0.55–0.64；
- 最近邻同类比例仅约 0.44–0.55；
- NN ratio 约 1.36，表示模拟细胞整体离真实流形更远；
- 异质性比约 1.0，没有过度平滑；
- 表达分布、marker 方向和比例检查仍通过。

结论：无条件模式保留了全局表达和 marker 信号，但没有充分保留细胞级
celltype 身份。原因是无条件生成使用一个全局 condition group，局部邻域
可在不同 celltype 间混合。这一限制是当前算法的真实行为，不能解释为脚本
评价错误。

## 8. 总体结论

1. 条件生成，包括 celltype 子集和 `celltype × ident` 多列边际约束，具有
   较强且稳定的生物学真实性。
2. 无条件生成适合总体表达分布和 marker 信号模拟，不适合需要准确恢复
   celltype 身份的基准任务。
3. 所有实验都保留了 marker 方向，并且模拟 marker AUROC 与真实 holdout
   接近；无条件模式的问题集中在 celltype 级身份和局部邻域。
4. 当前参考只有一个 sample/donor（`source=VKH`）。结果不能证明跨 donor、
   跨批次或跨疾病的泛化能力。
5. 数据相似不等于逐细胞复制；项目另有 exact-copy 和 source Jaccard
   诊断，但最终生物学真实性仍应以独立数据和湿实验为准。

## 9. 复现

```bash
# 全部实验
bash scripts/evaluate_experiments_biological.sh

# 单个实验
python evaluate_sim_biological.py \
  --real testdata/pbmc5k_querydata.h5ad \
  --sim outputs/experiments/experiment1/subset_10/simulated.h5ad \
  --outdir outputs/experiments/experiment1/subset_10/eval_bio_v2 \
  --max-cells-per-class 500 \
  --n-top-markers 30
```

每个 `eval_bio_v2/` 包含：

- `summary.json` / `summary.txt`；
- marker AUROC 表及 shuffled control；
- celltype confusion matrix；
- celltype 内异质性表；
- PCA 和 marker 可视化。
