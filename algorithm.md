# scRNARefGenerator 模拟算法说明

本文档描述当前实现，而不是理想化设计。关键代码位于：

- [`src/scrnarefgenerator/validation.py`](src/scrnarefgenerator/validation.py)
- [`src/scrnarefgenerator/conditions.py`](src/scrnarefgenerator/conditions.py)
- [`src/scrnarefgenerator/distribution.py`](src/scrnarefgenerator/distribution.py)
- [`src/scrnarefgenerator/geometry.py`](src/scrnarefgenerator/geometry.py)
- [`src/scrnarefgenerator/simulate.py`](src/scrnarefgenerator/simulate.py)
- [`src/scrnarefgenerator/assembly.py`](src/scrnarefgenerator/assembly.py)

## 1. 总体目标

给定参考 scRNA-seq raw counts `R` 和条件配额，生成新的 AnnData `S`：

```text
S.obsm / S.var_names / raw counts 可被下游 scRNA 工具直接读取
S 与 R 在表达分布、marker、celltype 身份和局部流形上相似
S 的 barcode、条件配额和部分细胞状态不等于参考的逐细胞复制
```

算法是非参数经验分布模型与条件邻域生成模型的组合。它不学习生物学因果结构。

## 2. 输入校验与 raw counts 解析

counts 检测顺序：

1. `adata.layers[counts_layer]`，默认 `layers["counts"]`；
2. `adata.raw.X`，要求 raw 基因与 `var_names` 完全一致；
3. `adata.X`。

矩阵必须：

- 有限；
- 非负；
- 整数容差 `< 1e-6`；
- 保持稀疏，内部转换为 CSR；
- obs/var 名称唯一；
- 基因顺序作为输出的唯一基因顺序。

归一化矩阵不会被静默当作 raw counts。

## 3. 条件规划

### 3.1 无条件模式

```python
simulate(condition_spec=None, n_cells=N)
```

所有 reference cells 构成一个全局 source pool。

### 3.2 联合条件

`ConditionSpec.joint` 直接指定条件 tuple 的精确细胞数：

```python
ConditionSpec(
    columns=["celltype", "ident"],
    joint={
        ("CD14", 1): 26,
        ("CD14", 2): 74,
        ("NK3", 1): 49,
        ("NK3", 2): 1,
    },
)
```

每个联合组源 pool 必须包含足够 cells，source barcode 全局无放回使用。

### 3.3 单列比例配额

`condition_spec_from_reference` 可以：

- 按参考类别频数降序选择 top-N celltypes；
- 或接受指定类别；
- 每类配额为 `floor(reference_count × fraction)`。

### 3.4 多列边际配额

对 `columns=["celltype", "ident"]`，用户分别指定两个边际：

```text
Σ_ident T[celltype, ident] = celltype_quota
Σ_celltype T[celltype, ident] = ident_quota
```

实现使用整数 MILP：

- 变量是每个观测联合组的整数细胞数；
- 约束包含精确总数、每个条件边际、每组参考容量；
- 目标联合分布按参考联合组容量比例初始化；
- 额外 deviation variables 最小化相对该目标的 L1 偏差。

若不可行，抛出 `InfeasibleConditionError`，不会静默修改用户配额。

## 4. 条件层级基因分布

对每个目标条件组 `g`：

### 4.1 全局基因概率

```text
p0_j = (Σ_i X_ij + ε) / (Σ_i library_i + Jε)
```

### 4.2 条件组收缩

```text
κ = prior_strength × median_library
p_gj = (Σ_{i∈g} X_ij + κ p0_j) / (Σ_{i∈g} library_i + κ)
```

该经验 Bayes 收缩让稀有类别自动向全局分布靠拢。

### 4.3 离散度

先将组内 counts 归一化为：

```text
y_ij = X_ij / library_i × median_library
```

逐基因计算 mean/variance 和 method-of-moments NB dispersion：

```text
θ_mom = μ² / max(var − μ, ε)
```

随后向 log-mean/log-dispersion trend 收缩。默认
`dispersion_n0=50`。

## 5. 条件内局部邻域几何

默认 `geometry_model="neighbor_preserving"`。

### 5.1 Latent 空间

对每个条件组：

1. library-size normalization；
2. `log1p`；
3. 选择 variance 最高的 2,000 genes；
4. TruncatedSVD，默认 20 components；
5. 标准化 SVD scores；
6. 在组内构建 `geometry_neighbors=30` 的 kNN graph。

拟合细胞可下采样到 `max_cells_for_fitting=10000`，但 latent transform
覆盖整个条件 source pool。

### 5.2 每个模拟细胞的 profile

设 source cell 为 `i`，同条件邻域为 `N_i`。邻域权重为：

```text
w_il ∝ exp(-d_il² / (2 σ_i²))
```

其中 `σ_i` 是该 source 邻域距离的中位数。

归一化 expression profile：

```text
q_source = X_i / library_i
q_neighbor = Σ_l w_il X_l / library_l
```

混合：

```text
q_local = (1 − λ) q_source + λ q_neighbor
q = normalize((1 − β) q_local + β p_g)
```

默认：

```text
λ = local_profile_mix = 0.50
β = group_prior_mix  = 0.10
```

这保留了细胞特异的共表达结构，同时防止完全复制 source cell。

## 6. 受控 Novelty

### 6.1 Latent covariance decoder

在条件组 training cells 上计算 SVD scores `Z`、gene loadings `B` 和
条件 covariance `Σ`：

```text
z = L ε,   L Lᵀ = Σ
perturbation = z Bᵀ
```

gene-space novelty factors 会逐基因归一化到单位 RMS，使 novelty strength
具有稳定的尺度含义。

### 6.2 Novel profile

```text
q_novel = normalize(p_g × exp(clip(perturbation, −3, 3)))
q_final = (1 − novelty_strength) q + novelty_strength q_novel
```

该步骤允许生成不是 source/邻居 profile 凸组合的细胞状态。

预设：

| mode | novelty strength |
|---|---:|
| fidelity | 0.00 |
| balanced | 0.25 |
| novel | 0.50 |

`novelty_strength` 可以显式覆盖预设。

## 7. Library size 与 counts 采样

选中 source cell 后：

```text
library_sim = library_source × exp(N(0, sigma_lib × (1 − similarity_strength)))
```

因此 `similarity_strength` 越接近 1，library perturbation 越小。

逐细胞期望 counts：

```text
μ_ij = library_sim,i × q_final,ij
```

默认 residual count model：

```text
X_sim,ij ~ Poisson(μ_ij)
```

原因：细胞间异质性已经由 source/邻域 profile 表示。若再次使用从边际
counts 拟合的完整 NB dispersion，会双重计入过度离散，导致 library size
收缩和零比例过高。

可选：

- `count_noise_model="nb"`：NB residual noise；
- `distribution_model="hierarchical_zinb"`：在 residual counts 上增加
  条件组零概率。

counts 以最多 128 cells 的 block 生成；不会构造全量 dense cells × genes
矩阵，最终拼接为 CSR integer matrix。

## 8. Barcode、metadata 与条件一致性

输出 barcode：

```text
source_barcode + barcode_suffix
```

默认 `_sim`。source barcode 全局无放回抽样；冲突时按配置报错或追加确定性
UUID。

输出 metadata 包括：

- 所有目标条件列；
- `source_barcode`；
- `sim_batch`、`sim_seed`；
- `similarity_strength`、`distribution_model`；
- `novelty_mode`、`novelty_strength`；
- 全部参考 metadata 的 `source_<column>`；
- 可由 `inherit_obs_cols` 指定直接继承的列，例如 `ident`。

组装时验证目标条件与 `source_<condition>` 完全一致。

## 9. 随机性

每个 attempt 使用：

```python
numpy.random.default_rng(SeedSequence([seed, attempt, 17]))
```

几何拟合也使用独立的 seed stream。禁止全局随机状态，因此相同参数、相同
attempt 和相同包版本可复现。

## 10. 质量门禁

生成后可计算：

- gene mean / zero / high-variable Spearman；
- library size KS；
- kNN distance ratio；
- cross-validated classifier AUC；
- 总体和逐条件门禁。

严格模式会调整 `similarity_strength` 并重试；否则保留候选和 warnings。

实验 runner 额外做 celltype holdout：

- 80% training cells 用于拟合和 source selection；
- 20% holdout 不参与生成；
- 生成数据与 holdout 比较。

## 11. 输出与导出

核心输出 `S`：

- `S.X` 和 `S.layers["counts"]`：相同 integer raw counts；
- `S.var_names`：与参考完全一致且顺序一致；
- `S.obs_names`：唯一生成 barcode；
- `S.uns["scRNARefGenerator"]`：JSON-safe manifest。

实验 runner 额外计算并保存：

- `obsm["X_pca"]`；
- `obsm["X_umap"]`。

`export_raw_counts()` 支持：

- Matrix Market：genes × cells；
- SciPy NPZ：cells × genes CSR；
- 对齐的 obs CSV 和 var CSV。

## 12. 默认参数

| 参数 | 默认值 |
|---|---:|
| geometry_model | neighbor_preserving |
| geometry_neighbors | 30 |
| local_profile_mix | 0.50 |
| group_prior_mix | 0.10 |
| geometry_latent_k | 20 |
| geometry_hvg_count | 2,000 |
| count_noise_model | poisson |
| novelty_mode | balanced |
| novelty_strength | 0.25 |
| prior_strength | 10 |
| sigma_lib | 0.15 |
| dispersion_n0 | 50 |
| max_cells_for_fitting | 10,000 |

## 13. 已知限制

1. **无条件模式缺少 celltype 分层**  
   无条件生成当前只有一个全局 condition group，kNN 可以跨 celltype
   建立邻域。这保留了总体 marker 信号，但 celltype 级身份迁移较弱。

2. **单一参考样本**  
   当前测试数据只有一个 `source=VKH`。随机 cell holdout 不能证明跨 donor、
   跨批次或跨疾病泛化。

3. **经验分布模型**  
   算法不学习因果调控、发育轨迹或扰动响应，不能宣称发现新的生物学机制。

4. **novelty 不等于真实性**  
   增加 novelty 可以降低 source 依赖，但外推细胞必须用 marker、celltype
   coherence、独立数据和湿实验验证。

5. **精确条件不是自然联合分布**  
   用户指定的边际或联合配额可能与参考自然比例不同；结果满足设计配额，
   但不一定代表真实人群构成。

## 14. 推荐使用方式

```python
from scrnarefgenerator import (
    GeneratorConfig,
    condition_spec_from_reference,
    scRNARefGenerator,
)

config = GeneratorConfig(
    seed=42,
    similarity_strength=0.7,
    novelty_mode="balanced",
    source_metadata=True,
    inherit_obs_cols=["ident"],
)

generator = scRNARefGenerator("testdata/pbmc5k_querydata.h5ad", config)
spec = condition_spec_from_reference(
    generator.reference,
    "celltype",
    top_n=10,
    fraction=0.60,
)
simulated = generator.simulate(spec)
```

实验与生物学评价：

```bash
bash scripts/run_experiments.sh
bash scripts/evaluate_experiments_biological.sh
```

评测结论和逐实验结果见 [`evaluation.md`](evaluation.md)。
