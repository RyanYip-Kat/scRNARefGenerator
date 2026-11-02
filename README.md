# scRNARefGenerator

`scRNARefGenerator` 从参考 scRNA-seq raw counts 生成具有精确条件配额的模拟
AnnData，同时保持参考基因及顺序，并记录随机种子、评估指标和实际配额。

详细文档：

- [模拟算法说明](algorithm.md)
- [实验结果与生物学评测报告](evaluation.md)
- [精选结果图表与汇总表](results/README.md)

## 安装

```bash
python -m pip install -e .
```

需要 Python 3.10 或更高版本。核心依赖为 `anndata`、NumPy、SciPy、
pandas 和 scikit-learn。

需要运行 UMAP 对比图时安装绘图依赖：

```bash
python -m pip install -e ".[plot]"
```

大型参考 `.h5ad` 不进入 Git 仓库。PBMC 集成测试和示例默认从以下相对路径读取：

```text
testdata/pbmc5k_querydata.h5ad
```

克隆仓库后需自行将该文件放入 `testdata/`。文件不存在时，真实数据集成测试会
自动 skip；使用 `--ref` 或 `--real` 参数时也可以指定其他路径。

## Python API

```python
from scrnarefgenerator import (
    GeneratorConfig,
    condition_spec_from_reference,
    scRNARefGenerator,
)

reference_path = "testdata/pbmc5k_querydata.h5ad"
generator = scRNARefGenerator(
    reference_path,
    GeneratorConfig(
        seed=42,
        similarity_strength=0.7,
        latent_k=0,
        strict_evaluation=False,
    ),
)

spec = condition_spec_from_reference(
    generator.reference,
    "celltype",
    top_n=10,
    fraction=0.60,
)
simulated = generator.simulate(
    spec,
    output_path="outputs/pbmc_sim.h5ad",
    report_path="outputs/pbmc_sim_report.json",
)
```

`condition_spec_from_reference()` 默认按参考细胞数降序选择前 10 个
`celltype`，并为每类生成 `floor(参考数量 × 0.60)` 个细胞。可通过
`categories=["CD14", "NK3", ...]` 指定类别；指定顺序会保留，且类别数量不能
超过 `top_n`。这里的整数取整规则固定为向下取整。

无限制模式：

```python
simulated = generator.simulate(condition_spec=None, n_cells=100)
```

生成后可以直接调用公共相似度 API：

```python
from scrnarefgenerator import EvaluationConfig, evaluate_similarity

report = evaluate_similarity(
    reference,
    simulated,
    condition_col="celltype",
    config=EvaluationConfig(matched_cells=1000),
)

print(report.passed)
print(report.overall.metrics)
print(report.groups["CD14"].metrics)
```

`report.as_dict()` 可直接序列化。若提供 `condition_col`，报告同时包含总体指标
和逐条件指标。

反抄袭/来源依赖诊断：

```python
from scrnarefgenerator import evaluate_novelty

novelty = evaluate_novelty(
    simulated,
    reference,
    condition_col="celltype",
)
print(novelty.metrics)
```

报告包括 exact source copy 比例、模拟细胞与 source/随机同类型参考的
Jaccard、source dependence gain 和 library ratio。

### source metadata 与条件一致性

每个模拟 barcode 由某个参考 source barcode 加上后缀生成，并从目标条件
对应的 source pool 中抽取。因此条件列始终与 `source_<condition>` 一致。
默认 `source_metadata=True` 会把参考的全部 obs 列复制为 `source_<col>`。

若需要让 `ident` 等 metadata 直接出现在输出中，可用
`inherit_obs_cols`：

```python
config = GeneratorConfig(
    source_metadata=True,
    inherit_obs_cols=["ident"],
)
generated = scRNARefGenerator(reference, config).simulate(condition_spec)

assert (generated.obs["ident"] == generated.obs["source_ident"]).all()
assert (
    generated.obs["celltype"].astype(str)
    == generated.obs["source_celltype"].astype(str)
).all()
```

条件列和内部输出列不能被其他 metadata 覆盖；缺失列或名称冲突会直接报错。

多列边际模式：

```python
spec = ConditionSpec(
    columns=["batch", "group"],
    marginal={
        "batch": {"A": 10, "B": 10},
        "group": {"x": 8, "y": 12},
    },
)
```

边际规划使用整数约束优化，同时满足每一列的精确配额和参考联合组合容量。
不可行条件会抛出 `InfeasibleConditionError`，不会静默修改配额。
分配目标同时尽量接近参考联合分布比例，避免把目标细胞集中到少数
`celltype × ident` 联合组合。需要精确控制联合组合时，应直接使用 `joint`。

`celltype` 与 `ident` 同时受限：

```python
spec = ConditionSpec(
    columns=["celltype", "ident"],
    marginal={
        "celltype": {"CD14": 100, "NK3": 50},
        "ident": {1: 75, 2: 75},
    },
)
```

两列各自总量必须相等；生成结果会严格满足两个边际，并报告实际联合配额。

### 导出 raw 矩阵和 metadata

```python
from scrnarefgenerator import export_raw_counts

artifacts = export_raw_counts(
    generated,
    "outputs/export",
    prefix="sim",
    matrix_format="mtx",  # genes x cells; 同时写出 obs/var CSV
)
print(artifacts.matrix, artifacts.obs, artifacts.var)
```

- `mtx`：`sim_matrix.mtx` 为 genes × cells，`sim_obs.csv` 为细胞 metadata，
  `sim_var.csv` 为基因 metadata；
- `npz`：`sim_matrix.npz` 为 cells × genes 的 SciPy CSR 稀疏矩阵；
- 导出始终使用校验后的 raw integer counts，不会导出 normalized `X`。

## CLI

```bash
scrnarefgen simulate \
  --ref testdata/pbmc5k_querydata.h5ad \
  --out outputs/pbmc_sim.h5ad \
  --condition-col celltype \
  --condition-json '{"CD14": 100, "NK3": 100}' \
  --seed 42 \
  --latent-k 0 \
  --strict-evaluation false \
  --report outputs/pbmc_sim_report.json
```

多列 CLI 使用以条件列名为键的边际 JSON：

```bash
scrnarefgen simulate \
  --ref reference.h5ad \
  --out simulated.h5ad \
  --condition-col batch \
  --condition-col group \
  --condition-json '{"batch":{"A":10,"B":10},"group":{"x":8,"y":12}}'
```

## 模型

默认 `hierarchical_nb` 使用全局基因概率与条件组概率的两层经验贝叶斯收缩，
并对基因离散度使用矩估计和均值-离散度趋势收缩。每个模拟细胞继承选中的
参考 library size，在加入可控扰动后重新归一化基因权重，再按基因分块进行
NB 采样。

默认几何模型 `neighbor_preserving` 还会在每个 `celltype` 内拟合 PCA 邻域，
将源细胞局部 profile、近邻 profile 和组均值混合成细胞特异的基因权重。
由于细胞间异质性已经由局部 profile 表达，默认计数层使用 Poisson 残差
`count_noise_model="poisson"`，避免再次施加完整边际 NB dispersion 导致
过度稀疏。需要传统 NB 残差时可设置为 `count_noise_model="nb"`。

新颖性与保真度通过 `novelty_mode` 控制：

- `fidelity`：不添加 latent novelty，最大程度保留局部参考结构；
- `balanced`：默认，使用条件协方差约束的 latent perturbation；
- `novel`：更强的新颖性混合，用于探索性生成。

默认强度分别为 `0 / 0.25 / 0.50`，也可以用 `novelty_strength` 显式覆盖。
扰动在条件潜空间中采样，再通过经验协方差解码回 gene space，因此不会只
复制 source/近邻 profile 的凸组合。

可选项：

- `distribution_model="hierarchical_zinb"`：加入按组收缩的逐基因零概率；
- `latent_k > 0`：从 log-normalized counts 的 TruncatedSVD 得分中加入
  可控共表达扰动；
- `similarity_strength`：越大越保留源细胞的 library 和 latent 特征。

所有输出保持 CSR 整数 counts，不会构造完整稠密细胞×基因矩阵。

## 输出契约

输出 AnnData 包含：

- `X` 和 `layers["counts"]`：相同的非负整数 raw counts；
- `var_names`：与参考完全一致且顺序一致；
- `obs_names`：`source_barcode + barcode_suffix`；
- `source_barcode`、`sim_batch`、`sim_seed`；
- `similarity_strength`、`distribution_model` 和请求的条件列；
- `uns["scRNARefGenerator"]` 中可被 h5ad 序列化的 manifest。

JSON 报告包含实际配额、源池大小、逐次尝试的相似度参数、评估指标、违规项
和警告。默认不覆盖已有输出；设置 `overwrite=True` 或 CLI 的 `--overwrite`
才会替换。

## 质量门禁

评估使用与模拟样本量匹配的确定性参考子集，计算：

- 基因均值 Spearman 相关；
- 零比例 Spearman 相关；
- library-size KS；
- 参考/模拟 kNN 距离比；
- 交叉验证分类器 AUC；
- 高变基因均值相关。

`strict_evaluation=True` 时，不通过门禁会调整 similarity strength 并重试；
最终仍失败则抛出 `EvaluationGateError`。探索性分析可使用
`strict_evaluation=False`，此时保留指标和警告并返回候选结果。

## 端到端测试流程与 UMAP

`scripts/pipeline.py` 会读取固定 PBMC 参考，默认按细胞数降序选择前 10 个
`celltype`，每类生成参考数量的 60%，调用公共相似度 API，并输出共享 UMAP、
celltype 分面、library-size ECDF：

```bash
python scripts/pipeline.py \
  --outdir outputs/pipeline \
  --top-n 10 \
  --fraction 0.60 \
  --formats png pdf svg \
  --overwrite
```

只检查指定类别时，使用 `--conditions`：

```bash
python scripts/pipeline.py \
  --outdir outputs/pipeline_subset \
  --conditions "CD14" "NK3" "CD4 Naive" "CD4 Tem" "CD8 Naive" \
               "NK2" "CD8 CTL" "CD8 Tem" "Naive BC" "Memory BC" \
  --top-n 10 \
  --fraction 0.60
```

主要输出：

```text
outputs/pipeline/
  simulated.h5ad
  generation_report.json
  similarity_evaluation.json
  pipeline_summary.json
  figures/
    umap_comparison.png/pdf/svg
    library_size_ecdf.png/pdf/svg
    umap_embedding.csv
```

参考和模拟数据在归一化 counts 上拼接后只拟合一次共享 UMAP。参考细胞按模拟
配额分层抽样，避免参考数据量更大时在图中淹没模拟数据。celltype 颜色在参考
和模拟面板中一致。

pipeline 默认按条件做 80%/20% 分层拆分。训练集用于拟合邻域和选择 source
barcode，holdout 不参与生成；相似度报告同时包含模拟数据对 holdout 的指标，
以及训练集对 holdout 的自然参考基线。

`similarity_evaluation.json` 还包含 `novelty` 区块，以及 `geometry` 中的
实际 novelty mode 和 effective strength。

## 批量实验

运行完整实验网格：

```bash
bash scripts/run_experiments.sh
```

脚本执行三组固定设置：

- Experiment 1：`celltype` 子集数 `5/8/10/12/15/all`，每类取参考数量的
  60%，`similarity_strength=0.70`；
- Experiment 2：`celltype` 与 `ident` 的每个类别分别取 60%，通过精确边际
  MILP 分配联合配额，并扫描 `similarity_strength=0.5–0.9`；
- Experiment 3：无条件生成 `1000/3000/5000/10000/20000` 个细胞。

默认输出到 `outputs/experiments/`。每个设置独立保存：

```text
simulated.h5ad                 # 含 obsm["X_pca"], obsm["X_umap"]
experiment_summary.json
reference_existing_umap.png    # 参考 obsm["X_umap"]
generated_umap.png             # 对生成数据重新拟合的 UMAP
shared_umap/
  umap_comparison.png/pdf/svg  # 参考与模拟共同嵌入后的直接对比
  library_size_ecdf.png/pdf/svg
  umap_embedding.csv
```

推荐以 `umap_comparison` 判断分布重合度；`reference_existing_umap` 和
`generated_umap` 是分别拟合的诊断图，其坐标系不能直接做数值比较。

这两个独立诊断图默认只绘制模拟数据指定的条件类别，并在每个类别内最多抽样
200 个细胞。可通过 runner 参数调整：

```bash
python scripts/run_experiment.py \
  --experiment 1 \
  --outdir outputs/example \
  --subset 5 \
  --plot-cells-per-condition 100
```

重新运行并覆盖已有实验：

```bash
FORCE=1 bash scripts/run_experiments.sh
```

## 生物学真实性评价

单个模拟结果：

```bash
python evaluate_sim_biological.py \
  --real testdata/pbmc5k_querydata.h5ad \
  --sim outputs/experiments/experiment1/subset_10/simulated.h5ad \
  --outdir outputs/experiments/experiment1/subset_10/eval_bio_v2 \
  --max-cells-per-class 500 \
  --n-top-markers 30
```

批量评价所有实验：

```bash
bash scripts/evaluate_experiments_biological.sh
```

只评价某一组：

```bash
EXPERIMENT_FILTER=experiment2 \
  bash scripts/evaluate_experiments_biological.sh
```

强制重算：

```bash
FORCE=1 bash scripts/evaluate_experiments_biological.sh
```

评价逻辑：

- 从模拟数据的 `source_barcode` 排除训练参考，构造真实 holdout；
- 将 training/holdout/simulation 每个 celltype 平衡抽样；
- 从 training reference 学习 marker，在 holdout 和模拟中独立计算 AUROC；
- 用 training reference 拟合 PCA 和分类器，分别检验 holdout 与模拟的 celltype 迁移；
- 比较 celltype 内 PCA 异质性、最近邻身份、NN 距离和表达分布；
- 条件实验的 celltype 配额不作为生物学真实性惩罚。

每个实验目录下输出 `eval_bio_v2/`，包括 `summary.json`、marker 表、混淆矩阵、
异质性表和 PCA/marker 图。全部结果汇总到：

```text
outputs/experiments/biological_summary_all.csv
outputs/experiments/biological_summary_all.json
```

注意：这是干实验筛查，不能替代独立 donor、扰动实验或湿实验验证。

## 验证

```bash
PYTHONPATH=src python -m pytest -q
```

集成测试读取固定 PBMC5k 数据，显式选择少量 `celltype` 配额，并验证稀疏
整数输出、基因顺序、条件配额与 JSON 报告。

## 限制

- 模拟数据不编码真实因果机制，不应直接用于推断生物学因果效应。
- 默认不做归一化、基因筛选或批次校正。
- 边际规划规模取决于观测到的联合条件数；类别组合极大时 MILP 求解可能较慢。
- 高维 latent 模型会使用更多内存，建议先在小配额上验证。
