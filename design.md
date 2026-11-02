# `scRNARefGenerator` 设计文档（Codex 实现版）

## 0. 文档目的

本文档用于指导 Codex 实现 Python 包 `scRNARefGenerator`。包的目标是：以参考 scRNA 数据 `R` 为输入，在条件约束下生成新的 scRNA 数据 `S`，输出 raw counts 矩阵和细胞 metadata。`S` 的分布应与 `R` 相似但不相同，且满足 barcode、基因、条件配额等硬约束。

测试参考数据固定为：

```text
testdata/pbmc5k_querydata.h5ad
```

---

## 1. 目标与非目标

### 1.1 目标

1. 支持无限制生成、单列条件生成、多列联合条件生成、多列边际条件生成。
2. 输出 `S`：
   - `X` / `layers["counts"]` 为非负整数 raw counts；
   - `var_names` 与 `R` 完全一致且顺序一致；
   - `obs_names` = 参考 barcode + 后缀，默认 `_sim`；
   - `obs` 包含条件列、来源 barcode、模拟批次、seed、相似度参数等。
3. 提供“分层条件分布生成器”，估计并采样条件相关的 library size、基因均值、离散度、零比例和低维共表达结构。
4. 提供相似度/差异度评估门禁，防止过度复制参考或过度偏离参考。
5. 所有随机过程可复现，输出 manifest 和评估报告。
6. 提供 CLI 和 Python API，便于测试和下游集成。

### 1.2 非目标

1. 不保证模拟数据具有真实生物学因果结构。
2. 不默认做批次校正、归一化或基因筛选。
3. 不修改参考数据文件。
4. 不输出浮点归一化矩阵作为 counts。

---

## 2. 参考数据与测试数据契约

### 2.1 输入 `R` 要求

输入为 `anndata.AnnData` 或 `.h5ad` 路径。

counts 检测优先级：

```text
1. adata.layers[counts_layer]        # 默认 counts_layer="counts"
2. adata.raw.X                       # 要求 raw.var_names 与 adata.var_names 一致
3. adata.X                           # 仅当其为非负整数矩阵
```

校验规则：

- counts 必须非负，且满足 `max(abs(X - round(X))) < 1e-6`；
- `obs_names` 唯一；
- `var_names` 唯一且与输出基因一致；
- metadata 行数与 counts 列数一致；
- 条件列存在，NA 需显式处理：默认报错，配置 `na_policy="error" | "drop"`；
- 类别数、每类可用细胞数必须足够。

若无法检测到 raw counts，抛出 `CountsNotFoundError`，不得静默把归一化矩阵当 counts。

### 2.2 测试数据使用方式

测试固定读取：

```python
adata = anndata.read_h5ad("testdata/pbmc5k_querydata.h5ad")
```

测试不硬编码具体 metadata 列名。自动选择条件列规则：

1. 优先选择 `obs` 中 categorical / object 类型列；
2. 类别数在 `[2, 10]`；
3. 每类细胞数 ≥ 10；
4. 若找不到，测试中复制 `adata` 并添加确定性合成列 `test_celltype`、`test_group` 用于多列测试，不修改原始文件。

---

## 3. 包结构建议

```text
src/scrnarefgenerator/
  __init__.py
  api.py
  config.py
  io.py
  validation.py
  conditions.py
  distribution/
    __init__.py
    base.py
    nb.py
    zinb.py
  generator.py
  evaluation.py
  report.py
  cli.py
  utils.py

tests/
  conftest.py
  test_validation.py
  test_conditions.py
  test_generator_pbmc5k.py
  test_evaluation.py
  test_cli.py
```

依赖建议：

```text
anndata, numpy, scipy, pandas, scikit-learn
可选：scanpy（仅用于可选 PCA 加速，不作为硬依赖）
```

---

## 4. 公共 API 与配置

### 4.1 Python API

```python
from scrnarefgenerator import scRNARefGenerator, ConditionSpec, GeneratorConfig

gen = scRNARefGenerator(
    ref=adata_or_path,
    config=GeneratorConfig(
        counts_layer="counts",
        condition_cols=["celltype"],
        barcode_suffix="_sim",
        seed=42,
        similarity_strength=0.7,
        distribution_model="hierarchical_nb",
        latent_k=20,
        prior_strength=10.0,
        na_policy="error",
    ),
)

S = gen.simulate(
    condition_spec=ConditionSpec(
        columns=["celltype"],
        joint={("c1",): 1234, ("c2",): 4332},
        marginal=None,
    ),
    n_cells=None,
    output_path="out/sim.h5ad",
    report_path="out/sim_report.json",
)
```

无限制模式：

```python
S = gen.simulate(condition_spec=None, n_cells=1000)
```

### 4.2 CLI

```bash
scrnarefgen simulate \
  --ref testdata/pbmc5k_querydata.h5ad \
  --out out/pbmc5k_sim.h5ad \
  --condition-col celltype \
  --condition-json '{"celltype":{"c1":200,"c2":300}}' \
  --seed 42 \
  --similarity-strength 0.7 \
  --report out/pbmc5k_sim_report.json
```

---

## 5. 数据模型

### 5.1 `ConditionSpec`

```python
@dataclass
class ConditionSpec:
    columns: list[str]
    joint: dict[tuple[str, ...], int] | None = None
    marginal: dict[str, dict[str, int]] | None = None
```

语义：

- 单列条件视为 `joint={(category,): count}`；
- 多列推荐使用 `joint`；
- 多列 `marginal` 需要分配为联合配额，分配算法见 7.2；
- 不可同时提供 `joint` 和 `marginal`；
- 所有配额必须为正整数。

### 5.2 输出 metadata 必含字段

| 字段 | 含义 |
|---|---|
| `source_barcode` | 来源参考 barcode |
| `sim_batch` | 模拟批次 ID |
| `sim_seed` | 随机种子 |
| `similarity_strength` | 相似度强度 |
| `distribution_model` | 分布模型名 |
| 条件列 | 模拟条件值 |
| 可选 `source_<col>` | 来源细胞原始 metadata |

---

## 6. 总体流程

```text
1. 读取并校验 R
2. 解析 ConditionSpec
3. 构建条件抽样池并做可行性校验
4. 按层级拟合条件分布参数
5. 为每个目标细胞选择源 barcode、library size、潜变量
6. 生成候选 counts
7. 执行精确配额选择与 barcode 命名
8. 评估相似度/差异度门禁
9. 必要时调整 similarity_strength 重试
10. 输出 AnnData、manifest、报告
```

---

## 7. 核心算法设计

## 7.1 counts 解析与校验算法

```python
def resolve_counts(adata, counts_layer="counts"):
    if counts_layer in adata.layers:
        X = adata.layers[counts_layer]
        var_names = adata.var_names
    elif adata.raw is not None and is_nonnegative_integer(adata.raw.X):
        if not adata.raw.var_names.equals(adata.var_names):
            raise GeneAlignmentError(...)
        X = adata.raw.X
        var_names = adata.var_names
    elif is_nonnegative_integer(adata.X):
        X = adata.X
        var_names = adata.var_names
    else:
        raise CountsNotFoundError(...)

    X = ensure_csr_or_csc(X)
    X = round_and_cast_int(X)
    validate_unique(adata.obs_names, adata.var_names)
    return X, var_names
```

必须检查：

- `X.min() >= 0`；
- 整数容差；
- 稀疏矩阵保持稀疏；
- 不复制大矩阵，除非必要。

---

## 7.2 条件解析与可行性校验

### 7.2.1 单列 / 联合条件

对每个 `key, n_key`：

1. 从 `R.obs` 筛出满足 `key` 的源细胞索引池 `P_key`；
2. 校验 `P_key` 非空；
3. 校验 `n_key <= len(P_key)`；
4. 校验所有 `n_key` 总和 ≤ `R.n_obs`；
5. 全局源 barcode 无放回使用。

### 7.2.2 多列边际条件分配算法

输入：

- 参考联合计数表 `C[a,b]`；
- 目标行边际 `r_a`、列边际 `c_b`。

步骤：

1. 校验每个边际 ≤ 参考对应边际；
2. 用 IPF 拟合联合表 `T`，使其行/列边际匹配目标；
3. 对 `T` 取 floor，余数用 largest remainder 分配，保证行/列边际精确；
4. 校验每个格子 `T[a,b] <= C[a,b]`；
5. 若不可行，抛出 `InfeasibleConditionError`，并返回建议的 `joint` 配额或冲突格子。

伪代码：

```python
T = C.astype(float)
for _ in range(max_iter):
    T *= target_row / T.sum(axis=1)[:, None]
    T *= target_col / T.sum(axis=0)[None, :]
T_int = largest_remainder_round(T, target_row, target_col)
if any(T_int > C):
    raise InfeasibleConditionError(...)
```

**禁止**静默修改用户配额。

---

## 7.3 分层条件分布估计

默认模型：`hierarchical_nb`。

层级：

```text
global -> 每个条件列边际 -> 联合条件组
```

对每个条件组 `g` 估计：

1. library size 分布；
2. 基因比例 `p_gj`；
3. 基因离散度 `θ_gj`；
4. 可选低维潜变量 `z` 和载荷 `B`；
5. 可选零膨胀概率 `π_gj`。

### 7.3.1 library size

对参考组 `g`：

```text
lib_i = sum_j X_ij
```

生成时对选中源 barcode 使用其参考 library size 作为基准：

```text
lib_sim = lib_source * exp(N(0, sigma_lib * s))
```

其中 `s = similarity_strength`，`sigma_lib` 默认 0.15。这样既保留 library size 分布，又引入可控差异。

### 7.3.2 基因比例 `p_gj`

全局：

```text
p0_j = (sum_i X_ij + alpha) / (sum_i lib_i + alpha * J)
```

组 `g` 向父层收缩：

```text
p_gj = (sum_{i in g} X_ij + kappa * p_parent_j) /
       (sum_{i in g} lib_i + kappa)
```

`kappa = prior_strength * median_lib`，默认 `prior_strength=10`。稀有组自动向全局收缩。

### 7.3.3 离散度 `θ_gj`

NB 方差：

```text
Var(X) = mu + mu^2 / theta
```

对组 `g` 的归一化 counts：

```text
y_ij = X_ij / lib_i * median_lib
```

逐基因 MoM：

```text
m_j = mean(y_ij)
v_j = var(y_ij)
theta_mom_j = m_j^2 / max(v_j - m_j, eps)   if v_j > m_j
theta_mom_j = large_value                  otherwise
```

再向 mean-dispersion trend 收缩：

```text
theta_gj = exp((1-w) * log(theta_mom_j) + w * log(theta_trend_j))
w = n_g / (n_g + n0)
```

`n0` 默认 50。零方差基因使用 trend 值。

### 7.3.4 低维共表达结构

在参考全局或组内做 PCA / TruncatedSVD：

```text
Y = log1p(X / lib * median_lib)
Z = PCA(Y, k=latent_k)      # 细胞得分
B = PCA.components_.T       # 基因载荷
```

生成细胞时：

```text
z_sim = sqrt(1 - rho) * z_source + sqrt(rho) * epsilon
epsilon ~ N(0, Sigma_z)
rho = clip(s * latent_mix, 0, 1)
```

因子效应：

```text
mu_ij = lib_sim_i * p_gj * exp(beta * (z_sim_i @ B_j))
```

`beta` 默认 0.2，`latent_mix` 默认 0.5。若未启用 latent，则 `mu_ij = lib_sim_i * p_gj`。

---

## 7.4 counts 采样

默认 NB：

```text
X_sim_ij ~ NegativeBinomial(mean=mu_ij, dispersion=theta_gj)
```

实现：

```text
n = theta_gj
p = theta_gj / (theta_gj + mu_ij)
X_sim_ij ~ NB(n, p)
```

可选 ZINB：

```text
zero_mask_ij ~ Bernoulli(pi_gj)
X_sim_ij = 0 if zero_mask else NB(mu_ij, theta_gj)
```

`pi_gj` 由组内零比例向父层收缩得到。默认不用 ZINB，避免过度零化。

生成后检查：

- 非负整数；
- 不等于源 barcode 的原始 counts；若完全相同，重采样该细胞；
- 保持稀疏。

---

## 7.5 精确配额选择与 barcode 命名

对每个条件组 `g`：

1. 从源池 `P_g` 无放回抽取 `n_g` 个 `source_barcode`；
2. 生成对应 `n_g` 个模拟细胞；
3. 汇总所有组，得到精确总数；
4. barcode：

```text
sim_barcode = source_barcode + barcode_suffix
```

默认 `_sim`。若 `sim_barcode` 已存在于 `R.obs_names` 或组内重复，抛出 `BarcodeSuffixCollisionError`，建议用户更换后缀。不得静默追加随机字符串，除非配置 `collision_policy="append_uuid"`。

---

## 7.6 相似度/差异度评估门禁

在共同基因和 log1p 归一化空间计算：

| 指标 | 目的 | 默认门禁 |
|---|---|---|
| gene mean Spearman | 均值分布相似 | ≥ 0.80 |
| zero proportion Spearman | 稀疏性相似 | ≥ 0.70 |
| library size KS | library 分布相似 | KS ≤ 0.20 |
| PCA centroid distance | 全局位置 | 报告 |
| kNN 距离比 `median NN(S,R)/median NN(R,R)` | 防止过度复制/偏离 | 0.5–2.0 |
| R vs S 分类器 AUC | 可区分性 | 0.55–0.90 |
| marker / 高变基因均值相关 | 生物信号 | ≥ 0.75 |

门禁可配置。`simulate` 默认最多重试 `max_retries=3`：

```text
若过于相似（AUC 过低、NN 距离过小）：增大 s
若过于偏离（mean/zero 相关过低、AUC 过高）：减小 s
```

每次重试记录到报告。最终若仍失败，默认抛出 `EvaluationGateError`；配置 `strict_evaluation=False` 时仅警告并输出。

---

## 8. 输出规范

输出 `AnnData`：

```text
S.X                 raw counts，CSR
S.layers["counts"]  与 X 相同
S.obs_names         source_barcode + suffix
S.var_names         与 R 完全一致
S.obs               条件列 + source_barcode + sim_batch + sim_seed + similarity_strength + distribution_model
S.uns["scRNARefGenerator"] manifest
```

manifest 包含：

```json
{
  "version": "0.1.0",
  "ref_path": "testdata/pbmc5k_querydata.h5ad",
  "seed": 42,
  "condition_spec": {},
  "actual_quota": {},
  "similarity_strength": 0.7,
  "distribution_model": "hierarchical_nb",
  "retries": 0,
  "evaluation": {}
}
```

同时输出 JSON 报告，包含所有评估指标、警告、实际配额、源池大小、拒绝/重采样次数。

---

## 9. 错误与警告

必须定义的异常：

```text
CountsNotFoundError
InvalidCountsError
GeneAlignmentError
DuplicateObsNamesError
DuplicateVarNamesError
ConditionColumnNotFoundError
ConditionCategoryNotFoundError
InfeasibleConditionError
BarcodeSuffixCollisionError
EvaluationGateError
```

警告：

- 稀有条件组：`n_g < min_cells_per_group`；
- 条件组参考细胞数接近目标数；
- 启用 ZINB 导致零比例明显升高；
- 评估指标接近阈值；
- 输出路径已存在。

---

## 10. 测试计划：`testdata/pbmc5k_querydata.h5ad`

### 10.1 Fixture

```python
@pytest.fixture(scope="session")
def ref_adata():
    adata = anndata.read_h5ad("testdata/pbmc5k_querydata.h5ad")
    return adata
```

### 10.2 必测用例

1. **counts 检测**
   - 能解析 `layers["counts"]`、`raw.X` 或整数 `X`；
   - 若均不存在，抛出 `CountsNotFoundError`。

2. **基础校验**
   - barcode 唯一、基因唯一；
   - 输出基因名和顺序与 R 完全一致；
   - 输出 counts 为非负整数。

3. **单列条件**
   - 自动选择第一个可用 categorical 列；
   - 每类配额取 `min(可用数, 50)`；
   - 断言实际配额精确、barcode 唯一、后缀正确、总数 ≤ R。

4. **无限制模式**
   - `n_cells=min(100, R.n_obs)`；
   - 输出数量精确。

5. **多列联合条件**
   - 若参考无合适两列，测试中在副本添加合成列；
   - 提供联合配额，断言精确匹配。

6. **多列边际条件**
   - 构造可行边际，断言 IPF 分配后行/列边际精确；
   - 构造不可行边际，断言 `InfeasibleConditionError`。

7. **barcode 后缀冲突**
   - 人为构造含 `_sim` 的 barcode，断言报错或按配置处理。

8. **可复现性**
   - 相同 seed 两次生成结果完全一致；
   - 不同 seed 结果不同。

9. **相似度与差异度**
   - 使用宽松阈值断言 `mean Spearman ≥ 0.7`、`zero Spearman ≥ 0.6`；
   - 断言生成 counts 不整体等于参考；
   - 断言分类器 AUC 在宽松区间内或仅报告。

10. **CLI smoke test**
    - 运行 CLI 生成小样本，检查输出文件和报告存在。

---

## 11. 重点防漏洞清单

1. barcode 加后缀后重复；
2. 原 barcode 已含 `_sim`；
3. metadata 与 counts 行列不对齐；
4. 基因名重复或顺序改变；
5. 归一化矩阵误当 counts；
6. 多列边际条件不可行但静默改配额；
7. 条件类别不存在或配额超过源池；
8. 稀有组硬拟合导致异常分布；
9. 生成 counts 与参考某细胞完全相同；
10. 过度偏离参考或过度复制参考；
11. 输出浮点 counts；
12. 随机种子未记录；
13. 大矩阵内存爆炸；
14. `uns` 中不可序列化对象；
15. 输出文件覆盖未提示。

---

## 12. 性能与可复现性

- counts 保持稀疏，按基因分块生成；
- 组内 `sum` 用稀疏矩阵乘法；
- PCA 使用 `TruncatedSVD` 或可选 scanpy；
- 所有随机数来自 `numpy.random.default_rng(seed)`，不得使用全局随机状态；
- 输出 manifest 记录包版本、seed、参数、实际配额；
- 对超大参考数据支持 `max_cells_for_fitting` 下采样拟合分布，但抽样池仍用全量索引。

---

## 13. Codex 实现优先级与验收标准

### P0

1. 输入校验与 counts 解析；
2. ConditionSpec 解析与可行性校验；
3. 单列条件 + 无限制生成；
4. 分层 NB 分布估计与采样；
5. barcode 命名、metadata、AnnData 输出；
6. 基础测试通过 `testdata/pbmc5k_querydata.h5ad`。

### P1

1. 多列联合条件；
2. 多列边际 IPF 分配；
3. 评估报告与相似度门禁；
4. CLI。

### P2

1. ZINB 插件；
2. latent factor 共表达建模；
3. 重试策略和报告 HTML；
4. 性能优化。

验收标准：

```text
- 使用 testdata/pbmc5k_querydata.h5ad 可成功生成 S；
- S.var_names 与 R 完全一致；
- S.obs_names 唯一且均为 source + "_sim"；
- S.X 为非负整数 raw counts；
- 条件配额精确满足；
- 同 seed 可复现；
- 评估报告包含相似度指标；
- 不可行条件、counts 缺失、barcode 冲突均有明确异常。
```

