# Condition Metadata and Export API Design

## Goal

Make generated condition columns, inherited source metadata, multi-column
marginal allocation, and raw-count export explicit and verifiable.

## Source Alignment

Every generated cell already selects a source barcode from the exact target
condition pool. The output condition columns are assigned from that target
group, so a generated `celltype` always corresponds to the source barcode's
reference `celltype`.

When `source_metadata=True`:

- every reference obs column is copied as `source_<column>`;
- selected columns in `GeneratorConfig.inherit_obs_cols` are additionally
  copied under their original names;
- condition and internal output column names cannot be inherited without an
  explicit conflict error.

Example:

```python
GeneratorConfig(
    source_metadata=True,
    inherit_obs_cols=["ident"],
)
```

This produces both `source_ident` and `ident`. The `ident` value is inherited
from the source barcode and is consistent with the selected source barcode.

## Joint and Marginal Conditions

Single-column or explicit joint quotas continue to use exact joint groups.

For multi-column marginal quotas:

- the requested per-column category counts remain exact;
- every joint combination must have enough source cells;
- integer allocation minimizes deviation from the reference joint proportions;
- the resulting actual joint quota remains visible in the report.

The MILP variables are joint cell counts plus nonnegative deviation variables.
The objective minimizes total absolute deviation from a proportional target
joint table while enforcing exact row/column marginals and source capacities.

## Raw Export

Add:

```python
export_raw_counts(
    data,
    output_dir,
    prefix="counts",
    matrix_format="mtx",
    overwrite=False,
) -> ExportArtifacts
```

Outputs:

```text
counts_matrix.mtx   # genes x cells
counts_obs.csv      # cell metadata, rows aligned to matrix columns
counts_var.csv      # gene metadata, rows aligned to matrix rows
```

`matrix_format="npz"` writes SciPy sparse CSR with cells x genes instead.
Export always resolves validated integer raw counts rather than normalized X.

## Validation

- Generated condition columns equal source condition columns.
- Inherited `ident` equals source `source_ident`.
- Barcode, condition, and inherited metadata row order remain aligned.
- Multi-column marginal plans satisfy exact margins and stay near reference
  support proportions.
- Raw exports reload with cell/gene order matching metadata CSVs.

## Validated Result

The full PBMC output now contains every source metadata column, including
`source_ident`, and an inherited `ident` column. `ident == source_ident` and
the controlled `celltype` column equals `source_celltype` for all 30,009
generated cells.

For a real `celltype × ident` marginal request:

```text
celltype: CD14=100, NK3=50
ident:    1=75, 2=75

joint allocation:
CD14/1 = 26
CD14/2 = 74
NK3/1  = 49
NK3/2  = 1
```

Both marginals are exact and the joint allocation follows the reference
combination support.
