# Ejercicio Semana 1 — Qué pide y cómo resolverlo

Working notes for `Ejercicio_Semana_1_Enunciado.pdf`.
Build one reproducible pipeline: NumPy abstraction → vectorized stats → Pandas analysis → Parquet round trip.

- **Modality:** individual · **Suggested time:** 2–3 h
- **Deadline:** 18h30, Tuesday 25 August 2026 (D2L → folder *Ejercicios 1*)
- **Dataset:** `data/server_measurements.csv` — 300 rows, 6 servers × 50 observations
- **Columns:** `server`, `timestamp`, `gpu_utilization`, `cpu_utilization`, `memory_gb`, `power_w`, `temperature_c`

---

## 1. Current state of the repo

```
├── data/server_measurements.csv     ✅ provided (300 rows)
├── src/week1_exercise/
│   ├── __init__.py                  ✅ exports the public API — do not change
│   ├── __main__.py                  ✅ driver already written — do not change
│   └── pipeline.py                  ⬅️  ALL YOUR WORK IS HERE (5 × NotImplementedError)
├── tests/test_public.py             ✅ 9 tests — do not modify (enunciado says so)
├── pyproject.toml                   ✅ numpy≥2.0, pandas≥2.2, pyarrow≥17, pytest≥8
├── uv.lock                          ❌ missing — first `uv sync` generates it
├── git repo                         ❌ not initialized yet
└── PLAN.md                          ⬅️  this file
```

**Everything you must write lives in `src/week1_exercise/pipeline.py`.** The constants at the top are already
given and are the source of truth:

```python
FEATURES = ("gpu_utilization", "cpu_utilization", "memory_gb")
WEIGHTS  = np.array([0.50, 0.30, 0.20], dtype=float)
```

`__main__.py` already wires the whole flow, so its calls define the exact signatures you must honor:

```python
batch                 = build_measurements(df)          # print(batch) → uses __repr__
_, load_score, _      = compute_load_score(batch)       # returns a 3-tuple, score is index 1
analysis              = enrich_dataframe(df, load_score)
summary               = build_summary(analysis)
restored              = save_and_validate_parquet(analysis, output_path)   # must RETURN a DataFrame
```

---

## 2. The five tasks and their real contract

The PDF describes the tasks in prose; `tests/test_public.py` defines them precisely. **Where the two
disagree, the test wins** — it's the grading mechanism. Below is the merged contract.

### Tarea 1 — `ServerMeasurements` + `build_measurements`
*Tests: `test_batch_contract`, `test_invalid_batch_rejected`, `test_feature_count_must_match_columns`,
`test_non_numeric_values_rejected`, `test_batch_repr` (5 tests)*

Attribute names are fixed by the tests: **`.values`** and **`.feature_names`**.

| Requirement | Where it's checked |
|---|---|
| `.values` is a 2-D NumPy array | `batch.values.shape == (len(df), len(FEATURES))` |
| `.values` has a **floating** dtype | `np.issubdtype(batch.values.dtype, np.floating)` |
| `.feature_names` is a **tuple**, equal to `FEATURES` | `isinstance(..., tuple)` |
| 1-D input → `ValueError` | `ServerMeasurements(np.array([1.0, 2.0]), ("a",))` |
| `ncols != len(names)` → `ValueError` | 3 columns vs `("gpu", "cpu")` |
| non-numeric input → `ValueError` | array of strings `["high", "50", "20"]` |
| `__repr__` contains `ServerMeasurements`, `shape=(2, 3)`, and each feature name | substring assertions |

`build_measurements` = select `FEATURES` from the DataFrame → NumPy float matrix → wrap in the class.

### Tarea 2 — `compute_load_score`
*Test: `test_load_score_shapes_and_finiteness` (1 test)*

Returns a 3-tuple **in this order** (per the docstring and `__main__`): `(zscores, load_score, means)`.

- `zscores.shape == (300, 3)`, `load_score.shape == (300,)`, `means.shape == (3,)`
- Z-score per column; guard against division by zero when a column's std is 0
- All values finite
- `zscores.mean(axis=0) ≈ 0` and `zscores.std(axis=0) ≈ 1` — **tolerance is `atol=1e-10`**
- `load_score == zscores @ WEIGHTS` — fully vectorized, **no `for` loop over observations**
- Expected means: `[57.99313333 50.54163333 29.45533333]`

### Tarea 3 — `enrich_dataframe`
*Test: `test_enriched_dataframe_contract` (1 test)*

- Returns a **new** DataFrame; the original must stay untouched → start from `df.copy()`
- Adds `load_score` and `requires_review`
- Rule: `load_score > 1.5` **or** `temperature_c > 80`
- `requires_review` dtype must be exactly **`bool`**, not `object`
- Same row count as the input

### Tarea 4 — `build_summary`
*Test: `test_summary_contract` (1 test)*

Group by `server`, one row per server. Column list and **order** are asserted exactly:

```python
["server", "observations", "mean_power_w", "max_temperature_c", "mean_load", "review_count"]
```

- `observations` = row count per server
- `mean_power_w` = mean of `power_w` · `max_temperature_c` = max of `temperature_c`
- `mean_load` = mean of `load_score` · `review_count` = count of `requires_review == True`
- `set(summary["server"])` must equal the set of servers in the source (all 6 present)

### Tarea 5 — `save_and_validate_parquet`
*Test: `test_parquet_roundtrip` (1 test)*

Write to `output/server_analysis.parquet` via PyArrow/Parquet, read it back, verify, **and return the
restored DataFrame** (`__main__` does `len(restored)`).

The test compares your file against `pa.Table.from_pandas(out, preserve_index=False)` and asserts
`num_rows`, `column_names`, `schema`, and full `.equals()`. So **write without the pandas index** —
`preserve_index=False` (or `to_parquet(..., index=False)`). Verify at minimum: rows before == rows after,
columns before == columns after, schema before == schema after, values equal.

---

## 3. Traps worth knowing before you start

1. **The PDF prose contradicts the test on column names.** Tarea 4's paragraph says `mean_power` and
   `max_temperature`, but both the PDF's own sample output *and* `test_summary_contract` require
   **`mean_power_w`** and **`max_temperature_c`**. Use the suffixed names.
2. **`observations ` in the prose has a stray trailing space.** The test wants `"observations"` clean.
3. **Order your validations: `ndim` first, then column count.** `test_invalid_batch_rejected` passes a 1-D
   array *with* one name. If you check `values.shape[1] != len(names)` first, you get `IndexError`, but the
   test demands `ValueError`.
4. **Detect non-numeric by dtype, not by try/except.** The test passes `np.array([["high","50","20"], ...])`,
   whose dtype is `<U4`. Check `np.issubdtype(values.dtype, np.number)` and raise `ValueError`.
5. **Use population std (`ddof=0`)** — NumPy's default. `ddof=1` drifts far outside the `atol=1e-10` window.
6. **`requires_review` must be real `bool`.** Combining conditions can yield `object` dtype; finish with
   `.astype(bool)`. Use `|` (not `or`) for the element-wise OR, and parenthesize each comparison.
7. **`output/` does not exist.** `save_and_validate_parquet` must `mkdir(parents=True, exist_ok=True)` on the
   parent, otherwise `uv run week1-exercise` fails even though pytest passes (the test uses `tmp_path`).
8. **`.gitignore` ignores `output/`, but the deliverable requires the generated Parquet file.** Resolve this
   before zipping — either force-add it (`git add -f output/server_analysis.parquet`) or drop that
   `.gitignore` line. Easy way to lose points.
9. **The `23` index in the Tarea 4 sample output is a PDF copy-paste artifact.** A 6-row per-server summary
   cannot have index 23. Ignore it; only the columns and values matter.
10. **`__main__.py` prints "Parquet verificado: ..."** while the PDF shows "Round trip verificado: ... /
    Filas: True / ...". The tests don't check stdout. If you want the PDF's exact output, print it from
    inside `save_and_validate_parquet`; don't edit the scaffold's contract.

---

## 4. Gameplan

### Step 0 — Environment and Git (10 min)
```bash
uv sync --group dev          # generates uv.lock
git init
git add .
git commit -m "Inicializa ejercicio de semana 1 con estructura y dataset"
uv run pytest -q             # baseline: 9 failed (NotImplementedError) — expected
```

### Step 1 — Tarea 1 (~35 min)
Implement `ServerMeasurements.__init__` (validate: ndim → dtype numeric → column/name count, in that order),
`__repr__` (`f"ServerMeasurements(shape={...}, features={...})"`), then `build_measurements`
(`df[list(FEATURES)].to_numpy(dtype=float)`).
```bash
uv run pytest -v -k "batch or feature_count or non_numeric"   # → 5 passed, 4 deselected
```

### Step 2 — Tarea 2 (~25 min)
Column means and stds with `axis=0`; replace zero stds with 1 before dividing (`np.where`); `zscores @ WEIGHTS`.
Return `(zscores, load_score, means)`.
```bash
uv run pytest -v -k "load_score"                              # → 1 passed, 8 deselected
```

### Step 3 — Tarea 3 (~15 min)
`out = df.copy()`, assign both columns, `.astype(bool)` on the flag.
```bash
uv run pytest -v -k "enriched_dataframe"                      # → 1 passed, 8 deselected
```

**Commit checkpoint** (satisfies the "at least two commits" requirement):
```bash
git commit -am "Implementa ServerMeasurements, load_score y DataFrame enriquecido"
```

### Step 4 — Tarea 4 (~20 min)
`groupby("server")` with a named aggregation, then `reset_index()` and reorder to the exact column list.
`review_count` is just the sum of the boolean column.
```bash
uv run pytest -v -k "summary"                                 # → 1 passed, 8 deselected
```

### Step 5 — Tarea 5 (~25 min)
`mkdir` the parent → `pa.Table.from_pandas(df, preserve_index=False)` → `pq.write_table` → `pq.read_table`
→ compare rows / column names / schema / `.equals()` → return `restored.to_pandas()`.
```bash
uv run pytest -v -k "parquet"                                 # → 1 passed, 8 deselected
```

### Step 6 — Full verification
```bash
uv run pytest -v            # → 9 passed
uv run week1-exercise       # → prints the batch repr, the summary, creates output/server_analysis.parquet
```
Cross-check against the PDF: repr is `ServerMeasurements(shape=(300, 3), features=('gpu_utilization',
'cpu_utilization', 'memory_gb'))` and means are `[57.99313333 50.54163333 29.45533333]`.

### Step 7 — Package for submission
```bash
git commit -am "Completa resumen por servidor y round trip Parquet"
git log --oneline > git_history.txt
```

---

## 5. Submission checklist

- [ ] `uv run pytest -v` → **9 passed**
- [ ] `uv run week1-exercise` runs clean and produces `output/server_analysis.parquet`
- [ ] At least **2 commits** with descriptive messages
- [ ] `git_history.txt` generated **last**, after the final commit
- [ ] Zip contains: source code, `pyproject.toml`, **`uv.lock`**, `git_history.txt`,
      `data/server_measurements.csv`, and **`output/server_analysis.parquet`** (watch trap #8)
- [ ] Zip excludes: `.venv/`, `__pycache__/`, `.pytest_cache/`, `.DS_Store`
- [ ] Uploaded to D2L → *Ejercicios 1* before 18h30, 25 Aug 2026
