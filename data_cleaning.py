"""
Data Cleaning for Cancer Regression Dataset
Target: target_deathrate (mean per-capita cancer mortality per 100k, by US county)

Notes on variable selection:
- avgdeathsperyear / avganncount: 2009-2013 lagged predictors for a ~2015 target.
  Using lags of the DV and related counts is standard practice — not leakage.
- geography: extract state for fixed effects to absorb unobserved state-level variation.
"""

import csv
import math
import statistics
import json

# ── 1. Load raw data ──────────────────────────────────────────────────────────
with open("cancer_reg.csv") as f:
    raw = list(csv.DictReader(f))

print(f"Raw shape: {len(raw)} rows x {len(raw[0])} columns\n")

# ── 2. Columns to DROP and rationale ─────────────────────────────────────────
DROP = {
    # Redundancy
    "binnedinc":               "categorical bin of medincome — redundant",
    "pctmarriedhouseholds":    "near-duplicate of percentmarried (r>0.97)",
    "pctempprivcoverage":      "subset of pctprivatecoverage — redundant",
    # Missing data
    "pctsomecol18_24":         "75% missing — not salvageable",
    "pctprivatecoveragealone": "20% missing and redundant with pctprivatecoverage",
    # Corrupted
    "medianage":               "max=624 — data entry error; use medianagemale/female instead",
    # geography handled separately (state fixed effects extracted before dropping)
}

print("=== Columns Dropped ===")
for col, reason in DROP.items():
    print(f"  - {col}: {reason}")
print(f"  - geography: string identifier — state extracted as fixed effects, raw column dropped")

# ── 3. Extract state fixed effects from geography before dropping ─────────────
print("\n=== State Fixed Effects (from geography) ===")
states = sorted(set(row["geography"].split(", ")[-1].strip() for row in raw))
print(f"  {len(states)} unique states found")

# One-hot encode states (drop first to avoid perfect multicollinearity)
reference_state = states[0]
state_cols = [f"state_{s.replace(' ', '_')}" for s in states[1:]]
print(f"  Reference state (dropped): {reference_state}")
print(f"  Dummy columns added: {len(state_cols)}")

# ── 4. Build working dataset ──────────────────────────────────────────────────
def to_float(val):
    try:
        return float(val.strip()) if val.strip() != "" else None
    except ValueError:
        return None

numeric_cols = [c for c in raw[0].keys() if c not in DROP and c != "geography"]
data = []
for row in raw:
    record = {c: to_float(row[c]) for c in numeric_cols}
    # Add state dummies
    state = row["geography"].split(", ")[-1].strip()
    for s in states[1:]:
        record[f"state_{s.replace(' ', '_')}"] = 1.0 if state == s else 0.0
    data.append(record)

all_cols = numeric_cols + state_cols
print(f"\nAfter processing: {len(all_cols)} columns ({len(numeric_cols)} numeric + {len(state_cols)} state dummies)")

# ── 5. Missing value audit ────────────────────────────────────────────────────
print("\n=== Missing Values in Numeric Columns ===")
any_missing = False
for col in numeric_cols:
    n_missing = sum(1 for r in data if r[col] is None)
    if n_missing:
        print(f"  {col}: {n_missing} missing ({100*n_missing/len(data):.1f}%)")
        any_missing = True
if not any_missing:
    print("  None (except pctemployed16_over below)")

# ── 6. Impute pctemployed16_over with median (5% missing) ────────────────────
col = "pctemployed16_over"
valid_vals = [r[col] for r in data if r[col] is not None]
median_val = statistics.median(valid_vals)
imputed = sum(1 for r in data if r[col] is None)
for r in data:
    if r[col] is None:
        r[col] = median_val
print(f"\n=== Imputation ===")
print(f"  {col}: {imputed} values filled with median ({median_val:.2f})")

# ── 7. Outlier handling ───────────────────────────────────────────────────────
print("\n=== Outlier Handling ===")
target_vals = sorted(r["target_deathrate"] for r in data)
q1 = target_vals[int(0.25 * len(target_vals))]
q3 = target_vals[int(0.75 * len(target_vals))]
iqr = q3 - q1
lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
outliers = [r for r in data if r["target_deathrate"] < lower or r["target_deathrate"] > upper]
print(f"  target_deathrate: {len(outliers)} extreme outliers (3×IQR) — retained (real counties)")

# ── 8. Log1p transforms for right-skewed features ────────────────────────────
print("\n=== Log1p Transforms (right-skewed features) ===")
for col in ["popest2015", "studypercap", "avganncount", "avgdeathsperyear"]:
    orig_vals = [r[col] for r in data]
    orig_mean = statistics.mean(orig_vals)
    orig_std  = statistics.stdev(orig_vals)
    for r in data:
        r[col] = math.log1p(r[col])
    new_vals = [r[col] for r in data]
    new_mean = statistics.mean(new_vals)
    new_std  = statistics.stdev(new_vals)
    print(f"  {col}: mean {orig_mean:.1f}→{new_mean:.3f}, std {orig_std:.1f}→{new_std:.3f}")

# ── 9. Final stats ────────────────────────────────────────────────────────────
target = [r["target_deathrate"] for r in data]
print(f"\n=== Final Dataset ===")
print(f"  Rows   : {len(data)}")
print(f"  Columns: {len(all_cols)}  ({len(numeric_cols)} numeric + {len(state_cols)} state FE dummies)")
print(f"\n  target_deathrate  min={min(target):.2f}  max={max(target):.2f}  "
      f"mean={statistics.mean(target):.2f}  std={statistics.stdev(target):.2f}")

# ── 10. Write cleaned CSV ─────────────────────────────────────────────────────
out_path = "cancer_reg_clean.csv"
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=all_cols)
    writer.writeheader()
    writer.writerows(data)
print(f"\nCleaned data written to: {out_path}")

# ── 11. Save cleaning summary ─────────────────────────────────────────────────
summary = {
    "original_shape": {"rows": len(raw), "cols": len(raw[0])},
    "final_shape": {"rows": len(data), "cols": len(all_cols)},
    "dropped_columns": DROP,
    "geography_handling": {
        "action": "extracted state as fixed effects (one-hot, reference dropped)",
        "n_states": len(states),
        "reference_state": reference_state,
        "n_dummies_added": len(state_cols)
    },
    "retained_with_correction": {
        "avgdeathsperyear": "lagged predictor (2009-2013 avg vs ~2015 target) — valid",
        "avganncount":      "lagged predictor (2009-2013 avg vs ~2015 target) — valid"
    },
    "imputed": {
        "pctemployed16_over": {"method": "median", "value": round(median_val, 4), "n_filled": imputed}
    },
    "log1p_transformed": ["popest2015", "studypercap", "avganncount", "avgdeathsperyear"],
    "outliers_retained": {
        "target_deathrate": {"rule": "3xIQR", "n_flagged": len(outliers), "action": "retained"}
    },
    "kept_numeric_columns": numeric_cols,
    "state_fe_columns": state_cols
}
with open("cleaning_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Cleaning summary written to: cleaning_summary.json")
