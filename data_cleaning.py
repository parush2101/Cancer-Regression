"""
Data Cleaning for Cancer Regression Dataset
Target: target_deathrate (mean per-capita cancer mortality per 100k, by US county)
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
    # Leakage: raw death count is essentially the numerator of target_deathrate
    "avgdeathsperyear":        "leakage — raw death count directly computes target rate",
    # Redundancy / noise
    "avganncount":             "raw diagnosis count; incidencerate (per-100k rate) is better",
    "binnedinc":               "categorical bin of medincome — redundant",
    "pctmarriedhouseholds":    "r>0.97 correlation with percentmarried — duplicate",
    "pctempprivcoverage":      "subset of pctprivatecoverage — redundant",
    # Missing data
    "pctsomecol18_24":         "75% missing — not salvageable",
    "pctprivatecoveragealone": "20% missing and redundant with pctprivatecoverage",
    # Corrupted
    "medianage":               "max=624 — data entry error; use medianagemale/female instead",
    # Non-numeric identifier
    "geography":               "county/state string identifier — no ordinal meaning",
}

print("=== Columns Dropped ===")
for col, reason in DROP.items():
    print(f"  - {col}: {reason}")

# ── 3. Build working dataset (drop unwanted columns) ─────────────────────────
def to_float(val):
    try:
        return float(val.strip()) if val.strip() != "" else None
    except ValueError:
        return None

kept_cols = [c for c in raw[0].keys() if c not in DROP]
data = []
for row in raw:
    record = {c: to_float(row[c]) for c in kept_cols}
    data.append(record)

print(f"\nAfter dropping: {len(kept_cols)} columns remain\n")

# ── 4. Missing value audit on kept columns ────────────────────────────────────
print("=== Missing Values in Kept Columns ===")
any_missing = False
for col in kept_cols:
    n_missing = sum(1 for r in data if r[col] is None)
    if n_missing:
        print(f"  {col}: {n_missing} missing ({100*n_missing/len(data):.1f}%)")
        any_missing = True
if not any_missing:
    print("  None (except pctemployed16_over below)")

# ── 5. Impute pctemployed16_over with median (5% missing) ────────────────────
col = "pctemployed16_over"
valid_vals = [r[col] for r in data if r[col] is not None]
median_val = statistics.median(valid_vals)
imputed = sum(1 for r in data if r[col] is None)
for r in data:
    if r[col] is None:
        r[col] = median_val
print(f"\n=== Imputation ===")
print(f"  {col}: {imputed} values filled with median ({median_val:.2f})")

# ── 6. Outlier handling ───────────────────────────────────────────────────────
print("\n=== Outlier Handling ===")

# target_deathrate: flag extreme outliers (>3 IQR) but keep — they are real counties
target_vals = [r["target_deathrate"] for r in data]
q1 = sorted(target_vals)[int(0.25 * len(target_vals))]
q3 = sorted(target_vals)[int(0.75 * len(target_vals))]
iqr = q3 - q1
lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
outliers = [r for r in data if r["target_deathrate"] < lower or r["target_deathrate"] > upper]
print(f"  target_deathrate: {len(outliers)} extreme outliers (3×IQR rule) — retained (real counties)")

# popest2015 and studypercap: log1p transform to reduce right skew
print("\n=== Log1p Transforms (right-skewed features) ===")
for col in ["popest2015", "studypercap", "avganncount_removed"]:
    if col == "avganncount_removed":
        continue
for col in ["popest2015", "studypercap"]:
    orig_vals = [r[col] for r in data]
    orig_mean = statistics.mean(orig_vals)
    orig_std  = statistics.stdev(orig_vals)
    for r in data:
        r[col] = math.log1p(r[col])
    new_vals = [r[col] for r in data]
    new_mean = statistics.mean(new_vals)
    new_std  = statistics.stdev(new_vals)
    print(f"  {col}: mean {orig_mean:.1f}→{new_mean:.3f}, std {orig_std:.1f}→{new_std:.3f}")

# ── 7. Final shape & quick stats on target ────────────────────────────────────
target = [r["target_deathrate"] for r in data]
print(f"\n=== Final Dataset ===")
print(f"  Rows   : {len(data)}")
print(f"  Columns: {len(kept_cols)}")
print(f"\n  target_deathrate  min={min(target):.2f}  max={max(target):.2f}  "
      f"mean={statistics.mean(target):.2f}  std={statistics.stdev(target):.2f}")

# ── 8. Write cleaned CSV ──────────────────────────────────────────────────────
out_path = "cancer_reg_clean.csv"
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=kept_cols)
    writer.writeheader()
    writer.writerows(data)

print(f"\nCleaned data written to: {out_path}")

# ── 9. Save cleaning summary as JSON (for reproducibility) ───────────────────
summary = {
    "original_shape": {"rows": len(raw), "cols": len(raw[0])},
    "final_shape": {"rows": len(data), "cols": len(kept_cols)},
    "dropped_columns": DROP,
    "imputed": {
        "pctemployed16_over": {"method": "median", "value": round(median_val, 4), "n_filled": imputed}
    },
    "log1p_transformed": ["popest2015", "studypercap"],
    "outliers_retained": {
        "target_deathrate": {"rule": "3xIQR", "n_flagged": len(outliers), "action": "retained"}
    },
    "kept_columns": kept_cols
}

with open("cleaning_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("Cleaning summary written to: cleaning_summary.json")
