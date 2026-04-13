"""
OLS, Ridge, and Lasso regression on cancer_reg_clean.csv
Target: target_deathrate
Evaluation: 5-fold cross-validated RMSE, MAE, R²
Tuning: Ridge and Lasso alpha selected via inner CV (RidgeCV / LassoCV)
"""

import csv
import math
import json
import statistics

import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── 1. Load cleaned data ──────────────────────────────────────────────────────
with open("cancer_reg_clean.csv") as f:
    rows = list(csv.DictReader(f))

feature_cols = [c for c in rows[0].keys() if c != "target_deathrate"]
X = np.array([[float(r[c]) for c in feature_cols] for r in rows])
y = np.array([float(r["target_deathrate"]) for r in rows])

print(f"Dataset: {X.shape[0]} rows x {X.shape[1]} features\n")

# ── 2. Cross-validation setup ─────────────────────────────────────────────────
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Alpha grid for Ridge and Lasso inner CV
alphas_ridge = np.logspace(-3, 5, 100)
alphas_lasso = np.logspace(-3, 3, 100)

# ── 3. Define pipelines (StandardScaler + model) ──────────────────────────────
# Scaling is essential for Ridge and Lasso so regularisation is applied fairly;
# applied to OLS too for consistency.
models = {
    "OLS":   Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    "Ridge": Pipeline([("scaler", StandardScaler()), ("model", RidgeCV(alphas=alphas_ridge, cv=5))]),
    "Lasso": Pipeline([("scaler", StandardScaler()), ("model", LassoCV(alphas=alphas_lasso, cv=5, max_iter=10000, random_state=42))]),
}

# ── 4. 5-fold CV for each model ───────────────────────────────────────────────
scoring = {
    "rmse": "neg_root_mean_squared_error",
    "mae":  "neg_mean_absolute_error",
    "r2":   "r2",
}

results = {}
print("Running 5-fold cross-validation...\n")

for name, pipe in models.items():
    cv_res = cross_validate(pipe, X, y, cv=kf, scoring=scoring, return_train_score=True)

    cv_rmse  = -cv_res["test_rmse"].mean()
    cv_mae   = -cv_res["test_mae"].mean()
    cv_r2    = cv_res["test_r2"].mean()
    tr_rmse  = -cv_res["train_rmse"].mean()
    tr_r2    = cv_res["train_r2"].mean()

    # Fit on full data to get tuned alpha and non-zero coef count
    pipe.fit(X, y)
    model = pipe.named_steps["model"]

    alpha      = round(float(model.alpha_), 6) if hasattr(model, "alpha_") else None
    coef       = model.coef_
    n_nonzero  = int(np.sum(np.abs(coef) > 1e-8))
    y_pred_full = pipe.predict(X)
    full_r2    = r2_score(y, y_pred_full)
    full_rmse  = math.sqrt(mean_squared_error(y, y_pred_full))

    results[name] = {
        "cv_rmse":   round(cv_rmse, 4),
        "cv_mae":    round(cv_mae, 4),
        "cv_r2":     round(cv_r2, 4),
        "train_rmse": round(tr_rmse, 4),
        "train_r2":  round(tr_r2, 4),
        "full_r2":   round(full_r2, 4),
        "full_rmse": round(full_rmse, 4),
        "best_alpha": alpha,
        "n_nonzero_coefs": n_nonzero,
    }

    print(f"{'─'*50}")
    print(f"  {name}")
    print(f"  CV RMSE  : {cv_rmse:.4f}   (train: {tr_rmse:.4f})")
    print(f"  CV MAE   : {cv_mae:.4f}")
    print(f"  CV R²    : {cv_r2:.4f}   (train: {tr_r2:.4f})")
    if alpha is not None:
        print(f"  Best α   : {alpha}")
    print(f"  Non-zero coefs: {n_nonzero} / {len(coef)}")

# ── 5. Results table ──────────────────────────────────────────────────────────
print("\n")
print("=" * 72)
print(f"{'Model':<10} | {'CV RMSE':>9} | {'CV MAE':>8} | {'CV R²':>7} | {'Train R²':>9} | {'α':>10} | {'Coefs≠0':>8}")
print("=" * 72)
for name, r in results.items():
    alpha_str = f"{r['best_alpha']:.4f}" if r["best_alpha"] is not None else "  N/A"
    print(f"{name:<10} | {r['cv_rmse']:>9.4f} | {r['cv_mae']:>8.4f} | {r['cv_r2']:>7.4f} | "
          f"{r['train_r2']:>9.4f} | {alpha_str:>10} | {r['n_nonzero_coefs']:>8}")
print("=" * 72)

# ── 6. Top 10 coefficients by absolute magnitude (OLS and Ridge on full fit) ──
print("\n=== Top 10 Features by |Coefficient| (full-data fit, scaled) ===\n")
for name in ["OLS", "Ridge", "Lasso"]:
    pipe = models[name]
    coef = pipe.named_steps["model"].coef_
    top10 = sorted(zip(feature_cols, coef), key=lambda x: abs(x[1]), reverse=True)[:10]
    print(f"  {name}:")
    for feat, val in top10:
        print(f"    {feat:<40} {val:>+.4f}")
    print()

# ── 7. Save results ───────────────────────────────────────────────────────────
with open("model_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Results saved to model_results.json")
