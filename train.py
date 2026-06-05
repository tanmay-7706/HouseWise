"""
train.py — Train & evaluate five regression models for HouseWise.

Usage:
    python train.py
    python train.py --csv /path/to/housing.csv

Outputs (all written to the script's directory):
    best_model.pkl           — best pipeline (scaler + model)
    scaler.pkl               — fitted StandardScaler
    model_lr.pkl  … model_gb.pkl  — each model individually
    feature_names.json       — ordered feature list
    feature_importance.json  — per-model feature importances
    metrics.json             — evaluation results for all models
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate

from eda import HousingEDA

warnings.filterwarnings("ignore")

DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns and return the augmented DataFrame."""
    df = df.copy()
    df["rooms_per_household"] = df["AveRooms"] / df["AveOccup"].replace(0, np.nan)
    df["bedrooms_ratio"] = df["AveBedrms"] / df["AveRooms"].replace(0, np.nan)
    df["population_per_household"] = df["Population"] / df["AveOccup"].replace(0, np.nan)
    df["income_per_room"] = df["MedInc"] / df["AveRooms"].replace(0, np.nan)
    # Fill any resulting NaN / inf with column median
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)
    return df


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def _evaluate(name, model, X_tr, y_tr, X_te, y_te, n_features) -> dict:
    y_pred = model.predict(X_te)
    rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
    mae = float(mean_absolute_error(y_te, y_pred))
    r2 = float(r2_score(y_te, y_pred))
    n = len(y_te)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)

    cv = cross_val_score(model, X_tr, y_tr, cv=5, scoring="r2")

    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "adjusted_r2": round(adj_r2, 4),
        "cv_r2_mean": round(float(cv.mean()), 4),
        "cv_r2_std": round(float(cv.std()), 4),
        "y_test": y_te.tolist(),
        "y_pred": y_pred.tolist(),
    }


# ---------------------------------------------------------------------------
# Feature importance extraction
# ---------------------------------------------------------------------------

def _feature_importances(model, feature_names: list[str]) -> list[dict]:
    """Return a sorted list of {feature, importance} dicts."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
    else:
        return []
    total = importances.sum() or 1.0
    items = [
        {"feature": f, "importance": round(float(v / total), 6)}
        for f, v in zip(feature_names, importances)
    ]
    items.sort(key=lambda x: x["importance"], reverse=True)
    return items


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(csv_path: str | None = None) -> None:
    print("=" * 60)
    print("  HouseWise — Model Training Pipeline")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    eda = HousingEDA()
    df = eda.load_data(csv_path)
    print(f"\n📂 Dataset shape: {df.shape}")
    print(f"   Missing values: {df.isnull().sum().sum()}")
    ps = eda.price_stats(df)
    print(f"   Price — mean: {ps['mean']:.4f}  median: {ps['median']:.4f}  "
          f"std: {ps['std']:.4f}")

    # ------------------------------------------------------------------
    # 3. Feature engineering
    # ------------------------------------------------------------------
    print("\n🔧 Feature engineering …")
    df = engineer_features(df)

    # ------------------------------------------------------------------
    # 4. Cap outliers at 99th percentile
    # ------------------------------------------------------------------
    cap = df["Price"].quantile(0.99)
    df["Price"] = df["Price"].clip(upper=cap)
    print(f"   Price capped at 99th percentile: {cap:.4f}")

    # ------------------------------------------------------------------
    # 5-6. Scaling & split
    # ------------------------------------------------------------------
    feature_cols = [c for c in df.columns if c != "Price"]
    X = df[feature_cols].values
    y = df["Price"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, DIR / "scaler.pkl")

    with open(DIR / "feature_names.json", "w") as fh:
        json.dump(feature_cols, fh)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=42,
    )
    print(f"   Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ------------------------------------------------------------------
    # 7. Train models
    # ------------------------------------------------------------------
    models: dict[str, object] = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.001, max_iter=10_000),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42,
        ),
    }
    short_names = {
        "Linear Regression": "lr",
        "Ridge Regression": "ridge",
        "Lasso Regression": "lasso",
        "Random Forest": "rf",
        "Gradient Boosting": "gb",
    }

    results: dict[str, dict] = {}
    importances: dict[str, list[dict]] = {}

    for name, mdl in models.items():
        print(f"\n🚀 Training {name} …")
        mdl.fit(X_train, y_train)
        res = _evaluate(name, mdl, X_train, y_train, X_test, y_test,
                        X_train.shape[1])
        results[name] = res
        importances[name] = _feature_importances(mdl, feature_cols)

    # ------------------------------------------------------------------
    # 8. Comparison table
    # ------------------------------------------------------------------
    best_name = max(results, key=lambda k: results[k]["r2"])

    rows = []
    for name, m in results.items():
        tag = " ★" if name == best_name else ""
        rows.append([
            f"{name}{tag}",
            f"{m['r2']:.4f}",
            f"{m['rmse']:.4f}",
            f"{m['mae']:.4f}",
            f"{m['adjusted_r2']:.4f}",
            f"{m['cv_r2_mean']:.4f} ± {m['cv_r2_std']:.4f}",
        ])

    print("\n" + "=" * 60)
    print("  MODEL COMPARISON")
    print("=" * 60)
    print(tabulate(
        rows,
        headers=["Model", "R²", "RMSE", "MAE", "Adj R²", "CV R² (mean ± std)"],
        tablefmt="fancy_grid",
    ))
    print(f"\n🏆 Best model: {best_name} (R² = {results[best_name]['r2']:.4f})")

    # ------------------------------------------------------------------
    # 9. Save artefacts
    # ------------------------------------------------------------------
    for name, mdl in models.items():
        fname = f"model_{short_names[name]}.pkl"
        joblib.dump(mdl, DIR / fname)
        print(f"💾 Saved {fname}")

    joblib.dump(models[best_name], DIR / "best_model.pkl")
    print(f"💾 Saved best_model.pkl  ({best_name})")

    # Strip heavy arrays before saving metrics JSON
    metrics_for_json: dict[str, dict] = {}
    for name, m in results.items():
        metrics_for_json[name] = {k: v for k, v in m.items()
                                  if k not in ("y_test", "y_pred")}

    # Keep y_test/y_pred only for best model (for residual plots)
    best_res = results[best_name]

    # Sample 500 points for the scatter plot to keep JSON manageable
    n_sample = min(500, len(best_res["y_test"]))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(best_res["y_test"]), size=n_sample, replace=False)
    sampled_actual = [best_res["y_test"][i] for i in idx]
    sampled_pred   = [best_res["y_pred"][i] for i in idx]

    payload = {
        "best_model": best_name,
        "dataset_size": len(df),
        "n_features": len(feature_cols),
        "feature_names": feature_cols,
        "price_stats": ps,
        "models": metrics_for_json,
        "best_actual_sample": sampled_actual,
        "best_pred_sample": sampled_pred,
    }
    with open(DIR / "metrics.json", "w") as fh:
        json.dump(payload, fh, indent=2)
    print("💾 Saved metrics.json")

    # ------------------------------------------------------------------
    # 10. Feature importance
    # ------------------------------------------------------------------
    with open(DIR / "feature_importance.json", "w") as fh:
        json.dump(importances, fh, indent=2)
    print("💾 Saved feature_importance.json")

    print("\n✅ Training complete!\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HouseWise models")
    parser.add_argument("--csv", type=str, default=None,
                        help="Optional path to a housing CSV")
    args = parser.parse_args()
    main(csv_path=args.csv)
