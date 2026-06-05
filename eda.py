"""
eda.py — Exploratory Data Analysis helpers for HouseWise.

Provides a HousingEDA class that wraps the California Housing dataset
and exposes convenience methods for stats, distributions, outlier
detection, and geographical aggregation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing


class HousingEDA:
    """Stateless helper that loads data once and returns analysis artefacts."""

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    @staticmethod
    def load_data(csv_path: str | None = None) -> pd.DataFrame:
        """
        Return the California Housing dataset as a tidy DataFrame.

        If *csv_path* is provided and the file exists, load from CSV instead
        (expects the same feature columns + a 'Price' target column).
        """
        if csv_path is not None:
            from pathlib import Path

            p = Path(csv_path)
            if p.exists():
                return pd.read_csv(p)

        housing = fetch_california_housing(as_frame=True)
        df = housing.frame.copy()                        # type: ignore[union-attr]
        df.rename(columns={"MedHouseVal": "Price"}, inplace=True)
        return df

    # ------------------------------------------------------------------
    # Descriptive statistics
    # ------------------------------------------------------------------

    @staticmethod
    def basic_stats(df: pd.DataFrame) -> dict:
        """Shape, missing values, dtypes, and describe() output."""
        return {
            "shape": df.shape,
            "missing_values": df.isnull().sum().to_dict(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "describe": df.describe().to_dict(),
        }

    @staticmethod
    def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
        """Pearson correlations for all numeric columns."""
        return df.select_dtypes(include="number").corr()

    @staticmethod
    def feature_distributions(df: pd.DataFrame) -> pd.DataFrame:
        """Per-column summary: mean, std, min, 25 %, 50 %, 75 %, max, skew."""
        stats = df.describe().T
        stats["skew"] = df.skew(numeric_only=True)
        return stats

    # ------------------------------------------------------------------
    # Outlier detection (IQR)
    # ------------------------------------------------------------------

    @staticmethod
    def outlier_detection(df: pd.DataFrame) -> dict[str, int]:
        """Return the count of IQR-based outliers per numeric column."""
        counts: dict[str, int] = {}
        for col in df.select_dtypes(include="number").columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            counts[col] = int(((df[col] < lower) | (df[col] > upper)).sum())
        return counts

    # ------------------------------------------------------------------
    # Geographical helpers
    # ------------------------------------------------------------------

    @staticmethod
    def geographical_stats(
        df: pd.DataFrame, lat_bins: int = 10, lon_bins: int = 10,
    ) -> pd.DataFrame:
        """
        Bin Latitude/Longitude and compute mean price per region.
        """
        tmp = df.copy()
        tmp["lat_bin"] = pd.cut(tmp["Latitude"], bins=lat_bins)
        tmp["lon_bin"] = pd.cut(tmp["Longitude"], bins=lon_bins)
        return (
            tmp.groupby(["lat_bin", "lon_bin"], observed=False)["Price"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "avg_price", "count": "num_properties"})
        )

    # ------------------------------------------------------------------
    # Price helpers
    # ------------------------------------------------------------------

    @staticmethod
    def price_stats(df: pd.DataFrame) -> dict:
        """Key price statistics."""
        prices = df["Price"]
        return {
            "min": round(float(prices.min()), 4),
            "max": round(float(prices.max()), 4),
            "mean": round(float(prices.mean()), 4),
            "median": round(float(prices.median()), 4),
            "std": round(float(prices.std()), 4),
            "p25": round(float(prices.quantile(0.25)), 4),
            "p75": round(float(prices.quantile(0.75)), 4),
            "p90": round(float(prices.quantile(0.90)), 4),
            "p99": round(float(prices.quantile(0.99)), 4),
        }


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    eda = HousingEDA()
    df = eda.load_data()
    print(f"Shape: {df.shape}")
    print(f"\nPrice stats:\n{eda.price_stats(df)}")
    print(f"\nOutlier counts:\n{eda.outlier_detection(df)}")
