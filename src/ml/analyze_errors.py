"""
Module 2 - Diagnostic: break down prediction error by premium price band.

Loads the already-trained model, re-runs it on the test split, and shows
average error separately for cheap vs expensive policies - this proves
concretely why one global RMSE number is misleading on this data, and
what the error actually looks like at a given price level.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "datasets" / "combined_quotation.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "quote_predictor.pkl"

NUMERIC_FEATURES = [
    "customer_age", "city_tier", "city_risk_score", "manufacturing_year",
    "vehicle_age_years", "engine_cc", "idv", "ncb_percent",
    "claim_history_count", "num_addons",
]
CATEGORICAL_FEATURES = ["vehicle_type", "vehicle_make", "segment", "fuel_type", "policy_type"]
TARGET = "annual_premium"


def main():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    pipeline = joblib.load(MODEL_PATH)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    y_log = np.log1p(y)

    X_train, X_test, y_train_log, y_test_log, y_train, y_test = train_test_split(
        X, y_log, y, test_size=0.2, random_state=42
    )

    y_pred = np.expm1(pipeline.predict(X_test))

    results = pd.DataFrame({
        "actual": y_test.values,
        "predicted": y_pred,
    })
    results["abs_error"] = (results["actual"] - results["predicted"]).abs()
    results["pct_error"] = results["abs_error"] / results["actual"] * 100

    bands = [
        (0, 5000, "Under Rs.5,000 (cheap policies)"),
        (5000, 20000, "Rs.5,000 - 20,000 (typical policies)"),
        (20000, 100000, "Rs.20,000 - 1,00,000 (higher-end)"),
        (100000, float("inf"), "Above Rs.1,00,000 (luxury/high-IDV)"),
    ]

    print("=== Error by premium price band ===\n")
    print(f"{'Band':<38} {'Count':>8} {'Avg Actual':>12} {'Avg Abs Error':>14} {'Avg % Error':>12}")
    for lo, hi, label in bands:
        subset = results[(results["actual"] >= lo) & (results["actual"] < hi)]
        if len(subset) == 0:
            continue
        print(f"{label:<38} {len(subset):>8,} Rs.{subset['actual'].mean():>9,.0f} "
              f"Rs.{subset['abs_error'].mean():>11,.0f} {subset['pct_error'].mean():>11.2f}%")

    print(f"\nOverall MAPE: {results['pct_error'].mean():.2f}%")
    print(f"Overall RMSE: Rs.{np.sqrt((results['abs_error']**2).mean()):,.2f}")

    print("\n=== Worked example: a ~Rs.1,000 policy ===")
    near_1000 = results[(results["actual"] >= 900) & (results["actual"] <= 1100)]
    if len(near_1000) > 0:
        print(f"Rows with actual premium ~Rs.1,000: {len(near_1000)}")
        print(f"Average absolute error for these: Rs.{near_1000['abs_error'].mean():,.2f}")
        print(f"Average % error for these: {near_1000['pct_error'].mean():.2f}%")
        print(f"\nSample rows:")
        print(near_1000[["actual", "predicted", "abs_error", "pct_error"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()