"""
Module 2 - Premium Quote Predictor: feature engineering + model training.

Trains ONE combined regressor (car + bike, with vehicle_type as a feature)
on the merged quotation dataset, evaluates it, and saves the trained
pipeline + a SHAP summary plot.
"""

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # no display needed - just save the plot to a file
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "datasets" / "combined_quotation.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "quote_predictor.pkl"
SHAP_PLOT_PATH = PROJECT_ROOT / "models" / "shap_summary.png"

# --- Feature configuration ---
NUMERIC_FEATURES = [
    "customer_age", "city_tier", "city_risk_score", "manufacturing_year",
    "vehicle_age_years", "engine_cc", "idv", "ncb_percent",
    "claim_history_count", "num_addons",
]
CATEGORICAL_FEATURES = ["vehicle_type", "vehicle_make", "segment", "fuel_type", "policy_type"]
TARGET = "annual_premium"

# Columns that are algebraically/near-perfectly derived from the target -
# NEVER include these as features, or the model "cheats" instead of learning
# real pricing drivers. (See gst_amount's 0.9995 correlation with the target.)
LEAKAGE_COLUMNS = [
    "od_premium_before_ncb", "ncb_discount_amount", "tp_premium",
    "addon_premium", "gst_amount",
]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def build_pipeline() -> Pipeline:
    """Preprocessing (encode categoricals) + RandomForest regressor, as one pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=16,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", model),
    ])
    return pipeline


def train_and_evaluate(df: pd.DataFrame):
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    # annual_premium is heavily right-skewed (mean >> median) - predict in
    # log space so the model isn't dominated by a handful of very expensive
    # policies, then convert back to rupees for evaluation.
    y_log = np.log1p(y)

    X_train, X_test, y_train_log, y_test_log, y_train, y_test = train_test_split(
        X, y_log, y, test_size=0.2, random_state=42
    )

    print(f"\nTrain size: {len(X_train):,}  |  Test size: {len(X_test):,}")

    pipeline = build_pipeline()
    print("\nTraining RandomForestRegressor...")
    pipeline.fit(X_train, y_train_log)

    # Predict in log space, convert back to rupees
    y_pred_log = pipeline.predict(X_test)
    y_pred = np.expm1(y_pred_log)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100

    print("\n=== Evaluation (on actual rupee scale) ===")
    print(f"R2 score:  {r2:.4f}   (target: > 0.85)")
    print(f"RMSE:      Rs.{rmse:,.2f}   (target: within Rs.2,000 - "
          f"expect this to be far higher due to the wide Rs.500-Rs.16L premium range; see MAPE below)")
    print(f"MAPE:      {mape:.2f}%   (typical prediction is off by this % of the true premium)")

    return pipeline, X_train, X_test, y_test, y_pred


def save_model(pipeline: Pipeline):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


def explain_with_shap(pipeline: Pipeline, X_test: pd.DataFrame, sample_size: int = 500):
    """Generate and save a SHAP summary plot showing which features drove
    predictions, and how (matches the project's explainability requirement).
    """
    print("\nGenerating SHAP explanations (this can take a minute)...")

    preprocessor = pipeline.named_steps["preprocessor"]
    regressor = pipeline.named_steps["regressor"]

    X_sample = X_test.sample(min(sample_size, len(X_test)), random_state=42)
    X_sample_transformed = preprocessor.transform(X_sample)

    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(regressor)
    shap_values = explainer.shap_values(X_sample_transformed)

    plt.figure()
    shap.summary_plot(
        shap_values, X_sample_transformed, feature_names=feature_names, show=False
    )
    plt.tight_layout()
    SHAP_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(SHAP_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP summary plot saved to {SHAP_PLOT_PATH}")


def main():
    print("=== Module 2: Premium Quote Predictor - Training ===\n")

    df = load_data()

    missing_leakage = [c for c in LEAKAGE_COLUMNS if c not in df.columns]
    if missing_leakage:
        print(f"Note: expected leakage columns not found: {missing_leakage}")
    else:
        print(f"Confirmed excluding leakage columns: {LEAKAGE_COLUMNS}")

    pipeline, X_train, X_test, y_test, y_pred = train_and_evaluate(df)
    save_model(pipeline)
    explain_with_shap(pipeline, X_test)

    print("\n=== Training complete ===")


if __name__ == "__main__":
    main()