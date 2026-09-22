"""
Module 2 - Single-quote prediction.

Loads the already-trained pipeline and predicts ONE premium for ONE set
of customer inputs (unlike train_quote_model.py, which evaluates a whole
test batch). This is the function the Streamlit quote form will call.
Every prediction is logged to the quotations table.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.db.database import get_session
from src.db.models import Quotation

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "quote_predictor.pkl"

_pipeline = None  # loaded once, reused across calls (loading a .pkl is slow-ish)


def _load_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline


def predict_quote(inputs: dict, log_to_db: bool = True) -> float:
    """
    inputs must contain exactly these keys (same as NUMERIC_FEATURES +
    CATEGORICAL_FEATURES in train_quote_model.py):

      customer_age, city_tier, city_risk_score, manufacturing_year,
      vehicle_age_years, engine_cc, idv, ncb_percent, claim_history_count,
      num_addons, vehicle_type, vehicle_make, segment, fuel_type, policy_type

    Returns the predicted annual_premium in rupees.
    """
    pipeline = _load_pipeline()

    X = pd.DataFrame([inputs])
    pred_log = pipeline.predict(X)[0]
    predicted_premium = float(np.expm1(pred_log))

    if log_to_db:
        session = get_session()
        try:
            record = Quotation(predicted_premium=predicted_premium, **inputs)
            session.add(record)
            session.commit()
        finally:
            session.close()

    return predicted_premium


if __name__ == "__main__":
    # Quick manual test - one realistic car policy
    sample_input = {
        "customer_age": 35,
        "city_tier": 1,
        "city_risk_score": 0.65,
        "manufacturing_year": 2021,
        "vehicle_age_years": 4,
        "engine_cc": 1197,
        "idv": 450000,
        "ncb_percent": 20,
        "claim_history_count": 0,
        "num_addons": 2,
        "vehicle_type": "car",
        "vehicle_make": "Hyundai",
        "segment": "Hatchback",
        "fuel_type": "Petrol",
        "policy_type": "Comprehensive",
    }
    premium = predict_quote(sample_input)
    print(f"Predicted annual premium: Rs.{premium:,.2f}")
    print("Logged to quotations table.")