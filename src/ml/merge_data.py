"""
Module 2 - Step 1: Merge car + bike quotation CSVs into one combined dataset,
with a vehicle_type column so a single model can learn from both.
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"


def main():
    car = pd.read_csv(DATASETS_DIR / "acko_car_quotation.csv")
    bike = pd.read_csv(DATASETS_DIR / "acko_bike_quotation.csv")

    car["vehicle_type"] = "car"
    bike["vehicle_type"] = "bike"

    combined = pd.concat([car, bike], ignore_index=True, sort=False)

    out_path = DATASETS_DIR / "combined_quotation.csv"
    combined.to_csv(out_path, index=False)

    print(f"Combined shape: {combined.shape}")
    print(f"vehicle_type counts:\n{combined['vehicle_type'].value_counts()}")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()