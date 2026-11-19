"""Load, standardize, validate and split the real crop-yield dataset."""
from pathlib import Path
import re
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "crop_yield.csv"
TARGET = "Yield"
NUMERIC_FEATURES = ["Crop_Year", "Annual_Rainfall", "Area", "Pesticide", "Fertilizer"]
CATEGORICAL_FEATURES = ["Crop", "Season", "State"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

ALIASES = {
    "crop": "Crop", "season": "Season", "crop_year": "Crop_Year",
    "year": "Crop_Year", "state": "State", "annual_rainfall": "Annual_Rainfall",
    "rainfall": "Annual_Rainfall", "area": "Area", "pesticide": "Pesticide",
    "fertilizer": "Fertilizer", "yield": "Yield", "crop_yield": "Yield",
    "production": "Production"
}

def canonical_name(name):
    key = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
    return ALIASES.get(key, str(name).strip())

def load_dataset(path=DATA_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Real dataset not found at {path}. Download the Mendeley dataset, "
            "extract it and save the observations CSV as data/crop_yield.csv. "
            "See data/README.md. No synthetic fallback is used."
        )
    df = pd.read_csv(path)
    df.columns = [canonical_name(c) for c in df.columns]
    required = set(FEATURES + [TARGET])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}. "
                         f"Found columns: {list(df.columns)}")
    # Keep only documented model fields. Production is intentionally excluded.
    df = df[FEATURES + [TARGET]].copy()
    for col in NUMERIC_FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    df = df.replace([float("inf"), float("-inf")], pd.NA)
    # Rows with no target cannot train a supervised regressor.
    df = df.dropna(subset=[TARGET])
    # Physically impossible/non-useful target and area values are removed.
    df = df[(df[TARGET] >= 0) & (df["Area"] > 0)]
    if len(df) < 100:
        raise ValueError(f"Only {len(df)} usable rows remain. Check the selected CSV.")
    return df.reset_index(drop=True)

def split_dataset(df, random_state=42):
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.20, random_state=random_state)
