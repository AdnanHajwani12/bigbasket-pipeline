# src/bigbasket/data.py
import pandas as pd
from typing import Optional

def load_data(path: str) -> pd.DataFrame:
    """Load dataset (Excel). Raise FileNotFoundError if missing."""
    return pd.read_excel(path)

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create/normalize discount_pct and rating columns used by the pipeline.
    This function is defensive: it handles a few possible column schemas.
    """
    df = df.copy()

    # create discount_pct robustly
    if {"price", "discounted_price"}.issubset(df.columns):
        df["discount_pct"] = (df["price"] - df["discounted_price"]) / df["price"] * 100
    elif "discount" in df.columns:
        df["discount_pct"] = df["discount"].astype(float)
    else:
        # fallback: if there's no discount info, create zeros
        df["discount_pct"] = 0.0

    # rating column fallback
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    elif "product_rating" in df.columns:
        df["rating"] = pd.to_numeric(df["product_rating"], errors="coerce")
    else:
        # If no rating, fill NaNs (some experiments require rating)
        df["rating"] = df.get("rating", pd.NA)

    # Drop entries without rating
    df = df.dropna(subset=["rating"])
    return df