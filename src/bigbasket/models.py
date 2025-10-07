# src/bigbasket/models.py
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None  # CI can skip XGB if not installed

def _make_pipeline(model):
    cat_cols = ["category", "brand"]  # if present
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ],
        remainder="passthrough",
    )
    return Pipeline([("pre", preprocessor), ("model", model)])

def train_smoke_models(df: pd.DataFrame, target_col="rating", test_size=0.2, random_state=42) -> Dict[str, Dict]:
    """
    Train Ridge, RandomForest, and XGB (if available) on a small dataset (smoke).
    Returns a dict with model objects and metrics.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    results = {}

    # Ridge (simple linear)
    ridge = _make_pipeline(Ridge(random_state=random_state))
    ridge.fit(X_train, y_train)
    preds = ridge.predict(X_test)
    results["ridge"] = {"model": ridge, "mse": mean_squared_error(y_test, preds)}

    # Random Forest
    rf = _make_pipeline(RandomForestRegressor(n_estimators=50, random_state=random_state))
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    results["rf"] = {"model": rf, "mse": mean_squared_error(y_test, preds)}

    # XGBoost (if available)
    if XGBRegressor is not None:
        xgb = _make_pipeline(XGBRegressor(n_estimators=50, random_state=random_state, verbosity=0))
        xgb.fit(X_train, y_train)
        preds = xgb.predict(X_test)
        results["xgb"] = {"model": xgb, "mse": mean_squared_error(y_test, preds)}
    else:
        results["xgb"] = {"model": None, "mse": None, "note": "xgboost not installed"}

    return results

def save_model(pipeline, path: str):
    joblib.dump(pipeline, path)