# tests/test_models.py
import pandas as pd
from bigbasket.models import train_smoke_models

def test_train_smoke_models_runs():
    # small synthetic dataset
    df = pd.DataFrame({
        "discount_pct": [10, 20, 5, 0, 30, 15],
        "rating": [4.2, 3.8, 4.7, 5.0, 3.2, 4.0],
        "category": ["g","g","s","b","s","g"],
        "brand": ["a","b","a","c","b","a"],
    })
    results = train_smoke_models(df, target_col="rating", test_size=0.3, random_state=0)
    assert "ridge" in results and results["ridge"]["model"] is not None
    assert "rf" in results and results["rf"]["model"] is not None
    assert "xgb" in results  # may be None if XGB not installed in the environment