# tests/test_data.py
import pandas as pd
from bigbasket.data import prepare_features

def test_prepare_features_basic():
    df = pd.DataFrame({
        "price": [100, 200],
        "discounted_price": [90, 160],
        "rating": [4.5, 3.8],
        "category": ["a","b"],
        "brand": ["x","y"]
    })
    out = prepare_features(df)
    assert "discount_pct" in out.columns
    assert "rating" in out.columns
    assert len(out) == 2
    assert out["discount_pct"].iloc[0] > 0