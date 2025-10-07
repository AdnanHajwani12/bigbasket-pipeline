# src/bigbasket/train.py
import os
import argparse
import pandas as pd
from pathlib import Path
from bigbasket.data import load_data, prepare_features
from bigbasket.models import train_smoke_models, save_model

def make_synthetic(n=500):
    import numpy as np
    np.random.seed(0)
    categories = ["grocery", "snacks", "beverages"]
    brands = ["brandA", "brandB", "brandC"]
    discount_pct = np.random.uniform(0, 50, size=n)
    rating = 4.0 - 0.02 * discount_pct + np.random.normal(0, 0.3, size=n)
    rating = np.clip(rating, 1, 5)
    import pandas as pd
    return pd.DataFrame({
        "discount_pct": discount_pct,
        "rating": rating,
        "category": np.random.choice(categories, size=n),
        "brand": np.random.choice(brands, size=n)
    })

def main(data_path: str = None, outdir: str = "models"):
    if data_path and Path(data_path).exists():
        df = load_data(data_path)
        df = prepare_features(df)
        # Keep only columns used
        df = df[["discount_pct", "rating", "category", "brand"]]
    else:
        print("No dataset found; generating synthetic data for smoke training.")
        df = make_synthetic(500)

    results = train_smoke_models(df, target_col="rating")
    os.makedirs(outdir, exist_ok=True)
    for name, info in results.items():
        model = info.get("model")
        if model is not None:
            outpath = Path(outdir) / f"{name}_model.joblib"
            save_model(model, outpath)
            print(f"Saved {name} to {outpath} (mse={info.get('mse')})")
        else:
            print(f"{name} not saved: {info.get('note')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="Path to Excel data file", default=None)
    parser.add_argument("--outdir", help="Directory to save models", default="models")
    args = parser.parse_args()
    main(args.data, args.outdir)