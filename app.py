# Exp5_Streamlit_App.py
# Final cleaned version — Streamlit app for Experiment 5 with robust data cleaning

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import io
import shap
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Optional: XGBoost
try:
    from xgboost import XGBRegressor
    has_xgb = True
except Exception:
    has_xgb = False

# LIME
try:
    from lime.lime_tabular import LimeTabularExplainer
    has_lime = True
except Exception:
    has_lime = False

# Fairlearn
try:
    from fairlearn.metrics import MetricFrame
    from sklearn.metrics import mean_absolute_error as mae
    has_fairlearn = True
except Exception:
    has_fairlearn = False

st.set_page_config(page_title="Exp5: Ratings & Discounts — Responsible AI", layout="wide")
st.title("📦 Relationship between Discounts and Product Ratings")

st.markdown(
    """
    This app trains regression models (Ridge, Random Forest, XGBoost) to predict product ratings,
    shows SHAP global explanations, LIME local explanations (if installed), and performs a fairness
    audit across `brand` groups using group-wise MAE and R². The app also helps generate an API/Docker
    snippet for deployment.
    """
)

# -----------------------
# Utilities & Example data
# -----------------------
@st.cache_data
def load_example_data(n=500, seed=42):
    rng = np.random.default_rng(seed)
    brands = ["BrandA", "BrandB", "BrandC", "BrandD"]
    categories = ["Grocery", "PersonalCare", "Beverages"]
    df = pd.DataFrame({
        'product_id': np.arange(n),
        'sale_price': rng.uniform(20, 500, n),
        'market_price': rng.uniform(25, 600, n),
        'discount': rng.uniform(0, 0.5, n),
        'category': rng.choice(categories, n),
        'sub_category': rng.choice(['sub1','sub2','sub3'], n),
        'brand': rng.choice(brands, n),
        'type': rng.choice(['Pack','Loose'], n)
    })
    brand_effect = {'BrandA': 0.3, 'BrandB': -0.1, 'BrandC': 0.0, 'BrandD': 0.1}
    df['rating'] = (
        4.0 - 0.002 * df['sale_price'] + 1.5 * df['discount'] + df['brand'].map(brand_effect) +
        rng.normal(0, 0.2, n)
    )
    df['rating'] = df['rating'].clip(1.0, 5.0)
    return df

def build_preprocessor(numeric_features, categorical_features):
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numeric_features),
        ('cat', cat_pipeline, categorical_features)
    ], remainder='drop')
    return preprocessor

def train_models(X_train, y_train, preprocessor):
    results = {}
    ridge = Pipeline([
        ('pre', preprocessor),
        ('reg', RidgeCV(alphas=np.logspace(-3, 3, 7), cv=5))
    ])
    ridge.fit(X_train, y_train)
    results['Ridge'] = ridge

    rf = Pipeline([
        ('pre', preprocessor),
        ('reg', RandomForestRegressor(n_estimators=200, random_state=42))
    ])
    rf.fit(X_train, y_train)
    results['RandomForest'] = rf

    if has_xgb:
        xgb = Pipeline([
            ('pre', preprocessor),
            ('reg', XGBRegressor(n_estimators=200, random_state=42, verbosity=0))
        ])
        xgb.fit(X_train, y_train)
        results['XGBoost'] = xgb

    return results

def evaluate_models(models, X_test, y_test):
    rows = []
    for name, model in models.items():
        preds = model.predict(X_test)
        rows.append({
            'model': name,
            'MAE': mean_absolute_error(y_test, preds),
            'R2': r2_score(y_test, preds)
        })
    return pd.DataFrame(rows)

# -----------------------
# Sidebar: Upload / Config
# -----------------------
st.sidebar.header("1) Data & Settings")
upload_format = st.sidebar.radio("Upload format", ['Use example data', 'Upload CSV/Excel'])

if upload_format == 'Use example data':
    df = load_example_data()
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx'])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()
    else:
        st.info("No file uploaded — using example dataset.")
        df = load_example_data()

st.sidebar.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

st.subheader("Preview of raw data")
st.dataframe(df.head())

# -----------------------
# Column mapping UI
# -----------------------
with st.expander("Column mapping (advanced)", expanded=True):
    cols = list(df.columns)
    sale_col = st.selectbox('sale_price column', options=cols, index=cols.index('sale_price') if 'sale_price' in cols else 0)
    market_col = st.selectbox('market_price column', options=cols, index=cols.index('market_price') if 'market_price' in cols else 0)
    discount_col = st.selectbox('discount column', options=cols, index=cols.index('discount') if 'discount' in cols else 0)
    category_col = st.selectbox('category column', options=cols, index=cols.index('category') if 'category' in cols else 0)
    subcat_col = st.selectbox('sub_category column', options=cols, index=cols.index('sub_category') if 'sub_category' in cols else 0)
    brand_col = st.selectbox('brand column', options=cols, index=cols.index('brand') if 'brand' in cols else 0)
    type_col = st.selectbox('type column', options=cols, index=cols.index('type') if 'type' in cols else 0)
    rating_col = st.selectbox('rating column', options=cols, index=cols.index('rating') if 'rating' in cols else 0)

# -----------------------
# Build working dataframe
# -----------------------
# Safely construct work_df using selected mapping (preserve original df)
work_df = pd.DataFrame()
work_df['sale_price'] = df[sale_col]
work_df['market_price'] = df[market_col]
work_df['discount'] = df[discount_col]
work_df['category'] = df[category_col].astype(str)
work_df['sub_category'] = df[subcat_col].astype(str)
work_df['brand'] = df[brand_col].astype(str)
work_df['type'] = df[type_col].astype(str)
# rating will be attached after cleaning validation

# -----------------------
# Conservative cleaning & diagnostics
# -----------------------
st.markdown("### 🔎 Data diagnostics & conservative cleaning")

# Keep backup in session_state
if 'work_df_backup' not in st.session_state:
    st.session_state['work_df_backup'] = work_df.copy()
    st.session_state['orig_df'] = df.copy()

orig_cols = list(work_df.columns)
st.write("Detected columns used for modeling:", orig_cols)

# Helper: attempt numeric conversion and report %
def percent_numeric(series):
    conv = pd.to_numeric(series, errors='coerce')
    return 100.0 * conv.notna().sum() / max(1, len(conv))

def mild_clean_series(s, is_percent=False):
    s2 = s.astype(str).str.strip()
    # remove common currency symbols and whitespace and commas
    s2 = s2.str.replace(r'[₹$€,£]', '', regex=True)
    s2 = s2.str.replace(',', '', regex=True)
    s2 = s2.str.replace(r'[\(\)]', '', regex=True)
    # remove trailing non-numeric after numeric prefix
    s2 = s2.str.replace(r'^([0-9]+(?:\.[0-9]+)?).*$', r'\1', regex=True)
    s2 = s2.str.replace(r'\s+', '', regex=True)
    # drop empty strings
    s2 = s2.replace('', pd.NA)
    num = pd.to_numeric(s2, errors='coerce')
    if is_percent:
        # convert whole numbers >1 to fraction (e.g., 15 -> 0.15)
        num = num.where(num.isna(), num.where(num <= 1, num / 100))
    return num

# Work on a copy
clean = work_df.copy()
problem_examples = {}

for col in ['sale_price', 'market_price', 'discount']:
    if col not in clean.columns:
        continue
    orig_series = clean[col]
    p_direct = percent_numeric(orig_series)
    st.write(f"Column `{col}` direct numeric parse: {p_direct:.1f}%")
    if p_direct >= 60.0:
        # majority numeric - keep numeric conversion
        clean[col] = pd.to_numeric(orig_series, errors='coerce').astype(float)
    else:
        # try mild cleaning
        st.write(f"Attempting mild cleaning for `{col}` ...")
        is_pct = (col == 'discount')
        conv = mild_clean_series(orig_series, is_percent=is_pct)
        p_conv = 100.0 * conv.notna().sum() / max(1, len(conv))
        st.write(f"After mild cleaning `{col}` numeric parse: {p_conv:.1f}%")
        # accept mild cleaning if it improves and crosses threshold
        if p_conv >= max(40.0, p_direct):
            clean[col] = conv.astype(float)
        else:
            # keep best effort (direct conv) but record problems
            conv_direct = pd.to_numeric(orig_series, errors='coerce')
            clean[col] = conv_direct.astype(float)
            # capture sample problematic values where both conversions failed
            mask_bad = conv_direct.isna()
            examples = orig_series[mask_bad].astype(str).head(20).tolist()
            if examples:
                problem_examples[col] = examples

# Attach rating (after cleaning done) — ensure rating exists in original df
if rating_col in st.session_state.get('orig_df', df).columns:
    clean['rating'] = st.session_state.get('orig_df', df)[rating_col]
else:
    # fallback: try from df variable
    if rating_col in df.columns:
        clean['rating'] = df[rating_col]
    else:
        st.error("⚠️ Rating column not found. Please select the correct rating column in the Column mapping expander.")
        st.stop()

work_df_clean = clean.copy()

# Report problems and provide cleaned CSV for inspection
if problem_examples:
    st.warning("Found problematic / non-numeric samples in numeric columns (examples shown). You may need to fix these rows manually if they are important.")
    for k, v in problem_examples.items():
        st.write(f"Column `{k}` examples (up to 20):")
        st.write(v)
else:
    st.success("Numeric cleaning performed. No large problematic samples detected (or mild cleaning fixed them).")

st.write("Summary statistics (after conservative cleaning):")
st.dataframe(work_df_clean[['sale_price','market_price','discount']].describe().transpose())

# Download cleaned preview
csv_buf = work_df_clean.head(200).to_csv(index=False).encode('utf-8')
st.download_button("Download cleaned preview (first 200 rows)", data=csv_buf, file_name="bigbasket_cleaned_preview.csv", mime="text/csv")

# -----------------------
# Modeling choices
# -----------------------
st.sidebar.header("2) Modeling choices")
train_size = st.sidebar.slider('Train set fraction', 0.5, 0.9, 0.75)
random_state = st.sidebar.number_input('Random seed', value=42, step=1)
run_train = st.sidebar.button('Train models')

# Build X and y from cleaned dataframe
if 'rating' not in work_df_clean.columns:
    st.error("Rating column missing from cleaned data. Please map rating column correctly.")
    st.stop()

X = work_df_clean.drop(columns=['rating'])
y = work_df_clean['rating']

# Quick safeguard: ensure at least some numeric data present
num_missing_frac = X[['sale_price','market_price','discount']].isna().mean()
if num_missing_frac.max() >= 0.95:
    st.error("Too many missing numeric values after cleaning (>=95%). Please fix your input file or mapping before training.")
    st.stop()

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=int(random_state))

numeric_features = ['sale_price','market_price','discount']
categorical_features = ['category','sub_category','type']
preprocessor = build_preprocessor(numeric_features, categorical_features)

models = None
results_df = None

if run_train:
    with st.spinner('Training models — Ridge, RandomForest' + (', XGBoost' if has_xgb else '')):
        try:
            import numpy as np
            import pandas as pd
            import sklearn
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import OneHotEncoder
            from sklearn.compose import ColumnTransformer
            from sklearn.pipeline import Pipeline
            from sklearn.linear_model import Ridge
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import mean_squared_error, r2_score

            # Optional XGBoost
            if has_xgb:
                from xgboost import XGBRegressor

            # -----------------------
            # 1️⃣ Clean & drop NaNs in target
            # -----------------------
            non_na_mask_train = y_train.notna()
            X_train = X_train[non_na_mask_train]
            y_train = y_train[non_na_mask_train]

            non_na_mask_test = y_test.notna()
            X_test = X_test[non_na_mask_test]
            y_test = y_test[non_na_mask_test]

            if X_train.empty or X_test.empty:
                st.error("No valid rows left after removing NaN targets.")
                st.stop()

            # -----------------------
            # 2️⃣ Identify features
            # -----------------------
            numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
            categorical_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

            st.write(f"Numeric features: {len(numeric_features)}, Categorical features: {len(categorical_features)}")

            # -----------------------
            # 3️⃣ Build transformers (version-safe)
            # -----------------------
            skl_version = tuple(map(int, sklearn.__version__.split(".")[:2]))
            if skl_version >= (1, 2):
                cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            else:
                cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

            numeric_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median"))
            ])

            categorical_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", cat_encoder)
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, numeric_features),
                    ("cat", categorical_transformer, categorical_features)
                ],
                remainder='drop'
            )

            # -----------------------
            # 4️⃣ Define models
            # -----------------------
            models = {
                "Ridge": Pipeline([
                    ("preprocessor", preprocessor),
                    ("model", Ridge(random_state=42))
                ]),
                "RandomForest": Pipeline([
                    ("preprocessor", preprocessor),
                    ("model", RandomForestRegressor(n_estimators=150, random_state=42))
                ])
            }

            if has_xgb:
                models["XGBoost"] = Pipeline([
                    ("preprocessor", preprocessor),
                    ("model", XGBRegressor(
                        random_state=42, n_estimators=200, learning_rate=0.1,
                        tree_method="hist", eval_metric="rmse"
                    ))
                ])

            # -----------------------
            # 5️⃣ Train + Evaluate safely
            # -----------------------
            results = []
            for name, model in models.items():
                # Drop any rows with remaining NaNs (just in case)
                X_train_valid = X_train.dropna()
                y_train_valid = y_train.loc[X_train_valid.index]
                X_test_valid = X_test.dropna()
                y_test_valid = y_test.loc[X_test_valid.index]

                if X_train_valid.empty or X_test_valid.empty:
                    st.warning(f"Skipping {name}: No valid rows after NaN cleanup.")
                    continue

                model.fit(X_train_valid, y_train_valid)
                preds = model.predict(X_test_valid)

                # Drop any NaN predictions (safeguard)
                mask_valid = ~np.isnan(preds)
                preds = preds[mask_valid]
                y_true = y_test_valid.iloc[mask_valid]

                if len(preds) == 0:
                    st.warning(f"{name} produced all-NaN predictions, skipping.")
                    continue

                rmse = float(np.sqrt(np.mean((y_true - preds) ** 2)))
                r2 = float(r2_score(y_true, preds))
                results.append({"Model": name, "RMSE": rmse, "R2": r2})

            # -----------------------
            # 6️⃣ Show results
            # -----------------------
            if len(results) == 0:
                st.error("All models failed due to NaN contamination.")
            else:
                results_df = pd.DataFrame(results).sort_values(by="R2", ascending=False)
                st.success("✅ Training complete!")
                st.dataframe(results_df)

        except Exception as e:
            import traceback
            st.error(f"Training failed: {e}")
            st.code(traceback.format_exc())
            st.stop()





# -----------------------
# Post-training UI
# -----------------------
if models is not None:
    st.header('📈 Model comparison')
    st.table(results_df.set_index('Model'))

    model_choice = st.selectbox('Choose model for explanations', options=list(models.keys()))
    model = models[model_choice]

    # Predictions table
    preds = model.predict(X_test)
    st.subheader('Sample predictions (first 10)')
    sample_df = X_test.reset_index(drop=True).loc[:9].copy()
    sample_df['true_rating'] = y_test.reset_index(drop=True).loc[:9]
    sample_df['predicted_rating'] = preds[:10]
    st.dataframe(sample_df)

    # SHAP global explanations (defensive)
    st.subheader('🔍 SHAP global explanations')
    try:
        pre = model.named_steps.get('pre', preprocessor)
        # prepare background: sample from train
        n_bg = min(200, len(X_train))
        try:
            bg = pre.transform(X_train.sample(n_bg, random_state=1))
            if hasattr(shap, 'TreeExplainer') and 'RandomForest' in model_choice:
                explainer = shap.TreeExplainer(model.named_steps['model'])
                shap_values = explainer.shap_values(bg)
                fig = plt.figure(figsize=(8,6))
                shap.summary_plot(explainer.shap_values(pre.transform(X_test.sample(min(200,len(X_test)), random_state=1))),
                                  pre.transform(X_test.sample(min(200,len(X_test)), random_state=1)), show=False)
                st.pyplot(fig)
            else:
                explainer = shap.Explainer(model.named_steps['model'], bg)
                vals = explainer(pre.transform(X_test.sample(min(200,len(X_test)), random_state=1)))
                fig = plt.figure(figsize=(8,6))
                shap.summary_plot(vals, pre.transform(X_test.sample(min(200,len(X_test)), random_state=1)), show=False)
                st.pyplot(fig)
        except Exception as e:
            st.write("Could not generate SHAP plots (helpful when model/package combos mismatch):", e)
    except Exception as e:
        st.warning(f'SHAP failed: {e}')

    # Dependence plot simple scatter for discount vs centered predictions
    st.write('Dependence plot: discount vs centered predicted rating')
    try:
        centered = model.predict(X_test) - np.mean(model.predict(X_test))
        fig2 = plt.figure(figsize=(7,5))
        plt.scatter(X_test['discount'], centered, alpha=0.5)
        plt.xlabel('discount')
        plt.ylabel('centered predicted rating')
        st.pyplot(fig2)
    except Exception as e:
        st.write('Could not produce dependence scatter:', e)

    # LIME local explanations
    st.subheader('🧾 LIME — local explanations')
    if not has_lime:
        st.info('LIME not installed. To enable LIME, pip install lime and refresh.')
    else:
        try:
            pre = model.named_steps['pre']
            X_train_tr = pre.transform(X_train)
            # Attempt to build feature_names for Lime
            feat_names = numeric_features.copy()
            try:
                ohe = pre.named_transformers_['cat'].named_steps['ohe']
                cat_names = list(ohe.get_feature_names_out(categorical_features))
                feat_names += cat_names
            except Exception:
                # fallback: use positional names
                feat_names = [f'f{i}' for i in range(X_train_tr.shape[1])]
            explainer = LimeTabularExplainer(X_train_tr, feature_names=feat_names, verbose=False, mode='regression')
            idx = st.number_input('Choose an integer index (0..n_test-1) for a test sample to explain', min_value=0, max_value=max(0, len(X_test)-1), value=0)
            x_instance = X_test.reset_index(drop=True).loc[idx:idx]
            x_tr = pre.transform(x_instance)
            exp = explainer.explain_instance(x_tr.ravel(), model.predict, num_features=8)
            html = exp.as_html()
            st.components.v1.html(html, height=400, scrolling=True)
        except Exception as e:
            st.warning('LIME explanation failed: ' + str(e))

    # Fairness audit across brand groups
    st.subheader('⚖️ Fairness audit by brand (group-wise MAE & R²)')
    if not has_fairlearn:
        st.info('Fairlearn not installed. To enable fairness metrics, pip install fairlearn.')
    else:
        try:
            preds_all = model.predict(X_test)
            groups = X_test['brand'].reset_index(drop=True)
            mf = MetricFrame(metrics={'mae': lambda y_true, y_pred: mae(y_true, y_pred), 'r2': r2_score},
                             y_true=y_test.reset_index(drop=True), y_pred=preds_all, sensitive_features=groups)
            st.write('Overall MAE: ', mean_absolute_error(y_test, preds_all))
            st.write('Overall R²: ', r2_score(y_test, preds_all))
            st.write('Group-wise metrics:')
            st.dataframe(mf.by_group)
            # Simple pre-processing mitigation: reweight by inverse brand frequency (demonstration)
            st.markdown('**Bias mitigation (pre-processing): reweighting by inverse brand frequency — demo**')
            weights = X_train['brand'].value_counts()
            inv_freq = 1.0 / weights
            sample_weights = X_train['brand'].map(inv_freq).astype(float)
            # retrain ridge with sample weights (fit on transformed data)
            mitig_ridge = Pipeline([('pre', preprocessor), ('reg', RidgeCV(alphas=np.logspace(-3,3,7), cv=5))])
            try:
                mitig_ridge.named_steps['reg'].fit(preprocessor.fit_transform(X_train), y_train, sample_weight=sample_weights)
                preds_mitig = mitig_ridge.predict(X_test)
                mf2 = MetricFrame(metrics={'mae': lambda y_true, y_pred: mae(y_true, y_pred), 'r2': r2_score},
                                  y_true=y_test.reset_index(drop=True), y_pred=preds_mitig, sensitive_features=X_test['brand'].reset_index(drop=True))
                st.write('Group-wise metrics after reweighting (Ridge):')
                st.dataframe(mf2.by_group)
                st.write('Overall MAE after reweighting:', mean_absolute_error(y_test, preds_mitig))
                st.write('Overall R² after reweighting:', r2_score(y_test, preds_mitig))
            except Exception as e:
                st.write("Could not apply reweighting mitigation (demo):", e)
        except Exception as e:
            st.warning('Fairness audit failed: ' + str(e))

    # Save model button (download selected model)
    st.subheader('💾 Save a trained model')
    model_name = st.text_input('Filename for model (e.g. model.pkl)', value='exp5_model.pkl')
    if st.button('Save selected model'):
        try:
            buf = io.BytesIO()
            pickle.dump(models[model_choice], buf)
            buf.seek(0)
            st.download_button('Download model pickle', data=buf, file_name=model_name)
        except Exception as e:
            st.error("Could not save model: " + str(e))

    # Generate Dockerfile + FastAPI stub
    st.subheader('🐳 Generate Dockerfile + FastAPI stub')
    if st.button('Generate API + Dockerfile'):
        api_code = """
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

class RequestItem(BaseModel):
    sale_price: float
    market_price: float
    discount: float
    category: str
    sub_category: str
    brand: str
    type: str

app = FastAPI()
model = pickle.load(open('model.pkl', 'rb'))

@app.get('/health')
def health():
    return {'status':'ok'}

@app.post('/predict')
def predict(item: RequestItem):
    df = pd.DataFrame([item.dict()])
    preds = model.predict(df)
    return {'predicted_rating': float(preds[0])}
"""
        dockerfile = """
# Dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY api.py ./
COPY model.pkl ./
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
"""
        st.download_button('Download api.py', data=api_code, file_name='api.py')
        st.download_button('Download Dockerfile', data=dockerfile, file_name='Dockerfile')
        st.success('API stub and Dockerfile generated for download')

else:
    st.info('Train models to enable explanations, fairness audit and deployment helpers.')

st.sidebar.markdown('''
### Notes & next steps
- This app assumes `rating` is continuous (regression). For stock-out classification, use a separate pipeline.
- To enable full LIME/XGBoost/Fairlearn functionality: `pip install lime xgboost fairlearn`.
- To deploy: save the model, create an API (api.py), build Docker image and run.
''')