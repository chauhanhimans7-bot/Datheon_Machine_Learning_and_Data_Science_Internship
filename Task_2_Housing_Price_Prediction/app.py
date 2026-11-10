import os

import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_URL = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
LOCAL_CSV = "housing.csv"  # optional offline fallback, place next to app.py
MODEL_PATH = "housing_model_pred.joblib"  # saved from the notebook

st.set_page_config(page_title="California Housing Price Predictor", page_icon="🏠")


@st.cache_resource(show_spinner="Training model (first run only)...")
def load_model():
    """Load the saved model if it matches this scikit-learn version; otherwise retrain and re-save."""
    if os.path.exists(MODEL_PATH):
        try:
            saved = joblib.load(MODEL_PATH)
            if saved.get("sklearn_version") == sklearn.__version__:
                return saved["model"], saved["metrics"]
        except Exception:
            pass  # unreadable or incompatible file -> retrain below

    try:
        df = pd.read_csv(DATA_URL)
    except Exception:
        df = pd.read_csv(LOCAL_CSV)

    X = df.drop(columns=["median_house_value"])
    y = df["median_house_value"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    num_features = X_train.select_dtypes(include=np.number).columns.tolist()
    cat_features = X_train.select_dtypes(exclude=np.number).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), num_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_features),
    ])

    model = Pipeline([
        ("preprocess", preprocessor),
        ("model", HistGradientBoostingRegressor(
            l2_regularization=0.1,
            learning_rate=0.1,
            max_depth=None,
            max_leaf_nodes=63,
            min_samples_leaf=20,
            random_state=42,
        )),
    ])
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    metrics = {
        "RMSE": root_mean_squared_error(y_test, pred),
        "MAE": mean_absolute_error(y_test, pred),
        "R²": r2_score(y_test, pred),
    }
    try:
        joblib.dump(
            {"model": model, "metrics": metrics, "sklearn_version": sklearn.__version__},
            MODEL_PATH,
        )
    except Exception:
        pass
    return model, metrics


model, metrics = load_model()

st.title("🏠 California Housing Price Predictor")
st.caption(
    "Predicts the **median house value** of a California census block group "
    "using a tuned HistGradientBoostingRegressor."
)

with st.form("input_form"):
    st.subheader("Location")
    c1, c2, c3 = st.columns(3)
    latitude = c1.number_input("Latitude", 32.5, 42.0, 37.86, step=0.01, format="%.2f")
    longitude = c2.number_input("Longitude", -124.4, -114.3, -122.22, step=0.01, format="%.2f")
    ocean_proximity = c3.selectbox(
        "Ocean proximity", ["NEAR BAY", "<1H OCEAN", "INLAND", "NEAR OCEAN", "ISLAND"]
    )

    st.subheader("Block details")
    c1, c2 = st.columns(2)
    median_income = c1.number_input(
        "Median income (in $10,000s)", 0.5, 15.0, 8.30, step=0.1,
        help="3.5 means about $35,000",
    )
    housing_median_age = c2.number_input("Housing median age (years)", 1, 52, 21)

    c1, c2 = st.columns(2)
    total_rooms = c1.number_input("Total rooms", 1, 40000, 7099, step=50)
    total_bedrooms = c2.number_input("Total bedrooms", 1, 6500, 1106, step=10)

    c1, c2 = st.columns(2)
    population = c1.number_input("Population", 1, 36000, 2401, step=50)
    households = c2.number_input("Households", 1, 6100, 1138, step=10)

    submitted = st.form_submit_button("Predict price", type="primary", use_container_width=True)

if submitted:
    row = pd.DataFrame([{
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": float(housing_median_age),
        "total_rooms": float(total_rooms),
        "total_bedrooms": float(total_bedrooms),
        "population": float(population),
        "households": float(households),
        "median_income": median_income,
        "ocean_proximity": ocean_proximity,
    }])
    price = float(model.predict(row)[0])

    st.success(f"Estimated median house value: **${price:,.0f}**")
    st.caption(f"Typical error on unseen data is about ±${metrics['MAE']:,.0f} (MAE).")
    st.map(pd.DataFrame({"lat": [latitude], "lon": [longitude]}), zoom=6)

with st.expander("Model performance (test set)"):
    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", f"${metrics['RMSE']:,.0f}")
    c2.metric("MAE", f"${metrics['MAE']:,.0f}")
    c3.metric("R²", f"{metrics['R²']:.3f}")
    st.caption(
        "Data is from the 1990 California census, so values are far below today's prices. "
        "Target is capped at $500,001."
    )