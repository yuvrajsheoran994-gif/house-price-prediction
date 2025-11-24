# train_model.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from joblib import dump
from lightgbm import LGBMRegressor   # ← FAST MODEL

DATA_PATH = "india_housing_prices.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df = df.dropna(subset=["Price_in_Lakhs"])
    return df


def build_and_train(df):

    # 🔥 REMOVE extremely high-cardinality text columns
    drop_cols = ["Locality", "Amenities"]
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    # 🔥 Use only lightweight features
    features = [
        "State", "City", "Property_Type",
        "BHK", "Size_in_SqFt", "Year_Built",
        "Furnished_Status", "Floor_No", "Total_Floors",
        "Parking_Space", "Security", "Facing", "Owner_Type"
    ]

    features = [f for f in features if f in df.columns]

    X = df[features].copy()
    y = df["Price_in_Lakhs"].astype(float)

    # 🔥 FAST train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X_train.columns if c not in numeric_features]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        sparse_threshold=0.3
    )

    # ⚡ SUPER FAST LightGBM model
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LGBMRegressor(
            n_estimators=250,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            random_state=42
        ))
    ])

    print("Training LightGBM model (VERY FAST)...")
    model.fit(X_train, y_train)
    print("Training completed.")

    # Score
    score = model.score(X_test, y_test)
    print(f"R^2 score: {score:.4f}")

    # Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    dump(model, MODEL_PATH)

    print(f"Model saved at: {MODEL_PATH}")

    # Save form metadata
    meta = {
        "states": sorted(df["State"].dropna().unique().tolist()),
        "property_types": sorted(df["Property_Type"].dropna().unique().tolist())
    }

    # City mapping
    city_map = {}
    for s in meta["states"]:
        city_map[s] = sorted(df.loc[df["State"] == s, "City"].dropna().unique().tolist())

    import json
    with open(os.path.join(MODEL_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "city_map": city_map}, f, indent=2)

    print("meta.json saved successfully.")


if __name__ == "__main__":
    df = load_data()
    build_and_train(df)
