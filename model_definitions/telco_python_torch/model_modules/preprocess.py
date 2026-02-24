import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def to_float32_dense(Xt):
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    return Xt.astype(np.float32)

def infer_column_types(X: pd.DataFrame):
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    return num_cols, cat_cols

def build_preprocess(num_cols, cat_cols):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop"
    )
    return preprocess

def fit_transform_preprocess(preprocess, X_train: pd.DataFrame):
    Xt = preprocess.fit_transform(X_train)
    return to_float32_dense(Xt)

def transform_preprocess(preprocess, X: pd.DataFrame):
    Xt = preprocess.transform(X)
    return to_float32_dense(Xt)
