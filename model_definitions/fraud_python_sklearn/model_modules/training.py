import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from teradataml import DataFrame
from aoa import (
    record_training_stats,
    save_plot,
    tmo_create_context,
    ModelContext
)

import joblib


def train(context: ModelContext, **kwargs):
    tmo_create_context()

    feature_names = context.dataset_info.feature_names
    target_name = context.dataset_info.target_names[0]

    # read training dataset from Teradata and convert to pandas
    train_df = DataFrame.from_query(context.dataset_info.sql)
    train_pdf = train_df.to_pandas(all_rows=True)

    # split data into X and y
    X_train = train_pdf[feature_names]
    y_train = train_pdf[target_name]

    # Identify feature types (or replace with your own explicit lists)
    cat_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    print("Categorical columns:", cat_cols)
    print("Numerical columns:", num_cols)

    print("Starting training...")

    # ---------------------------------------------------------------------
    # Preprocessing
    # - Numeric: impute missing + standardize
    # - Categorical: impute missing + one-hot encode
    # ---------------------------------------------------------------------
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
        remainder="drop"  # or "passthrough" if you want to keep other columns
    )

    # ---------------------------------------------------------------------
    # Model pipeline
    # - class_weight="balanced" is often helpful for fraud imbalance
    # - solver="liblinear" or "saga" are common; saga works well for larger sparse one-hot matrices
    # ---------------------------------------------------------------------
    model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", LogisticRegression(
            max_iter=context.hyperparams["max_iter"],
            solver="liblinear",
            class_weight="balanced",
            n_jobs=-1
        ))
    ])

    # Fit on TRAIN only
    model.fit(X_train, y_train)
    print("Finished training")

    # export model artefacts
    joblib.dump(model, f"{context.artifact_output_path}/model.joblib")

    print("Saved trained model")

    print("Recording training stats")

    _categorical_stat_cols_list = [target_name]+cat_cols
    print(_categorical_stat_cols_list)
    record_training_stats(train_df,
                          features=feature_names,
                          targets=[target_name],
                          categorical=_categorical_stat_cols_list,
                        #   feature_importance=feature_importance,
                          context=context)

    print("All done!")
