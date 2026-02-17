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

import pandas as pd
import matplotlib.pyplot as plt

def get_feature_names_from_column_transformer(ct):
    """
    Returns output feature names from a fitted ColumnTransformer that may contain Pipelines.
    Works for SimpleImputer/StandardScaler + OneHotEncoder pipelines.
    """
    feature_names = []

    for name, transformer, cols in ct.transformers_:
        if name == "remainder" and transformer == "drop":
            continue
        if transformer == "drop":
            continue

        # If it's a Pipeline, the last step is the real transformer (e.g., OneHotEncoder)
        if hasattr(transformer, "named_steps"):
            last = list(transformer.named_steps.values())[-1]
        else:
            last = transformer

        if hasattr(last, "get_feature_names_out"):
            # OneHotEncoder and some others
            try:
                names = last.get_feature_names_out(cols)
            except TypeError:
                names = last.get_feature_names_out()
            feature_names.extend(names.tolist())
        else:
            # For scalers/imputers: names are the original columns
            if isinstance(cols, (list, tuple, np.ndarray, pd.Index)):
                feature_names.extend(list(cols))
            else:
                feature_names.append(str(cols))

    return np.array(feature_names)

def plot_logreg_feature_importance(fitted_pipeline, img_filename, top_n=30, use_abs=True, title=None):
    """
    fitted_pipeline: sklearn Pipeline with steps: preprocess (ColumnTransformer) + model (LogisticRegression)
    """
    preprocess = fitted_pipeline.named_steps["preprocess"]
    model = fitted_pipeline.named_steps["model"]

    feature_names = get_feature_names_from_column_transformer(preprocess)

    # Binary classification: coef_ shape (1, n_features)
    coefs = model.coef_.ravel()
    if use_abs:
        importances = np.abs(coefs)
        sort_idx = np.argsort(importances)[::-1]
    else:
        importances = coefs
        sort_idx = np.argsort(np.abs(importances))[::-1]  # sort by magnitude but keep sign

    top_idx = sort_idx[:top_n]

    plot_df = pd.DataFrame({
        "feature": feature_names[top_idx],
        "coef": coefs[top_idx],
        "importance": importances[top_idx],
    })

    plt.figure(figsize=(10, max(4, top_n * 0.25)))
    plt.barh(plot_df["feature"][::-1], plot_df["coef"][::-1])  # signed bars
    plt.xlabel("Coefficient (log-odds units)")
    plt.title(title or f"Top {top_n} Important Features")
    plt.tight_layout()
    # plt.show()
    fig = plt.gcf()
    fig.savefig(img_filename, dpi=500)
    plt.clf()

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

    # features importance
    plot_logreg_feature_importance(model, f"{context.artifact_output_path}/feature_importance")

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
