import os, sys

# Adds the folder that contains `model_modules/` to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))  # -> $model_local_path
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import joblib

from sklearn.model_selection import train_test_split
from teradataml import DataFrame
from aoa import tmo_create_context, ModelContext, record_training_stats

from model_modules.preprocess import infer_column_types, build_preprocess, fit_transform_preprocess, transform_preprocess
from model_modules.data import make_loaders
from model_modules.model import LogisticRegressionTorch
from model_modules.engine import get_device, fit_with_early_stopping

def train(context: ModelContext, **kwargs):
    tmo_create_context()

    # Repro
    torch.manual_seed(42)
    np.random.seed(42)

    device = get_device()
    print(device)

    feature_names = context.dataset_info.feature_names
    target_name = context.dataset_info.target_names[0]

    # Read training dataset from Teradata -> pandas
    train_df = DataFrame.from_query(context.dataset_info.sql)
    train_pdf = train_df.to_pandas(all_rows=True)

    # Split X/y
    X = train_pdf[feature_names]
    y = train_pdf[target_name].map({'Yes': 1, 'No': 0}).values.astype(np.int64)

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")

    # Preprocess (fit on train only!)
    num_cols, cat_cols = infer_column_types(X_train)
    print("Categorical columns:", cat_cols)
    print("Numerical columns:", num_cols)

    preprocess = build_preprocess(num_cols=num_cols, cat_cols=cat_cols)

    X_train_p = fit_transform_preprocess(preprocess, X_train)
    X_val_p   = transform_preprocess(preprocess, X_val)

    y_train_f = y_train.astype(np.float32)
    y_val_f   = y_val.astype(np.float32)

    input_dim = X_train_p.shape[1]
    print("Input dim after preprocessing:", input_dim)

    # Model
    model = LogisticRegressionTorch(input_dim).to(device)

    # Loaders
    batch_size = int(context.hyperparams["batch_size"])
    print("batch_size:", batch_size)
    train_loader, val_loader = make_loaders(X_train_p, y_train_f, X_val_p, y_val_f, batch_size)

    # Train with early stopping
    stats = fit_with_early_stopping(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        y_val_np=y_val,   # keep as int/0-1 for sklearn metrics
        device=device,
        lr=1e-3,
        weight_decay=1e-4,
        patience=5,
        max_epochs=50,
    )

    print("Finished training", stats)

    # Save artifacts (model weights + preprocess)
    torch.save(model.state_dict(), f"{context.artifact_output_path}/model.pt")
    joblib.dump(preprocess, f"{context.artifact_output_path}/preprocess.joblib")

    _categorical_stat_cols_list = [target_name]+cat_cols
    print(_categorical_stat_cols_list)
    record_training_stats(train_df,
                          features=feature_names,
                          targets=[target_name],
                          categorical=_categorical_stat_cols_list,
                        #   feature_importance=feature_importance,
                          context=context)

    print("Saved trained model and preprocessing artifacts")
    print("All done!")
