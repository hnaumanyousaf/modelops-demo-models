import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix

from teradataml import DataFrame
from aoa import (
    record_training_stats,
    save_plot,
    tmo_create_context,
    ModelContext
)

import joblib
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

torch.manual_seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------
# Dataset / DataLoader
# ---------------------------------------------------------------------
class TabularDataset(Dataset):
    def __init__(self, X_np: np.ndarray, y_np: np.ndarray):
        self.X = torch.from_numpy(X_np)
        self.y = torch.from_numpy(y_np).view(-1, 1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ---------------------------------------------------------------------
# Logistic Regression in PyTorch: Linear layer + BCEWithLogitsLoss
# ---------------------------------------------------------------------
class LogisticRegressionTorch(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)  # logits

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
    y_train = y_train.map({'Yes': 1, 'No': 0}).values

    # Identify feature types (or replace with your own explicit lists)
    cat_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    print("Categorical columns:", cat_cols)
    print("Numerical columns:", num_cols)

    print("Starting preprocessing...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.20, random_state=42, stratify=y_train
    )

    print(f"Train: {X_train.shape}, Val: {X_val.shape}")

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

    X_train_p = preprocess.fit_transform(X_train)
    X_test_p  = preprocess.transform(X_val)

    # The output is often a scipy sparse matrix because of one-hot encoding.
    # We'll convert to dense float32. If your data is huge, see note below for sparse handling.
    X_train_p = X_train_p.toarray().astype(np.float32) if hasattr(X_train_p, "toarray") else X_train_p.astype(np.float32)
    X_test_p  = X_test_p.toarray().astype(np.float32)  if hasattr(X_test_p, "toarray") else X_test_p.astype(np.float32)

    y_train_t = y_train.astype(np.float32)
    y_test_t  = y_val.astype(np.float32)

    input_dim = X_train_p.shape[1]
    print("Input dim after preprocessing:", input_dim)

    print("Starting training...")
    # ---------------------------------------------------------------------
    # Model pipeline
    # - class_weight="balanced" is often helpful for fraud imbalance
    # - solver="liblinear" or "saga" are common; saga works well for larger sparse one-hot matrices
    # ---------------------------------------------------------------------
    model = LogisticRegressionTorch(input_dim).to(device)

    # Handle class imbalance via pos_weight (recommended)
    # pos_weight = (#negative / #positive)
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)  # weight_decay ~ L2

    # ---------------------------------------------------------------------
    # Train / Eval loops
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def predict_proba(loader: DataLoader) -> np.ndarray:
        model.eval()
        probs = []
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            p = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            probs.append(p)
        return np.concatenate(probs, axis=0)

    def train_one_epoch(loader: DataLoader) -> float:
        model.train()
        total_loss = 0.0
        n = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            bs = xb.size(0)
            total_loss += loss.item() * bs
            n += bs
        return total_loss / max(n, 1)

    # ------------------------------------------------------
    # Fit with simple early stopping on validation PR AUC 
    # ------------------------------------------------------
    best_state = None
    best_val_ap = -1.0
    patience = 5
    pat_left = patience
    max_epochs = 50

    batch_size = int(context.hyperparams["batch_size"])
    print("batch_size:", batch_size)

    train_loader = DataLoader(TabularDataset(X_train_p, y_train_t), batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(TabularDataset(X_test_p, y_test_t),   batch_size=batch_size, shuffle=False)

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(train_loader)

        val_proba = predict_proba(test_loader)
        val_auc = roc_auc_score(y_val, val_proba)
        val_ap = average_precision_score(y_val, val_proba)

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.5f} | val_auc={val_auc:.5f} | val_ap={val_ap:.5f}")

        if val_ap > best_val_ap + 1e-5:
            best_val_ap = val_ap
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            pat_left = patience
        else:
            pat_left -= 1
            if pat_left <= 0:
                print("Early stopping.")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    print("Finished training")

    # export model artefacts
    # joblib.dump(model, f"{context.artifact_output_path}/model.joblib")
    torch.save(model.state_dict(), f"{context.artifact_output_path}/model.pt")
    joblib.dump(preprocess, f"{context.artifact_output_path}/preprocess.joblib")

    print("Saved trained model")

    # features importance
    # plot_logreg_feature_importance(model, f"{context.artifact_output_path}/feature_importance")

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
