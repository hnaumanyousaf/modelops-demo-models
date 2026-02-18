import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.manual_seed(42)
# np.random.seed(42)

from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from teradataml import DataFrame, copy_to_sql
from aoa import (
    record_evaluation_stats,
    save_plot,
    tmo_create_context,
    ModelContext
)

import joblib
import json
import numpy as np
import pandas as pd


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

class LogisticRegressionTorch(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)  # logits

def evaluate(context: ModelContext, **kwargs):

    tmo_create_context()
    batch_size = context.hyperparams["batch_size"]

    # model = joblib.load(f"{context.artifact_input_path}/model.joblib")
    preprocess = joblib.load(f"{context.artifact_input_path}/preprocess.joblib")

    feature_names = context.dataset_info.feature_names
    target_name = context.dataset_info.target_names[0]

    test_df = DataFrame.from_query(context.dataset_info.sql)
    test_pdf = test_df.to_pandas(all_rows=True)

    X_test = test_pdf[feature_names]
    y_test = test_pdf[target_name].map({'Yes': 1, 'No': 0}).values
    y_test_t  = y_test.astype(np.float32)

    X_test_p  = preprocess.transform(X_test)
    X_test_p  = X_test_p.toarray().astype(np.float32)  if hasattr(X_test_p, "toarray") else X_test_p.astype(np.float32)
    test_loader  = DataLoader(TabularDataset(X_test_p, y_test_t),   batch_size=batch_size, shuffle=False)

    print("Loading model")

    input_dim = X_test_p.shape[1]
    print("Input dim after preprocessing:", input_dim)

    model = LogisticRegressionTorch(input_dim).to(device)
    model.load_state_dict(torch.load(f"{context.artifact_input_path}/model.pt"))
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

    print("Scoring")
    y_proba = predict_proba(test_loader)
    threshold = 0.5
    y_pred = (y_proba >= threshold).astype(int)

    y_pred_tdf = pd.DataFrame(y_pred, columns=[target_name])
    y_pred_tdf["CustomerID"] = test_pdf["CustomerID"].values

    evaluation = {
        'Accuracy': '{:.2f}'.format(metrics.accuracy_score(y_test, y_pred)),
        'Recall': '{:.2f}'.format(metrics.recall_score(y_test, y_pred)),
        'Precision': '{:.2f}'.format(metrics.precision_score(y_test, y_pred)),
        'f1-score': '{:.2f}'.format(metrics.f1_score(y_test, y_pred))
    }

    with open(f"{context.artifact_output_path}/metrics.json", "w+") as f:
        json.dump(evaluation, f)

    # ConfusionMatrixDisplay.from_estimator(model, X_test, y_test_t)
    # save_plot('Confusion Matrix', context=context)

    # RocCurveDisplay.from_estimator(model, X_test, y_test)
    # save_plot('ROC Curve', context=context)

    predictions_table = "predictions_tmp"
    copy_to_sql(df=y_pred_tdf, table_name=predictions_table, index=False, if_exists="replace", temporary=True)

    record_evaluation_stats(features_df=test_df,
                            predicted_df=DataFrame.from_query(f"SELECT * FROM {predictions_table}"),
                            # feature_importance=feature_importance,
                            context=context)

    print("All done!")
