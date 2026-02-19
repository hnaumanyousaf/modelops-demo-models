import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from teradataml import copy_to_sql, DataFrame
from aoa import (
    record_scoring_stats,
    tmo_create_context,
    ModelContext
)

import joblib
import pandas as pd
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def score(context: ModelContext, **kwargs):

    tmo_create_context()
    batch_size = int(context.hyperparams["batch_size"])

    # model = joblib.load(f"{context.artifact_input_path}/model.joblib")
    preprocess = joblib.load(f"{context.artifact_input_path}/preprocess.joblib")

    feature_names = context.dataset_info.feature_names
    target_name = context.dataset_info.target_names[0]
    entity_key = context.dataset_info.entity_key

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
    predictions_pdf = (y_proba >= threshold).astype(int)

    print("Finished Scoring")

    # store the predictions
    predictions_pdf = pd.DataFrame(predictions_pdf, columns=[target_name])
    predictions_pdf[entity_key] = test_pdf.index.values
    # add job_id column so we know which execution this is from if appended to predictions table
    predictions_pdf["job_id"] = context.job_id
    predictions_pdf = predictions_pdf[["job_id", entity_key, target_name]]

    predictions_pdf["json_report"] = ""
    predictions_pdf = predictions_pdf[["job_id", entity_key, target_name, "json_report"]]

    copy_to_sql(df=predictions_pdf,
                schema_name=context.dataset_info.predictions_database,
                table_name=context.dataset_info.predictions_table,
                index=False,
                if_exists="append")

    print("Saved predictions in Teradata")

    # calculate stats
    predictions_df = DataFrame.from_query(f"""
        SELECT 
            * 
        FROM {context.dataset_info.get_predictions_metadata_fqtn()} 
            WHERE job_id = '{context.job_id}'
    """)

    # print(test_pdf)
    # print(predictions_df)

    record_scoring_stats(features_df=test_df,
                         predicted_df=predictions_df,
                         context=context)

    print("All done!")
