import os, sys
# Ensure the folder containing `model_modules/` is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))  # -> $model_local_path
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import numpy as np
import pandas as pd
import joblib
import torch

from sklearn import metrics
from teradataml import DataFrame, copy_to_sql
from aoa import (
    record_evaluation_stats,
    save_plot,
    tmo_create_context,
    ModelContext
)

from model_modules.data import TabularDataset
from model_modules.model import LogisticRegressionTorch
from model_modules.preprocess import transform_preprocess
from model_modules.engine import get_device, predict_proba


def evaluate(context: ModelContext, **kwargs):
    tmo_create_context()

    device = get_device()
    batch_size = int(context.hyperparams["batch_size"])

    # Load preprocessing
    preprocess = joblib.load(f"{context.artifact_input_path}/preprocess.joblib")

    feature_names = context.dataset_info.feature_names
    target_name = context.dataset_info.target_names[0]

    # Read evaluation dataset from Teradata -> pandas
    test_df = DataFrame.from_query(context.dataset_info.sql)
    test_pdf = test_df.to_pandas(all_rows=True)

    # Prepare X/y
    X_test = test_pdf[feature_names]
    y_test = test_pdf[target_name].map({'Yes': 1, 'No': 0}).values.astype(np.int64)
    y_test_f = y_test.astype(np.float32)

    # Apply same preprocessing used in training
    X_test_p = transform_preprocess(preprocess, X_test)

    # Build loader
    test_loader = torch.utils.data.DataLoader(
        TabularDataset(X_test_p, y_test_f),
        batch_size=batch_size,
        shuffle=False
    )

    # Load model
    input_dim = X_test_p.shape[1]
    print("Input dim after preprocessing:", input_dim)

    model = LogisticRegressionTorch(input_dim).to(device)
    state = torch.load(f"{context.artifact_input_path}/model.pt", map_location=device)
    model.load_state_dict(state)

    # Predict
    print("Scoring")
    y_proba = predict_proba(model, test_loader, device=device)

    threshold = float(context.hyperparams.get("threshold", 0.5))
    y_pred = (y_proba >= threshold).astype(np.int64)

    y_pred_tdf = pd.DataFrame(y_pred, columns=[target_name])
    y_pred_tdf["CustomerID"] = test_pdf["CustomerID"].values

    # # Example: join predictions back for reporting (keep if you need it)
    # if "CustomerID" in test_pdf.columns:
    #     y_pred_tdf = pd.DataFrame({
    #         "CustomerID": test_pdf["CustomerID"].values,
    #         "y_pred": y_pred,
    #         "y_proba": y_proba
    #     })
    #     # If you want to persist to Teradata later, you can use copy_to_sql()

    evaluation = {
        "Accuracy": float(metrics.accuracy_score(y_test, y_pred)),
        "Recall": float(metrics.recall_score(y_test, y_pred, zero_division=0)),
        "Precision": float(metrics.precision_score(y_test, y_pred, zero_division=0)),
        "f1-score": float(metrics.f1_score(y_test, y_pred, zero_division=0)),
    }

    # Save metrics artifact
    with open(f"{context.artifact_output_path}/metrics.json", "w") as f:
        json.dump(evaluation, f, indent=2)

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
