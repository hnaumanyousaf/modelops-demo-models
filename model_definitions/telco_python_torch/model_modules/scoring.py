import os, sys
# Ensure the folder containing `model_modules/` is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))  # -> $model_local_path
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import joblib
import torch

from teradataml import copy_to_sql, DataFrame
from aoa import (
    record_scoring_stats,
    tmo_create_context,
    ModelContext
)

from model_modules.data import TabularDataset
from model_modules.model import LogisticRegressionTorch
from model_modules.preprocess import transform_preprocess
from model_modules.engine import get_device, predict_proba


def score(context: ModelContext, **kwargs):
    tmo_create_context()

    device = get_device()
    batch_size = int(context.hyperparams["batch_size"])

    # Load preprocessing + optional metadata
    preprocess = joblib.load(f"{context.artifact_input_path}/preprocess.joblib")

    feature_names = context.dataset_info.feature_names
    target_name = context.dataset_info.target_names[0]
    entity_key = context.dataset_info.entity_key

    # Read scoring dataset from Teradata -> pandas
    test_df = DataFrame.from_query(context.dataset_info.sql)
    test_pdf = test_df.to_pandas(all_rows=True)

    # Prepare X (and y only if present; scoring may not have labels)
    X_test = test_pdf[feature_names]

    y_test_present = target_name in test_pdf.columns
    if y_test_present:
        y_test = test_pdf[target_name].map({'Yes': 1, 'No': 0}).values.astype(np.int64)
        y_test_f = y_test.astype(np.float32)
    else:
        y_test = None
        y_test_f = None

    # Apply same preprocessing used in training
    X_test_p = transform_preprocess(preprocess, X_test)

    # Build loader (labels optional)
    test_loader = torch.utils.data.DataLoader(
        TabularDataset(X_test_p, y_test_f) if y_test_present else TabularDataset(X_test_p, None),
        batch_size=batch_size,
        shuffle=False
    )

    # Load model
    input_dim = X_test_p.shape[1]
    print("Input dim after preprocessing:", input_dim)

    model = LogisticRegressionTorch(input_dim).to(device)
    state = torch.load(f"{context.artifact_input_path}/model.pt", map_location=device)
    model.load_state_dict(state)

    # Predict probabilities and classes
    print("Scoring")
    y_proba = predict_proba(model, test_loader, device=device)

    threshold = float(context.hyperparams.get("threshold", 0.5))
    y_pred = (y_proba >= threshold).astype(np.int64)
    print("Finished Scoring")

    # Store predictions to Teradata
    # IMPORTANT: use entity_key column values from the input if it exists;
    # otherwise fall back to row index (older behavior).
    if entity_key in test_pdf.columns:
        entity_vals = test_pdf[entity_key].values
    else:
        entity_vals = test_pdf.index.values

    predictions_pdf = pd.DataFrame({
        "job_id": context.job_id,
        entity_key: entity_vals,
        target_name: y_pred,
        "json_report": ""  # required by AOA metadata schema
    })[["job_id", entity_key, target_name, "json_report"]]

    copy_to_sql(
        df=predictions_pdf,
        schema_name=context.dataset_info.predictions_database,
        table_name=context.dataset_info.predictions_table,
        index=False,
        if_exists="append"
    )

    print("Saved predictions in Teradata")

    # Calculate + record scoring stats (AOA expects predicted_df from metadata view)
    predictions_df = DataFrame.from_query(f"""
        SELECT *
        FROM {context.dataset_info.get_predictions_metadata_fqtn()}
        WHERE job_id = '{context.job_id}'
    """)

    record_scoring_stats(
        features_df=test_df,
        predicted_df=predictions_df,
        context=context
    )

    print("All done!")
