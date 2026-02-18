import numpy as np
import pandas as pd
import string

from teradataml import copy_to_sql
from tmo import (tmo_create_context, ModelContext)

np.random.seed(None)


# -----------------------------
# Helper generators
# -----------------------------
def generate_customer_id_sequential(n, start=1):
    letters = string.ascii_uppercase
    ids = []
    for i in range(start, start + n):
        # base-26 “code” for the suffix
        x = i
        suffix = []
        for _ in range(5):
            suffix.append(letters[x % 26])
            x //= 26
        ids.append(f"{i:04d}-" + "".join(suffix))
    return ids


def generate_yes_no(n, p_yes=0.5):
    return np.random.choice(['Yes', 'No'], size=n, p=[p_yes, 1 - p_yes])


def generate_gender(n):
    return np.random.choice(['Male', 'Female'], size=n)


def generate_contract(n):
    return np.random.choice(
        ['Month-to-month', 'One year', 'Two year'],
        size=n,
        p=[0.55, 0.25, 0.20]
    )


def generate_payment(n):
    return np.random.choice([
        'Electronic check',
        'Mailed check',
        'Bank transfer (automatic)',
        'Credit card (automatic)'
    ], size=n)


def generate_internet(n):
    return np.random.choice(
        ['DSL', 'Fiber optic', 'No'],
        size=n,
        p=[0.3, 0.5, 0.2]
    )

def shift_probabilities(base_probs, shift_vector, min_prob=0.01):
    probs = np.array(base_probs) + np.array(shift_vector)
    probs = np.clip(probs, min_prob, None)
    return probs / probs.sum()

# -----------------------------
# Base Data Creation
# -----------------------------

def generate_base_telco(n):
    df = pd.DataFrame()

    df["CustomerID"] = generate_customer_id_sequential(n)
    df["Gender"] = generate_gender(n)
    df["SeniorCitizen"] = generate_yes_no(n, 0.20)
    df["Partner"] = generate_yes_no(n, 0.5)
    df["Dependents"] = generate_yes_no(n, 0.3)

    df["Tenure"] = np.random.randint(0, 72, n)

    df["PhoneService"] = generate_yes_no(n, 0.9)
    df["MultipleLines"] = generate_yes_no(n, 0.45)

    df["InternetService"] = generate_internet(n)

    internet_users = df["InternetService"] != "No"

    def internet_feature():
        return np.where(
            internet_users,
            generate_yes_no(n, 0.4),
            "No internet service"
        )

    df["OnlineSecurity"] = internet_feature()
    df["OnlineBackup"] = internet_feature()
    df["DeviceProtection"] = internet_feature()
    df["TechSupport"] = internet_feature()
    df["StreamingTV"] = internet_feature()
    df["StreamingMovies"] = internet_feature()

    df["Contract"] = generate_contract(n)
    df["PaperlessBilling"] = generate_yes_no(n, 0.6)
    df["PaymentMethod"] = generate_payment(n)

    # Monthly charges influenced by internet + contract
    base_charge = np.random.normal(70, 25, n)

    fiber_mask = df["InternetService"] == "Fiber optic"
    base_charge[fiber_mask] += 25

    long_contract = df["Contract"] == "Two year"
    base_charge[long_contract] -= 10

    df["MonthlyCharges"] = np.round(np.clip(base_charge, 18, 130), 2)

    df["TotalCharges"] = np.round(df["MonthlyCharges"] * df["Tenure"], 2)

    return df

# -----------------------------
# Churn Generation
# -----------------------------

def generate_churn(df, concept_strength=1.0, target_rate=0.25, rng=None):
    rng = rng or np.random.default_rng()

    # --- risk score (log-odds-ish) ---
    risk = np.zeros(len(df), dtype=float)

    risk += (df["Contract"] == "Month-to-month") * 1.5 * concept_strength
    risk += (df["InternetService"] == "Fiber optic") * 0.8 * concept_strength
    risk += (df["TechSupport"] == "No") * 0.6 * concept_strength
    risk += (df["Tenure"] < 12) * 1.0 * concept_strength
    risk += (df["MonthlyCharges"] > 90) * 0.7 * concept_strength

    # --- calibrate intercept so mean(prob) ~= target_rate ---
    # Start with intercept=0, compute mean probability
    p0 = 1 / (1 + np.exp(-(risk)))
    mean_p0 = float(p0.mean())

    # Convert target and current mean probs to logit space; delta is the intercept shift
    eps = 1e-6
    mean_p0 = np.clip(mean_p0, eps, 1 - eps)
    target_rate = np.clip(target_rate, eps, 1 - eps)

    logit = lambda p: np.log(p / (1 - p))
    intercept_shift = logit(target_rate) - logit(mean_p0)

    # Apply intercept shift
    p = 1 / (1 + np.exp(-(risk + intercept_shift)))

    churn = rng.binomial(1, p)
    return np.where(churn == 1, "Yes", "No")

# -----------------------------
# Drift Injection
# -----------------------------

def apply_drift(df,
                covariate_shift=0.0,
                prior_shift=0.0,
                concept_shift=0.0):

    df = df.copy()

    # -------------------------
    # Covariate drift
    # -------------------------
    if covariate_shift > 0:
        df["MonthlyCharges"] *= (1 + np.random.normal(0, covariate_shift, len(df)))

        # More fiber adoption
        new_probs = shift_probabilities(
            base_probs=[0.3, 0.5, 0.2],
            shift_vector=[-covariate_shift/2, covariate_shift, -covariate_shift/2]
        )

        df["InternetService"] = np.random.choice(
            ['DSL', 'Fiber optic', 'No'],
            size=len(df),
            p=new_probs
        )

        # fiber_prob = min(0.5 + covariate_shift, 0.9)
        # df["InternetService"] = np.random.choice(
        #     ['DSL', 'Fiber optic', 'No'],
        #     size=len(df),
        #     p=[0.3 - covariate_shift/2, fiber_prob, 0.7 - fiber_prob]
        # )

    # -------------------------
    # Concept drift
    # -------------------------
    concept_strength = 1 + concept_shift
    df["Churn"] = generate_churn(df, concept_strength)

    # -------------------------
    # Prior drift (overall rate)
    # -------------------------
    if prior_shift > 0:
        flip = np.random.rand(len(df)) < prior_shift
        df.loc[flip, "Churn"] = np.where(
            df.loc[flip, "Churn"] == "Yes",
            "No",
            "Yes"
        )

    return df

# ----------------------------------
# TRAIN DATA GENERATION FUNCTION
# ----------------------------------

def generate_training_data(n=5000,
                           covariate_drift=0.02,
                           prior_drift=0.01,
                           concept_drift=0.01):

    df = generate_base_telco(n)
    df["Churn"] = generate_churn(df)

    df = apply_drift(
        df,
        covariate_shift=covariate_drift,
        prior_shift=prior_drift,
        concept_shift=concept_drift
    )

    return df

# ----------------------------------
# TRAIN DATA GENERATION FUNCTION
# ----------------------------------

def generate_test_data(n=2000,
                       covariate_drift=0.15,
                       prior_drift=0.05,
                       concept_drift=0.10):

    df = generate_base_telco(n)
    df["Churn"] = generate_churn(df)

    df = apply_drift(
        df,
        covariate_shift=covariate_drift,
        prior_shift=prior_drift,
        concept_shift=concept_drift
    )

    return df


# ----------------------------------
# Tasks to push data to Teradata
# ----------------------------------

def run_train_data_task(context: ModelContext, **kwargs):
    tmo_create_context()

    train_df = generate_training_data()
    test_df = generate_test_data()

    train_df['train'] = 1
    test_df['train'] = 0

    df_combine = pd.concat([train_df, test_df], axis=0)
    # Write pandas DataFrame to a Teradata table
    copy_to_sql(df = df_combine, table_name = 'telco_model_data', if_exists = 'replace')

    # Store build properties as a file artifact
    with open(f"{context.artifact_output_path}/build_properties.txt", "w") as f:
        f.write(str(kwargs))

def run_score_data_task(context: ModelContext, **kwargs):
    tmo_create_context()

    test_df = generate_test_data(n=2000,
                       covariate_drift=0.20,
                       prior_drift=0.10,
                       concept_drift=0.15)
    test_df['train'] = 2

    # Write pandas DataFrame to a Teradata table
    try:
        copy_to_sql(df = test_df, table_name = 'telco_model_data', if_exists = 'append')
        print('Scorring data appended to existing model data table')
    except:
        copy_to_sql(df = test_df, table_name = 'telco_model_data', if_exists = 'replace')
        print('Scorring data appended to new model data table')


    # Store build properties as a file artifact
    with open(f"{context.artifact_output_path}/build_properties.txt", "w") as f:
        f.write(str(kwargs))

