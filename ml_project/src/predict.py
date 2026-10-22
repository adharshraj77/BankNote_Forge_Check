"""
Prediction Module for BankNote Authentication.

Handles inference using saved model and preprocessing pipeline.
Ensures the same transformations are applied at prediction time.
"""

import pandas as pd
import numpy as np
import joblib
import os

FEATURE_COLUMNS = ["variance", "skewness", "curtosis", "entropy"]
CLASS_LABELS = {0: "Genuine", 1: "Forged"}


def load_artifacts(model_dir: str) -> dict:
    """Load the saved model and preprocessing pipeline."""
    model_path = os.path.join(model_dir, "best_model.joblib")
    pipeline_path = os.path.join(model_dir, "preprocessing_pipeline.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Pipeline not found at: {pipeline_path}")

    return {
        "model": joblib.load(model_path),
        "pipeline": joblib.load(pipeline_path),
    }


def validate_input(data: dict) -> tuple:
    """Validate input data and return cleaned values with any errors."""
    errors = []

    for col in FEATURE_COLUMNS:
        if col not in data:
            errors.append(f"Missing feature: {col}")
        else:
            try:
                val = float(data[col])
                if np.isnan(val) or np.isinf(val):
                    errors.append(f"Invalid value for {col}: must be a finite number")
            except (ValueError, TypeError):
                errors.append(f"Invalid value for {col}: must be numeric")

    return errors


def predict_single(
    data: dict, model, pipeline
) -> dict:
    """Make prediction for a single banknote sample."""
    errors = validate_input(data)
    if errors:
        return {"success": False, "errors": errors}

    # Create DataFrame from input
    input_df = pd.DataFrame([{col: float(data[col]) for col in FEATURE_COLUMNS}])

    # Apply preprocessing
    input_scaled = pd.DataFrame(pipeline.transform(input_df), columns=FEATURE_COLUMNS)

    # Make prediction
    prediction = model.predict(input_scaled)[0]
    label = CLASS_LABELS[prediction]

    result = {
        "success": True,
        "prediction": int(prediction),
        "label": label,
        "input_values": {col: float(data[col]) for col in FEATURE_COLUMNS},
    }

    # Add probability if available
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_scaled)[0]
        result["probability_genuine"] = float(proba[0])
        result["probability_forged"] = float(proba[1])
        result["confidence"] = float(max(proba))

    return result


def predict_batch(
    df: pd.DataFrame, model, pipeline
) -> pd.DataFrame:
    """Make predictions for a batch of samples."""
    # Validate columns
    missing_cols = set(FEATURE_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Select and order features
    X = df[FEATURE_COLUMNS].copy()

    # Handle missing values
    if X.isnull().any().any():
        raise ValueError("Input data contains missing values. Please clean the data first.")

    # Apply preprocessing
    X_scaled = pd.DataFrame(pipeline.transform(X), columns=FEATURE_COLUMNS, index=X.index)

    # Make predictions
    predictions = model.predict(X_scaled)
    results_df = df.copy()
    results_df["prediction"] = predictions
    results_df["predicted_label"] = [CLASS_LABELS[p] for p in predictions]

    # Add probabilities if available
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X_scaled)
        results_df["prob_genuine"] = probas[:, 0]
        results_df["prob_forged"] = probas[:, 1]
        results_df["confidence"] = np.max(probas, axis=1)

    return results_df
