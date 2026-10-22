"""
Data Preprocessing Module for BankNote Authentication.

Handles data loading, cleaning, validation, and preprocessing pipeline
construction. All learned transformations are fitted only on training data
to prevent data leakage.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

# Column definitions
FEATURE_COLUMNS = ["variance", "skewness", "curtosis", "entropy"]
TARGET_COLUMN = "class"
FEATURE_DESCRIPTIONS = {
    "variance": "Variance of Wavelet Transformed image",
    "skewness": "Skewness of Wavelet Transformed image",
    "curtosis": "Kurtosis of Wavelet Transformed image",
    "entropy": "Entropy of image",
}
CLASS_LABELS = {0: "Genuine", 1: "Forged"}


def load_data(filepath: str) -> pd.DataFrame:
    """Load and validate the banknote dataset."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")

    df = pd.read_csv(filepath)

    # Validate expected columns
    expected_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_cols = set(expected_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    return df


def analyze_data_quality(df: pd.DataFrame) -> dict:
    """Perform comprehensive data quality analysis."""
    analysis = {
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
        "duplicates": df.duplicated().sum(),
        "duplicate_percentage": df.duplicated().sum() / len(df) * 100,
        "class_distribution": df[TARGET_COLUMN].value_counts().to_dict(),
        "class_balance_ratio": df[TARGET_COLUMN].value_counts().min()
        / df[TARGET_COLUMN].value_counts().max(),
        "basic_stats": df.describe().to_dict(),
    }
    return analysis


def detect_outliers_iqr(df: pd.DataFrame, columns: list, factor: float = 1.5) -> dict:
    """Detect outliers using the IQR method."""
    outlier_info = {}
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        outlier_info[col] = {
            "count": len(outliers),
            "percentage": len(outliers) / len(df) * 100,
            "lower_bound": lower,
            "upper_bound": upper,
        }
    return outlier_info


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset: handle duplicates and missing values."""
    df_clean = df.copy()

    # Remove exact duplicates
    n_duplicates = df_clean.duplicated().sum()
    if n_duplicates > 0:
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)

    # Check for missing values (this dataset has none, but handle gracefully)
    if df_clean.isnull().sum().sum() > 0:
        # For numerical features, impute with median (robust to outliers)
        for col in FEATURE_COLUMNS:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)

    return df_clean


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> dict:
    """
    Split data into train, validation, and test sets.
    Stratified split to maintain class balance.
    """
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # First split: train+val vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Second split: train vs val
    val_relative_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_relative_size,
        random_state=random_state,
        stratify=y_train_val,
    )

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }


def build_preprocessing_pipeline(scaler_type: str = "standard") -> Pipeline:
    """Build a preprocessing pipeline. Fitted only on training data."""
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")

    pipeline = Pipeline([("scaler", scaler)])
    return pipeline


def preprocess_data(splits: dict, scaler_type: str = "standard") -> dict:
    """
    Apply preprocessing pipeline to the split data.
    Fits ONLY on training data, transforms all sets.
    """
    pipeline = build_preprocessing_pipeline(scaler_type)

    # Fit on training data only
    X_train_scaled = pd.DataFrame(
        pipeline.fit_transform(splits["X_train"]),
        columns=FEATURE_COLUMNS,
        index=splits["X_train"].index,
    )

    X_val_scaled = pd.DataFrame(
        pipeline.transform(splits["X_val"]),
        columns=FEATURE_COLUMNS,
        index=splits["X_val"].index,
    )

    X_test_scaled = pd.DataFrame(
        pipeline.transform(splits["X_test"]),
        columns=FEATURE_COLUMNS,
        index=splits["X_test"].index,
    )

    return {
        "X_train": X_train_scaled,
        "X_val": X_val_scaled,
        "X_test": X_test_scaled,
        "y_train": splits["y_train"],
        "y_val": splits["y_val"],
        "y_test": splits["y_test"],
        "pipeline": pipeline,
    }


def save_pipeline(pipeline: Pipeline, filepath: str) -> None:
    """Save the preprocessing pipeline to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(pipeline, filepath)


def load_pipeline(filepath: str) -> Pipeline:
    """Load a saved preprocessing pipeline."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Pipeline not found at: {filepath}")
    return joblib.load(filepath)
