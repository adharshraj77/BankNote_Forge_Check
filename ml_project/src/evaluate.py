"""
Model Evaluation Module for BankNote Authentication.

Provides comprehensive evaluation metrics, visualization data,
and model diagnostics for classification models.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.inspection import permutation_importance
import warnings

warnings.filterwarnings("ignore")


def compute_classification_metrics(
    y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray = None
) -> dict:
    """Compute comprehensive classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(
            y_true, y_pred, target_names=["Genuine", "Forged"], output_dict=True
        ),
    }

    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        metrics["pr_auc"] = average_precision_score(y_true, y_proba)

        fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
        metrics["roc_curve"] = {"fpr": fpr, "tpr": tpr, "thresholds": roc_thresholds}

        precision_arr, recall_arr, pr_thresholds = precision_recall_curve(
            y_true, y_proba
        )
        metrics["pr_curve"] = {
            "precision": precision_arr,
            "recall": recall_arr,
            "thresholds": pr_thresholds,
        }

    return metrics


def compute_feature_importance(model, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Compute feature importance using model-based and permutation methods.
    """
    importance_data = {}

    # Model-based importance (for tree-based models)
    if hasattr(model, "feature_importances_"):
        importance_data["model_importance"] = model.feature_importances_

    # Coefficient-based importance (for linear models)
    if hasattr(model, "coef_"):
        importance_data["coefficients"] = np.abs(model.coef_[0])

    # Permutation importance (model-agnostic)
    perm_imp = permutation_importance(
        model, X, y, n_repeats=10, random_state=42, n_jobs=-1
    )
    importance_data["permutation_importance"] = perm_imp.importances_mean
    importance_data["permutation_std"] = perm_imp.importances_std

    return pd.DataFrame(importance_data, index=X.columns)


def check_overfitting(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict:
    """Check for overfitting by comparing train and validation performance."""
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)

    train_f1 = f1_score(y_train, train_pred)
    val_f1 = f1_score(y_val, val_pred)

    gap_acc = train_acc - val_acc
    gap_f1 = train_f1 - val_f1

    return {
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "accuracy_gap": gap_acc,
        "train_f1": train_f1,
        "val_f1": val_f1,
        "f1_gap": gap_f1,
        "is_overfitting": gap_acc > 0.05,
        "is_underfitting": val_acc < 0.7,
        "diagnosis": _get_diagnosis(gap_acc, val_acc),
    }


def _get_diagnosis(gap: float, val_acc: float) -> str:
    """Generate a human-readable diagnosis."""
    if val_acc < 0.7:
        return "[Warning] Underfitting: Model performance is too low. Consider more complex models or feature engineering."
    elif gap > 0.1:
        return "[Warning] Significant Overfitting: Large gap between train and validation. Consider regularization."
    elif gap > 0.05:
        return "[Warning] Mild Overfitting: Some gap between train and validation. Monitor closely."
    elif gap < 0.01:
        return "[Optimal] Good Fit: Train and validation performance are well-aligned."
    else:
        return "[Good] Acceptable Fit: Minor gap between train and validation."


def evaluate_on_test(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Final evaluation on held-out test set."""
    y_pred = model.predict(X_test)
    y_proba = None

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test)

    metrics = compute_classification_metrics(y_test, y_pred, y_proba)
    return metrics
