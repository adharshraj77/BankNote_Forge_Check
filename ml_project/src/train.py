"""
Model Training Module for BankNote Authentication.

Trains multiple classification models, performs hyperparameter tuning,
and selects the best model based on cross-validation performance.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    make_scorer,
)
import joblib
import os
import time
import warnings

warnings.filterwarnings("ignore")


def get_baseline_models() -> dict:
    """Return a dictionary of baseline models for comparison."""
    models = {
        "Logistic Regression": LogisticRegression(
            random_state=42, max_iter=1000, solver="lbfgs"
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42, n_estimators=100
        ),
        "SVM (RBF)": SVC(random_state=42, probability=True, kernel="rbf"),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "AdaBoost": AdaBoostClassifier(random_state=42, n_estimators=100),
    }
    return models


def train_and_evaluate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cv_folds: int = 5,
) -> pd.DataFrame:
    """
    Train all baseline models and evaluate using cross-validation
    and validation set performance.
    """
    models = get_baseline_models()
    results = []

    for name, model in models.items():
        start_time = time.time()

        # Cross-validation on training data
        cv_accuracy = cross_val_score(
            model, X_train, y_train, cv=cv_folds, scoring="accuracy"
        )
        cv_f1 = cross_val_score(
            model, X_train, y_train, cv=cv_folds, scoring="f1"
        )
        cv_roc_auc = cross_val_score(
            model, X_train, y_train, cv=cv_folds, scoring="roc_auc"
        )

        # Fit on full training data and evaluate on validation
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        y_val_proba = (
            model.predict_proba(X_val)[:, 1]
            if hasattr(model, "predict_proba")
            else model.decision_function(X_val)
        )

        train_time = time.time() - start_time

        results.append(
            {
                "Model": name,
                "CV_Accuracy_Mean": cv_accuracy.mean(),
                "CV_Accuracy_Std": cv_accuracy.std(),
                "CV_F1_Mean": cv_f1.mean(),
                "CV_F1_Std": cv_f1.std(),
                "CV_ROC_AUC_Mean": cv_roc_auc.mean(),
                "CV_ROC_AUC_Std": cv_roc_auc.std(),
                "Val_Accuracy": accuracy_score(y_val, y_val_pred),
                "Val_F1": f1_score(y_val, y_val_pred),
                "Val_ROC_AUC": roc_auc_score(y_val, y_val_proba),
                "Training_Time_s": train_time,
            }
        )

    return pd.DataFrame(results).sort_values("CV_ROC_AUC_Mean", ascending=False)


def get_tuning_params(model_name: str) -> dict:
    """Return hyperparameter search space for the given model."""
    param_grids = {
        "Random Forest": {
            "n_estimators": [50, 100, 200, 300],
            "max_depth": [None, 5, 10, 15, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
        },
        "Gradient Boosting": {
            "n_estimators": [50, 100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 5, 7, 10],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "subsample": [0.8, 0.9, 1.0],
        },
        "SVM (RBF)": {
            "C": [0.1, 1, 10, 100],
            "gamma": ["scale", "auto", 0.01, 0.1],
            "kernel": ["rbf"],
        },
        "KNN": {
            "n_neighbors": [3, 5, 7, 9, 11, 15],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan", "minkowski"],
        },
        "Logistic Regression": {
            "C": [0.001, 0.01, 0.1, 1, 10, 100],
            "penalty": ["l2"],
            "solver": ["lbfgs", "liblinear"],
        },
    }
    return param_grids.get(model_name, {})


def tune_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 50,
    cv_folds: int = 5,
) -> dict:
    """Perform hyperparameter tuning using RandomizedSearchCV."""
    models = get_baseline_models()
    model = models[model_name]
    param_grid = get_tuning_params(model_name)

    if not param_grid:
        # No tuning params available, just fit and return
        model.fit(X_train, y_train)
        return {"best_model": model, "best_params": {}, "best_score": None}

    search = RandomizedSearchCV(
        model,
        param_distributions=param_grid,
        n_iter=min(n_iter, 50),
        cv=cv_folds,
        scoring="roc_auc",
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )

    search.fit(X_train, y_train)

    return {
        "best_model": search.best_estimator_,
        "best_params": search.best_params_,
        "best_score": search.best_score_,
    }


def save_model(model, filepath: str) -> None:
    """Save trained model to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)


def load_model(filepath: str):
    """Load a saved model from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model not found at: {filepath}")
    return joblib.load(filepath)
