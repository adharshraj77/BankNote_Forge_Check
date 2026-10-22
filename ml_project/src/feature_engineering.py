"""
Feature Engineering Module for BankNote Authentication.

Provides feature analysis, creation, selection, and multicollinearity
analysis. Since the dataset contains 4 pre-extracted wavelet features,
engineering focuses on interaction terms and statistical feature selection.
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, f_classif
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

FEATURE_COLUMNS = ["variance", "skewness", "curtosis", "entropy"]


def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Pearson correlation matrix for feature columns."""
    return df[FEATURE_COLUMNS].corr()


def compute_feature_target_correlation(df: pd.DataFrame, target: str = "class") -> pd.Series:
    """Compute point-biserial correlation between features and binary target."""
    correlations = {}
    for col in FEATURE_COLUMNS:
        corr, pval = stats.pointbiserialr(df[target], df[col])
        correlations[col] = {"correlation": corr, "p_value": pval}
    return pd.DataFrame(correlations).T


def compute_mutual_information(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Compute mutual information between features and target."""
    mi_scores = mutual_info_classif(X, y, random_state=42)
    return pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)


def compute_anova_f_scores(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Compute ANOVA F-scores for feature selection."""
    f_scores, p_values = f_classif(X, y)
    return pd.DataFrame(
        {"f_score": f_scores, "p_value": p_values}, index=X.columns
    ).sort_values("f_score", ascending=False)


def check_multicollinearity(df: pd.DataFrame, threshold: float = 0.8) -> dict:
    """
    Check for multicollinearity using correlation matrix.
    Returns pairs of features with correlation above threshold.
    """
    corr_matrix = df[FEATURE_COLUMNS].corr().abs()
    high_corr_pairs = []

    for i in range(len(FEATURE_COLUMNS)):
        for j in range(i + 1, len(FEATURE_COLUMNS)):
            if corr_matrix.iloc[i, j] > threshold:
                high_corr_pairs.append(
                    {
                        "feature_1": FEATURE_COLUMNS[i],
                        "feature_2": FEATURE_COLUMNS[j],
                        "correlation": corr_matrix.iloc[i, j],
                    }
                )

    return {
        "correlation_matrix": corr_matrix,
        "high_correlation_pairs": high_corr_pairs,
        "has_multicollinearity": len(high_corr_pairs) > 0,
    }


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features between the original wavelet features.
    These can capture non-linear relationships.
    """
    df_eng = df.copy()

    # Ratio features (with small epsilon to avoid division by zero)
    eps = 1e-8
    df_eng["var_skew_ratio"] = df_eng["variance"] / (df_eng["skewness"].abs() + eps)
    df_eng["var_kurt_ratio"] = df_eng["variance"] / (df_eng["curtosis"].abs() + eps)

    # Product interactions
    df_eng["var_x_skew"] = df_eng["variance"] * df_eng["skewness"]
    df_eng["var_x_entropy"] = df_eng["variance"] * df_eng["entropy"]

    # Magnitude features
    df_eng["feature_magnitude"] = np.sqrt(
        df_eng["variance"] ** 2
        + df_eng["skewness"] ** 2
        + df_eng["curtosis"] ** 2
        + df_eng["entropy"] ** 2
    )

    return df_eng


def get_feature_importance_summary(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Generate a comprehensive feature importance summary."""
    mi_scores = compute_mutual_information(X, y)
    anova = compute_anova_f_scores(X, y)

    summary = pd.DataFrame(
        {
            "mutual_information": mi_scores,
            "anova_f_score": anova["f_score"],
            "anova_p_value": anova["p_value"],
        }
    )

    # Rank features
    summary["mi_rank"] = summary["mutual_information"].rank(ascending=False)
    summary["anova_rank"] = summary["anova_f_score"].rank(ascending=False)
    summary["avg_rank"] = (summary["mi_rank"] + summary["anova_rank"]) / 2

    return summary.sort_values("avg_rank")
