"""
Main Training Script for BankNote Authentication.

Orchestrates the full ML pipeline: data loading, EDA, preprocessing,
model training, evaluation, and artifact saving.

Usage:
    python run_training.py
"""

import sys
import os
import io

# Ensure UTF-8 output encoding on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import json
from src.preprocessing import (
    load_data,
    analyze_data_quality,
    detect_outliers_iqr,
    clean_data,
    split_data,
    preprocess_data,
    save_pipeline,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from src.feature_engineering import (
    compute_correlation_matrix,
    compute_mutual_information,
    compute_anova_f_scores,
    check_multicollinearity,
    get_feature_importance_summary,
)
from src.train import train_and_evaluate_models, tune_model, save_model
from src.evaluate import (
    compute_classification_metrics,
    compute_feature_importance,
    check_overfitting,
    evaluate_on_test,
)
import joblib


def main():
    print("=" * 70)
    print("  BankNote Authentication - ML Pipeline")
    print("=" * 70)

    # --- Step 1: Load Data ---
    print("\n[Step 1] Loading Data...")
    data_path = os.path.join(os.path.dirname(__file__), "data", "BankNoteAuthentication.csv")
    df = load_data(data_path)
    print(f"   Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")

    # --- Step 2: Data Quality Analysis ---
    print("\n[Step 2] Data Quality Analysis...")
    quality = analyze_data_quality(df)
    print(f"   Missing values: {sum(quality['missing_values'].values())}")
    print(f"   Duplicates: {quality['duplicates']} ({quality['duplicate_percentage']:.1f}%)")
    print(f"   Class distribution: {quality['class_distribution']}")
    print(f"   Class balance ratio: {quality['class_balance_ratio']:.3f}")

    # Outlier detection
    outliers = detect_outliers_iqr(df, FEATURE_COLUMNS)
    print("\n   Outlier Analysis (IQR method):")
    for col, info in outliers.items():
        print(f"     {col}: {info['count']} outliers ({info['percentage']:.1f}%)")

    # --- Step 3: Clean Data ---
    print("\n[Step 3] Cleaning Data...")
    df_clean = clean_data(df)
    print(f"   After cleaning: {df_clean.shape[0]} rows ({df.shape[0] - df_clean.shape[0]} removed)")

    # --- Step 4: Feature Analysis ---
    print("\n[Step 4] Feature Analysis...")
    corr_matrix = compute_correlation_matrix(df_clean)
    multicoll = check_multicollinearity(df_clean)
    print(f"   Multicollinearity detected: {multicoll['has_multicollinearity']}")
    if multicoll['high_correlation_pairs']:
        for pair in multicoll['high_correlation_pairs']:
            print(f"     {pair['feature_1']} <-> {pair['feature_2']}: {pair['correlation']:.3f}")

    mi_scores = compute_mutual_information(
        df_clean[FEATURE_COLUMNS], df_clean[TARGET_COLUMN]
    )
    print("\n   Mutual Information Scores:")
    for feat, score in mi_scores.items():
        print(f"     {feat}: {score:.4f}")

    # --- Step 5: Split & Preprocess ---
    print("\n[Step 5] Splitting & Preprocessing...")
    splits = split_data(df_clean)
    print(f"   Train: {len(splits['X_train'])} | Val: {len(splits['X_val'])} | Test: {len(splits['X_test'])}")

    processed = preprocess_data(splits, scaler_type="standard")
    print("   StandardScaler fitted on training data")

    # --- Step 6: Train Models ---
    print("\n[Step 6] Training & Comparing Models...")
    results_df = train_and_evaluate_models(
        processed["X_train"],
        processed["y_train"],
        processed["X_val"],
        processed["y_val"],
    )

    print("\n   Model Comparison (sorted by CV ROC-AUC):")
    print("   " + "-" * 65)
    for _, row in results_df.iterrows():
        print(
            f"   {row['Model']:<22} | CV ROC-AUC: {row['CV_ROC_AUC_Mean']:.4f} +/- {row['CV_ROC_AUC_Std']:.4f} | Val Acc: {row['Val_Accuracy']:.4f}"
        )

    # --- Step 7: Tune Best Model ---
    best_model_name = results_df.iloc[0]["Model"]
    print(f"\n[Step 7] Hyperparameter Tuning ({best_model_name})...")
    tuning_result = tune_model(
        best_model_name, processed["X_train"], processed["y_train"]
    )
    best_model = tuning_result["best_model"]
    print(f"   Best params: {tuning_result['best_params']}")
    if tuning_result["best_score"]:
        print(f"   Best CV ROC-AUC: {tuning_result['best_score']:.4f}")

    # --- Step 8: Evaluate Best Model ---
    print("\n[Step 8] Evaluating Tuned Model...")

    # Overfitting check
    overfit = check_overfitting(
        best_model,
        processed["X_train"],
        processed["y_train"],
        processed["X_val"],
        processed["y_val"],
    )
    print(f"   {overfit['diagnosis']}")
    print(f"   Train Accuracy: {overfit['train_accuracy']:.4f} | Val Accuracy: {overfit['val_accuracy']:.4f}")

    # Test set evaluation
    test_metrics = evaluate_on_test(
        best_model, processed["X_test"], processed["y_test"]
    )
    print(f"\n   Test Set Results:")
    print(f"     Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"     Precision: {test_metrics['precision']:.4f}")
    print(f"     Recall:    {test_metrics['recall']:.4f}")
    print(f"     F1-Score:  {test_metrics['f1_score']:.4f}")
    if "roc_auc" in test_metrics:
        print(f"     ROC-AUC:   {test_metrics['roc_auc']:.4f}")

    print(f"\n   Confusion Matrix:")
    cm = test_metrics["confusion_matrix"]
    print(f"     Genuine correctly identified: {cm[0][0]}")
    print(f"     Genuine misclassified:        {cm[0][1]}")
    print(f"     Forged correctly identified:   {cm[1][1]}")
    print(f"     Forged misclassified:          {cm[1][0]}")

    # Feature importance
    feat_imp = compute_feature_importance(
        best_model, processed["X_test"], processed["y_test"]
    )
    print("\n   Feature Importance (Permutation):")
    for feat in feat_imp.sort_values("permutation_importance", ascending=False).index:
        print(f"     {feat}: {feat_imp.loc[feat, 'permutation_importance']:.4f}")

    # --- Step 9: Save Artifacts ---
    print("\n[Step 9] Saving Artifacts...")
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(model_dir, exist_ok=True)

    save_model(best_model, os.path.join(model_dir, "best_model.joblib"))
    save_pipeline(processed["pipeline"], os.path.join(model_dir, "preprocessing_pipeline.joblib"))

    # Save model comparison results
    results_df.to_csv(os.path.join(model_dir, "model_comparison.csv"), index=False)

    # Save test metrics
    serializable_metrics = {
        k: v for k, v in test_metrics.items()
        if k not in ["confusion_matrix", "roc_curve", "pr_curve", "classification_report"]
    }
    serializable_metrics["confusion_matrix"] = cm.tolist()
    serializable_metrics["best_model_name"] = best_model_name
    serializable_metrics["best_params"] = {
        str(k): str(v) for k, v in tuning_result["best_params"].items()
    }

    with open(os.path.join(model_dir, "test_metrics.json"), "w") as f:
        json.dump(serializable_metrics, f, indent=2)

    # Save feature importance
    feat_imp.to_csv(os.path.join(model_dir, "feature_importance.csv"))

    # Save overfitting diagnostics
    overfit_serializable = {k: v for k, v in overfit.items() if isinstance(v, (int, float, str, bool))}
    with open(os.path.join(model_dir, "overfitting_check.json"), "w") as f:
        json.dump(overfit_serializable, f, indent=2)

    print(f"   [OK] Model saved to: {model_dir}/best_model.joblib")
    print(f"   [OK] Pipeline saved to: {model_dir}/preprocessing_pipeline.joblib")
    print(f"   [OK] Metrics saved to: {model_dir}/test_metrics.json")

    print("\n" + "=" * 70)
    print("  [SUCCESS] Training Pipeline Complete!")
    print(f"  Best Model: {best_model_name}")
    print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    if "roc_auc" in test_metrics:
        print(f"  Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
