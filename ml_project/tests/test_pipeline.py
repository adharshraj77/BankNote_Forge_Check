"""
Tests for preprocessing and prediction modules.
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing import (
    load_data,
    analyze_data_quality,
    clean_data,
    split_data,
    preprocess_data,
    build_preprocessing_pipeline,
    detect_outliers_iqr,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from src.predict import (
    validate_input,
    predict_single,
    predict_batch,
    load_artifacts,
    FEATURE_COLUMNS as PRED_FEATURES,
)
from src.feature_engineering import (
    compute_correlation_matrix,
    check_multicollinearity,
    compute_mutual_information,
)


# ─── Fixtures ───

@pytest.fixture
def sample_data():
    """Create a small sample dataset for testing."""
    np.random.seed(42)
    n = 100
    data = {
        "variance": np.random.randn(n) * 2,
        "skewness": np.random.randn(n) * 3,
        "curtosis": np.random.randn(n) * 4,
        "entropy": np.random.randn(n),
        "class": np.random.choice([0, 1], size=n),
    }
    return pd.DataFrame(data)


@pytest.fixture
def real_data():
    """Load the actual dataset if available."""
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "BankNoteAuthentication.csv")
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None


# ─── Data Loading Tests ───

class TestDataLoading:
    def test_load_data_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_data("nonexistent_file.csv")

    def test_load_real_data(self, real_data):
        if real_data is None:
            pytest.skip("Real dataset not available")
        assert real_data.shape[0] > 0
        assert all(col in real_data.columns for col in FEATURE_COLUMNS + [TARGET_COLUMN])

    def test_data_types(self, real_data):
        if real_data is None:
            pytest.skip("Real dataset not available")
        for col in FEATURE_COLUMNS:
            assert real_data[col].dtype in [np.float64, np.int64]


# ─── Data Quality Tests ───

class TestDataQuality:
    def test_analyze_data_quality(self, sample_data):
        quality = analyze_data_quality(sample_data)
        assert "shape" in quality
        assert "missing_values" in quality
        assert "duplicates" in quality
        assert quality["shape"] == (100, 5)

    def test_no_missing_values_real_data(self, real_data):
        if real_data is None:
            pytest.skip("Real dataset not available")
        assert real_data.isnull().sum().sum() == 0

    def test_class_distribution(self, real_data):
        if real_data is None:
            pytest.skip("Real dataset not available")
        class_counts = real_data[TARGET_COLUMN].value_counts()
        assert set(class_counts.index) == {0, 1}


# ─── Cleaning Tests ───

class TestCleaning:
    def test_clean_removes_duplicates(self):
        df = pd.DataFrame({
            "variance": [1.0, 1.0, 2.0],
            "skewness": [2.0, 2.0, 3.0],
            "curtosis": [3.0, 3.0, 4.0],
            "entropy": [0.5, 0.5, 0.6],
            "class": [0, 0, 1],
        })
        cleaned = clean_data(df)
        assert len(cleaned) == 2

    def test_clean_preserves_valid_data(self, sample_data):
        cleaned = clean_data(sample_data)
        assert len(cleaned) <= len(sample_data)


# ─── Splitting Tests ───

class TestSplitting:
    def test_split_sizes(self, sample_data):
        splits = split_data(sample_data, test_size=0.2, val_size=0.1)
        total = len(splits["X_train"]) + len(splits["X_val"]) + len(splits["X_test"])
        assert total == len(sample_data)

    def test_split_no_data_leakage(self, sample_data):
        splits = split_data(sample_data)
        train_idx = set(splits["X_train"].index)
        val_idx = set(splits["X_val"].index)
        test_idx = set(splits["X_test"].index)
        assert len(train_idx & val_idx) == 0
        assert len(train_idx & test_idx) == 0
        assert len(val_idx & test_idx) == 0

    def test_stratified_split(self, sample_data):
        splits = split_data(sample_data)
        original_ratio = sample_data[TARGET_COLUMN].mean()
        train_ratio = splits["y_train"].mean()
        assert abs(original_ratio - train_ratio) < 0.1


# ─── Preprocessing Tests ───

class TestPreprocessing:
    def test_pipeline_scales_data(self, sample_data):
        splits = split_data(sample_data)
        processed = preprocess_data(splits)
        # Check that training data is approximately standardized
        for col in FEATURE_COLUMNS:
            assert abs(processed["X_train"][col].mean()) < 0.2
            assert abs(processed["X_train"][col].std() - 1.0) < 0.3

    def test_pipeline_preserves_shape(self, sample_data):
        splits = split_data(sample_data)
        processed = preprocess_data(splits)
        assert processed["X_train"].shape == splits["X_train"].shape
        assert processed["X_test"].shape == splits["X_test"].shape


# ─── Outlier Detection Tests ───

class TestOutlierDetection:
    def test_detect_outliers(self, sample_data):
        outliers = detect_outliers_iqr(sample_data, FEATURE_COLUMNS)
        for col in FEATURE_COLUMNS:
            assert col in outliers
            assert "count" in outliers[col]
            assert outliers[col]["count"] >= 0


# ─── Prediction Input Validation Tests ───

class TestPredictionValidation:
    def test_valid_input(self):
        data = {"variance": 1.0, "skewness": 2.0, "curtosis": 3.0, "entropy": 0.5}
        errors = validate_input(data)
        assert len(errors) == 0

    def test_missing_feature(self):
        data = {"variance": 1.0, "skewness": 2.0}
        errors = validate_input(data)
        assert len(errors) > 0

    def test_invalid_value(self):
        data = {"variance": "abc", "skewness": 2.0, "curtosis": 3.0, "entropy": 0.5}
        errors = validate_input(data)
        assert len(errors) > 0

    def test_nan_value(self):
        data = {"variance": float("nan"), "skewness": 2.0, "curtosis": 3.0, "entropy": 0.5}
        errors = validate_input(data)
        assert len(errors) > 0


# ─── Feature Engineering Tests ───

class TestFeatureEngineering:
    def test_correlation_matrix(self, sample_data):
        corr = compute_correlation_matrix(sample_data)
        assert corr.shape == (4, 4)
        # Diagonal should be 1
        for col in FEATURE_COLUMNS:
            assert abs(corr.loc[col, col] - 1.0) < 1e-10

    def test_multicollinearity_check(self, sample_data):
        result = check_multicollinearity(sample_data)
        assert "has_multicollinearity" in result
        assert isinstance(result["has_multicollinearity"], bool)

    def test_mutual_information(self, sample_data):
        mi = compute_mutual_information(
            sample_data[FEATURE_COLUMNS], sample_data[TARGET_COLUMN]
        )
        assert len(mi) == 4
        assert all(v >= 0 for v in mi.values)


# --- Inference Tests with Trained Artifacts ---

class TestInferencePipeline:
    def test_load_artifacts(self):
        model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        artifacts = load_artifacts(model_dir)
        assert "model" in artifacts
        assert "pipeline" in artifacts
        assert hasattr(artifacts["model"], "predict")
        assert hasattr(artifacts["pipeline"], "transform")

    def test_predict_single_genuine(self):
        model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        artifacts = load_artifacts(model_dir)
        # Typical genuine banknote features (high variance, e.g. 3.6, -2.8, 1.2, 0.4)
        sample = {"variance": 3.6216, "skewness": 8.6661, "curtosis": -2.8073, "entropy": -0.44699}
        result = predict_single(sample, artifacts["model"], artifacts["pipeline"])
        assert result["success"] is True
        assert result["prediction"] in [0, 1]
        assert result["label"] in ["Genuine", "Forged"]
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_single_forged(self):
        model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        artifacts = load_artifacts(model_dir)
        # Typical forged banknote features (negative variance)
        sample = {"variance": -3.5637, "skewness": -8.3827, "curtosis": 12.393, "entropy": -1.2823}
        result = predict_single(sample, artifacts["model"], artifacts["pipeline"])
        assert result["success"] is True
        assert result["prediction"] in [0, 1]
        assert result["label"] in ["Genuine", "Forged"]
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_batch(self, real_data):
        if real_data is None:
            pytest.skip("Real dataset not available")
        model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        artifacts = load_artifacts(model_dir)
        batch = real_data.head(20)
        predictions_df = predict_batch(batch, artifacts["model"], artifacts["pipeline"])
        assert len(predictions_df) == 20
        assert "prediction" in predictions_df.columns
        assert "predicted_label" in predictions_df.columns
        assert "confidence" in predictions_df.columns
        assert all(predictions_df["confidence"] >= 0.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
