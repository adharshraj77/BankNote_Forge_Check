# 🏦 BankNote Authentication — ML Application

A production-quality machine learning application that authenticates banknotes as **Genuine** or **Forged** using wavelet-transformed image features. Includes a premium Streamlit web interface for real-time predictions, model insights, and data exploration.

---

## 📋 Project Overview

| Property | Detail |
|----------|--------|
| **Problem Type** | Binary Classification |
| **Dataset** | UCI BankNote Authentication (1,372 samples) |
| **Features** | 4 wavelet-transformed image attributes |
| **Target** | Class (0 = Genuine, 1 = Forged) |
| **Best Model** | Selected via automated comparison of 8 algorithms |
| **Interface** | Premium Streamlit web application |

## 🗂️ Project Structure

```
ml_project/
├── app.py                      # Streamlit web application
├── run_training.py             # Training pipeline orchestrator
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── data/
│   └── BankNoteAuthentication.csv  # Dataset
├── notebooks/
│   └── BankNote_Authentication_Analysis.ipynb  # Interactive EDA & Modeling Notebook
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        # Data loading, cleaning, scaling
│   ├── feature_engineering.py  # Feature analysis & creation
│   ├── train.py                # Model training & tuning
│   ├── evaluate.py             # Evaluation metrics & diagnostics
│   └── predict.py              # Inference pipeline
├── models/                     # Saved model artifacts
│   ├── best_model.joblib
│   ├── preprocessing_pipeline.joblib
│   ├── model_comparison.csv
│   ├── test_metrics.json
│   ├── feature_importance.csv
│   └── overfitting_check.json
└── tests/
    └── test_pipeline.py        # Automated test suite (25 test cases)
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.9 or higher
- pip package manager

### 2. Installation

```bash
cd ml_project

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (macOS/Linux)
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Train the Model

```bash
python run_training.py
```

This will:
- Load and analyze the dataset
- Clean data and detect outliers
- Train and compare 8 ML algorithms
- Tune the best model's hyperparameters
- Evaluate on a held-out test set
- Save the model and preprocessing pipeline

### 4. Launch the Application

```bash
streamlit run app.py
```

The application will open at `http://localhost:8501`.

### 5. Run Tests

```bash
pytest tests/ -v
```

## 📊 Dataset Description

The dataset contains features extracted from wavelet-transformed images of banknotes:

| Feature | Description |
|---------|-------------|
| `variance` | Variance of Wavelet Transformed image |
| `skewness` | Skewness of Wavelet Transformed image |
| `curtosis` | Kurtosis of Wavelet Transformed image |
| `entropy` | Entropy of image |
| `class` | Target: 0 = Genuine, 1 = Forged |

**Key Characteristics:**
- 1,372 samples (763 genuine, 610 forged)
- No missing values
- Some duplicate records (handled during preprocessing)
- Reasonably balanced classes (ratio ≈ 0.80)
- All features are continuous numerical values

## 🤖 Machine Learning Pipeline

### Preprocessing
- Duplicate removal
- StandardScaler normalization (fitted on training data only)
- Stratified train/validation/test split (70/10/20)

### Models Evaluated
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting
5. SVM (RBF kernel)
6. K-Nearest Neighbors
7. Naive Bayes
8. AdaBoost

### Selection Criteria
- 5-fold cross-validated ROC-AUC on training data
- Validation set accuracy and F1-score
- Overfitting diagnostics (train vs. validation gap)
- Final evaluation on held-out test set

### Feature Engineering
- Correlation analysis and multicollinearity check
- Mutual Information scoring
- ANOVA F-score ranking
- Permutation-based importance analysis

### 🏆 Model Benchmark & Test Set Results

| Model | 5-Fold CV ROC-AUC | Validation Accuracy | Status |
|:------|:------------------:|:--------------------:|:-------|
| **SVM (RBF)** | **1.0000 ± 0.0000** | **100.00%** | **Selected Best Model** |
| Logistic Regression | 0.9999 ± 0.0002 | 100.00% | Runner-up |
| Random Forest | 0.9998 ± 0.0002 | 100.00% | Strong Ensemble |
| Gradient Boosting | 0.9998 ± 0.0003 | 100.00% | Strong Ensemble |
| AdaBoost | 0.9997 ± 0.0005 | 100.00% | Good Boosting |
| KNN | 0.9990 ± 0.0019 | 100.00% | Strong Instance-based |
| Decision Tree | 0.9800 ± 0.0042 | 100.00% | Baseline Tree |
| Naive Bayes | 0.9406 ± 0.0104 | 81.48% | Baseline Probabilistic |

#### Final Held-Out Test Set Metrics (Tuned SVM RBF):
- **Accuracy**: `99.26%` (268 / 270 correct)
- **Recall (Forged)**: `100.00%` (122 / 122 forged banknotes detected — **0 false negatives**)
- **Precision**: `98.39%`
- **F1-Score**: `99.19%`
- **ROC-AUC**: `1.0000`
- **PR-AUC**: `1.0000`
- **Overfitting Diagnosis**: `[Optimal] Good Fit: Train and validation performance are well-aligned.`

## 🌐 Streamlit Application

The app includes four premium pages:

### 📊 Dashboard
- Executive KPI summary (Accuracy, F1, ROC-AUC, Precision, Recall)
- Dataset overview and class distribution
- Feature distribution histograms by class
- Model comparison bar charts
- Correlation heatmap

### 🔮 Prediction Workspace
- **Single Prediction**: Enter 4 features → get instant authentication result
- **Batch Prediction**: Upload CSV → authenticate multiple banknotes
- Confidence scores with gauge visualizations
- Feature context analysis
- Downloadable results (JSON/CSV)

### 🧠 Model Insights
- Performance gauge charts
- Interactive confusion matrix with interpretation
- Feature importance (permutation + model-based)
- Overfitting diagnostics (train vs. validation comparison)
- Selected model details and hyperparameters

### 🔍 Data Explorer
- Filterable data preview
- Per-class summary statistics
- Interactive scatter plots (choose X/Y features)
- Box plots by class
- Outlier analysis with IQR bounds
- Downloadable filtered data

## ⚠️ Limitations & Assumptions

1. **Dataset Scope**: Model trained on UCI BankNote Authentication dataset; performance on different banknote types or imaging conditions may vary.
2. **Feature Extraction**: Assumes wavelet-transformed features are pre-computed; the application does not process raw images.
3. **Class Balance**: The dataset is roughly balanced; no resampling was applied.
4. **Feature Count**: Only 4 features are used. The model's decision boundary is relatively simple, which aids interpretability but may limit performance on more complex forgery patterns.
5. **No Temporal Validation**: Data is not time-stamped; there is no temporal drift assessment.

## 🔮 Future Improvements

- Integrate raw image processing pipeline (Wavelet Transform computation)
- Add SHAP or LIME explanations for individual predictions
- Implement model monitoring and drift detection
- Add user authentication for the web application
- Support multiple currency types and denominations
- Deploy as a containerized microservice (Docker)
- Add A/B testing framework for model comparison in production

## 📄 License

This project is for educational and demonstration purposes.
Dataset source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/banknote+authentication)
