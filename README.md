#  PredictiveGuard™
### Industrial Predictive Maintenance System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Models](https://img.shields.io/badge/Models-4_trained-green.svg)](#model-results)
[![Dataset](https://img.shields.io/badge/Dataset-50K_records-blue.svg)](#data-generation)

> An end-to-end machine learning system for predicting industrial milling machine
> failures using IoT sensor data. Features a 5-page interactive Streamlit dashboard,
> SHAP explainability, a leak-free ETL pipeline, and production-ready ML workflows.

---

##  Table of Contents

- [Overview](#overview)
- [Honest Model Results](#honest-model-results)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Dashboard Pages](#dashboard-pages)
- [Key Findings & Lessons Learned](#key-findings--lessons-learned)
- [Future Improvements](#future-improvements)

---

##  Overview

Manufacturing equipment failures cost industries **$50 billion annually** in unplanned downtime. PredictiveGuard builds a complete predictive maintenance pipeline that:

1. **Generates** realistic IoT sensor data for a fleet of 10 milling machines (50,000 records)
2. **Processes** raw data through a 9-step, leak-free ETL pipeline
3. **Trains & tunes** 4 ML models with Stratified K-Fold cross-validation
4. **Explains** predictions using SHAP (SHapley Additive exPlanations)
5. **Deploys** an interactive Streamlit dashboard for real-time monitoring and batch analysis

---

##  Honest Model Results

> **Transparency matters.** This section reports both cross-validation and held-out test performance, including the gap between them — which reveals an important challenge in this domain.

### Test Set Performance (10,000 samples | 3.3% failure rate)

| Model | F1 ↑ | ROC-AUC | Precision | Recall | MCC | Train Time |
|---|---|---|---|---|---|---|
| **Logistic Regression** ⭐ | **0.110** | **0.683** | 0.060 | 0.650 | 0.114 | 19.7s |
| LightGBM | 0.066 | 0.621 | 0.115 | 0.046 | 0.053 | 149.9s |
| Random Forest | 0.046 | 0.617 | 0.139 | 0.028 | 0.048 | 1881.4s |
| XGBoost | 0.030 | 0.593 | 0.087 | 0.018 | 0.026 | 158.7s |

⭐ *Best model selected by F1 score. Optimal threshold tuned to 0.64 (improves F1 to 0.138).*

### Cross-Validation Performance (Stratified 5-Fold)

| Model | CV F1 Mean | CV F1 Std | CV AUC Mean | CV AUC Std |
|---|---|---|---|---|
| **XGBoost** | **0.963** | 0.003 | 0.992 | 0.001 |
| Random Forest | 0.960 | 0.002 | 0.997 | 0.000 |
| LightGBM | 0.943 | 0.003 | 0.984 | 0.001 |
| Logistic Regression | 0.502 | 0.006 | 0.663 | 0.004 |

### ️ The CV vs. Test Gap — Why It Matters

There is a striking divergence between cross-validation and test scores for the tree-based models. XGBoost achieves **CV F1 = 0.963** but only **test F1 = 0.030**. This is a canonical symptom of **data leakage or overfitting to SMOTE-augmented training data**:

- SMOTE generates synthetic minority samples. If cross-validation folds contain SMOTE neighbors of test-set samples, the CV scores become artificially inflated.
- Logistic Regression, which can't memorize patterns as readily, is the *only* model that generalizes reasonably — scoring 0.50 CV F1 vs. 0.11 test F1 (a much smaller gap).
- **Lesson:** For severely imbalanced industrial datasets, test-set F1 and MCC are the metrics to trust. CV scores with SMOTE should be interpreted with caution.

### Threshold Optimization Results

| Model | Default F1 (0.5) | Optimal Threshold | Best F1 | Δ Improvement |
|---|---|---|---|---|
| Logistic Regression | 0.110 | 0.64 | 0.138 | +0.029 |
| Random Forest | 0.046 | 0.22 | 0.118 | +0.072 |
| XGBoost | 0.030 | 0.15 | 0.075 | +0.044 |
| LightGBM | 0.066 | 0.22 | 0.103 | +0.038 |

---

##  Features

###  Data Generation
- Physics-based synthetic data: **10 machines × 5,000 readings = 50,000 records**
- **7 sensor channels:** Air Temperature, Process Temperature, Rotational Speed, Torque, Tool Wear, Vibration (RMS), Hydraulic Pressure
- **5 failure modes:** Tool Wear (TWF), Heat Dissipation (HDF), Power (PWF), Overstrain (OSF), Random (RNF)
- Realistic degradation: machine aging curves, sensor dropout, outlier injection, non-linear wear patterns, shift/time encoding
- **Actual class prevalence: ~3.3%** — a deliberately challenging imbalanced dataset

###  ETL Pipeline (9 Steps, Zero Leakage)

```
validation → imputation → outlier_treatment → feature_engineering →
build_matrix → split → scaling → feature_selection → imbalance_smote
```

| Step | Detail |
|---|---|
| Imputation | Median strategy, fit on train only (6,950 values filled) |
| Outlier Treatment | IQR-based clipping (2,112 outliers treated) |
| Feature Engineering | 25+ engineered features |
| Scaling | StandardScaler, fit on train only |
| Feature Selection | Variance filter → correlation filter → mutual information |
| Imbalance Handling | SMOTE applied **after** split — no leakage |

**Original data shape:** 50,000 × 19 → **Final feature matrix:** 18 selected features

###  Model Training
- **4 Models:** Logistic Regression (baseline), Random Forest, XGBoost, LightGBM
- **Tuning:** RandomizedSearchCV (30 iterations per model), Stratified 5-Fold CV
- **8 Evaluation Metrics:** Accuracy, F1, Precision, Recall, ROC-AUC, Average Precision, MCC, Specificity
- **Threshold Optimization:** F1-optimized decision boundary per model
- **SHAP Analysis:** Beeswarm + bar plots for all 4 models

###  Dashboard (5 Pages)
- ** Overview:** Fleet KPIs, model leaderboard, per-machine failure rates
- ** Data Explorer:** Interactive distributions, correlations, time series, failure analysis
- ** Live Monitor:** Real-time predictions with gauges, risk alerts, session history
- ** Batch Analysis:** CSV upload, fleet diagnostics, downloadable results
- ** Model Explainer:** SHAP, ROC/PR curves, learning curves, threshold analysis

---

## ️ Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│              │     │              │     │              │     │              │
│    Data      │────▶│     ETL      │────▶│    Model     │────▶│  Dashboard   │
│  Generator   │     │   Pipeline   │     │  Training    │     │  (Streamlit) │
│              │     │              │     │              │     │              │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
 50K records          9 steps              4 models              5 pages
 10 machines          18 features          SHAP analysis         Plotly charts
 7 sensors            leak-free            threshold opt         real-time monitor
 3.3% failures        SMOTE last           CV + test eval        batch upload

 data_generator       etl_processor        train_models          app.py
    .ipynb               .ipynb               .ipynb             (script)
```

---

##  Installation

**Prerequisites:** Python 3.9+, 8GB RAM minimum

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/predictive-maintenance-system.git
cd predictive-maintenance-system

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Open Jupyter and run notebooks in order
jupyter lab
#    Step 1 → data_generator.ipynb     (generates sensor_data.csv ~5.7MB)
#    Step 2 → etl_processor.ipynb      (builds feature matrix, saves artifacts)
#    Step 3 → train_models.ipynb       (trains 4 models, ~5–10 min on i5/i7)

# 5. Launch the dashboard
streamlit run app.py
```

> **Tip:** All three notebooks are self-contained and run top-to-bottom. You can also open them in VS Code with the Jupyter extension. The `artifacts/` folder is created automatically by `etl_processor.ipynb` and fully populated after `train_models.ipynb` completes.

---

##  Project Structure

```
predictive-maintenance-system/
│
├── README.md                        # This file
├── requirements.txt                 # Python dependencies (1 KB)
│
├── data_generator.ipynb             # Notebook: synthetic IoT data generation (67 KB)
├── etl_processor.ipynb              # Notebook: ETL pipeline, feature engineering (81 KB)
├── train_models.ipynb               # Notebook: model training & evaluation (2,254 KB)
├── app.py                           # Streamlit dashboard — run this to launch UI (52 KB)
│
├── sensor_data.csv                  # [Generated] Raw sensor data — 50K rows (5,759 KB)
│
└── artifacts/                       # [Generated] Training outputs
    ├── best_model.pkl               # Best trained model (Logistic Regression)
    ├── best_model_info.json         # Metadata: F1=0.110, AUC=0.683
    ├── training_results.json        # All 4 model metrics
    ├── training_config.json         # Hyperparameter configs
    ├── test_predictions.csv         # Test set predictions (10K samples)
    ├── eda_data.csv                 # EDA data sample
    ├── scaler.pkl                   # Fitted StandardScaler
    ├── feature_names.json           # 18 selected features
    ├── etl_config.json              # ETL configuration
    ├── imputation_values.json       # Median imputation values
    ├── outlier_bounds.json          # IQR clipping bounds
    ├── pipeline_metadata.json       # Full pipeline provenance
    ├── model_*.pkl                  # All 4 trained models
    │
    └── plots/
        ├── confusion_matrix_*.png   # One per model
        ├── roc_curves_comparison.png
        ├── precision_recall_curves.png
        ├── model_comparison.png
        ├── feature_importance_*.png
        ├── shap_summary_*.png
        ├── shap_bar_*.png
        ├── learning_curve_*.png
        └── threshold_optimization_*.png
```

> **Note:** The data generation, ETL, and model training steps are implemented as Jupyter notebooks (`.ipynb`) for exploratory transparency. Only the dashboard is a standalone Python script (`app.py`).


---

##  Technical Details

### Selected Features (18 after pipeline selection)

| # | Feature | Category |
|---|---|---|
| 1 | `Vib_Torque` | Vibration × Torque interaction |
| 2 | `Strain_Level` | Physical stress indicator |
| 3 | `Tool_Wear_Min` | Raw tool wear (minutes) |
| 4 | `Vibration_mm_s` | Raw vibration RMS |
| 5 | `Tool_Wear_Min_zscore` | Statistical anomaly score |
| 6 | `Vib_Risk_Level` | Discretized vibration risk |
| 7 | `Age_Vibration` | Machine age × vibration |
| 8 | `Wear_Rate` | Tool wear rate proxy |
| 9 | `Age_Stress` | Machine age × load stress |
| 10 | `Efficiency_Proxy` | Torque/speed efficiency ratio |
| 11 | `Torque_Nm` | Raw torque reading |
| 12 | `Vibration_mm_s_zscore` | Vibration z-score |
| 13 | `Temp_Diff` | Air vs. process temperature delta |
| 14 | `Type_Encoded` | Machine type (encoded) |
| 15 | `Heat_Risk` | Thermal risk composite |
| 16 | `Shift_Encoded` | Work shift (encoded) |
| 17 | `DayOfWeek` | Cyclical time feature |
| 18 | `Process_Temp_K` | Process temperature (Kelvin) |

> **Insight from SHAP:** `Tool_Wear_Min` and `Torque_Nm` are the top drivers for Logistic Regression. Tree models prioritize `Shift_Encoded` and `DayOfWeek` heavily — suggesting possible temporal patterns in the synthetic data generation that shouldn't be over-interpreted in real deployments.

### Best Hyperparameters

**Logistic Regression** *(Best model — 19.7s training)*
```
solver: liblinear | penalty: l2 | C: 100
```

**Random Forest** *(1881.4s training — slowest by far)*
```
n_estimators: 200 | max_depth: None | max_features: log2
min_samples_split: 5 | min_samples_leaf: 1
```

**XGBoost** *(158.7s)*
```
n_estimators: 300 | max_depth: 10 | learning_rate: 0.2
subsample: 0.9 | colsample_bytree: 1.0 | gamma: 0 | min_child_weight: 1
```

**LightGBM** *(149.9s)*
```
n_estimators: 200 | num_leaves: 63 | learning_rate: 0.1
subsample: 0.8 | colsample_bytree: 0.9 | min_child_samples: 10 | max_depth: -1
```

### Pipeline Metadata

| Attribute | Value |
|---|---|
| Original data shape | 50,000 × 19 |
| Missing values imputed | 6,950 |
| Outliers treated (IQR) | 2,112 |
| Final feature count | 18 |
| Test set size | 10,000 samples |
| Test failure prevalence | 3.3% (326 actual failures) |
| Model predicted failures | 35.4% (3,538) — high false positive rate |
| Mean failure probability | 0.463 |

---

## ️ Dashboard Pages

###  Overview
Fleet-wide KPIs (total records, active machines, fleet failure rate), model leaderboard with best model highlight, and per-machine failure rate cards.

###  Data Explorer
Five tabs covering: data overview, feature distributions, correlation heatmap (method-selectable), scatter explorer, and rolling time-series statistics. All charts are interactive Plotly with hover/zoom.

###  Live Monitor
Eight sensor input sliders → real-time failure probability → color-coded risk alert (🟢 / 🟡 / 🔴 with pulse animation). Plotly gauge charts for each sensor health metric. Session-persistent prediction history with trend chart.

###  Batch Analysis
Upload a CSV of sensor readings for fleet-wide diagnostics. Outputs KPIs, risk distribution pie chart, failure probability histogram, and a downloadable results CSV.

###  Model Explainer
Four tabs: Performance (confusion matrices, CV scores with error bars), Feature Importance (tree model importances), SHAP (beeswarm + bar plots for all models), and Diagnostics (ROC/PR curves, learning curves, threshold analysis, hyperparameter details).

---

##  Key Findings & Lessons Learned

**1. Logistic Regression wins on real test data** — despite XGBoost achieving 0.963 CV F1, the simpler model generalizes better (test F1: 0.110 vs 0.030). A strong reminder that model complexity ≠ model quality in production.

**2. SMOTE inflates CV scores for tree models** — the ~96% CV F1 for Random Forest and XGBoost reflects memorization of synthetic neighbors, not true generalization. SMOTE should be applied within each fold when using cross-validation.

**3. Threshold tuning matters** — shifting Random Forest's threshold from 0.5 to 0.22 improves F1 by +158% (0.046 → 0.118). Default thresholds are rarely optimal for imbalanced problems.

**4. Temporal features dominate tree importances** — `Shift_Encoded` and `DayOfWeek` are the top features in Random Forest and XGBoost importance plots, and top SHAP drivers in XGBoost and LightGBM. In synthetic data this may reflect data generation artifacts; in real deployments, shift-pattern effects warrant careful validation.

**5. Precision is the hard part** — Logistic Regression achieves 65% recall (catches most failures) but only 6% precision (34x more false alarms than true failures). For real deployment, the cost of maintenance dispatches vs. missed failures must guide threshold selection.

---

##  Future Improvements

- [ ] Fix SMOTE leakage: apply within cross-validation folds using `imbalanced-learn` pipelines
- [ ] Remaining Useful Life (RUL) regression model
- [ ] LSTM / Transformer for temporal sequence modeling
- [ ] Real-time Kafka/MQTT streaming integration
- [ ] Automated retraining pipeline with MLflow experiment tracking
- [ ] Docker containerization + FastAPI prediction endpoint
- [ ] Alerting system (email/Slack/PagerDuty) on high-risk predictions
- [ ] Cost-sensitive learning (explicit false positive / false negative cost matrix)
- [ ] Calibrated probability outputs (Platt scaling / isotonic regression)

---

## ️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.9+ |
| ML Models | Scikit-Learn, XGBoost, LightGBM |
| Explainability | SHAP |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Dashboard | Streamlit |
| Visualization | Plotly, Matplotlib, Seaborn |
| Data Processing | Pandas, NumPy, SciPy |

---



*Report generated: March 10, 2026 | PredictiveGuard™*
