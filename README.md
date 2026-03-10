#  PredictiveGuard™
### Industrial Predictive Maintenance System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Models](https://img.shields.io/badge/Models-4_trained-green.svg)](#model-results)
[![Dataset](https://img.shields.io/badge/Dataset-50K_records-blue.svg)](#data-generation)

> An end-to-end machine learning system for predicting industrial milling machine
> failures using IoT sensor data. Includes a 4-tab interactive Streamlit dashboard
> with model explainability, SHAP analysis, diagnostic curves, and one-click
> PDF report generation.

<img width="3433" height="1220" alt="image" src="https://github.com/user-attachments/assets/b58a5f70-517e-4584-b72d-80eac15398bc" />

---

## Table of Contents

- [Overview](#overview)
- [Honest Model Results](#honest-model-results)
- [Architecture](#architecture)
- [Dashboard — 4 Tabs In Detail](#dashboard--4-tabs-in-detail)
- [PDF Report Generation](#pdf-report-generation)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Key Findings & Lessons Learned](#key-findings--lessons-learned)
- [Future Improvements](#future-improvements)

---

##  Overview

Manufacturing equipment failures cost industries **$50 billion annually** in unplanned downtime. PredictiveGuard builds a complete predictive maintenance pipeline that:

1. **Generates** realistic IoT sensor data for a fleet of 10 milling machines (50,000 records)
2. **Processes** raw data through a 9-step, leak-free ETL pipeline
3. **Trains & tunes** 4 ML models with Stratified K-Fold cross-validation
4. **Explains** predictions using SHAP (SHapley Additive exPlanations)
5. **Deploys** a focused Model Explainability Dashboard with one-click PDF reporting

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

⭐ *Best model selected by F1. Optimal threshold tuned to 0.64 (improves F1 to 0.138).*

### Cross-Validation Performance (Stratified 5-Fold)

| Model | CV F1 Mean | CV F1 Std | CV AUC Mean | CV AUC Std |
|---|---|---|---|---|
| **XGBoost** | **0.963** | 0.003 | 0.992 | 0.001 |
| Random Forest | 0.960 | 0.002 | 0.997 | 0.000 |
| LightGBM | 0.943 | 0.003 | 0.984 | 0.001 |
| Logistic Regression | 0.502 | 0.006 | 0.663 | 0.004 |

###  The CV vs. Test Gap — Why It Matters

XGBoost achieves **CV F1 = 0.963** but only **test F1 = 0.030** — a near-total collapse. This is a canonical symptom of **SMOTE leakage into cross-validation folds**: synthetic minority samples generated from training data bleed into validation sets, inflating CV scores. Tree models memorize these patterns; Logistic Regression can't, which is exactly why it ends up being the only model that generalizes.

**Lesson:** Always apply SMOTE *within* each fold using an `imbalanced-learn` pipeline. Test-set F1 and MCC are the metrics to trust.

### Threshold Optimization Results

| Model | Default F1 (0.5) | Optimal Threshold | Best F1 | Δ Improvement |
|---|---|---|---|---|
| Logistic Regression | 0.110 | 0.64 | 0.138 | +0.029 |
| Random Forest | 0.046 | 0.22 | 0.118 | +0.072 |
| XGBoost | 0.030 | 0.15 | 0.075 | +0.044 |
| LightGBM | 0.066 | 0.22 | 0.103 | +0.038 |

---

##  Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│              │     │              │     │              │     │              │
│    Data      │────▶│     ETL      │────▶│    Model     │────▶│  Dashboard   │
│  Generator   │     │   Pipeline   │     │  Training    │     │  (Streamlit) │
│              │     │              │     │              │     │              │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
 50K records          9 steps              4 models              4 tabs
 10 machines          18 features          SHAP analysis         PDF export
 7 sensors            leak-free            threshold opt         radar chart
 3.3% failures        SMOTE last           CV + test eval        SHAP interactive

 data_generator       etl_processor        train_models          app.py
    .ipynb               .ipynb               .ipynb             (script)
```

---

##  Dashboard — 4 Tabs In Detail

The `app.py` dashboard is a **single-page, 4-tab Model Explainability interface** — purpose-built for post-training analysis and reporting. It reads all pre-generated artifacts from the `artifacts/` folder and requires no active model inference at runtime.

### Sidebar (Persistent Across All Tabs)
- **System status indicator** — green/red alert confirming all required artifacts are loaded
- **Best model KPIs** — F1, ROC-AUC, Precision, Recall, Optimal Threshold always visible
- **Pipeline provenance** — timestamp, original data shape, steps completed, missing values filled, outliers treated
- **Version badge** — v2.0

---

###  Tab 1 — Model Performance Overview

The performance hub. Everything needed to assess model quality at a glance:

- **5 KPI metric cards** — Best Model (with "WINNER" badge), F1 Score, ROC-AUC, Precision, Recall; color-coded (blue/purple/green/amber)
- **Full comparison table** — all 4 models: F1, AUC, Precision, Recall, Accuracy, MCC, CV F1 ± std, Training Time; best model marked with `>>`
- **Interactive radar chart** — Plotly spider chart comparing all 4 models across 6 metrics simultaneously with per-model fill colors
- **Confusion matrix** — best model's matrix with 4 breakdown metric cards (TN, TP, FP, FN); falls back to a dynamically-computed Plotly heatmap if the PNG is absent
- **Prediction probability distribution** — failure probability histogram split by true label (Normal vs Failure), optimal threshold plotted as a vertical line, marginal box plot included
- **"How to Interpret" expander** — in-dashboard guide covering all metrics, maintenance-specific guidance, and CV interpretation

---

###  Tab 2 — Feature Importance

Understand which sensor signals and engineered features drive predictions:

- **Per-model importance plot** — model selector dropdown; displays saved top-15 bar chart, or dynamically computes from the `.pkl` file if the plot is missing
- **Multi-model comparison chart** — grouped bar chart of top 10 features by average importance across all tree models, side by side
- **"How to Interpret" expander** — covers mean decrease in impurity (Random Forest) vs. gain (XGBoost/LightGBM)

---

###  Tab 3 — SHAP Explainability

The deepest explainability tab — shows not just *what* the model uses, but *how* and *in which direction*:

- **Side-by-side SHAP plots** — Global Importance bar chart and Beeswarm plot rendered together for any selected model
- **Interactive SHAP distribution** — per-feature dropdown → SHAP value histogram with marginal box plot and a "No Impact" zero-line marker
- **SHAP correlation matrix** — Plotly heatmap of pairwise SHAP value correlations, revealing which features the model treats as interchangeable
- **"How to Read SHAP" expander** — explains beeswarm dot encoding (color = feature value, x = impact), bar plot aggregation, and why SHAP is essential for auditing industrial ML

> **Why SHAP over feature importance?** Feature importance tells you *what* the model uses. SHAP tells you *how* and in *which direction* — critical when a false alarm in manufacturing costs thousands of dollars per hour in unnecessary maintenance dispatch.

---

###  Tab 4 — Curves & Diagnostics

Deep-dive diagnostic tools, model-selectable via dropdown:

- **ROC curves (all models)** — side-by-side layout from saved PNGs, or dynamically approximated from AUC values
- **Precision-Recall curves (all models)** — more informative than ROC for imbalanced datasets
- **Learning curves** — training vs. validation F1 as dataset size grows; diagnose overfitting or underfitting
- **Threshold optimization** — plot + 3-column metric cards (Optimal Threshold, Best F1, Improvement over default 0.5)
- **Overall model comparison bar chart** — all metrics, all models in one view
- **Hyperparameter expanders** — per-model collapsible tables showing best parameters, description, and training time
- **"How to Read Diagnostic Curves" expander** — in-dashboard guide covering ROC, PR, learning, and threshold interpretation

---

##  PDF Report Generation

A standout feature: a **one-click, 12-section PDF report** generated entirely in-browser using `fpdf2`.

**How to use:** Click "Generate PDF Report" → then "Download Report (PDF)"

The report includes:

| Section | Content |
|---|---|
| 1. Title Page | Best model metrics + report date |
| 2. Executive Summary | Model count, best model, F1 score |
| 3. Model Comparison | Test metrics table + CV table + comparison chart |
| 4. ROC & PR Curves | All-model comparison plots |
| 5. Confusion Matrices | All 4 models with TN/FP/FN/TP counts |
| 6. Feature Importance | Top-15 bar charts for all models |
| 7. SHAP Analysis | Bar + beeswarm plots for all models |
| 8. Learning Curves | All models |
| 9. Threshold Optimization | Plots + optimal threshold metrics per model |
| 10. Hyperparameters | Best params table per model |
| 11. Pipeline Metadata | Steps, imputation stats, outlier counts |
| 12. Test Set Analysis | Probability statistics + actual vs. predicted counts |
| + Feature List | All 18 selected features |

The PDF uses a custom `PDFReport` class with Unicode sanitization — special characters (™, —, ", ') never crash the renderer.

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
#    Step 3 → train_models.ipynb       (trains 4 models, ~5-10 min on i5/i7)

# 5. Launch the dashboard
streamlit run app.py
```

> **Tip:** The dashboard checks for required artifacts on startup and shows a clear sidebar error message if any are missing, telling you exactly which files to regenerate. VS Code with the Jupyter extension works as an alternative to `jupyter lab`.

---

##  Project Structure

```
predictive-maintenance-system/
│
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
│
├── data_generator.ipynb             # Notebook: synthetic IoT data generation (67 KB)
├── etl_processor.ipynb              # Notebook: ETL pipeline & feature engineering (81 KB)
├── train_models.ipynb               # Notebook: training, evaluation, SHAP (2,254 KB)
├── app.py                           # Streamlit dashboard — 4-tab explainability UI (52 KB)
│
├── sensor_data.csv                  # [Generated] Raw sensor data — 50K rows (5,759 KB)
│
└── artifacts/                       # [Generated] All training outputs
    ├── best_model.pkl               # Best trained model (Logistic Regression)
    ├── best_model_info.json         # Metrics, threshold, feature names
    ├── training_results.json        # All 4 models: metrics, CV, confusion matrices
    ├── training_config.json         # Hyperparameter search configs
    ├── test_predictions.csv         # Actual, Predicted, Failure_Probability columns
    ├── eda_data.csv                 # EDA data sample
    ├── scaler.pkl                   # Fitted StandardScaler (train-only)
    ├── feature_names.json           # 18 selected feature names
    ├── etl_config.json              # ETL configuration
    ├── imputation_values.json       # Median values (fit on train only)
    ├── outlier_bounds.json          # IQR clipping bounds
    ├── pipeline_metadata.json       # Full pipeline provenance + step log
    ├── model_*.pkl                  # All 4 trained model files
    │
    └── plots/                       # All saved visualizations
        ├── confusion_matrix_*.png
        ├── roc_curves_comparison.png
        ├── precision_recall_curves.png
        ├── model_comparison.png
        ├── feature_importance_*.png
        ├── shap_summary_*.png
        ├── shap_bar_*.png
        ├── shap_values_*.csv        # Raw SHAP values for interactive tab
        ├── learning_curve_*.png
        └── threshold_optimization_*.png
```

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

### ETL Pipeline (9 Steps, Zero Leakage)

```
validation → imputation → outlier_treatment → feature_engineering → build_matrix → split → scaling → feature_selection → imbalance_smote
```

| Step | Detail |
|---|---|
| Imputation | Median strategy, fit on train only — 6,950 values filled |
| Outlier Treatment | IQR-based clipping — 2,112 outliers treated |
| Feature Engineering | 25+ features engineered from 7 raw sensors |
| Scaling | StandardScaler, fit on train only |
| Feature Selection | Variance filter → correlation filter → mutual information |
| Imbalance Handling | SMOTE applied **after** split — no leakage |

### Best Hyperparameters

**Logistic Regression** *(Best model — 19.7s)*
```
solver: liblinear | penalty: l2 | C: 100
```
**Random Forest** *(1881.4s — slowest)*
```
n_estimators: 200 | max_depth: None | max_features: log2
min_samples_split: 5 | min_samples_leaf: 1
```
**XGBoost** *(158.7s)*
```
n_estimators: 300 | max_depth: 10 | learning_rate: 0.2
subsample: 0.9 | colsample_bytree: 1.0 | gamma: 0
```
**LightGBM** *(149.9s)*
```
n_estimators: 200 | num_leaves: 63 | learning_rate: 0.1
subsample: 0.8 | colsample_bytree: 0.9 | min_child_samples: 10
```

---

##  Key Findings & Lessons Learned

**1. Logistic Regression wins on real test data** — The simplest model generalizes best. XGBoost's 0.963 CV F1 collapsed to 0.030 on the test set. Complexity ≠ generalization.

**2. SMOTE inflates CV scores for tree models** — The fix: apply SMOTE *inside* each fold using `imbalanced-learn`'s `Pipeline`. Never fit SMOTE before the train/test split.

**3. Threshold tuning delivers outsized gains for free** — Shifting Random Forest's decision boundary from 0.5 → 0.22 improved F1 by +158% with zero retraining. The dashboard makes this visual and interactive.

**4. SHAP reveals what importance scores hide** — Feature importance flagged `Shift_Encoded` and `DayOfWeek` as top features in tree models. SHAP confirmed and clarified this — a signal that synthetic data has temporal artifacts that may not generalize to real deployments.

**5. Recall is the right objective for maintenance** — Logistic Regression catches 65% of real failures. The 6% precision (34× false alarms) is the cost. In production, the tradeoff should be driven by the ratio of dispatch cost to breakdown cost.

---

##  Future Improvements

- [ ] Fix SMOTE leakage: apply within each CV fold via `imbalanced-learn` Pipeline
- [ ] Cost-sensitive learning with explicit FP/FN cost matrix
- [ ] Calibrated probability outputs (Platt scaling / isotonic regression)
- [ ] Remaining Useful Life (RUL) regression model
- [ ] LSTM / Transformer for temporal sequence modeling
- [ ] Real-time Kafka/MQTT streaming integration
- [ ] Automated retraining pipeline with MLflow experiment tracking
- [ ] Docker containerization + FastAPI prediction endpoint
- [ ] Alerting system (email/Slack) triggered by high-risk predictions

---

##  Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.9+ |
| ML Models | Scikit-Learn, XGBoost, LightGBM |
| Explainability | SHAP |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Dashboard | Streamlit |
| Visualization | Plotly, Matplotlib, Seaborn |
| PDF Generation | fpdf2 |
| Data Processing | Pandas, NumPy, SciPy |

---

##  License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

##  Author

Built as a demonstration of production-aware ML engineering for industrial predictive maintenance — including honest reporting of model limitations, SHAP-based explainability, and automated PDF reporting baked directly into the dashboard.

---

*PredictiveGuard™ v2.0 | Built with Streamlit · Scikit-Learn · XGBoost · LightGBM · SHAP · Plotly · fpdf2*
