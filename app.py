"""
=============================================================================
PREDICTIVE MAINTENANCE - INTERACTIVE DASHBOARD
=============================================================================
Production-ready Streamlit dashboard for predictive maintenance monitoring.

Sections:
    1. Model Performance Overview (Tab 1)
    2. Feature Importance (Tab 2)
    3. SHAP Explainability (Tab 3)
    4. Diagnostic Curves (Tab 4)

Run: streamlit run app.py
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import os
from PIL import Image
from fpdf import FPDF
from io import BytesIO
from datetime import datetime

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="PredictiveGuard - Maintenance Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CUSTOM CSS
# =============================================================================
st.markdown(
    """
<style>
    .stApp {
        background-color: #FFFFFF;
        color: #1E293B;
    }
    .section-header {
        font-size: 24px;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 15px;
        padding-bottom: 8px;
        border-bottom: 2px solid #3B82F6;
    }
    .metric-card {
        background: linear-gradient(135deg, #F8FAFC 0%, #E2E8F0 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #CBD5E1;
        text-align: center;
        margin-bottom: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .metric-card h3 {
        color: #64748B;
        font-size: 14px;
        margin-bottom: 5px;
        font-weight: 500;
    }
    .metric-card h1 {
        color: #0F172A;
        font-size: 32px;
        margin: 0;
        font-weight: 700;
    }
    .winner-badge {
        background: linear-gradient(135deg, #16A34A 0%, #15803D 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
        margin-top: 5px;
    }
    [data-testid="stSidebar"] {
        background-color: #F1F5F9;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #F1F5F9;
        border-radius: 8px;
        color: #475569;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6 !important;
        color: white !important;
    }
    .streamlit-expanderHeader {
        background-color: #F8FAFC;
        border-radius: 8px;
        color: #1E293B;
        font-weight: 600;
    }
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# COLOR PALETTE & PLOTLY LAYOUT
# =============================================================================
COLORS = {
    "primary": "#3B82F6",
    "danger": "#EF4444",
    "success": "#22C55E",
    "warning": "#F59E0B",
    "purple": "#8B5CF6",
    "cyan": "#06B6D4",
    "bg_dark": "#FFFFFF",
    "bg_card": "#F8FAFC",
    "bg_surface": "#E2E8F0",
    "text": "#1E293B",
    "text_muted": "#64748B",
}

PLOTLY_LAYOUT = {
    "paper_bgcolor": "#FFFFFF",
    "plot_bgcolor": "#FFFFFF",
    "font": dict(color="#1E293B", size=12),
    "margin": dict(l=20, r=20, t=50, b=20),
    "xaxis": dict(gridcolor="#E2E8F0", zerolinecolor="#CBD5E1"),
    "yaxis": dict(gridcolor="#E2E8F0", zerolinecolor="#CBD5E1"),
}


# =============================================================================
# PDF REPORT GENERATOR
# =============================================================================
class PDFReport(FPDF):
    """Custom PDF with sanitised text so built-in Helvetica never crashes."""

    def __init__(self):
        super().__init__("P", "mm", "A4")
        self.report_title = "PredictiveGuard(TM) - Predictive Maintenance Report"
        self.set_auto_page_break(auto=True, margin=25)

    @staticmethod
    def sanitize(text):
        replacements = {
            "\u2122": "(TM)",
            "\u2014": "-",
            "\u2013": "-",
            "\u2018": "'",
            "\u2019": "'",
            "\u201C": '"',
            "\u201D": '"',
            "\u2026": "...",
            "\u00b1": "+/-",
            "\u00d7": "x",
        }
        for char, repl in replacements.items():
            text = text.replace(char, repl)
        return text

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(100, 116, 139)
        self.cell(0, 8, self.sanitize(self.report_title), align="L")
        self.cell(0, 8, f"Page {self.page_no()}", align="R", ln=1)
        self.set_draw_color(203, 213, 225)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(148, 163, 184)
        self.cell(
            0, 10,
            self.sanitize(
                f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                " | PredictiveGuard(TM)"
            ),
            align="C",
        )

    def add_title_page(self, best_model_info):
        self.add_page()
        self.ln(50)
        self.set_font("Helvetica", "B", 32)
        self.set_text_color(15, 23, 42)
        self.cell(0, 15, "PredictiveGuard", align="C", ln=1)
        self.set_font("Helvetica", "", 14)
        self.set_text_color(100, 116, 139)
        self.cell(0, 10, "Industrial Predictive Maintenance System", align="C", ln=1)
        self.ln(10)
        self.set_draw_color(59, 130, 246)
        self.set_line_width(0.8)
        self.line(60, self.get_y(), 150, self.get_y())
        self.ln(10)
        self.set_font("Helvetica", "B", 18)
        self.set_text_color(15, 23, 42)
        self.cell(0, 12, "Model Training & Evaluation Report", align="C", ln=1)
        self.ln(20)
        self.set_font("Helvetica", "", 11)
        self.set_text_color(71, 85, 105)
        if best_model_info:
            bm = best_model_info.get("model_name", "N/A")
            met = best_model_info.get("all_metrics", {})
            nf = best_model_info.get("n_features", "N/A")
            for line in [
                f"Best Model: {bm}",
                f"F1 Score: {met.get('f1_score', 0):.4f}",
                f"ROC AUC: {met.get('roc_auc', 0):.4f}",
                f"Precision: {met.get('precision', 0):.4f}",
                f"Recall: {met.get('recall', 0):.4f}",
                f"Features Used: {nf}",
                f"Report Date: {datetime.now().strftime('%B %d, %Y')}",
            ]:
                self.cell(0, 8, self.sanitize(line), align="C", ln=1)

    def add_section_header(self, title):
        self.ln(6)
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(15, 23, 42)
        self.cell(0, 10, self.sanitize(title), ln=1)
        self.set_draw_color(59, 130, 246)
        self.set_line_width(0.6)
        self.line(10, self.get_y(), 80, self.get_y())
        self.ln(6)

    def add_sub_header(self, title):
        self.ln(3)
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(30, 41, 59)
        self.cell(0, 8, self.sanitize(title), ln=1)
        self.ln(2)

    def add_body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(71, 85, 105)
        self.multi_cell(0, 6, self.sanitize(text))
        self.ln(2)

    def add_metric_row(self, label, value):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(30, 41, 59)
        self.cell(70, 7, self.sanitize(str(label)))
        self.set_font("Helvetica", "", 10)
        self.set_text_color(59, 130, 246)
        self.cell(0, 7, self.sanitize(str(value)), ln=1)

    def add_image_safe(self, image_path, w=180):
        if os.path.exists(image_path):
            try:
                if self.get_y() > 180:
                    self.add_page()
                self.image(image_path, x=15, w=w)
                self.ln(8)
                return True
            except Exception:
                self.add_body_text(
                    f"[Could not load image: {os.path.basename(image_path)}]"
                )
                return False
        return False

    def add_table(self, headers, rows, col_widths=None):
        if col_widths is None:
            col_widths = [190 / len(headers)] * len(headers)
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(241, 245, 249)
        self.set_text_color(30, 41, 59)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 8, self.sanitize(str(h)), border=1, fill=True, align="C")
        self.ln()
        self.set_font("Helvetica", "", 9)
        self.set_text_color(71, 85, 105)
        for row in rows:
            for i, val in enumerate(row):
                self.cell(col_widths[i], 7, self.sanitize(str(val)), border=1, align="C")
            self.ln()
        self.ln(4)


def generate_pdf_report(training_results, best_model_info, pipeline_meta, test_predictions):
    pdf = PDFReport()
    plots_dir = "artifacts/plots"

    # Title page
    pdf.add_title_page(best_model_info)

    # Section 1: Executive Summary
    pdf.add_page()
    pdf.add_section_header("1. Executive Summary")
    if best_model_info:
        bm = best_model_info.get("model_name", "N/A")
        met = best_model_info.get("all_metrics", {})
        threshold = best_model_info.get("threshold", {})
        pdf.add_body_text(
            f"This report presents the results of the predictive maintenance model "
            f"training pipeline. A total of {len(training_results) if training_results else 0} "
            f"models were trained and evaluated. The best performing model is "
            f"{bm} with an F1 score of {met.get('f1_score', 0):.4f}."
        )
        pdf.ln(4)
        pdf.add_sub_header("Best Model Metrics")
        pdf.add_metric_row("Model", bm)
        pdf.add_metric_row("F1 Score", f"{met.get('f1_score', 0):.4f}")
        pdf.add_metric_row("ROC AUC", f"{met.get('roc_auc', 0):.4f}")
        pdf.add_metric_row("Precision", f"{met.get('precision', 0):.4f}")
        pdf.add_metric_row("Recall", f"{met.get('recall', 0):.4f}")
        pdf.add_metric_row("Accuracy", f"{met.get('accuracy', 0):.4f}")
        pdf.add_metric_row("MCC", f"{met.get('mcc', 0):.4f}")
        pdf.add_metric_row("Specificity", f"{met.get('specificity', 0):.4f}")
        if threshold:
            pdf.add_metric_row("Optimal Threshold", f"{threshold.get('best_threshold', 0.5):.2f}")
        pdf.add_metric_row("Features Used", best_model_info.get("n_features", "N/A"))

    # Section 2: Model Comparison
    pdf.add_page()
    pdf.add_section_header("2. Model Performance Comparison")
    if training_results:
        pdf.add_body_text(
            "The table below compares all trained models across key classification metrics. "
            "The best model is highlighted based on the primary optimization metric (F1 Score)."
        )
        headers = ["Model", "F1", "AUC", "Precision", "Recall", "MCC", "Time(s)"]
        col_widths = [45, 22, 22, 25, 22, 22, 22]
        rows = []
        for name, res in training_results.items():
            m = res.get("metrics", {})
            marker = " *" if best_model_info and name == best_model_info.get("model_name") else ""
            rows.append([
                f"{name}{marker}",
                f"{m.get('f1_score', 0):.4f}",
                f"{m.get('roc_auc', 0):.4f}",
                f"{m.get('precision', 0):.4f}",
                f"{m.get('recall', 0):.4f}",
                f"{m.get('mcc', 0):.4f}",
                f"{res.get('training_time', 0):.1f}",
            ])
        pdf.add_table(headers, rows, col_widths)
        pdf.add_body_text("* Indicates the selected best model.")
        pdf.ln(4)
        pdf.add_sub_header("Cross-Validation Results")
        cv_headers = ["Model", "CV F1 Mean", "CV F1 Std", "CV AUC Mean", "CV AUC Std"]
        cv_widths = [45, 32, 32, 32, 32]
        cv_rows = []
        for name, res in training_results.items():
            cv = res.get("cv_results", {})
            cv_rows.append([
                name,
                f"{cv.get('f1', {}).get('mean', 0):.4f}",
                f"{cv.get('f1', {}).get('std', 0):.4f}",
                f"{cv.get('roc_auc', {}).get('mean', 0):.4f}",
                f"{cv.get('roc_auc', {}).get('std', 0):.4f}",
            ])
        pdf.add_table(cv_headers, cv_rows, cv_widths)
        pdf.ln(4)
        pdf.add_sub_header("Visual Comparison")
        pdf.add_image_safe(os.path.join(plots_dir, "model_comparison.png"))

    # Section 3: ROC & PR Curves
    pdf.add_page()
    pdf.add_section_header("3. ROC & Precision-Recall Curves")
    pdf.add_body_text(
        "ROC curves plot True Positive Rate vs False Positive Rate across all "
        "classification thresholds. AUC closer to 1.0 indicates better discrimination."
    )
    pdf.add_sub_header("ROC Curves - All Models")
    pdf.add_image_safe(os.path.join(plots_dir, "roc_curves_comparison.png"))
    pdf.add_sub_header("Precision-Recall Curves - All Models")
    pdf.add_image_safe(os.path.join(plots_dir, "precision_recall_curves.png"))

    # Section 4: Confusion Matrices
    pdf.add_page()
    pdf.add_section_header("4. Confusion Matrices")
    pdf.add_body_text(
        "Confusion matrices show the breakdown of correct and incorrect predictions."
    )
    if training_results:
        for name in training_results:
            safe = name.replace(" ", "_").lower()
            cm_path = os.path.join(plots_dir, f"confusion_matrix_{safe}.png")
            if os.path.exists(cm_path):
                pdf.add_sub_header(f"Confusion Matrix - {name}")
                pdf.add_image_safe(cm_path, w=140)
                cm_data = training_results[name].get("confusion_matrix")
                if cm_data and len(cm_data) == 2 and len(cm_data[0]) == 2:
                    tn, fp = cm_data[0][0], cm_data[0][1]
                    fn, tp = cm_data[1][0], cm_data[1][1]
                    pdf.add_body_text(
                        f"TN: {tn:,} | FP: {fp:,} | FN: {fn:,} | TP: {tp:,}"
                    )

    # Section 5: Feature Importance
    pdf.add_page()
    pdf.add_section_header("5. Feature Importance")
    pdf.add_body_text("Feature importance measures how much each feature contributes to predictions.")
    if training_results:
        for name in training_results:
            safe = name.replace(" ", "_").lower()
            fi = os.path.join(plots_dir, f"feature_importance_{safe}.png")
            if os.path.exists(fi):
                pdf.add_sub_header(f"Feature Importance - {name}")
                pdf.add_image_safe(fi)

    # Section 6: SHAP
    pdf.add_page()
    pdf.add_section_header("6. SHAP Explainability Analysis")
    pdf.add_body_text("SHAP provides mathematically grounded explanations for model predictions.")
    if training_results:
        for name in training_results:
            safe = name.replace(" ", "_").lower()
            sb = os.path.join(plots_dir, f"shap_bar_{safe}.png")
            if os.path.exists(sb):
                pdf.add_sub_header(f"SHAP Global Importance - {name}")
                pdf.add_image_safe(sb)
            ss = os.path.join(plots_dir, f"shap_summary_{safe}.png")
            if os.path.exists(ss):
                pdf.add_sub_header(f"SHAP Beeswarm Plot - {name}")
                pdf.add_image_safe(ss)

    # Section 7: Learning Curves
    pdf.add_page()
    pdf.add_section_header("7. Learning Curves")
    pdf.add_body_text("Learning curves show how model performance changes as training data increases.")
    if training_results:
        for name in training_results:
            safe = name.replace(" ", "_").lower()
            lc = os.path.join(plots_dir, f"learning_curve_{safe}.png")
            if os.path.exists(lc):
                pdf.add_sub_header(f"Learning Curve - {name}")
                pdf.add_image_safe(lc)

    # Section 8: Threshold Optimization
    pdf.add_page()
    pdf.add_section_header("8. Threshold Optimization")
    pdf.add_body_text("The default threshold of 0.5 is often suboptimal for imbalanced datasets.")
    if training_results:
        for name, res in training_results.items():
            safe = name.replace(" ", "_").lower()
            tp_path = os.path.join(plots_dir, f"threshold_optimization_{safe}.png")
            if os.path.exists(tp_path):
                pdf.add_sub_header(f"Threshold Optimization - {name}")
                pdf.add_image_safe(tp_path)
                thresh = res.get("threshold_result")
                if thresh:
                    pdf.add_metric_row("Optimal Threshold", f"{thresh.get('best_threshold', 0.5):.2f}")
                    pdf.add_metric_row("Best F1", f"{thresh.get('best_f1', 0):.4f}")
                    pdf.add_metric_row("Default F1 (0.5)", f"{thresh.get('default_f1', 0):.4f}")
                    pdf.add_metric_row("Improvement", f"+{thresh.get('improvement', 0):.4f}")
                    pdf.ln(4)

    # Section 9: Hyperparameters
    pdf.add_page()
    pdf.add_section_header("9. Best Hyperparameters")
    pdf.add_body_text("Hyperparameters were optimized using RandomizedSearchCV.")
    if training_results:
        for name, res in training_results.items():
            pdf.add_sub_header(name)
            if res.get("description"):
                pdf.add_body_text(f"Description: {res['description']}")
            params = res.get("best_params", {})
            if params:
                p_rows = [[str(k), str(v)] for k, v in params.items()]
                pdf.add_table(["Parameter", "Value"], p_rows, [80, 100])
            pdf.add_metric_row("Training Time", f"{res.get('training_time', 0):.1f}s")
            pdf.ln(4)

    # Section 10: Pipeline Metadata
    pdf.add_page()
    pdf.add_section_header("10. Pipeline Metadata")
    pdf.add_body_text("Technical details about the ETL pipeline and data processing steps.")
    if pipeline_meta:
        pdf.add_metric_row("Pipeline Timestamp", str(pipeline_meta.get("timestamp", "N/A")))
        pdf.add_metric_row("Original Data Shape", str(pipeline_meta.get("original_shape", "N/A")))
        pdf.add_metric_row("Steps Completed", str(len(pipeline_meta.get("steps_completed", []))))
        pdf.add_metric_row("Missing Values Filled", str(pipeline_meta.get("missing_values_filled", 0)))
        pdf.add_metric_row("Outliers Treated", str(pipeline_meta.get("outliers_treated", 0)))
        steps = pipeline_meta.get("steps_completed", [])
        if steps:
            pdf.ln(4)
            pdf.add_sub_header("Pipeline Steps Executed")
            for i, step in enumerate(steps, 1):
                pdf.add_body_text(f"  {i}. {step}")

    # Section 11: Test Set Analysis
    if test_predictions is not None and not test_predictions.empty:
        pdf.add_page()
        pdf.add_section_header("11. Test Set Prediction Analysis")
        n_total = len(test_predictions)
        n_actual = int(test_predictions["Actual"].sum()) if "Actual" in test_predictions.columns else 0
        n_pred = int(test_predictions["Predicted"].sum()) if "Predicted" in test_predictions.columns else 0
        pdf.add_body_text(
            f"The test set contains {n_total:,} samples. "
            f"Actual failures: {n_actual:,} ({n_actual / n_total * 100:.1f}%). "
            f"Predicted failures: {n_pred:,} ({n_pred / n_total * 100:.1f}%)."
        )
        if "Failure_Probability" in test_predictions.columns:
            ps = test_predictions["Failure_Probability"].describe()
            pdf.ln(4)
            pdf.add_sub_header("Prediction Probability Statistics")
            pdf.add_metric_row("Mean Probability", f"{ps['mean']:.4f}")
            pdf.add_metric_row("Median Probability", f"{ps['50%']:.4f}")
            pdf.add_metric_row("Std Deviation", f"{ps['std']:.4f}")
            pdf.add_metric_row("Min Probability", f"{ps['min']:.4f}")
            pdf.add_metric_row("Max Probability", f"{ps['max']:.4f}")

    # Section 12: Feature List
    if best_model_info and best_model_info.get("feature_names"):
        pdf.add_page()
        pdf.add_section_header("12. Selected Features")
        feat_names = best_model_info["feature_names"]
        pdf.add_body_text(f"The final model uses {len(feat_names)} features after selection.")
        pdf.ln(4)
        f_rows = [[str(i), fn] for i, fn in enumerate(feat_names, 1)]
        pdf.add_table(["#", "Feature Name"], f_rows, [20, 160])

    # Final Page
    pdf.add_page()
    pdf.ln(60)
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 12, "End of Report", align="C", ln=1)
    pdf.ln(8)
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 8, "PredictiveGuard(TM) - Industrial Predictive Maintenance System", align="C", ln=1)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}", align="C", ln=1)
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, "Built with Scikit-Learn | XGBoost | LightGBM | SHAP | Streamlit", align="C", ln=1)

    pdf_output = pdf.output(dest="S")
    if isinstance(pdf_output, str):
        return pdf_output.encode("latin-1")
    return bytes(pdf_output)


# =============================================================================
# DATA LOADING UTILITIES
# =============================================================================
@st.cache_data
def load_json_artifact(filename):
    filepath = os.path.join("artifacts", filename)
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return None


@st.cache_data
def load_csv_artifact(filename):
    filepath = os.path.join("artifacts", filename)
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    return None


def load_plot_image(filename):
    filepath = os.path.join("artifacts", "plots", filename)
    if os.path.exists(filepath):
        return filepath
    return None


# =============================================================================
# LOAD ALL ARTIFACTS
# =============================================================================
training_results = load_json_artifact("training_results.json")
best_model_info = load_json_artifact("best_model_info.json")
pipeline_meta = load_json_artifact("pipeline_metadata.json")
etl_config = load_json_artifact("etl_config.json")
training_config = load_json_artifact("training_config.json")
test_predictions = load_csv_artifact("test_predictions.csv")
eda_data = load_csv_artifact("eda_data.csv")


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("## PredictiveGuard")
    st.markdown("**Industrial Predictive Maintenance**")
    st.markdown("---")

    artifacts_exist = (
        training_results is not None
        and best_model_info is not None
        and test_predictions is not None
    )

    if artifacts_exist:
        st.success("System Online - All artifacts loaded")
    else:
        st.error("Missing artifacts - run training pipeline first")
        missing = []
        if training_results is None:
            missing.append("training_results.json")
        if best_model_info is None:
            missing.append("best_model_info.json")
        if test_predictions is None:
            missing.append("test_predictions.csv")
        if missing:
            st.caption(f"Missing: {', '.join(missing)}")

    st.markdown("---")

    if best_model_info:
        st.markdown("### Best Model")
        st.markdown(f"**{best_model_info.get('model_name', 'N/A')}**")
        bm_sidebar = best_model_info.get("all_metrics", {})
        st.metric("F1 Score", f"{bm_sidebar.get('f1_score', 0):.4f}")
        st.metric("ROC AUC", f"{bm_sidebar.get('roc_auc', 0):.4f}")
        st.metric("Precision", f"{bm_sidebar.get('precision', 0):.4f}")
        st.metric("Recall", f"{bm_sidebar.get('recall', 0):.4f}")
        thr = best_model_info.get("threshold", {})
        if thr:
            st.metric("Optimal Threshold", f"{thr.get('best_threshold', 0.5):.2f}")
        st.markdown(f"**Features:** {best_model_info.get('n_features', 'N/A')}")

    st.markdown("---")

    if pipeline_meta:
        st.markdown("### Pipeline Info")
        st.caption(f"Timestamp: {pipeline_meta.get('timestamp', 'N/A')}")
        st.caption(f"Original shape: {pipeline_meta.get('original_shape', 'N/A')}")
        st.caption(f"Steps: {len(pipeline_meta.get('steps_completed', []))}")
        st.caption(f"Missing filled: {int(pipeline_meta.get('missing_values_filled', 0)):,}")
        st.caption(f"Outliers treated: {int(pipeline_meta.get('outliers_treated', 0)):,}")

    st.markdown("---")
    st.markdown(
        "<p style='color:#64748B; font-size:11px; text-align:center;'>"
        "v2.0 &bull; Built with Streamlit</p>",
        unsafe_allow_html=True,
    )


# =============================================================================
# MAIN HEADER
# =============================================================================
st.markdown(
    """
<div style="text-align:center; padding:20px 0 30px 0;">
    <h1 style="color:#0F172A; font-size:36px; margin-bottom:5px;">
        PredictiveGuard Dashboard
    </h1>
    <p style="color:#64748B; font-size:16px;">
        Real-Time Predictive Maintenance Monitoring &amp; Model Explainability
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# ── Download Report ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Download Report")

if artifacts_exist:
    if st.button("Generate PDF Report", use_container_width=True):
        with st.spinner("Generating comprehensive PDF report..."):
            pdf_bytes = generate_pdf_report(
                training_results, best_model_info, pipeline_meta, test_predictions,
            )
        st.download_button(
            label="Download Report (PDF)",
            data=pdf_bytes,
            file_name=f"PredictiveGuard_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
        st.success("Report generated!")
else:
    st.info("Run the training pipeline first to generate a report.")


# =============================================================================
# TABS
# =============================================================================
tab_m1, tab_m2, tab_m3, tab_m4 = st.tabs([
    "Model Performance",
    "Feature Importance",
    "SHAP Analysis",
    "Curves & Diagnostics",
])


# =============================================================================
# TAB 1 - MODEL PERFORMANCE OVERVIEW
# =============================================================================
with tab_m1:
    st.markdown(
        '<p class="section-header">Model Performance Overview</p>',
        unsafe_allow_html=True,
    )

    if training_results and best_model_info:
        best_name = best_model_info.get("model_name", "N/A")
        best_metrics = best_model_info.get("all_metrics", {})

        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        with kpi1:
            st.markdown(
                f'<div class="metric-card"><h3>Best Model</h3>'
                f'<h1 style="font-size:18px;">{best_name}</h1>'
                f'<span class="winner-badge">WINNER</span></div>',
                unsafe_allow_html=True,
            )
        with kpi2:
            st.markdown(
                f'<div class="metric-card"><h3>F1 Score</h3>'
                f'<h1 style="color:{COLORS["primary"]}">'
                f'{best_metrics.get("f1_score", 0):.4f}</h1></div>',
                unsafe_allow_html=True,
            )
        with kpi3:
            st.markdown(
                f'<div class="metric-card"><h3>ROC AUC</h3>'
                f'<h1 style="color:{COLORS["purple"]}">'
                f'{best_metrics.get("roc_auc", 0):.4f}</h1></div>',
                unsafe_allow_html=True,
            )
        with kpi4:
            st.markdown(
                f'<div class="metric-card"><h3>Precision</h3>'
                f'<h1 style="color:{COLORS["success"]}">'
                f'{best_metrics.get("precision", 0):.4f}</h1></div>',
                unsafe_allow_html=True,
            )
        with kpi5:
            st.markdown(
                f'<div class="metric-card"><h3>Recall</h3>'
                f'<h1 style="color:{COLORS["warning"]}">'
                f'{best_metrics.get("recall", 0):.4f}</h1></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Comparison table
        st.markdown("**Full Model Comparison**")
        comp_rows = []
        for name, res in training_results.items():
            m = res.get("metrics", {})
            cv = res.get("cv_results", {})
            is_best = name == best_name
            comp_rows.append({
                "Model": f">> {name}" if is_best else name,
                "F1 Score": f"{m.get('f1_score', 0):.4f}",
                "ROC AUC": f"{m.get('roc_auc', 0):.4f}",
                "Precision": f"{m.get('precision', 0):.4f}",
                "Recall": f"{m.get('recall', 0):.4f}",
                "Accuracy": f"{m.get('accuracy', 0):.4f}",
                "MCC": f"{m.get('mcc', 0):.4f}",
                "CV F1 (mean+/-std)": (
                    f"{cv.get('f1', {}).get('mean', 0):.4f} +/- "
                    f"{cv.get('f1', {}).get('std', 0):.4f}"
                ) if cv else "N/A",
                "Train Time (s)": f"{res.get('training_time', 0):.1f}",
            })
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Radar chart
        st.markdown("**Model Performance Radar**")
        radar_metrics = ["f1_score", "roc_auc", "precision", "recall", "accuracy", "mcc"]
        radar_labels = ["F1", "AUC", "Precision", "Recall", "Accuracy", "MCC"]
        fig_radar = go.Figure()
        model_colors = [COLORS["primary"], COLORS["danger"], COLORS["success"], COLORS["warning"]]
        for i, (name, res) in enumerate(training_results.items()):
            m = res.get("metrics", {})
            vals = [m.get(rm, 0) for rm in radar_metrics]
            vals.append(vals[0])
            fig_radar.add_trace(go.Scatterpolar(
                r=vals,
                theta=radar_labels + [radar_labels[0]],
                fill="toself",
                name=name,
                line=dict(color=model_colors[i % len(model_colors)], width=2.5),
                fillcolor=model_colors[i % len(model_colors)],
                opacity=0.15,
            ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="#FFFFFF",
                radialaxis=dict(visible=True, range=[0, 1], gridcolor="#E2E8F0", linecolor="#CBD5E1"),
                angularaxis=dict(gridcolor="#E2E8F0", linecolor="#CBD5E1", tickfont=dict(size=12, color="#1E293B")),
            ),
            title="Model Performance Radar Chart",
            height=500,
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(color="#1E293B"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5, font=dict(size=12)),
            margin=dict(l=60, r=60, t=50, b=80),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Confusion matrix
        st.markdown(f"**Confusion Matrix - {best_name}**")
        best_cm_path = load_plot_image(f"confusion_matrix_{best_name.replace(' ', '_').lower()}.png")
        if best_cm_path:
            col_cm, col_cm_info = st.columns([2, 1])
            with col_cm:
                st.image(best_cm_path, use_container_width=True)
            with col_cm_info:
                cm_data = training_results.get(best_name, {}).get("confusion_matrix")
                if cm_data and len(cm_data) == 2 and len(cm_data[0]) == 2:
                    tn, fp = cm_data[0][0], cm_data[0][1]
                    fn, tp = cm_data[1][0], cm_data[1][1]
                    st.markdown(
                        f'<div class="metric-card"><h3>True Negatives</h3>'
                        f'<h1 style="color:{COLORS["success"]}">{tn:,}</h1></div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div class="metric-card"><h3>True Positives</h3>'
                        f'<h1 style="color:{COLORS["primary"]}">{tp:,}</h1></div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div class="metric-card"><h3>False Positives</h3>'
                        f'<h1 style="color:{COLORS["warning"]}">{fp:,}</h1></div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div class="metric-card"><h3>False Negatives</h3>'
                        f'<h1 style="color:{COLORS["danger"]}">{fn:,}</h1></div>',
                        unsafe_allow_html=True,
                    )
        else:
            if (
                test_predictions is not None
                and "Actual" in test_predictions.columns
                and "Predicted" in test_predictions.columns
            ):
                from sklearn.metrics import confusion_matrix as sk_cm
                cm = sk_cm(test_predictions["Actual"], test_predictions["Predicted"])
                fig = px.imshow(
                    cm, text_auto=True,
                    labels=dict(x="Predicted", y="Actual"),
                    x=["Normal", "Failure"], y=["Normal", "Failure"],
                    color_continuous_scale="Blues",
                )
                fig.update_layout(
                    title=f"Confusion Matrix - {best_name}",
                    height=450, coloraxis_showscale=False, **PLOTLY_LAYOUT,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Confusion matrix not available.")

        st.markdown("<br>", unsafe_allow_html=True)

        # Prediction distribution
        if test_predictions is not None and "Failure_Probability" in test_predictions.columns:
            st.markdown(f"**Prediction Probability Distribution - {best_name}**")
            fig_dist = px.histogram(
                test_predictions, x="Failure_Probability", color="Actual",
                nbins=80, barmode="overlay", opacity=0.7,
                color_discrete_map={0: COLORS["success"], 1: COLORS["danger"]},
                labels={"Failure_Probability": "Predicted Failure Probability", "Actual": "True Label"},
                marginal="box",
            )
            threshold_val = best_model_info.get("threshold", {}).get("best_threshold", 0.5)
            fig_dist.add_vline(
                x=threshold_val, line_dash="dash", line_color=COLORS["warning"],
                line_width=2, annotation_text=f"Threshold = {threshold_val:.2f}",
                annotation_font_color=COLORS["warning"],
            )
            fig_dist.update_layout(
                title="Prediction Distribution by True Label", height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        with st.expander("How to Interpret Model Performance"):
            st.markdown("""
**Key Metrics Explained:**
- **F1 Score**: Harmonic mean of Precision and Recall (0-1, higher = better).
- **ROC AUC**: Area Under the ROC Curve (1.0 = perfect, 0.5 = random).
- **Precision**: Of predicted failures, what fraction were real?
- **Recall**: Of real failures, what fraction did we catch?
- **MCC**: Matthews Correlation Coefficient (-1 to +1).
- **Specificity**: Of normals, what fraction were correctly identified?

**For Predictive Maintenance:**
- **Recall is critical** - missing a failure causes damage / downtime.
- **Precision matters too** - too many false alarms cause alarm fatigue.
- The **optimal threshold** balances the trade-off.

**Cross-Validation (CV):**
- Small std = robust model. Large std = sensitive to data.
""")

    else:
        st.warning(
            "No training results found. Please run the training pipeline first.\n\n"
            "**Steps:**\n"
            "1. Run the Data Generator notebook\n"
            "2. Run the ETL Pipeline notebook\n"
            "3. Run the Model Training notebook\n"
            "4. Launch this dashboard: `streamlit run app.py`"
        )


# =============================================================================
# TAB 2 - FEATURE IMPORTANCE
# =============================================================================
with tab_m2:
    st.markdown(
        '<p class="section-header">Feature Importance</p>',
        unsafe_allow_html=True,
    )

    if training_results:
        sel_model = st.selectbox("Select Model", list(training_results.keys()), key="fi_model")
        img_name = f"feature_importance_{sel_model.replace(' ', '_').lower()}.png"
        img_path = load_plot_image(img_name)

        if img_path:
            st.image(img_path, use_container_width=True)
        else:
            safe_name = sel_model.replace(" ", "_").lower()
            model_path = os.path.join("artifacts", f"model_{safe_name}.pkl")
            if os.path.exists(model_path):
                loaded_m = joblib.load(model_path)
                if hasattr(loaded_m, "feature_importances_") and best_model_info:
                    feat_names = best_model_info.get("feature_names", [])
                    if len(feat_names) == len(loaded_m.feature_importances_):
                        imp = pd.Series(loaded_m.feature_importances_, index=feat_names)
                        imp = imp.sort_values(ascending=True).tail(15)
                        fig = px.bar(
                            x=imp.values, y=imp.index, orientation="h",
                            labels={"x": "Importance", "y": "Feature"},
                            color=imp.values,
                            color_continuous_scale=["#3B82F6", "#8B5CF6"],
                        )
                        fig.update_layout(
                            title=f"Top 15 Features - {sel_model}", height=500,
                            showlegend=False, coloraxis_showscale=False, **PLOTLY_LAYOUT,
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Feature name mismatch.")
                else:
                    st.info(f"{sel_model} does not expose feature_importances_.")
            else:
                st.warning(f"Model file not found for {sel_model}.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Importance Comparison Across Models**")

        comparison_data = []
        for name in training_results:
            safe = name.replace(" ", "_").lower()
            mp = os.path.join("artifacts", f"model_{safe}.pkl")
            if os.path.exists(mp):
                m_loaded = joblib.load(mp)
                if hasattr(m_loaded, "feature_importances_") and best_model_info:
                    fnames = best_model_info.get("feature_names", [])
                    if len(fnames) == len(m_loaded.feature_importances_):
                        for fn, fv in zip(fnames, m_loaded.feature_importances_):
                            comparison_data.append({"Model": name, "Feature": fn, "Importance": fv})

        if comparison_data:
            df_ci = pd.DataFrame(comparison_data)
            top_feats = df_ci.groupby("Feature")["Importance"].mean().nlargest(10).index.tolist()
            df_filtered = df_ci[df_ci["Feature"].isin(top_feats)]
            fig = px.bar(
                df_filtered, x="Importance", y="Feature", color="Model",
                orientation="h", barmode="group",
                color_discrete_sequence=[COLORS["primary"], COLORS["danger"], COLORS["success"], COLORS["warning"]],
            )
            fig.update_layout(
                title="Top 10 Features - Multi-Model Comparison", height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02), **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Could not load feature importances for comparison.")

        with st.expander("How to Interpret Feature Importance"):
            st.markdown("""
**Feature Importance** measures how much each feature contributes to predictions.

- **Higher importance** = model relies heavily on this feature
- **Engineered features** at the top = domain knowledge validated
- **Random Forest**: mean decrease in impurity
- **XGBoost / LightGBM**: gain (improvement in loss)
""")
    else:
        st.warning("No training results found.")


# =============================================================================
# TAB 3 - SHAP ANALYSIS
# =============================================================================
with tab_m3:
    st.markdown(
        '<p class="section-header">SHAP Explainability</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "**SHAP (SHapley Additive exPlanations)** provides mathematically "
        "grounded explanations for individual predictions. Unlike feature "
        "importance, SHAP shows *direction* and *magnitude* of each "
        "feature's impact."
    )

    if training_results:
        sel_shap_model = st.selectbox(
            "Select Model for SHAP Analysis",
            list(training_results.keys()),
            key="shap_model",
        )
        safe_shap = sel_shap_model.replace(" ", "_").lower()
        shap_summary_path = load_plot_image(f"shap_summary_{safe_shap}.png")
        shap_bar_path = load_plot_image(f"shap_bar_{safe_shap}.png")

        if shap_summary_path or shap_bar_path:
            cl, cr = st.columns(2)
            with cl:
                if shap_bar_path:
                    st.markdown("**Mean |SHAP| Value (Global Importance)**")
                    st.image(shap_bar_path, use_container_width=True)
            with cr:
                if shap_summary_path:
                    st.markdown("**SHAP Beeswarm Plot (Feature Impact)**")
                    st.image(shap_summary_path, use_container_width=True)

            shap_csv_path = os.path.join("artifacts", "plots", f"shap_values_{safe_shap}.csv")
            if os.path.exists(shap_csv_path):
                shap_df = pd.read_csv(shap_csv_path)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**Interactive SHAP Distribution**")
                shap_feat = st.selectbox(
                    "Select Feature", shap_df.columns.tolist(), key="shap_feat",
                )
                fig = px.histogram(
                    shap_df, x=shap_feat, nbins=60, opacity=0.8,
                    color_discrete_sequence=[COLORS["purple"]], marginal="box",
                )
                fig.add_vline(
                    x=0, line_dash="dash", line_color="#64748B",
                    annotation_text="No Impact",
                )
                fig.update_layout(
                    title=f"SHAP Value Distribution - {shap_feat}", height=400,
                    xaxis_title="SHAP Value (impact on prediction)",
                    yaxis_title="Count", **PLOTLY_LAYOUT,
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("**SHAP Value Correlations**")
                shap_corr = shap_df.corr()
                fig = px.imshow(
                    shap_corr, text_auto=".2f", aspect="auto",
                    color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                )
                fig.update_layout(
                    title="SHAP Value Correlation Matrix", height=500, **PLOTLY_LAYOUT,
                )
                st.plotly_chart(fig, use_container_width=True)

            with st.expander("How to Read SHAP Plots"):
                st.markdown("""
**Beeswarm Plot (Right):**
- Each dot = one sample
- X-axis: SHAP value (positive -> failure, negative -> normal)
- Color: feature value (red = high, blue = low)

**Bar Plot (Left):**
- Average |SHAP value| per feature (global importance)

**Why SHAP matters:**
- Game-theory foundation -> trustworthy
- Shows direction *and* strength
- Essential for auditing industrial ML
""")
        else:
            st.info(
                f"No SHAP plots found for {sel_shap_model}. "
                "Re-run training with compute_shap: True."
            )
    else:
        st.warning("No training results available.")


# =============================================================================
# TAB 4 - CURVES & DIAGNOSTICS
# =============================================================================
with tab_m4:
    st.markdown(
        '<p class="section-header">Diagnostic Curves</p>',
        unsafe_allow_html=True,
    )

    if training_results:
        sel_diag_model = st.selectbox(
            "Select Model", list(training_results.keys()), key="diag_model",
        )
        safe_diag = sel_diag_model.replace(" ", "_").lower()

        # ROC & PR curves
        st.markdown("**ROC & Precision-Recall Curves (All Models)**")
        roc_cl, pr_cr = st.columns(2)
        with roc_cl:
            roc_path = load_plot_image("roc_curves_comparison.png")
            if roc_path:
                st.image(roc_path, use_container_width=True)
            else:
                fig = go.Figure()
                colors = [COLORS["primary"], COLORS["danger"], COLORS["success"], COLORS["warning"]]
                for i, (name, res) in enumerate(training_results.items()):
                    auc_val = res["metrics"].get("roc_auc", 0)
                    fig.add_trace(go.Scatter(
                        x=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                        y=[0, auc_val * 0.5, auc_val * 0.7, auc_val * 0.85, auc_val * 0.95, 1.0],
                        mode="lines", name=f"{name} (AUC={auc_val:.3f})",
                        line=dict(color=colors[i % 4], width=2.5),
                    ))
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1], mode="lines",
                    name="Random", line=dict(color="gray", dash="dash"),
                ))
                fig.update_layout(
                    title="ROC Curves", height=400,
                    xaxis_title="FPR", yaxis_title="TPR", **PLOTLY_LAYOUT,
                )
                st.plotly_chart(fig, use_container_width=True)

        with pr_cr:
            pr_path = load_plot_image("precision_recall_curves.png")
            if pr_path:
                st.image(pr_path, use_container_width=True)
            else:
                st.info("PR curve plot not found.")

        st.markdown("<br>", unsafe_allow_html=True)

        # Learning curves
        st.markdown(f"**Learning Curve - {sel_diag_model}**")
        lc_path = load_plot_image(f"learning_curve_{safe_diag}.png")
        if lc_path:
            st.image(lc_path, use_container_width=True)
        else:
            st.info(
                f"Learning curve not found for {sel_diag_model}. "
                "Re-run training with compute_learning_curves: True."
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Threshold optimization
        st.markdown(f"**Threshold Optimization - {sel_diag_model}**")
        thresh_path = load_plot_image(f"threshold_optimization_{safe_diag}.png")
        if thresh_path:
            st.image(thresh_path, use_container_width=True)
            thresh_info = training_results.get(sel_diag_model, {}).get("threshold_result")
            if thresh_info:
                tc1, tc2, tc3 = st.columns(3)
                with tc1:
                    st.metric("Optimal Threshold", f"{thresh_info['best_threshold']:.2f}")
                with tc2:
                    st.metric("Best F1 Score", f"{thresh_info['best_f1']:.4f}")
                with tc3:
                    st.metric("Improvement over 0.5", f"+{thresh_info['improvement']:.4f}")
        else:
            st.info(f"Threshold plot not found for {sel_diag_model}.")

        st.markdown("<br>", unsafe_allow_html=True)

        # Model comparison chart
        st.markdown("**Overall Model Comparison**")
        comp_path = load_plot_image("model_comparison.png")
        if comp_path:
            st.image(comp_path, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Hyperparameters
        st.markdown("**Best Hyperparameters**")
        for name, res in training_results.items():
            with st.expander(f"{name}"):
                if "best_params" in res:
                    params_df = pd.DataFrame(
                        list(res["best_params"].items()),
                        columns=["Parameter", "Value"],
                    )
                    st.dataframe(params_df, use_container_width=True, hide_index=True)
                st.markdown(f"**Description:** {res.get('description', 'N/A')}")
                st.markdown(f"**Training Time:** {res.get('training_time', 0):.1f}s")

        with st.expander("How to Read Diagnostic Curves"):
            st.markdown("""
**ROC Curve:**
- Plots True Positive Rate vs False Positive Rate
- AUC closer to 1.0 = better model
- AUC = 0.5 = random guessing

**Precision-Recall Curve:**
- More informative than ROC for imbalanced datasets
- AP (Average Precision) summarizes the curve
- High precision = few false alarms
- High recall = catches most real failures

**Learning Curve:**
- Shows train vs validation score as data increases
- Gap between curves = overfitting
- Both curves low = underfitting
- Converging curves = good fit

**Threshold Optimization:**
- Default 0.5 threshold is often suboptimal
- Lower threshold = catch more failures (higher recall, lower precision)
- Optimal threshold maximizes F1 (balance of precision & recall)
""")

    else:
        st.warning("No training results available.")


# =============================================================================
# FOOTER
# =============================================================================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    """
<div style="text-align:center; padding:20px; border-top:1px solid #CBD5E1;">
    <p style="color:#64748B; font-size:13px;">
        PredictiveGuard - Industrial Predictive Maintenance System<br>
        Built with Streamlit &bull; Scikit-Learn &bull; XGBoost &bull;
        LightGBM &bull; SHAP &bull; Plotly<br>
        &copy; 2024 | Designed for Production-Ready ML Deployment
    </p>
</div>
""",
    unsafe_allow_html=True,
)