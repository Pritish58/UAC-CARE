import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Model Performance | UAC-CareAI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

REPORT_DIR = BASE_DIR / "reports"

DATA_DIR_1 = BASE_DIR / "Data" / "Processed"
DATA_DIR_2 = BASE_DIR / "Data" / "processed"


# ============================================================
# THEME
# ============================================================

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

dark = st.session_state.dark_mode


if dark:

    BG = "#07111F"
    CARD = "#0F1B2D"
    CARD_ALT = "#13243A"
    TEXT = "#F5F9FC"
    MUTED = "#91A5B8"
    BORDER = "#243A53"

    BLUE = "#60A5FA"
    CYAN = "#38BDF8"
    GREEN = "#34D399"
    ORANGE = "#FBBF24"
    RED = "#FB7185"

else:

    BG = "#F4F7FB"
    CARD = "#FFFFFF"
    CARD_ALT = "#F8FAFC"
    TEXT = "#14283D"
    MUTED = "#60758A"
    BORDER = "#DCE5EE"

    BLUE = "#1769E0"
    CYAN = "#087EA4"
    GREEN = "#159957"
    ORANGE = "#C77700"
    RED = "#D83A3A"


# ============================================================
# GLOBAL STYLE
#
# IMPORTANT:
# Only CSS is placed inside unsafe_allow_html.
# No dashboard content is generated through HTML.
# ============================================================

st.markdown(
    f"""
    <style>

    .stApp {{
        background-color: {BG};
        color: {TEXT};
    }}

    .block-container {{
        max-width: 1500px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }}

    h1, h2, h3, h4 {{
        color: {TEXT} !important;
    }}

    p, label {{
        color: {MUTED};
    }}

    /* --------------------------------------------------------
       STREAMLIT METRICS
    -------------------------------------------------------- */

    [data-testid="stMetric"] {{
        background-color: {CARD};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 16px 18px;
        min-height: 115px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.035);
    }}

    [data-testid="stMetricLabel"] {{
        color: {MUTED} !important;
        font-size: 10px !important;
        font-weight: 800 !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }}

    [data-testid="stMetricValue"] {{
        color: {TEXT} !important;
        font-size: 27px !important;
        font-weight: 850 !important;
    }}

    [data-testid="stMetricDelta"] {{
        font-size: 11px !important;
    }}

    /* --------------------------------------------------------
       CONTAINERS
    -------------------------------------------------------- */

    [data-testid="stVerticalBlockBorderWrapper"] {{
        background-color: {CARD};
        border-color: {BORDER} !important;
        border-radius: 15px;
    }}

    /* --------------------------------------------------------
       DATAFRAME
    -------------------------------------------------------- */

    [data-testid="stDataFrame"] {{
        border: 1px solid {BORDER};
        border-radius: 12px;
        overflow: hidden;
    }}

    /* --------------------------------------------------------
       ALERTS
    -------------------------------------------------------- */

    [data-testid="stAlert"] {{
        border-radius: 12px;
    }}

    /* --------------------------------------------------------
       DIVIDERS
    -------------------------------------------------------- */

    hr {{
        border-color: {BORDER} !important;
    }}

    /* --------------------------------------------------------
       SIDEBAR
    -------------------------------------------------------- */

    [data-testid="stSidebar"] {{
        background-color: {CARD};
    }}

    /* --------------------------------------------------------
       BUTTONS
    -------------------------------------------------------- */

    .stButton > button {{
        border-radius: 9px;
        border: 1px solid {BORDER};
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_report(filename):

    candidates = [
        REPORT_DIR / filename,
        BASE_DIR / "Reports" / filename
    ]

    for path in candidates:

        if path.exists():
            return path

    return None


def find_data(filename):

    candidates = [
        DATA_DIR_1 / filename,
        DATA_DIR_2 / filename
    ]

    for path in candidates:

        if path.exists():
            return path

    return None


def load_csv(path):

    if path is None:
        return pd.DataFrame()

    try:

        return pd.read_csv(path)

    except Exception:

        return pd.DataFrame()


def find_column(df, candidates):

    if df is None or df.empty:
        return None

    # Exact match
    for candidate in candidates:

        if candidate in df.columns:
            return candidate

    # Case-insensitive match
    lookup = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for candidate in candidates:

        key = str(candidate).strip().lower()

        if key in lookup:
            return lookup[key]

    # Partial matching
    for col in df.columns:

        col_low = str(col).lower()

        for candidate in candidates:

            if str(candidate).lower() in col_low:

                return col

    return None


def safe_numeric(series):

    return pd.to_numeric(
        series,
        errors="coerce"
    )


def chart_layout(fig, height=400):

    fig.update_layout(

        template=(
            "plotly_dark"
            if dark
            else "plotly_white"
        ),

        height=height,

        margin=dict(
            l=25,
            r=25,
            t=65,
            b=30
        ),

        plot_bgcolor="rgba(0,0,0,0)",

        paper_bgcolor="rgba(0,0,0,0)",

        hovermode="x unified"
    )

    return fig


def section_header(number, title, description):

    st.caption(
        number
    )

    st.subheader(
        title
    )

    st.write(
        description
    )


# ============================================================
# LOAD REPORT FILES
# ============================================================

load_results_path = find_report(
    "load_model_results.csv"
)

load_predictions_path = find_report(
    "load_predictions.csv"
)

stress_results_path = find_report(
    "stress_model_results.csv"
)

stress_confusion_path = find_report(
    "stress_confusion_matrix.csv"
)

final_validation_path = find_report(
    "final_ml_validation_report.csv"
)

anomaly_summary_path = find_report(
    "anomaly_summary.csv"
)

cluster_summary_path = find_report(
    "cluster_summary.csv"
)

shap_importance_path = find_report(
    "shap_feature_importance.csv"
)


load_results = load_csv(
    load_results_path
)

load_predictions = load_csv(
    load_predictions_path
)

stress_results = load_csv(
    stress_results_path
)

stress_confusion = load_csv(
    stress_confusion_path
)

final_validation = load_csv(
    final_validation_path
)

anomaly_summary = load_csv(
    anomaly_summary_path
)

cluster_summary = load_csv(
    cluster_summary_path
)

shap_importance = load_csv(
    shap_importance_path
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.subheader(
        "🎯 UAC-CareAI"
    )

    st.caption(
        "Model Evaluation Lab"
    )

    st.divider()

    st.write(
        "This module evaluates trained predictive models "
        "using validation results generated by the ML pipeline."
    )

    st.divider()

    st.write(
        "**Evaluation Scope**"
    )

    st.caption(
        "• Load prediction"
    )

    st.caption(
        "• Stress classification"
    )

    st.caption(
        "• Prediction reliability"
    )

    st.caption(
        "• Model validation"
    )

    st.caption(
        "• Explainability inventory"
    )

    st.divider()

    st.write(
        "**Appearance**"
    )

    dark_mode = st.toggle(
        "🌙 Dark Mode",
        value=st.session_state.dark_mode
    )

    if dark_mode != st.session_state.dark_mode:

        st.session_state.dark_mode = dark_mode

        st.rerun()

    st.divider()

    st.caption(
        "UAC-CareAI v1.0"
    )

    st.caption(
        "Machine Learning • Validation • Explainable AI"
    )


# ============================================================
# PAGE HEADER
# ============================================================

st.caption(
    "UAC-CAREAI  /  MACHINE LEARNING"
)

st.title(
    "Model Performance"
)

st.write(
    "A dedicated evaluation layer for measuring predictive "
    "accuracy, comparing candidate algorithms, validating "
    "classification performance and identifying the models "
    "retained for operational decision support."
)


# ============================================================
# MODEL EVALUATION STATUS
# ============================================================

with st.container(border=True):

    st.subheader(
        "Evaluation Status"
    )

    status_cols = st.columns(4)

    with status_cols[0]:

        if not load_results.empty:

            st.success(
                "✓ Load evaluation connected"
            )

        else:

            st.error(
                "✗ Load evaluation missing"
            )

    with status_cols[1]:

        if not load_predictions.empty:

            st.success(
                "✓ Prediction output connected"
            )

        else:

            st.warning(
                "⚠ Prediction output missing"
            )

    with status_cols[2]:

        if not stress_results.empty:

            st.success(
                "✓ Stress evaluation connected"
            )

        else:

            st.warning(
                "⚠ Stress evaluation missing"
            )

    with status_cols[3]:

        if not final_validation.empty:

            st.success(
                "✓ Validation report connected"
            )

        else:

            st.warning(
                "⚠ Validation report missing"
            )

    st.write(
        "The Model Performance page evaluates trained models "
        "using previously generated validation reports. "
        "Future time-series forecasting is handled separately "
        "by the Forecasting module."
    )


# ============================================================
# FIND BEST REGRESSION MODEL
# ============================================================

best_regression = None

regression_model_col = find_column(
    load_results,
    [
        "Model",
        "model",
        "Algorithm",
        "algorithm"
    ]
)

regression_rmse_col = find_column(
    load_results,
    [
        "RMSE",
        "rmse",
        "Root Mean Squared Error"
    ]
)

regression_mae_col = find_column(
    load_results,
    [
        "MAE",
        "mae",
        "Mean Absolute Error"
    ]
)

regression_r2_col = find_column(
    load_results,
    [
        "R2",
        "R²",
        "R2 Score",
        "R² Score"
    ]
)


if (
    not load_results.empty
    and regression_rmse_col is not None
):

    temp = load_results.copy()

    temp[regression_rmse_col] = safe_numeric(
        temp[regression_rmse_col]
    )

    temp = temp.dropna(
        subset=[
            regression_rmse_col
        ]
    )

    if not temp.empty:

        best_regression = temp.loc[
            temp[regression_rmse_col].idxmin()
        ]


# ============================================================
# FIND BEST CLASSIFIER
# ============================================================

best_classifier = None

classifier_model_col = find_column(
    stress_results,
    [
        "Model",
        "model",
        "Algorithm",
        "algorithm"
    ]
)

classifier_f1_col = None

if not stress_results.empty:

    f1_candidates = [

        column

        for column in stress_results.columns

        if "f1" in str(column).lower()

    ]

    if f1_candidates:

        classifier_f1_col = f1_candidates[0]


if (
    not stress_results.empty
    and classifier_f1_col is not None
):

    temp = stress_results.copy()

    temp[classifier_f1_col] = safe_numeric(
        temp[classifier_f1_col]
    )

    temp = temp.dropna(
        subset=[
            classifier_f1_col
        ]
    )

    if not temp.empty:

        best_classifier = temp.loc[
            temp[classifier_f1_col].idxmax()
        ]


# ============================================================
# SECTION 01
# EXECUTIVE EVALUATION
# ============================================================

st.divider()

section_header(
    "01 / EXECUTIVE EVALUATION",
    "Which models performed best?",
    "The selected models are determined directly from the "
    "validation reports generated by the ML pipeline."
)


winner_cols = st.columns(2)


# ------------------------------------------------------------
# BEST LOAD MODEL
# ------------------------------------------------------------

with winner_cols[0]:

    with st.container(border=True):

        st.caption(
            "BEST LOAD PREDICTION MODEL"
        )

        if best_regression is not None:

            if regression_model_col:

                model_name = str(
                    best_regression[
                        regression_model_col
                    ]
                )

            else:

                model_name = "Selected Regression Model"


            st.subheader(
                f"🏆 {model_name}"
            )

            st.write(
                "Selected using the lowest validation RMSE."
            )

            metric_cols = st.columns(3)


            rmse_value = (
                best_regression[
                    regression_rmse_col
                ]
                if regression_rmse_col
                else np.nan
            )


            mae_value = (
                best_regression[
                    regression_mae_col
                ]
                if regression_mae_col
                else np.nan
            )


            r2_value = (
                best_regression[
                    regression_r2_col
                ]
                if regression_r2_col
                else np.nan
            )


            with metric_cols[0]:

                if pd.notna(rmse_value):

                    st.metric(
                        "RMSE",
                        f"{float(rmse_value):,.2f}"
                    )


            with metric_cols[1]:

                if pd.notna(mae_value):

                    st.metric(
                        "MAE",
                        f"{float(mae_value):,.2f}"
                    )


            with metric_cols[2]:

                if pd.notna(r2_value):

                    st.metric(
                        "R²",
                        f"{float(r2_value):.3f}"
                    )

        else:

            st.warning(
                "Regression model results are unavailable."
            )


# ------------------------------------------------------------
# BEST STRESS CLASSIFIER
# ------------------------------------------------------------

with winner_cols[1]:

    with st.container(border=True):

        st.caption(
            "BEST STRESS CLASSIFIER"
        )

        if best_classifier is not None:

            if classifier_model_col:

                model_name = str(
                    best_classifier[
                        classifier_model_col
                    ]
                )

            else:

                model_name = "Selected Classifier"


            st.subheader(
                f"🎯 {model_name}"
            )

            st.write(
                "Selected using the highest validation F1 score."
            )


            if classifier_f1_col:

                f1_value = best_classifier[
                    classifier_f1_col
                ]

                if pd.notna(f1_value):

                    st.metric(
                        "F1 Score",
                        f"{float(f1_value):.3f}"
                    )

        else:

            st.warning(
                "Stress classification results are unavailable."
            )


# ============================================================
# SECTION 02
# LOAD PREDICTION BENCHMARK
# ============================================================

st.divider()

section_header(
    "02 / LOAD PREDICTION",
    "Regression Model Benchmark",
    "Candidate regression algorithms are compared using "
    "RMSE, MAE and R². Lower error and higher R² indicate "
    "stronger predictive performance."
)


if not load_results.empty:

    display_results = load_results.copy()

    st.dataframe(
        display_results,
        use_container_width=True,
        hide_index=True
    )


    chart_cols = st.columns(2)


    # --------------------------------------------------------
    # RMSE
    # --------------------------------------------------------

    with chart_cols[0]:

        if (
            regression_model_col
            and regression_rmse_col
        ):

            chart_df = load_results.copy()

            chart_df[
                regression_rmse_col
            ] = safe_numeric(
                chart_df[
                    regression_rmse_col
                ]
            )


            fig = px.bar(
                chart_df,
                x=regression_model_col,
                y=regression_rmse_col,
                title="RMSE Comparison — Lower is Better",
                text_auto=".2f"
            )


            fig.update_layout(
                xaxis_title="Model",
                yaxis_title="RMSE"
            )


            chart_layout(
                fig,
                390
            )


            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displayModeBar": False
                }
            )


    # --------------------------------------------------------
    # R2
    # --------------------------------------------------------

    with chart_cols[1]:

        if (
            regression_model_col
            and regression_r2_col
        ):

            chart_df = load_results.copy()

            chart_df[
                regression_r2_col
            ] = safe_numeric(
                chart_df[
                    regression_r2_col
                ]
            )


            fig = px.bar(
                chart_df,
                x=regression_model_col,
                y=regression_r2_col,
                title="R² Comparison — Higher is Better",
                text_auto=".3f"
            )


            fig.update_layout(
                xaxis_title="Model",
                yaxis_title="R²"
            )


            chart_layout(
                fig,
                390
            )


            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displayModeBar": False
                }
            )

else:

    st.warning(
        "load_model_results.csv was not found."
    )


# ============================================================
# SECTION 03
# ACTUAL VS PREDICTED
# ============================================================

st.divider()

section_header(
    "03 / PREDICTION RELIABILITY",
    "Actual vs Predicted System Load",
    "A visual validation of the selected machine-learning "
    "load predictor using held-out observations."
)


if not load_predictions.empty:

    prediction_date_col = find_column(
        load_predictions,
        [
            "Date",
            "date"
        ]
    )

    actual_col = find_column(
        load_predictions,
        [
            "Actual",
            "Actual Load",
            "Actual System Load",
            "y_true"
        ]
    )

    predicted_col = find_column(
        load_predictions,
        [
            "Predicted",
            "Predicted Load",
            "Predicted System Load",
            "y_pred"
        ]
    )


    if (
        prediction_date_col
        and actual_col
        and predicted_col
    ):

        prediction_df = load_predictions.copy()


        prediction_df[
            prediction_date_col
        ] = pd.to_datetime(
            prediction_df[
                prediction_date_col
            ],
            errors="coerce"
        )


        prediction_df[
            actual_col
        ] = safe_numeric(
            prediction_df[
                actual_col
            ]
        )


        prediction_df[
            predicted_col
        ] = safe_numeric(
            prediction_df[
                predicted_col
            ]
        )


        prediction_df = prediction_df.dropna(
            subset=[
                prediction_date_col,
                actual_col,
                predicted_col
            ]
        )


        prediction_df = prediction_df.sort_values(
            prediction_date_col
        )


        fig = go.Figure()


        fig.add_trace(
            go.Scatter(
                x=prediction_df[
                    prediction_date_col
                ],
                y=prediction_df[
                    actual_col
                ],
                mode="lines",
                name="Actual Load",
                line=dict(
                    width=2.5
                )
            )
        )


        fig.add_trace(
            go.Scatter(
                x=prediction_df[
                    prediction_date_col
                ],
                y=prediction_df[
                    predicted_col
                ],
                mode="lines",
                name="Predicted Load",
                line=dict(
                    width=2,
                    dash="dash"
                )
            )
        )


        fig.update_layout(
            title="Held-Out Actual vs Predicted System Load",
            xaxis_title="Date",
            yaxis_title="System Load"
        )


        chart_layout(
            fig,
            460
        )


        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displayModeBar": False
            }
        )


        # ----------------------------------------------------
        # ERROR CALCULATIONS
        # ----------------------------------------------------

        prediction_df[
            "Prediction Error"
        ] = (
            prediction_df[
                actual_col
            ]
            -
            prediction_df[
                predicted_col
            ]
        )


        prediction_df[
            "Absolute Error"
        ] = prediction_df[
            "Prediction Error"
        ].abs()


        mean_error = prediction_df[
            "Prediction Error"
        ].mean()


        mean_absolute_error = prediction_df[
            "Absolute Error"
        ].mean()


        maximum_error = prediction_df[
            "Absolute Error"
        ].max()


        validation_rows = len(
            prediction_df
        )


        error_cols = st.columns(4)


        with error_cols[0]:

            st.metric(
                "Mean Error",
                f"{mean_error:,.2f}"
            )


        with error_cols[1]:

            st.metric(
                "MAE",
                f"{mean_absolute_error:,.2f}"
            )


        with error_cols[2]:

            st.metric(
                "Max Error",
                f"{maximum_error:,.2f}"
            )


        with error_cols[3]:

            st.metric(
                "Validation Rows",
                f"{validation_rows:,}"
            )


        # ----------------------------------------------------
        # ERROR DISTRIBUTION
        # ----------------------------------------------------

        st.subheader(
            "Prediction Error Distribution"
        )

        error_fig = px.histogram(
            prediction_df,
            x="Prediction Error",
            nbins=25,
            title="Distribution of Prediction Errors"
        )


        error_fig.add_vline(
            x=0,
            line_dash="dash"
        )


        error_fig.update_layout(
            xaxis_title="Prediction Error",
            yaxis_title="Frequency"
        )


        chart_layout(
            error_fig,
            380
        )


        st.plotly_chart(
            error_fig,
            use_container_width=True,
            config={
                "displayModeBar": False
            }
        )


    else:

        st.warning(
            "The prediction file was found, but the expected "
            "Date / Actual / Predicted columns could not be identified."
        )

        st.write(
            "Available columns:"
        )

        st.write(
            list(load_predictions.columns)
        )

else:

    st.warning(
        "load_predictions.csv was not found."
    )


# ============================================================
# SECTION 04
# STRESS CLASSIFICATION
# ============================================================

st.divider()

section_header(
    "04 / OPERATIONAL STRESS",
    "Stress Classification Benchmark",
    "Candidate classifiers are compared using classification "
    "metrics. Weighted F1 is used as the primary selection measure."
)


if not stress_results.empty:

    st.dataframe(
        stress_results,
        use_container_width=True,
        hide_index=True
    )


    if (
        classifier_model_col
        and classifier_f1_col
    ):

        stress_chart = stress_results.copy()

        stress_chart[
            classifier_f1_col
        ] = safe_numeric(
            stress_chart[
                classifier_f1_col
            ]
        )


        fig = px.bar(
            stress_chart,
            x=classifier_model_col,
            y=classifier_f1_col,
            title="Weighted F1 Score by Classifier",
            text_auto=".3f"
        )


        fig.update_layout(
            xaxis_title="Classifier",
            yaxis_title="Weighted F1"
        )


        chart_layout(
            fig,
            390
        )


        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displayModeBar": False
            }
        )

else:

    st.warning(
        "stress_model_results.csv was not found."
    )


# ============================================================
# SECTION 05
# CONFUSION MATRIX
# ============================================================

st.divider()

section_header(
    "05 / CLASSIFICATION VALIDATION",
    "Stress Classification Confusion Matrix",
    "The confusion matrix shows how the selected stress "
    "classifier distributes predictions across the operational classes."
)


if not stress_confusion.empty:

    confusion = stress_confusion.copy()


    # Try to identify a matrix-style numeric table.
    numeric_columns = confusion.select_dtypes(
        include=np.number
    ).columns.tolist()


    if len(numeric_columns) >= 2:

        fig = px.imshow(
            confusion[
                numeric_columns
            ],
            text_auto=True,
            aspect="auto",
            title="Stress Classification Confusion Matrix"
        )


        fig.update_layout(
            xaxis_title="Predicted Class",
            yaxis_title="Actual Class"
        )


        chart_layout(
            fig,
            420
        )


        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displayModeBar": False
            }
        )


    else:

        st.dataframe(
            confusion,
            use_container_width=True,
            hide_index=True
        )

else:

    st.info(
        "stress_confusion_matrix.csv is not available."
    )


# ============================================================
# SECTION 06
# AI COMPONENT INVENTORY
# ============================================================

st.divider()

section_header(
    "06 / AI COMPONENTS",
    "Model & Analytics Inventory",
    "UAC-CareAI uses complementary machine-learning patterns "
    "rather than relying on a single algorithm."
)


inventory = pd.DataFrame({

    "Component": [

        "Load Prediction",

        "Stress Classification",

        "Anomaly Detection",

        "Operational Clustering",

        "SARIMA Forecasting",

        "SHAP Explainability"

    ],

    "Purpose": [

        "Predict next-reporting system load",

        "Classify operational stress",

        "Identify unusual observations",

        "Discover recurring operational patterns",

        "Forecast future system load",

        "Explain feature influence"

    ],

    "Method": [

        "Regression model comparison",

        "Classification model comparison",

        "Isolation Forest",

        "K-Means",

        "SARIMA / SARIMAX",

        "SHAP"

    ],

    "Evaluation Focus": [

        "RMSE / MAE / R²",

        "Accuracy / Precision / Recall / F1",

        "Anomaly identification",

        "Silhouette score",

        "Forecast error",

        "Feature importance"

    ]

})


st.dataframe(
    inventory,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# ADDITIONAL MODEL OUTPUT STATUS
# ============================================================

st.divider()

section_header(
    "07 / ANALYTICS COVERAGE",
    "Supporting AI Modules",
    "The following project outputs provide additional analytical "
    "evidence beyond the core predictive benchmarks."
)


support_cols = st.columns(3)


# ------------------------------------------------------------
# ANOMALY
# ------------------------------------------------------------

with support_cols[0]:

    with st.container(border=True):

        st.subheader(
            "🚨 Anomaly Detection"
        )

        if not anomaly_summary.empty:

            st.success(
                "Isolation Forest output available"
            )

            st.write(
                "Operational observations are screened "
                "for unusual system behaviour."
            )

        else:

            st.warning(
                "Anomaly summary unavailable."
            )


# ------------------------------------------------------------
# CLUSTERING
# ------------------------------------------------------------

with support_cols[1]:

    with st.container(border=True):

        st.subheader(
            "🔎 Operational Patterns"
        )

        if not cluster_summary.empty:

            st.success(
                "K-Means output available"
            )

            st.write(
                "Historical operating conditions are "
                "grouped into interpretable patterns."
            )

        else:

            st.warning(
                "Cluster summary unavailable."
            )


# ------------------------------------------------------------
# SHAP
# ------------------------------------------------------------

with support_cols[2]:

    with st.container(border=True):

        st.subheader(
            "🧠 Explainable AI"
        )

        if not shap_importance.empty:

            st.success(
                "SHAP analysis available"
            )

            st.write(
                "Feature influence is available for "
                "the selected load prediction model."
            )

        else:

            st.warning(
                "SHAP importance unavailable."
            )


# ============================================================
# SECTION 08
# MODEL SELECTION LOGIC
# ============================================================

st.divider()

section_header(
    "08 / MODEL SELECTION",
    "Why Were These Models Selected?",
    "The selection rules used by the project provide a transparent "
    "basis for retaining the preferred predictive models."
)


with st.container(border=True):

    st.subheader(
        "Load Prediction"
    )

    st.write(
        "Regression candidates are compared using RMSE, MAE "
        "and R². The preferred load predictor is the model "
        "with the lowest validation RMSE."
    )


    st.divider()


    st.subheader(
        "Stress Classification"
    )

    st.write(
        "Classification candidates are compared using accuracy, "
        "precision, recall and F1. The preferred classifier is "
        "selected using the highest validation F1 score."
    )


    st.divider()


    st.subheader(
        "Anomaly Detection"
    )

    st.write(
        "Isolation Forest is used to identify observations "
        "whose operational characteristics differ substantially "
        "from normal historical patterns."
    )


    st.divider()


    st.subheader(
        "Operational Clustering"
    )

    st.write(
        "K-Means is used to group observations with similar "
        "workload, custody, transfer and intake-pressure characteristics."
    )


    st.divider()


    st.subheader(
        "Explainability"
    )

    st.write(
        "SHAP is used to examine which input variables have "
        "the strongest influence on model predictions."
    )


    st.divider()


    st.subheader(
        "Forecasting Separation"
    )

    st.write(
        "SARIMA/SARIMAX forecasting is intentionally handled "
        "on the separate Forecasting page. Model Performance "
        "evaluates predictive-model validation; Forecasting "
        "handles temporal future projections."
    )


# ============================================================
# SECTION 09
# VALIDATION STATUS
# ============================================================

st.divider()

section_header(
    "09 / VALIDATION STATUS",
    "ML Pipeline Validation",
    "Current analytical components represented in the project."
)


validation_cols = st.columns(4)


with validation_cols[0]:

    st.metric(
        "Load Model",
        "Validated"
        if not load_results.empty
        else "Unavailable",
        "Regression benchmark"
    )


with validation_cols[1]:

    st.metric(
        "Stress Model",
        "Validated"
        if not stress_results.empty
        else "Unavailable",
        "Classification benchmark"
    )


with validation_cols[2]:

    st.metric(
        "Anomaly Engine",
        "Active"
        if not anomaly_summary.empty
        else "Unavailable",
        "Isolation Forest"
    )


with validation_cols[3]:

    st.metric(
        "Explainability",
        "Active"
        if not shap_importance.empty
        else "Unavailable",
        "SHAP analysis"
    )


# ============================================================
# FINAL METHODOLOGY NOTE
# ============================================================

st.divider()

st.subheader(
    "Methodology Note"
)

st.info(
    "Performance values shown on this page are validation "
    "results generated from the UAC project dataset. They "
    "describe historical/held-out validation performance and "
    "should not be interpreted as guarantees of future "
    "operational performance."
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "UAC-CareAI • Model Evaluation Lab • "
    "Machine Learning • Validation • Explainable AI"
)