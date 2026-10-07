import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "Data" / "Processed"
MODEL_DIR = BASE_DIR / "Models"
REPORT_DIR = BASE_DIR / "reports"

FINAL_REPORT = REPORT_DIR / "final_ml_validation_report.csv"


# ============================================================
# HELPER
# ============================================================

validation_results = []


def check_file(name, path):
    exists = path.exists()

    validation_results.append({
        "Component": name,
        "Check": "File exists",
        "Status": "PASS" if exists else "FAIL",
        "Details": str(path)
    })

    return exists


def add_result(component, check, status, details):
    validation_results.append({
        "Component": component,
        "Check": check,
        "Status": status,
        "Details": details
    })


# ============================================================
# HEADER
# ============================================================

print("\n" + "=" * 70)
print("UAC CARE ANALYTICS - FINAL ML VALIDATION")
print("=" * 70)


# ============================================================
# 1. DATA VALIDATION
# ============================================================

print("\n[1] DATA VALIDATION")

cleaned_file = DATA_DIR / "uac_cleaned.csv"
features_file = DATA_DIR / "uac_features.csv"

if check_file("Data", "Cleaned dataset",) if False else False:
    pass

cleaned_exists = check_file(
    "Data",
    cleaned_file
)

features_exists = check_file(
    "Feature Engineering",
    features_file
)


if cleaned_exists:

    cleaned = pd.read_csv(
        cleaned_file
    )

    print(
        f"Cleaned dataset rows    : {len(cleaned)}"
    )

    print(
        f"Cleaned dataset columns : {len(cleaned.columns)}"
    )

    add_result(
        "Data",
        "No negative values",
        "PASS"
        if (
            cleaned.select_dtypes(
                include=np.number
            ) >= 0
        ).all().all()
        else "WARNING",
        "Checked numeric columns"
    )


if features_exists:

    features = pd.read_csv(
        features_file
    )

    print(
        f"Feature dataset rows    : {len(features)}"
    )

    print(
        f"Feature dataset columns : {len(features.columns)}"
    )

    add_result(
        "Feature Engineering",
        "Target exists",
        "PASS"
        if "Next Day System Load" in features.columns
        else "FAIL",
        "Next Day System Load"
    )


# ============================================================
# 2. LOAD PREDICTION
# ============================================================

print("\n[2] LOAD PREDICTION")

load_model = MODEL_DIR / "best_load_prediction_model.pkl"
load_results = REPORT_DIR / "load_model_results.csv"
load_predictions = REPORT_DIR / "load_predictions.csv"

check_file(
    "Load Prediction",
    load_model
)

check_file(
    "Load Prediction",
    load_results
)

check_file(
    "Load Prediction",
    load_predictions
)


if load_results.exists():

    results = pd.read_csv(
        load_results
    )

    print("\nLoad model comparison:")

    print(
        results.to_string(
            index=False
        )
    )

    if "RMSE" in results.columns:

        best_row = results.loc[
            results["RMSE"].idxmin()
        ]

        print(
            "\nBest model:",
            best_row.iloc[0]
        )

        print(
            "Best RMSE:",
            best_row["RMSE"]
        )

        add_result(
            "Load Prediction",
            "Model comparison completed",
            "PASS",
            f"Best RMSE = {best_row['RMSE']}"
        )


# ============================================================
# 3. STRESS CLASSIFICATION
# ============================================================

print("\n[3] STRESS CLASSIFICATION")

stress_model = MODEL_DIR / "best_stress_model.pkl"
stress_results = REPORT_DIR / "stress_model_results.csv"
stress_matrix = REPORT_DIR / "stress_confusion_matrix.csv"
stress_labels = DATA_DIR / "uac_stress_labels.csv"

check_file(
    "Stress Classification",
    stress_model
)

check_file(
    "Stress Classification",
    stress_results
)

check_file(
    "Stress Classification",
    stress_matrix
)

check_file(
    "Stress Classification",
    stress_labels
)


if stress_results.exists():

    stress_df = pd.read_csv(
        stress_results
    )

    print("\nStress model comparison:")

    print(
        stress_df.to_string(
            index=False
        )
    )

    add_result(
        "Stress Classification",
        "Model evaluation available",
        "PASS",
        "Stress results successfully loaded"
    )


# ============================================================
# 4. ANOMALY DETECTION
# ============================================================

print("\n[4] ANOMALY DETECTION")

anomaly_model = MODEL_DIR / "anomaly_detection_model.pkl"
anomaly_results = DATA_DIR / "uac_anomaly_results.csv"
anomaly_summary = REPORT_DIR / "anomaly_summary.csv"

check_file(
    "Anomaly Detection",
    anomaly_model
)

check_file(
    "Anomaly Detection",
    anomaly_results
)

check_file(
    "Anomaly Detection",
    anomaly_summary
)


if anomaly_results.exists():

    anomaly_df = pd.read_csv(
        anomaly_results
    )

    if "Anomaly Flag" in anomaly_df.columns:

        anomaly_count = int(
            anomaly_df["Anomaly Flag"].sum()
        )

        anomaly_rate = (
            anomaly_count
            / len(anomaly_df)
            * 100
        )

        print(
            f"\nAnomalies detected : {anomaly_count}"
        )

        print(
            f"Anomaly rate       : {anomaly_rate:.2f}%"
        )

        add_result(
            "Anomaly Detection",
            "Anomaly rate calculated",
            "PASS",
            f"{anomaly_rate:.2f}%"
        )


# ============================================================
# 5. CLUSTERING
# ============================================================

print("\n[5] OPERATIONAL CLUSTERING")

cluster_model = MODEL_DIR / "operational_clustering_model.pkl"
cluster_data = DATA_DIR / "uac_operational_clusters.csv"
cluster_summary = REPORT_DIR / "cluster_summary.csv"

# Support the filename created by clustering.py
if not cluster_model.exists():

    alternative_model = (
        MODEL_DIR
        / "operational_clustering_model.pkl"
    )

    cluster_model = alternative_model


check_file(
    "Operational Clustering",
    cluster_model
)

check_file(
    "Operational Clustering",
    cluster_data
)

check_file(
    "Operational Clustering",
    cluster_summary
)


if cluster_summary.exists():

    cluster_df = pd.read_csv(
        cluster_summary
    )

    print("\nCluster summary:")

    print(
        cluster_df.to_string(
            index=False
        )
    )

    add_result(
        "Operational Clustering",
        "Cluster summary available",
        "PASS",
        f"{len(cluster_df)} clusters"
    )


# ============================================================
# 6. TIME-SERIES FORECASTING
# ============================================================

print("\n[6] TIME-SERIES FORECASTING")

forecast_model = MODEL_DIR / "sarima_forecasting_model.pkl"
forecast_file = REPORT_DIR / "sarima_forecast.csv"
forecast_results = REPORT_DIR / "forecast_results.csv"

check_file(
    "SARIMA Forecasting",
    forecast_model
)

check_file(
    "SARIMA Forecasting",
    forecast_file
)

check_file(
    "SARIMA Forecasting",
    forecast_results
)


if forecast_results.exists():

    forecast_df = pd.read_csv(
        forecast_results
    )

    print("\nSARIMA evaluation:")

    print(
        forecast_df.to_string(
            index=False
        )
    )

    add_result(
        "SARIMA Forecasting",
        "Forecast evaluation available",
        "PASS",
        "MAE and RMSE available"
    )


# ============================================================
# 7. SHAP EXPLAINABILITY
# ============================================================

print("\n[7] SHAP EXPLAINABILITY")

shap_values = REPORT_DIR / "shap_values.csv"

shap_importance = (
    REPORT_DIR
    / "shap_feature_importance.csv"
)

shap_summary = (
    REPORT_DIR
    / "figures"
    / "shap_summary.png"
)

shap_bar = (
    REPORT_DIR
    / "figures"
    / "shap_feature_importance.png"
)


check_file(
    "Explainability",
    shap_values
)

check_file(
    "Explainability",
    shap_importance
)

check_file(
    "Explainability",
    shap_summary
)

check_file(
    "Explainability",
    shap_bar
)


if shap_importance.exists():

    shap_df = pd.read_csv(
        shap_importance
    )

    print("\nTop SHAP features:")

    print(
        shap_df.head(10).to_string(
            index=False
        )
    )

    add_result(
        "Explainability",
        "Feature importance available",
        "PASS",
        "SHAP importance successfully generated"
    )


# ============================================================
# 8. MODEL INVENTORY
# ============================================================

print("\n[8] MODEL INVENTORY")

models = [
    "best_load_prediction_model.pkl",
    "best_stress_model.pkl",
    "anomaly_detection_model.pkl",
    "operational_clustering_model.pkl",
    "sarima_forecasting_model.pkl"
]

for model_name in models:

    path = MODEL_DIR / model_name

    status = (
        "PASS"
        if path.exists()
        else "FAIL"
    )

    print(
        f"{model_name:<45} {status}"
    )


# ============================================================
# 9. FINAL VALIDATION SUMMARY
# ============================================================

validation_df = pd.DataFrame(
    validation_results
)


validation_df.to_csv(
    FINAL_REPORT,
    index=False
)


pass_count = (
    validation_df["Status"]
    == "PASS"
).sum()

fail_count = (
    validation_df["Status"]
    == "FAIL"
).sum()

warning_count = (
    validation_df["Status"]
    == "WARNING"
).sum()


print("\n" + "=" * 70)
print("FINAL VALIDATION SUMMARY")
print("=" * 70)

print(
    f"\nPASS    : {pass_count}"
)

print(
    f"WARNING : {warning_count}"
)

print(
    f"FAIL    : {fail_count}"
)


if fail_count == 0:

    print(
        "\nSTATUS: ALL REQUIRED ML OUTPUTS AVAILABLE"
    )

else:

    print(
        "\nSTATUS: SOME OUTPUTS REQUIRE ATTENTION"
    )


print(
    f"\nFinal report:"
)

print(
    FINAL_REPORT
)

print("\n" + "=" * 70)
print("FINAL ML VALIDATION COMPLETED")
print("=" * 70)