import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "Data"
    / "Processed"
    / "uac_features.csv"
)

MODEL_FILE = (
    BASE_DIR
    / "Models"
    / "anomaly_detection_model.pkl"
)

OUTPUT_FILE = (
    BASE_DIR
    / "Data"
    / "Processed"
    / "uac_anomaly_results.csv"
)

REPORT_FILE = (
    BASE_DIR
    / "reports"
    / "anomaly_summary.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

df["Date"] = pd.to_datetime(df["Date"])


# ============================================================
# FEATURES FOR ANOMALY DETECTION
# ============================================================

anomaly_features = [
    "Total System Load",
    "Net Intake Pressure",
    "Daily Load Change",
    "Discharge Offset Ratio",
    "Transfer to CBP Ratio",
    "7-Day Net Intake",
    "14-Day Net Intake",
    "Load Volatility 7D",
    "Load Volatility 14D"
]


# Keep only features that actually exist
available_features = [
    col for col in anomaly_features
    if col in df.columns
]

print("\n" + "=" * 65)
print("UAC CARE ANALYTICS - ANOMALY DETECTION")
print("=" * 65)

print("\nFeatures used:")
for feature in available_features:
    print(f" - {feature}")


# ============================================================
# PREPARE DATA
# ============================================================

X = df[available_features].copy()

# Replace infinite values
X = X.replace([np.inf, -np.inf], np.nan)

# Fill missing values using median
X = X.fillna(X.median())


# ============================================================
# STANDARDIZATION
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ============================================================
# ISOLATION FOREST
# ============================================================

model = IsolationForest(
    n_estimators=300,
    contamination=0.05,
    random_state=42,
    n_jobs=-1
)

model.fit(X_scaled)


# ============================================================
# PREDICT ANOMALIES
# ============================================================

predictions = model.predict(X_scaled)

# Isolation Forest:
#  1  = Normal
# -1  = Anomaly

df["Anomaly Flag"] = np.where(
    predictions == -1,
    1,
    0
)

df["Anomaly Status"] = np.where(
    df["Anomaly Flag"] == 1,
    "ANOMALY",
    "NORMAL"
)


# ============================================================
# ANOMALY SCORE
# ============================================================

# Higher decision function = more normal
# Therefore multiply by -1 so higher score = more anomalous

df["Anomaly Score"] = -model.decision_function(X_scaled)


# ============================================================
# ANOMALY SEVERITY
# ============================================================

score_75 = df["Anomaly Score"].quantile(0.75)
score_90 = df["Anomaly Score"].quantile(0.90)
score_97 = df["Anomaly Score"].quantile(0.97)


def classify_anomaly(score):
    if score >= score_97:
        return "CRITICAL"
    elif score >= score_90:
        return "HIGH"
    elif score >= score_75:
        return "MODERATE"
    else:
        return "NORMAL"


df["Anomaly Severity"] = df["Anomaly Score"].apply(
    classify_anomaly
)


# ============================================================
# SAVE MODEL
# ============================================================

model_package = {
    "model": model,
    "scaler": scaler,
    "features": available_features
}

joblib.dump(
    model_package,
    MODEL_FILE
)


# ============================================================
# SAVE RESULTS
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# ANOMALY SUMMARY
# ============================================================

total_records = len(df)

total_anomalies = int(
    df["Anomaly Flag"].sum()
)

anomaly_percentage = (
    total_anomalies / total_records
) * 100


severity_counts = (
    df["Anomaly Severity"]
    .value_counts()
)


summary = pd.DataFrame({
    "Metric": [
        "Total Records",
        "Total Anomalies",
        "Anomaly Percentage",
        "Critical Anomalies",
        "High Anomalies",
        "Moderate Anomalies"
    ],
    "Value": [
        total_records,
        total_anomalies,
        round(anomaly_percentage, 2),
        int(severity_counts.get("CRITICAL", 0)),
        int(severity_counts.get("HIGH", 0)),
        int(severity_counts.get("MODERATE", 0))
    ]
})

summary.to_csv(
    REPORT_FILE,
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 65)
print("ANOMALY DETECTION RESULTS")
print("=" * 65)

print(f"\nTotal records      : {total_records}")
print(f"Anomalies detected : {total_anomalies}")
print(f"Anomaly percentage : {anomaly_percentage:.2f}%")

print("\nSeverity distribution:")

for severity in ["NORMAL", "MODERATE", "HIGH", "CRITICAL"]:
    count = int(
        severity_counts.get(severity, 0)
    )

    print(
        f" {severity:<10}: {count}"
    )


# ============================================================
# TOP ANOMALOUS DAYS
# ============================================================

print("\nTop 10 anomalous dates:")

top_anomalies = (
    df[
        df["Anomaly Flag"] == 1
    ]
    .sort_values(
        "Anomaly Score",
        ascending=False
    )
    [
        [
            "Date",
            "Total System Load",
            "Net Intake Pressure",
            "Anomaly Score",
            "Anomaly Severity"
        ]
    ]
    .head(10)
)

print(top_anomalies.to_string(index=False))


print("\n" + "=" * 65)
print("ANOMALY DETECTION COMPLETED")
print("=" * 65)

print(f"\nModel   : {MODEL_FILE}")
print(f"Results : {OUTPUT_FILE}")
print(f"Summary : {REPORT_FILE}")