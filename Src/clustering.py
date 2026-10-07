import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

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
    / "operational_clustering_model.pkl"
)

OUTPUT_FILE = (
    BASE_DIR
    / "Data"
    / "Processed"
    / "uac_operational_clusters.csv"
)

REPORT_FILE = (
    BASE_DIR
    / "reports"
    / "cluster_summary.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

df["Date"] = pd.to_datetime(df["Date"])


# ============================================================
# FEATURES
# ============================================================

cluster_features = [
    "Total System Load",
    "Net Intake Pressure",
    "Children in CBP custody",
    "Children transferred out of CBP custody",
    "Children in HHS Care",
    "Children discharged from HHS Care",
    "7-Day Net Intake",
    "Load Volatility 7D"
]


available_features = [
    col for col in cluster_features
    if col in df.columns
]


print("\n" + "=" * 65)
print("UAC CARE ANALYTICS - OPERATIONAL CLUSTERING")
print("=" * 65)

print("\nFeatures used:")

for feature in available_features:
    print(f" - {feature}")


# ============================================================
# PREPARE DATA
# ============================================================

X = df[available_features].copy()

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(
    X.median()
)


# ============================================================
# STANDARDIZATION
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ============================================================
# FIND BEST NUMBER OF CLUSTERS
# ============================================================

print("\nTesting cluster sizes...")

cluster_scores = []

for k in range(2, 7):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    labels = model.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        labels
    )

    cluster_scores.append(
        {
            "Clusters": k,
            "Silhouette Score": score
        }
    )

    print(
        f"K = {k} | "
        f"Silhouette Score = {score:.4f}"
    )


scores_df = pd.DataFrame(
    cluster_scores
)


best_k = int(
    scores_df.loc[
        scores_df["Silhouette Score"].idxmax(),
        "Clusters"
    ]
)

best_score = float(
    scores_df.loc[
        scores_df["Silhouette Score"].idxmax(),
        "Silhouette Score"
    ]
)


print(
    f"\nBest number of clusters: {best_k}"
)

print(
    f"Best silhouette score   : {best_score:.4f}"
)


# ============================================================
# FINAL K-MEANS MODEL
# ============================================================

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=20
)

df["Operational Cluster"] = (
    kmeans.fit_predict(X_scaled)
)


# ============================================================
# CLUSTER SUMMARY
# ============================================================

summary_columns = [
    "Total System Load",
    "Net Intake Pressure",
    "Children in CBP custody",
    "Children transferred out of CBP custody",
    "Children in HHS Care",
    "Children discharged from HHS Care",
    "7-Day Net Intake",
    "Load Volatility 7D"
]

summary_columns = [
    col
    for col in summary_columns
    if col in df.columns
]


cluster_summary = (
    df.groupby("Operational Cluster")[
        summary_columns
    ]
    .mean()
    .round(2)
)


cluster_summary["Observations"] = (
    df["Operational Cluster"]
    .value_counts()
    .sort_index()
)


cluster_summary = cluster_summary.reset_index()


# ============================================================
# AUTOMATIC CLUSTER DESCRIPTION
# ============================================================

overall_load = df["Total System Load"].mean()
overall_pressure = df["Net Intake Pressure"].mean()


def describe_cluster(row):

    load = row["Total System Load"]
    pressure = row["Net Intake Pressure"]

    if (
        load >= overall_load
        and pressure >= overall_pressure
    ):
        return "High Load / High Intake Pressure"

    elif (
        load >= overall_load
        and pressure < overall_pressure
    ):
        return "High Load / Lower Intake Pressure"

    elif (
        load < overall_load
        and pressure >= overall_pressure
    ):
        return "Lower Load / High Intake Pressure"

    else:
        return "Stable / Lower Load"


cluster_summary["Operational Pattern"] = (
    cluster_summary.apply(
        describe_cluster,
        axis=1
    )
)


# ============================================================
# MERGE PATTERN INTO MAIN DATA
# ============================================================

pattern_map = dict(
    zip(
        cluster_summary["Operational Cluster"],
        cluster_summary["Operational Pattern"]
    )
)


df["Operational Pattern"] = (
    df["Operational Cluster"]
    .map(pattern_map)
)


# ============================================================
# SAVE MODEL
# ============================================================

model_package = {
    "model": kmeans,
    "scaler": scaler,
    "features": available_features,
    "best_k": best_k
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


cluster_summary.to_csv(
    REPORT_FILE,
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 65)
print("OPERATIONAL CLUSTER SUMMARY")
print("=" * 65)

print(
    cluster_summary.to_string(
        index=False
    )
)


print("\n" + "=" * 65)
print("OPERATIONAL CLUSTERING COMPLETED")
print("=" * 65)

print(f"\nModel   : {MODEL_FILE}")
print(f"Results : {OUTPUT_FILE}")
print(f"Summary : {REPORT_FILE}")