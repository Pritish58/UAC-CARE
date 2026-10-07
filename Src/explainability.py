import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from pathlib import Path


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
    / "best_load_prediction_model.pkl"
)

REPORT_DIR = (
    BASE_DIR
    / "reports"
    / "figures"
)

SHAP_VALUES_FILE = (
    BASE_DIR
    / "reports"
    / "shap_values.csv"
)

FEATURE_IMPORTANCE_FILE = (
    BASE_DIR
    / "reports"
    / "shap_feature_importance.csv"
)


REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

df["Date"] = pd.to_datetime(df["Date"])


# ============================================================
# TARGET
# ============================================================

TARGET = "Next Day System Load"


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(MODEL_FILE)


print("\n" + "=" * 65)
print("UAC CARE ANALYTICS - SHAP EXPLAINABILITY")
print("=" * 65)

print("\nModel loaded successfully.")


# ============================================================
# PREPARE FEATURES
# ============================================================

exclude_columns = [
    TARGET,
    "Date"
]

X = df.drop(
    columns=exclude_columns,
    errors="ignore"
)


# Keep numeric features only
X = X.select_dtypes(
    include=[np.number]
)


# Remove infinite values
X = X.replace(
    [np.inf, -np.inf],
    np.nan
)


# Remove rows with missing values
X = X.dropna()


# ============================================================
# IMPORTANT:
# MATCH MODEL FEATURES
# ============================================================

if hasattr(model, "feature_names_in_"):

    model_features = list(
        model.feature_names_in_
    )

    available_model_features = [
        col
        for col in model_features
        if col in X.columns
    ]

    X = X[
        available_model_features
    ]


print(
    "\nNumber of observations:",
    len(X)
)

print(
    "Number of features:",
    X.shape[1]
)


# ============================================================
# SAMPLE DATA FOR SHAP
# ============================================================

# Avoid unnecessarily large SHAP calculations

sample_size = min(
    300,
    len(X)
)

X_sample = X.sample(
    sample_size,
    random_state=42
)


# ============================================================
# SHAP EXPLAINER
# ============================================================

print("\nCalculating SHAP values...")

try:

    explainer = shap.TreeExplainer(
        model
    )

except Exception:

    print(
        "\nTreeExplainer could not be used."
    )

    print(
        "Using SHAP Explainer instead..."
    )

    explainer = shap.Explainer(
        model,
        X_sample
    )


shap_values = explainer(
    X_sample
)


# ============================================================
# HANDLE SHAP OUTPUT
# ============================================================

values = shap_values.values

if values.ndim == 3:

    values = values[:, :, 0]


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

mean_abs_shap = np.abs(
    values
).mean(axis=0)


feature_importance = pd.DataFrame({
    "Feature": X_sample.columns,
    "Mean Absolute SHAP": mean_abs_shap
})


feature_importance = (
    feature_importance
    .sort_values(
        "Mean Absolute SHAP",
        ascending=False
    )
)


# ============================================================
# SAVE FEATURE IMPORTANCE
# ============================================================

feature_importance.to_csv(
    FEATURE_IMPORTANCE_FILE,
    index=False
)


# ============================================================
# SAVE SHAP VALUES
# ============================================================

shap_values_df = pd.DataFrame(
    values,
    columns=X_sample.columns
)

shap_values_df.to_csv(
    SHAP_VALUES_FILE,
    index=False
)


# ============================================================
# SHAP SUMMARY PLOT
# ============================================================

print("\nCreating SHAP summary plot...")

plt.figure()

shap.summary_plot(
    values,
    X_sample,
    show=False
)

plt.tight_layout()

plt.savefig(
    REPORT_DIR
    / "shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SHAP BAR PLOT
# ============================================================

print(
    "Creating SHAP feature importance plot..."
)

plt.figure()

shap.summary_plot(
    values,
    X_sample,
    plot_type="bar",
    show=False
)

plt.tight_layout()

plt.savefig(
    REPORT_DIR
    / "shap_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# DISPLAY TOP FEATURES
# ============================================================

print("\n" + "=" * 65)
print("TOP FEATURES IN MODEL EXPLANATION")
print("=" * 65)

print(
    feature_importance
    .head(15)
    .to_string(index=False)
)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 65)
print("SHAP EXPLAINABILITY COMPLETED")
print("=" * 65)

print(
    f"\nSHAP values       : {SHAP_VALUES_FILE}"
)

print(
    f"Feature importance: {FEATURE_IMPORTANCE_FILE}"
)

print(
    f"Summary plot      : "
    f"{REPORT_DIR / 'shap_summary.png'}"
)

print(
    f"Importance plot   : "
    f"{REPORT_DIR / 'shap_feature_importance.png'}"
)