import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import joblib


# ============================================================
# UAC CARE ANALYTICS
# OPERATIONAL STRESS / RISK CLASSIFICATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "Data"
    / "Processed"
    / "uac_features.csv"
)

MODEL_DIR = BASE_DIR / "Models"

RESULT_FILE = (
    BASE_DIR
    / "reports"
    / "stress_model_results.csv"
)

CONFUSION_FILE = (
    BASE_DIR
    / "reports"
    / "stress_confusion_matrix.csv"
)


def create_stress_labels(df):

    """
    Create transparent stress labels using historical
    system-load and net-intake distributions.

    Stress score is based on:
        1. Total System Load
        2. Net Intake Pressure
        3. 7-Day Net Intake
    """

    # --------------------------------------------------------
    # Percentile thresholds
    # --------------------------------------------------------

    load_q75 = df["Total System Load"].quantile(0.75)
    load_q90 = df["Total System Load"].quantile(0.90)

    pressure_q75 = df["Net Intake Pressure"].quantile(0.75)
    pressure_q90 = df["Net Intake Pressure"].quantile(0.90)

    backlog_q75 = df["7-Day Net Intake"].quantile(0.75)
    backlog_q90 = df["7-Day Net Intake"].quantile(0.90)

    # --------------------------------------------------------
    # Stress components
    # --------------------------------------------------------

    df["Load Stress Component"] = pd.cut(
        df["Total System Load"],
        bins=[
            -np.inf,
            load_q75,
            load_q90,
            np.inf
        ],
        labels=[0, 1, 2]
    ).astype(float)

    df["Pressure Stress Component"] = pd.cut(
        df["Net Intake Pressure"],
        bins=[
            -np.inf,
            pressure_q75,
            pressure_q90,
            np.inf
        ],
        labels=[0, 1, 2]
    ).astype(float)

    df["Backlog Stress Component"] = pd.cut(
        df["7-Day Net Intake"],
        bins=[
            -np.inf,
            backlog_q75,
            backlog_q90,
            np.inf
        ],
        labels=[0, 1, 2]
    ).astype(float)

    # --------------------------------------------------------
    # Combined stress score
    # --------------------------------------------------------

    df["Stress Score"] = (
        df["Load Stress Component"]
        + df["Pressure Stress Component"]
        + df["Backlog Stress Component"]
    )

    # --------------------------------------------------------
    # Four risk categories
    # --------------------------------------------------------

    df["Stress Level"] = pd.cut(
        df["Stress Score"],
        bins=[
            -np.inf,
            1,
            3,
            5,
            np.inf
        ],
        labels=[
            "LOW",
            "MODERATE",
            "HIGH",
            "CRITICAL"
        ]
    )

    return df


def train_stress_models():

    print("\n" + "=" * 70)
    print("UAC CARE ANALYTICS - STRESS/RISK CLASSIFICATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values(
        "Date"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Create labels
    # --------------------------------------------------------

    df = create_stress_labels(df)

    print("\nStress distribution:")

    print(
        df["Stress Level"]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    FEATURES = [

        "Children apprehended and placed in CBP custody*",
        "Children in CBP custody",
        "Children transferred out of CBP custody",
        "Children in HHS Care",
        "Children discharged from HHS Care",

        "Total System Load",
        "Net Intake Pressure",

        "Discharge Offset Ratio",
        "Transfer to CBP Ratio",

        "Daily Load Change",
        "Daily Load Growth %",

        "Load Lag 1",
        "Load Lag 3",
        "Load Lag 7",
        "Load Lag 14",

        "Net Intake Lag 1",
        "Net Intake Lag 3",
        "Net Intake Lag 7",
        "Net Intake Lag 14",

        "Load Rolling Mean 7",
        "Load Rolling Mean 14",
        "Load Rolling Mean 30",

        "Load Rolling Std 7",
        "Load Rolling Std 14",
        "Load Rolling Std 30",

        "Net Intake Rolling Mean 7",
        "Net Intake Rolling Mean 14",
        "Net Intake Rolling Mean 30",

        "7-Day Net Intake",
        "14-Day Net Intake",

        "Load Volatility 7D",
        "Load Volatility 14D",

        "Day of Week",
        "Week of Year",
        "Month",
        "Quarter",
        "Year",
        "Is Weekend"
    ]

    TARGET = "Stress Level"

    model_df = df[
        FEATURES + [TARGET]
    ].dropna().copy()

    X = model_df[FEATURES]
    y = model_df[TARGET]

    # --------------------------------------------------------
    # Chronological split
    # --------------------------------------------------------

    split_index = int(
        len(model_df) * 0.80
    )

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print("\nTraining rows:", len(X_train))
    print("Testing rows:", len(X_test))

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    models = {

        "Logistic Regression": Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced"
                )
            )
        ]),

        "Random Forest": Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=12,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]),

        "Gradient Boosting": Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "model",
                GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=42
                )
            )
        ])
    }

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    results = []

    trained_models = {}

    predictions = {}

    for name, model in models.items():

        print("\nTraining:", name)

        model.fit(
            X_train,
            y_train
        )

        pred = model.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            pred
        )

        precision = precision_score(
            y_test,
            pred,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            pred,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            pred,
            average="weighted",
            zero_division=0
        )

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        })

        trained_models[name] = model

        predictions[name] = pred

        print(
            f"Accuracy : {accuracy:.4f}"
        )

        print(
            f"Precision: {precision:.4f}"
        )

        print(
            f"Recall   : {recall:.4f}"
        )

        print(
            f"F1 Score : {f1:.4f}"
        )

    # --------------------------------------------------------
    # Model comparison
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        "F1",
        ascending=False
    ).reset_index(drop=True)

    print("\n" + "=" * 70)
    print("STRESS MODEL COMPARISON")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Best model
    # --------------------------------------------------------

    best_model_name = (
        results_df.iloc[0]["Model"]
    )

    best_model = trained_models[
        best_model_name
    ]

    best_prediction = predictions[
        best_model_name
    ]

    print(
        "\nBest Stress Model:",
        best_model_name
    )

    # --------------------------------------------------------
    # Detailed classification report
    # --------------------------------------------------------

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            best_prediction,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    labels = [
        "LOW",
        "MODERATE",
        "HIGH",
        "CRITICAL"
    ]

    cm = confusion_matrix(
        y_test,
        best_prediction,
        labels=labels
    )

    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels
    )

    print("\nConfusion Matrix:")
    print(cm_df)

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    RESULT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        RESULT_FILE,
        index=False
    )

    cm_df.to_csv(
        CONFUSION_FILE
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model_file = (
        MODEL_DIR
        / "best_stress_model.pkl"
    )

    joblib.dump(
        best_model,
        model_file
    )

    # --------------------------------------------------------
    # Save labelled dataset
    # --------------------------------------------------------

    labelled_file = (
        BASE_DIR
        / "Data"
        / "Processed"
        / "uac_stress_labels.csv"
    )

    df.to_csv(
        labelled_file,
        index=False
    )

    print("\nFiles saved:")

    print(
        "Model:",
        model_file
    )

    print(
        "Results:",
        RESULT_FILE
    )

    print(
        "Confusion Matrix:",
        CONFUSION_FILE
    )

    print(
        "Labels:",
        labelled_file
    )

    print("\n" + "=" * 70)
    print("STRESS CLASSIFICATION COMPLETED")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train_stress_models()