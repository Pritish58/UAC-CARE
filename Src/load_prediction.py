import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import joblib


# ============================================================
# UAC CARE ANALYTICS
# NEXT-DAY SYSTEM LOAD PREDICTION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "Data"
    / "Processed"
    / "uac_features.csv"
)

MODEL_DIR = BASE_DIR / "Models"
REPORT_DIR = BASE_DIR / "reports" / "figures"

RESULT_FILE = (
    BASE_DIR
    / "reports"
    / "load_model_results.csv"
)


def train_models():

    print("\n" + "=" * 70)
    print("UAC CARE ANALYTICS - LOAD PREDICTION")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load feature dataset
    # --------------------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values("Date").reset_index(drop=True)

    print("\nDataset loaded.")
    print("Initial shape:", df.shape)

    # --------------------------------------------------------
    # 2. Define target
    # --------------------------------------------------------

    TARGET = "Next Day System Load"

    # --------------------------------------------------------
    # 3. Select features
    # --------------------------------------------------------

    FEATURES = [

        # Current operational state
        "Children apprehended and placed in CBP custody*",
        "Children in CBP custody",
        "Children transferred out of CBP custody",
        "Children in HHS Care",
        "Children discharged from HHS Care",

        # Current derived metrics
        "Total System Load",
        "Net Intake Pressure",
        "Discharge Offset Ratio",
        "Transfer to CBP Ratio",

        # Recent changes
        "Daily Load Change",
        "Daily Load Growth %",

        # Lag features
        "Load Lag 1",
        "Load Lag 3",
        "Load Lag 7",
        "Load Lag 14",

        "Net Intake Lag 1",
        "Net Intake Lag 3",
        "Net Intake Lag 7",
        "Net Intake Lag 14",

        "HHS Load Lag 1",
        "HHS Load Lag 3",
        "HHS Load Lag 7",
        "HHS Load Lag 14",

        # Rolling trends
        "Load Rolling Mean 7",
        "Load Rolling Mean 14",
        "Load Rolling Mean 30",

        "Load Rolling Std 7",
        "Load Rolling Std 14",
        "Load Rolling Std 30",

        "Net Intake Rolling Mean 7",
        "Net Intake Rolling Mean 14",
        "Net Intake Rolling Mean 30",

        # Backlog / pressure
        "7-Day Net Intake",
        "14-Day Net Intake",
        "Backlog Indicator",

        # Volatility
        "Load Volatility 7D",
        "Load Volatility 14D",

        # Calendar
        "Day",
        "Day of Week",
        "Week of Year",
        "Month",
        "Quarter",
        "Year",
        "Is Weekend"
    ]

    # --------------------------------------------------------
    # 4. Check features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_features:

        print("\nMissing features:")

        for feature in missing_features:
            print(" -", feature)

        raise ValueError(
            "Some ML features are missing from the dataset."
        )

    # --------------------------------------------------------
    # 5. Keep required columns
    # --------------------------------------------------------

    model_df = df[
        ["Date"] + FEATURES + [TARGET]
    ].copy()

    # --------------------------------------------------------
    # 6. Remove rows unavailable because of lags/rolling
    # --------------------------------------------------------

    before = len(model_df)

    model_df = model_df.dropna()

    removed = before - len(model_df)

    print("\nRows removed because of lag/rolling NaN:", removed)

    print("Final ML dataset:", model_df.shape)

    # --------------------------------------------------------
    # 7. X and y
    # --------------------------------------------------------

    X = model_df[FEATURES]

    y = model_df[TARGET]

    # --------------------------------------------------------
    # 8. Chronological train/test split
    # --------------------------------------------------------

    split_index = int(len(model_df) * 0.80)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    dates_test = model_df["Date"].iloc[split_index:]

    print("\nTime-based split:")
    print("Training rows:", len(X_train))
    print("Testing rows :", len(X_test))

    print(
        "Training period:",
        model_df["Date"].iloc[0],
        "to",
        model_df["Date"].iloc[split_index - 1]
    )

    print(
        "Testing period:",
        dates_test.iloc[0],
        "to",
        dates_test.iloc[-1]
    )

    # --------------------------------------------------------
    # 9. Define models
    # --------------------------------------------------------

    models = {

        "Linear Regression":
            LinearRegression(),

        "Decision Tree":
            DecisionTreeRegressor(
                max_depth=8,
                random_state=42
            ),

        "Random Forest":
            RandomForestRegressor(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),

        "Extra Trees":
            ExtraTreesRegressor(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),

        "Gradient Boosting":
            GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=42
            )
    }

    # --------------------------------------------------------
    # 10. Train and evaluate
    # --------------------------------------------------------

    results = []

    trained_models = {}

    predictions = {}

    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE")
    print("=" * 70)

    for name, model in models.items():

        print(f"\nTraining: {name}")

        model.fit(
            X_train,
            y_train
        )

        prediction = model.predict(X_test)

        mae = mean_absolute_error(
            y_test,
            prediction
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                prediction
            )
        )

        r2 = r2_score(
            y_test,
            prediction
        )

        results.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

        trained_models[name] = model
        predictions[name] = prediction

        print(f"MAE  : {mae:.2f}")
        print(f"RMSE : {rmse:.2f}")
        print(f"R²   : {r2:.4f}")

    # --------------------------------------------------------
    # 11. Model comparison
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "RMSE"
    ).reset_index(drop=True)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 12. Select best model
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

    print("\nBest Model:")
    print(best_model_name)

    # --------------------------------------------------------
    # 13. Save results
    # --------------------------------------------------------

    RESULT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        RESULT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # 14. Save best model
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model_file = (
        MODEL_DIR
        / "best_load_prediction_model.pkl"
    )

    joblib.dump(
        best_model,
        model_file
    )

    # --------------------------------------------------------
    # 15. Actual vs predicted
    # --------------------------------------------------------

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    prediction_df = pd.DataFrame({
        "Date": dates_test.values,
        "Actual": y_test.values,
        "Predicted": best_prediction
    })

    prediction_df.to_csv(
        BASE_DIR
        / "reports"
        / "load_predictions.csv",
        index=False
    )

    plt.figure(figsize=(15, 6))

    plt.plot(
        prediction_df["Date"],
        prediction_df["Actual"],
        label="Actual"
    )

    plt.plot(
        prediction_df["Date"],
        prediction_df["Predicted"],
        label="Predicted"
    )

    plt.title(
        f"Actual vs Predicted Next-Day System Load\n"
        f"Best Model: {best_model_name}"
    )

    plt.xlabel("Date")
    plt.ylabel("System Load")

    plt.legend()

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR
        / "actual_vs_predicted_load.png",
        dpi=200
    )

    plt.close()

    # --------------------------------------------------------
    # 16. Feature importance
    # --------------------------------------------------------

    if hasattr(
        best_model,
        "feature_importances_"
    ):

        importance = pd.DataFrame({
            "Feature": FEATURES,
            "Importance": best_model.feature_importances_
        })

        importance = importance.sort_values(
            "Importance",
            ascending=False
        )

        importance.to_csv(
            BASE_DIR
            / "reports"
            / "load_feature_importance.csv",
            index=False
        )

        # Top 15 features
        top_features = importance.head(15)

        plt.figure(figsize=(10, 7))

        plt.barh(
            top_features["Feature"][::-1],
            top_features["Importance"][::-1]
        )

        plt.title(
            f"Top Features - {best_model_name}"
        )

        plt.xlabel("Importance")

        plt.tight_layout()

        plt.savefig(
            REPORT_DIR
            / "load_feature_importance.png",
            dpi=200
        )

        plt.close()

    # --------------------------------------------------------
    # 17. Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LOAD PREDICTION COMPLETED")
    print("=" * 70)

    print("\nBest model:", best_model_name)

    print(
        "\nResults saved to:",
        RESULT_FILE
    )

    print(
        "Model saved to:",
        model_file
    )

    print(
        "Prediction file saved to:",
        BASE_DIR / "reports" / "load_predictions.csv"
    )

    print(
        "Charts saved to:",
        REPORT_DIR
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    train_models()