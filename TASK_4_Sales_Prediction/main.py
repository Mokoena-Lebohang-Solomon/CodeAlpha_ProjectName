import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==============================================================
#                    SALES PREDICTION
# ==============================================================

DATA_FOLDER = "data"
OUTPUT_FOLDER = "outputs"

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


def print_header():

    print("\n")
    print("=" * 70)
    print("|                    SALES PREDICTION                         |")
    print("=" * 70)
    print("|  Advertising Analysis | Regression | Forecasting | Insights |")
    print("=" * 70)
    print()


def print_section(title):

    print("\n" + "-" * 70)
    print(f"| {title}")
    print("-" * 70)


def load_data():

    print_section(
        "LOADING DATA"
    )

    file_path = os.path.join(
        DATA_FOLDER,
        "Advertising.csv"
    )

    if not os.path.exists(file_path):

        print("| ERROR: Advertising.csv not found.")

        return None

    data = pd.read_csv(
        file_path
    )

    # Remove unnecessary index column
    if "Unnamed: 0" in data.columns:

        data = data.drop(
            columns=["Unnamed: 0"]
        )

    print("| Dataset loaded successfully.")
    print(f"| Rows    : {len(data):,}")
    print(f"| Columns : {len(data.columns)}")

    return data


def clean_data(data):

    print_section(
        "DATA CLEANING"
    )

    data.columns = (
        data.columns
        .str.strip()
    )

    numeric_columns = [
        "TV",
        "Radio",
        "Newspaper",
        "Sales"
    ]

    for column in numeric_columns:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    before = len(data)

    data = data.dropna()

    after = len(data)

    print(
        f"| Rows before cleaning : {before}"
    )

    print(
        f"| Rows after cleaning  : {after}"
    )

    return data


def analyse_data(data):

    print_section(
        "ADVERTISING ANALYSIS"
    )

    correlations = (
        data
        .corr(numeric_only=True)
        ["Sales"]
        .sort_values(
            ascending=False
        )
    )

    print(
        "| Correlation with Sales"
    )

    print(
        "|-----------------------"
    )

    for column, value in correlations.items():

        print(
            f"| {column:<12} : {value:.3f}"
        )

    return correlations


def create_visualisations(data):

    print_section(
        "CREATING VISUALISATIONS"
    )

    sns.set_theme(
        style="whitegrid"
    )

    channels = [
        "TV",
        "Radio",
        "Newspaper"
    ]

    for channel in channels:

        plt.figure(
            figsize=(8, 6)
        )

        sns.regplot(
            data=data,
            x=channel,
            y="Sales",
            scatter_kws={
                "alpha": 0.65
            }
        )

        plt.title(
            f"{channel} Advertising vs Sales"
        )

        plt.xlabel(
            f"{channel} Advertising Spend"
        )

        plt.ylabel(
            "Sales"
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                OUTPUT_FOLDER,
                f"{channel.lower()}_sales.png"
            )
        )

        plt.close()

    # Correlation heatmap

    plt.figure(
        figsize=(8, 6)
    )

    sns.heatmap(
        data.corr(
            numeric_only=True
        ),
        annot=True,
        fmt=".2f"
    )

    plt.title(
        "Advertising and Sales Correlation"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            "correlation_heatmap.png"
        )
    )

    plt.close()

    print(
        "| Visualisations created successfully."
    )


def train_model(data):

    print_section(
        "TRAINING SALES PREDICTION MODEL"
    )

    features = [
        "TV",
        "Radio",
        "Newspaper"
    ]

    X = data[features]

    y = data["Sales"]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )
    )

    model = LinearRegression()

    print(
        "| Training Linear Regression..."
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    return (
        model,
        X_test,
        y_test,
        predictions
    )


def evaluate_model(
    model,
    X_test,
    y_test,
    predictions
):

    print_section(
        "MODEL EVALUATION"
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    print(
        "| Model Coefficients"
    )

    print(
        "|-------------------"
    )

    for feature, coefficient in zip(
        X_test.columns,
        model.coef_
    ):

        print(
            f"| {feature:<12} : {coefficient:.4f}"
        )

    print(
        f"\n| Intercept : {model.intercept_:.4f}"
    )

    print(
        "\n| Performance"
    )

    print(
        "|-------------------"
    )

    print(
        f"| MAE  : {mae:.3f}"
    )

    print(
        f"| RMSE : {rmse:.3f}"
    )

    print(
        f"| R²   : {r2:.3f}"
    )

    return mae, rmse, r2


def create_prediction_chart(
    y_test,
    predictions
):

    print_section(
        "CREATING PREDICTION CHART"
    )

    plt.figure(
        figsize=(8, 6)
    )

    sns.scatterplot(
        x=y_test,
        y=predictions
    )

    minimum = min(
        y_test.min(),
        predictions.min()
    )

    maximum = max(
        y_test.max(),
        predictions.max()
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--"
    )

    plt.xlabel(
        "Actual Sales"
    )

    plt.ylabel(
        "Predicted Sales"
    )

    plt.title(
        "Actual vs Predicted Sales"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            "actual_vs_predicted_sales.png"
        )
    )

    plt.close()

    print(
        "| Prediction chart created."
    )


def save_predictions(
    X_test,
    y_test,
    predictions
):

    results = X_test.copy()

    results["Actual_Sales"] = y_test

    results["Predicted_Sales"] = predictions

    results["Absolute_Error"] = (
        results["Actual_Sales"]
        -
        results["Predicted_Sales"]
    ).abs()

    results.to_csv(
        os.path.join(
            OUTPUT_FOLDER,
            "sales_predictions.csv"
        ),
        index=False
    )

    print(
        "| Prediction results saved."
    )


def main():

    print_header()

    data = load_data()

    if data is None:
        return

    data = clean_data(
        data
    )

    analyse_data(
        data
    )

    create_visualisations(
        data
    )

    (
        model,
        X_test,
        y_test,
        predictions
    ) = train_model(
        data
    )

    evaluate_model(
        model,
        X_test,
        y_test,
        predictions
    )

    create_prediction_chart(
        y_test,
        predictions
    )

    save_predictions(
        X_test,
        y_test,
        predictions
    )

    print_section(
        "PROJECT COMPLETED"
    )

    print(
        "| Sales prediction completed successfully."
    )

    print(
        "| Results saved inside the 'outputs' folder."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()