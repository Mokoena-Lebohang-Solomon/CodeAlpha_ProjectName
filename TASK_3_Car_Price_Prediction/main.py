import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==============================================================
#                  CAR PRICE PREDICTION
# ==============================================================

DATA_FOLDER = "data"
OUTPUT_FOLDER = "outputs"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def print_header():

    print("\n")
    print("=" * 70)
    print("|                 CAR PRICE PREDICTION                         |")
    print("=" * 70)
    print("|  Data Cleaning | Feature Engineering | Machine Learning    |")
    print("|  Regression    | Evaluation            | Prediction          |")
    print("=" * 70)
    print()


def print_section(title):

    print("\n" + "-" * 70)
    print(f"| {title}")
    print("-" * 70)


def load_data():

    print_section("LOADING DATA")

    file_path = os.path.join(
        DATA_FOLDER,
        "car data.csv"
    )

    if not os.path.exists(file_path):

        print("| ERROR: Dataset not found.")

        return None

    data = pd.read_csv(file_path)

    print("| Dataset loaded successfully.")
    print(f"| Rows    : {len(data):,}")
    print(f"| Columns : {len(data.columns)}")

    return data


def clean_and_engineer(data):

    print_section(
        "DATA CLEANING & FEATURE ENGINEERING"
    )

    data.columns = data.columns.str.strip()

    data = data.drop_duplicates()

    # Create vehicle age
    current_year = data["Year"].max()

    data["Car_Age"] = (
        current_year - data["Year"]
    )

    # Extract brand from car name
    data["Car_Brand"] = (
        data["Car_Name"]
        .str.split()
        .str[0]
        .str.title()
    )

    numeric_columns = [
        "Selling_Price",
        "Present_Price",
        "Driven_kms",
        "Owner",
        "Car_Age"
    ]

    for column in numeric_columns:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    data = data.dropna()

    print("| Duplicate records removed.")
    print("| Car_Age feature created.")
    print("| Car_Brand feature created.")
    print(f"| Final records : {len(data):,}")

    return data


def exploratory_analysis(data):

    print_section("EXPLORATORY DATA ANALYSIS")

    print("\n| Price Statistics")
    print("|-----------------------------")

    print(
        data[
            [
                "Selling_Price",
                "Present_Price",
                "Driven_kms"
            ]
        ].describe()
    )

    sns.set_theme(style="whitegrid")

    # Present price vs selling price

    plt.figure(figsize=(9, 6))

    sns.scatterplot(
        data=data,
        x="Present_Price",
        y="Selling_Price",
        hue="Fuel_Type"
    )

    plt.title(
        "Present Price vs Selling Price"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            "present_vs_selling_price.png"
        )
    )

    plt.close()

    # Mileage vs price

    plt.figure(figsize=(9, 6))

    sns.scatterplot(
        data=data,
        x="Driven_kms",
        y="Selling_Price"
    )

    plt.title(
        "Mileage vs Selling Price"
    )

    plt.xlabel("Driven Kilometres")

    plt.ylabel("Selling Price")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            "mileage_vs_price.png"
        )
    )

    plt.close()

    # Fuel type

    plt.figure(figsize=(8, 6))

    sns.boxplot(
        data=data,
        x="Fuel_Type",
        y="Selling_Price"
    )

    plt.title(
        "Selling Price by Fuel Type"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            "price_by_fuel_type.png"
        )
    )

    plt.close()

    print("| Visualisations created successfully.")


def train_model(data):

    print_section(
        "TRAINING MACHINE LEARNING MODEL"
    )

    features = [
        "Present_Price",
        "Driven_kms",
        "Car_Age",
        "Owner",
        "Fuel_Type",
        "Selling_type",
        "Transmission",
        "Car_Brand"
    ]

    target = "Selling_Price"

    X = data[features]

    y = data[target]

    categorical_features = [
        "Fuel_Type",
        "Selling_type",
        "Transmission",
        "Car_Brand"
    ]

    numerical_features = [
        "Present_Price",
        "Driven_kms",
        "Car_Age",
        "Owner"
    ]

    preprocessor = ColumnTransformer(
        transformers=[

            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            ),

            (
                "numerical",
                "passthrough",
                numerical_features
            )
        ]
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),

            (
                "model",
                model
            )
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print("| Training model...")

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_test
    )

    return (
        pipeline,
        X_test,
        y_test,
        predictions
    )


def evaluate_model(
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

    print("| Evaluation Metric | Result")
    print("|-------------------|--------")
    print(f"| MAE               | {mae:.3f}")
    print(f"| RMSE              | {rmse:.3f}")
    print(f"| R²                | {r2:.3f}")

    return mae, rmse, r2


def create_prediction_chart(
    y_test,
    predictions
):

    print_section(
        "CREATING PREDICTION VISUALISATION"
    )

    plt.figure(figsize=(8, 6))

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
        "Actual Selling Price"
    )

    plt.ylabel(
        "Predicted Selling Price"
    )

    plt.title(
        "Actual vs Predicted Car Prices"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            "actual_vs_predicted.png"
        )
    )

    plt.close()

    print("| Prediction chart created.")


def save_predictions(
    X_test,
    y_test,
    predictions
):

    results = X_test.copy()

    results["Actual_Price"] = y_test

    results["Predicted_Price"] = predictions

    results["Absolute_Error"] = (
        results["Actual_Price"]
        -
        results["Predicted_Price"]
    ).abs()

    results.to_csv(
        os.path.join(
            OUTPUT_FOLDER,
            "car_price_predictions.csv"
        ),
        index=False
    )

    print("| Prediction results saved.")


def main():

    print_header()

    data = load_data()

    if data is None:
        return

    data = clean_and_engineer(
        data
    )

    exploratory_analysis(
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

    print("| Car price prediction model completed.")
    print("| Results saved inside the 'outputs' folder.")
    print("=" * 70)


if __name__ == "__main__":
    main()