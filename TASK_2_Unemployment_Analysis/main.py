import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==============================================================
#                    UNEMPLOYMENT ANALYSIS
# ==============================================================

DATA_FOLDER = "data"
OUTPUT_FOLDER = "outputs"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def print_header():
    print("\n")
    print("=" * 70)
    print("|              UNEMPLOYMENT ANALYSIS WITH PYTHON              |")
    print("=" * 70)
    print("|  Data Cleaning | COVID-19 Analysis | Trends | Visualisation |")
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
        "Unemployment in India.csv"
    )

    if not os.path.exists(file_path):
        print("| ERROR: Dataset not found.")
        print("| Please place the CSV file inside the data folder.")
        return None

    data = pd.read_csv(file_path)

    print(f"| Dataset loaded successfully")
    print(f"| Number of rows    : {len(data):,}")
    print(f"| Number of columns : {len(data.columns)}")

    return data


def clean_data(data):
    print_section("DATA CLEANING")

    # Remove spaces from column names
    data.columns = data.columns.str.strip()

    # Convert date
    data["Date"] = pd.to_datetime(
        data["Date"],
        dayfirst=True,
        errors="coerce"
    )

    # Convert numerical columns
    numeric_columns = [
        "Estimated Unemployment Rate (%)",
        "Estimated Employed",
        "Estimated Labour Participation Rate (%)"
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    before = len(data)

    data = data.dropna(
        subset=[
            "Date",
            "Estimated Unemployment Rate (%)"
        ]
    )

    after = len(data)

    print(f"| Rows before cleaning : {before:,}")
    print(f"| Rows after cleaning  : {after:,}")
    print(f"| Rows removed         : {before - after:,}")

    return data


def analyse_unemployment(data):
    print_section("GENERAL ANALYSIS")

    average_rate = data[
        "Estimated Unemployment Rate (%)"
    ].mean()

    highest_rate = data[
        "Estimated Unemployment Rate (%)"
    ].max()

    lowest_rate = data[
        "Estimated Unemployment Rate (%)"
    ].min()

    print(f"| Average unemployment rate : {average_rate:.2f}%")
    print(f"| Highest unemployment rate : {highest_rate:.2f}%")
    print(f"| Lowest unemployment rate  : {lowest_rate:.2f}%")


def covid_analysis(data):
    print_section("COVID-19 IMPACT ANALYSIS")

    pre_covid = data[
        data["Date"] < pd.Timestamp("2020-03-01")
    ]

    covid_period = data[
        data["Date"] >= pd.Timestamp("2020-03-01")
    ]

    pre_average = pre_covid[
        "Estimated Unemployment Rate (%)"
    ].mean()

    covid_average = covid_period[
        "Estimated Unemployment Rate (%)"
    ].mean()

    difference = covid_average - pre_average

    percentage_change = (
        difference / pre_average
    ) * 100

    print("| Period                      | Average Rate")
    print("|-----------------------------|-------------")
    print(f"| Pre-COVID                   | {pre_average:.2f}%")
    print(f"| COVID period                | {covid_average:.2f}%")
    print(f"| Change                      | {difference:+.2f}%")
    print(f"| Percentage change           | {percentage_change:+.2f}%")

    return pre_average, covid_average


def regional_analysis(data):
    print_section("REGIONAL ANALYSIS")

    regional = (
        data.groupby("Region")[
            "Estimated Unemployment Rate (%)"
        ]
        .mean()
        .sort_values(ascending=False)
    )

    print("| Region                         | Average Rate")
    print("|--------------------------------|-------------")

    for region, rate in regional.head(10).items():
        print(f"| {region:<30} | {rate:.2f}%")

    regional.to_csv(
        os.path.join(
            OUTPUT_FOLDER,
            "regional_summary.csv"
        )
    )


def create_visualisations(data):
    print_section("CREATING VISUALISATIONS")

    sns.set_theme(style="whitegrid")

    # ----------------------------------------------------------
    # 1. Overall unemployment trend
    # ----------------------------------------------------------

    monthly = (
        data.groupby("Date")[
            "Estimated Unemployment Rate (%)"
        ]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(12, 6))

    sns.lineplot(
        data=monthly,
        x="Date",
        y="Estimated Unemployment Rate (%)",
        marker="o"
    )

    plt.axvline(
        pd.Timestamp("2020-03-01"),
        linestyle="--",
        label="COVID-19 period"
    )

    plt.title(
        "India Unemployment Rate Over Time"
    )

    plt.xlabel("Date")
    plt.ylabel("Unemployment Rate (%)")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            "unemployment_trend.png"
        )
    )

    plt.close()

    # ----------------------------------------------------------
    # 2. Regional unemployment
    # ----------------------------------------------------------

    regional = (
        data.groupby("Region")[
            "Estimated Unemployment Rate (%)"
        ]
        .mean()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )

    plt.figure(figsize=(12, 7))

    sns.barplot(
        data=regional,
        x="Estimated Unemployment Rate (%)",
        y="Region"
    )

    plt.title(
        "Average Unemployment Rate by Region"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            "regional_unemployment.png"
        )
    )

    plt.close()

    # ----------------------------------------------------------
    # 3. Rural vs Urban
    # ----------------------------------------------------------

    area_data = (
        data.groupby(
            ["Date", "Area"]
        )[
            "Estimated Unemployment Rate (%)"
        ]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(12, 6))

    sns.lineplot(
        data=area_data,
        x="Date",
        y="Estimated Unemployment Rate (%)",
        hue="Area",
        marker="o"
    )

    plt.title(
        "Rural vs Urban Unemployment"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            "rural_vs_urban.png"
        )
    )

    plt.close()

    print("| Visualisations created successfully.")


def main():

    print_header()

    data = load_data()

    if data is None:
        return

    data = clean_data(data)

    analyse_unemployment(data)

    covid_analysis(data)

    regional_analysis(data)

    create_visualisations(data)

    print_section("PROJECT COMPLETED")

    print("| Analysis completed successfully.")
    print("| Results saved inside the 'outputs' folder.")
    print("=" * 70)


if __name__ == "__main__":
    main()