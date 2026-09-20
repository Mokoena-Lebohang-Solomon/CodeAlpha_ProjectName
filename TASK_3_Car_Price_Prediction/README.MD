# Task 3 – Car Price Prediction with Machine Learning

## Objective
Build a regression model that predicts used-car selling prices from vehicle characteristics.

## Dataset
`car data.csv`

The supplied dataset contains:
- Car name
- Year
- Present price
- Selling price
- Driven kilometres
- Fuel type
- Selling type
- Transmission
- Owner count

The dataset does not contain a separate numerical "brand goodwill" feature, so the project derives `Car_Brand` from `Car_Name`.

## Model
A Random Forest Regressor is used because it can model non-linear relationships between vehicle characteristics and price.

## Workflow
1. Load and clean data.
2. Remove duplicate records.
3. Engineer `Car_Age`.
4. Extract `Car_Brand`.
5. One-hot encode categorical variables.
6. Split data into training and testing sets.
7. Train the regression model.
8. Evaluate with MAE, RMSE and R².
9. Plot actual versus predicted prices.

## Run
```bash
pip install -r requirements.txt
python main.py
```

Outputs are saved in `outputs/`.
