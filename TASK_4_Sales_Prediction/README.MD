# Task 4 – Sales Prediction using Python

## Objective
Analyse advertising expenditure and predict sales using regression.

## Dataset
`Advertising.csv`

The supplied dataset contains:
- TV advertising spend
- Radio advertising spend
- Newspaper advertising spend
- Sales

It does not contain target segment or platform columns, so the model focuses on the advertising variables that are actually available.

## Workflow
1. Clean the CSV and remove its index column.
2. Explore relationships between advertising spend and sales.
3. Calculate correlations.
4. Train a Linear Regression model.
5. Evaluate using MAE, RMSE and R².
6. Compare actual and predicted sales.
7. Save prediction results and charts.

## Run
```bash
pip install -r requirements.txt
python main.py
```

## Business interpretation
The regression coefficients indicate the model's estimated association between each advertising channel and sales while the other included channels are held constant. These results should be treated as predictive associations, not proof that increasing one channel will automatically cause the same increase in sales.
