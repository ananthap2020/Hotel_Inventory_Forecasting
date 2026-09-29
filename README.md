# Hotel Inventory Forecasting & Optimization

## Project Overview

The Hotel Inventory Forecasting & Optimization project is a machine-learning-based system designed to forecast alcohol consumption in hotel bars and generate inventory replenishment recommendations.

The system analyzes historical consumption data for different bars, alcohol types, and brands. It uses time-series features and a Random Forest regression model to predict future demand.

The predicted demand is then converted into:

- Safety Stock
- Reorder Point
- 7-Day PAR Level
- Recommended Order Quantity

---

## Business Problem

Hotel bars need to maintain enough inventory to satisfy customer demand while avoiding unnecessary overstock.

Two major inventory problems are:

1. Stockouts
   - Products may become unavailable when demand is high.

2. Overstock
   - Excess inventory occupies storage space and working capital.

This project provides data-driven inventory recommendations based on historical consumption and forecast uncertainty.

---

## Objectives

The main objectives of the project are:

- Analyze historical hotel bar consumption.
- Forecast daily product demand.
- Compare machine-learning forecasting with a seasonal-naive baseline.
- Calculate safety stock.
- Calculate reorder points.
- Calculate 7-day PAR levels.
- Generate recommended order quantities.
- Provide actionable inventory recommendations for each bar and product.

---

## Dataset

The dataset contains:

- 6,575 records
- 366 dates
- 6 bars
- 5 alcohol types
- 16 brands
- 96 bar-product combinations

### Dataset Columns

- Date Time Served
- Bar Name
- Alcohol Type
- Brand Name
- Opening Balance (ml)
- Purchase (ml)
- Consumed (ml)
- Closing Balance (ml)

Total recorded consumption:

**1,968,681.66 ml**

---

## Technologies Used

### Programming Language

- Python

### Libraries

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

### Machine Learning

- Random Forest Regression

### Development Environment

- VS Code
- Python Virtual Environment

---

## Data Processing

The following preprocessing steps were performed:

1. Loaded the CSV dataset.
2. Converted the date/time column into a date field.
3. Aggregated consumption into daily demand.
4. Created a complete daily time series.
5. Filled missing daily demand with zero.
6. Created lag features.
7. Created rolling demand features.
8. Created calendar features.
9. Performed a chronological train-test split.

---

## Feature Engineering

The model uses:

### Lag Features

- 1-day lag
- 7-day lag
- 14-day lag
- 28-day lag

### Rolling Features

- 7-day rolling demand
- 14-day rolling demand

### Calendar Features

- Day of week
- Month
- Weekend indicator

Categorical features include:

- Bar Name
- Alcohol Type
- Brand Name

---

## Machine Learning Model

The primary model is:

### Random Forest Regressor

Configuration:

- Number of trees: 300
- Maximum depth: 15
- Minimum samples per leaf: 2
- Random state: 42

The model was evaluated using a chronological 80/20 split.

---

## Baseline Model

A 7-day seasonal-naive model was used as a baseline.

The baseline predicts demand using the corresponding day from the previous week.

This provides a simple reference point for determining whether the machine-learning model provides additional value.

---

## Model Results

| Metric | Random Forest | Seasonal Naive |
|---|---:|---:|
| MAE | 98.05 ml | 97.52 ml |
| RMSE | 146.22 ml | 203.95 ml |

The Random Forest produced a slightly higher MAE but a substantially lower RMSE.

This means the Random Forest reduced larger forecast errors, while the seasonal-naive baseline had slightly better average absolute error.

Therefore, the Random Forest should not be considered universally superior based only on these two metrics.

---

## Inventory Optimization

The forecasting results are converted into inventory planning values.

### Safety Stock

Safety stock protects against forecast uncertainty.

The implementation uses:

- 95% service level
- z-value = 1.65
- 1-day lead time

### Reorder Point

```text
Reorder Point =
Predicted Daily Demand × Lead Time
+ Safety Stock