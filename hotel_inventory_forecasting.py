# ============================================================
# HOTEL BAR INVENTORY FORECASTING PROJECT
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings("ignore")


# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

file_path = "Consumption Dataset - Google Sheets.csv"

df = pd.read_csv(file_path)

print("=" * 60)
print("HOTEL BAR INVENTORY FORECASTING PROJECT")
print("=" * 60)

print("\nDataset loaded successfully!")
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])


# ------------------------------------------------------------
# 2. DISPLAY FIRST 5 ROWS
# ------------------------------------------------------------

print("\nFirst 5 rows:")
print(df.head())


# ------------------------------------------------------------
# 3. DATASET INFORMATION
# ------------------------------------------------------------

print("\nDataset information:")
print(df.info())


# ------------------------------------------------------------
# 4. CHECK MISSING VALUES
# ------------------------------------------------------------

print("\nMissing values:")
print(df.isnull().sum())


# ------------------------------------------------------------
# 5. CHECK DUPLICATES
# ------------------------------------------------------------

print("\nDuplicate rows:")
print(df.duplicated().sum())


# ------------------------------------------------------------
# 6. CONVERT DATE/TIME
# ------------------------------------------------------------

df["Date Time Served"] = pd.to_datetime(
    df["Date Time Served"]
)

df["Date"] = df["Date Time Served"].dt.date
df["Date"] = pd.to_datetime(df["Date"])


# ------------------------------------------------------------
# 7. BASIC DATASET INFORMATION
# ------------------------------------------------------------

print("\nDate range:")
print(
    df["Date"].min(),
    "to",
    df["Date"].max()
)

print("\nNumber of bars:")
print(df["Bar Name"].nunique())

print("\nBars:")
print(df["Bar Name"].unique())

print("\nNumber of alcohol types:")
print(df["Alcohol Type"].nunique())

print("\nAlcohol types:")
print(df["Alcohol Type"].unique())

print("\nNumber of brands:")
print(df["Brand Name"].nunique())

print("\nBrands:")
print(df["Brand Name"].unique())


# ------------------------------------------------------------
# 8. TOTAL CONSUMPTION
# ------------------------------------------------------------

total_consumption = df["Consumed (ml)"].sum()

print("\nTotal consumption:")
print(f"{total_consumption:,.2f} ml")


# ------------------------------------------------------------
# 9. CONSUMPTION BY ALCOHOL TYPE
# ------------------------------------------------------------

consumption_by_type = (
    df.groupby("Alcohol Type")["Consumed (ml)"]
    .sum()
    .sort_values(ascending=False)
)

print("\nConsumption by alcohol type:")
print(consumption_by_type)


# ------------------------------------------------------------
# 10. CONSUMPTION BY BRAND
# ------------------------------------------------------------

consumption_by_brand = (
    df.groupby("Brand Name")["Consumed (ml)"]
    .sum()
    .sort_values(ascending=False)
)

print("\nConsumption by brand:")
print(consumption_by_brand)


# ------------------------------------------------------------
# 11. CONSUMPTION BY BAR
# ------------------------------------------------------------

consumption_by_bar = (
    df.groupby("Bar Name")["Consumed (ml)"]
    .sum()
    .sort_values(ascending=False)
)

print("\nConsumption by bar:")
print(consumption_by_bar)


# ------------------------------------------------------------
# 12. DAILY TOTAL CONSUMPTION
# ------------------------------------------------------------

daily_consumption = (
    df.groupby("Date")["Consumed (ml)"]
    .sum()
)

print("\nDaily consumption:")
print(daily_consumption.head())


# ------------------------------------------------------------
# 13. PLOT CONSUMPTION BY ALCOHOL TYPE
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

consumption_by_type.plot(kind="bar")

plt.title("Total Consumption by Alcohol Type")
plt.xlabel("Alcohol Type")
plt.ylabel("Consumed (ml)")
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 14. PLOT CONSUMPTION BY BAR
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

consumption_by_bar.plot(kind="bar")

plt.title("Total Consumption by Bar")
plt.xlabel("Bar")
plt.ylabel("Consumed (ml)")
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 15. PLOT CONSUMPTION BY BRAND
# ------------------------------------------------------------

plt.figure(figsize=(12, 6))

consumption_by_brand.plot(kind="bar")

plt.title("Total Consumption by Brand")
plt.xlabel("Brand")
plt.ylabel("Consumed (ml)")
plt.xticks(rotation=60)
plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 16. DAILY CONSUMPTION TREND
# ------------------------------------------------------------

plt.figure(figsize=(14, 6))

plt.plot(
    daily_consumption.index,
    daily_consumption.values
)

plt.title("Daily Alcohol Consumption")
plt.xlabel("Date")
plt.ylabel("Consumed (ml)")
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()

# ============================================================
# 17. CREATE DAILY DEMAND DATASET
# ============================================================

daily_demand = (
    df.groupby(
        ["Date", "Bar Name", "Alcohol Type", "Brand Name"],
        as_index=False
    )["Consumed (ml)"]
    .sum()
)

print("\n" + "=" * 60)
print("DAILY DEMAND DATASET")
print("=" * 60)

print("\nFirst 20 rows:")
print(daily_demand.head(20))

print("\nDaily demand shape:")
print(daily_demand.shape)

# ============================================================
# 18. UNIQUE BAR-BRAND COMBINATIONS
# ============================================================

combinations = (
    daily_demand[
        ["Bar Name", "Alcohol Type", "Brand Name"]
    ]
    .drop_duplicates()
)

print("\nNumber of unique bar-brand combinations:")
print(len(combinations))

print("\nBar-brand combinations:")
print(combinations.to_string(index=False))

# ============================================================
# 19. CREATE COMPLETE DAILY TIME SERIES
# ============================================================

print("\n" + "=" * 60)
print("CREATING COMPLETE DAILY TIME SERIES")
print("=" * 60)

# Actual bar-brand combinations that exist in the dataset
bar_brand = daily_demand[
    ["Bar Name", "Alcohol Type", "Brand Name"]
].drop_duplicates()

# All dates in the historical period
all_dates = pd.DataFrame({
    "Date": pd.date_range(
        start=df["Date"].min(),
        end=df["Date"].max(),
        freq="D"
    )
})

# Cross join dates with actual bar-brand combinations
complete_daily = all_dates.merge(
    bar_brand,
    how="cross"
)

# Merge the actual consumption values
complete_daily = complete_daily.merge(
    daily_demand,
    on=[
        "Date",
        "Bar Name",
        "Alcohol Type",
        "Brand Name"
    ],
    how="left"
)

# No recorded consumption for that bar-brand-date
# is treated as zero demand.
complete_daily["Consumed (ml)"] = (
    complete_daily["Consumed (ml)"]
    .fillna(0)
)

# Sort chronologically
complete_daily = complete_daily.sort_values(
    ["Bar Name", "Brand Name", "Date"]
).reset_index(drop=True)

print("\nOriginal daily dataset:")
print(daily_demand.shape)

print("\nComplete daily dataset:")
print(complete_daily.shape)

print("\nExpected rows:")
print(366 * 96)

print("\nFirst 20 rows:")
print(complete_daily.head(20))

# ============================================================
# 20. VERIFY COMPLETE TIME SERIES
# ============================================================

print("\nNumber of unique dates:")
print(complete_daily["Date"].nunique())

print("\nNumber of bar-brand combinations:")
print(
    complete_daily[
        ["Bar Name", "Brand Name"]
    ].drop_duplicates().shape[0]
)

print("\nNumber of rows:")
print(len(complete_daily))

print("\nMissing consumption values:")
print(
    complete_daily["Consumed (ml)"].isnull().sum()
)

# ============================================================
# 21. EXAMPLE DEMAND TREND
# ============================================================

example_bar = "Smith's Bar"
example_brand = "Captain Morgan"

example_data = complete_daily[
    (complete_daily["Bar Name"] == example_bar) &
    (complete_daily["Brand Name"] == example_brand)
].copy()

plt.figure(figsize=(14, 6))

plt.plot(
    example_data["Date"],
    example_data["Consumed (ml)"]
)

plt.title(
    f"Daily Demand - {example_bar} - {example_brand}"
)

plt.xlabel("Date")
plt.ylabel("Consumed (ml)")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# ============================================================
# 22. FEATURE ENGINEERING
# ============================================================

model_data = complete_daily.copy()

# Previous-day demand
model_data["lag_1"] = (
    model_data
    .groupby(["Bar Name", "Brand Name"])["Consumed (ml)"]
    .shift(1)
)

# Demand 7 days ago
model_data["lag_7"] = (
    model_data
    .groupby(["Bar Name", "Brand Name"])["Consumed (ml)"]
    .shift(7)
)

# Demand 14 days ago
model_data["lag_14"] = (
    model_data
    .groupby(["Bar Name", "Brand Name"])["Consumed (ml)"]
    .shift(14)
)

# Demand 28 days ago
model_data["lag_28"] = (
    model_data
    .groupby(["Bar Name", "Brand Name"])["Consumed (ml)"]
    .shift(28)
)

# 7-day rolling average
model_data["rolling_7"] = (
    model_data
    .groupby(["Bar Name", "Brand Name"])["Consumed (ml)"]
    .transform(
        lambda x: x.shift(1).rolling(7).mean()
    )
)

# 14-day rolling average
model_data["rolling_14"] = (
    model_data
    .groupby(["Bar Name", "Brand Name"])["Consumed (ml)"]
    .transform(
        lambda x: x.shift(1).rolling(14).mean()
    )
)

# Calendar features
model_data["day_of_week"] = (
    model_data["Date"].dt.dayofweek
)

model_data["month"] = (
    model_data["Date"].dt.month
)

model_data["is_weekend"] = (
    model_data["day_of_week"] >= 5
).astype(int)

print("\nFeature-engineered data:")
print(model_data.head(40))

# ============================================================
# 23. REMOVE INITIAL LAG ROWS
# ============================================================

model_data = model_data.dropna().reset_index(drop=True)

print("\nModel dataset shape:")
print(model_data.shape)

print("\nRemaining missing values:")
print(model_data.isnull().sum().sum())

# ============================================================
# 24. TIME-BASED TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 60)
print("TIME-BASED TRAIN / TEST SPLIT")
print("=" * 60)

# Use the last 20% of dates for testing
unique_dates = sorted(
    model_data["Date"].unique()
)

split_index = int(len(unique_dates) * 0.80)

train_end_date = unique_dates[split_index]

train_data = model_data[
    model_data["Date"] < train_end_date
].copy()

test_data = model_data[
    model_data["Date"] >= train_end_date
].copy()

print("\nTraining period:")
print(
    train_data["Date"].min(),
    "to",
    train_data["Date"].max()
)

print("\nTesting period:")
print(
    test_data["Date"].min(),
    "to",
    test_data["Date"].max()
)

print("\nTraining rows:")
print(len(train_data))

print("\nTesting rows:")
print(len(test_data))

# ============================================================
# 25. DEFINE MODEL FEATURES
# ============================================================

feature_columns = [
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_7",
    "rolling_14",
    "day_of_week",
    "month",
    "is_weekend"
]

# Add categorical information
categorical_columns = [
    "Bar Name",
    "Alcohol Type",
    "Brand Name"
]

# Convert categorical variables into numerical columns
model_encoded = pd.get_dummies(
    model_data,
    columns=categorical_columns,
    dtype=int
)

# Recreate train/test split after encoding
train_encoded = model_encoded[
    model_encoded["Date"] < train_end_date
].copy()

test_encoded = model_encoded[
    model_encoded["Date"] >= train_end_date
].copy()

# Find the generated categorical columns
encoded_feature_columns = [
    column
    for column in model_encoded.columns
    if column.startswith("Bar Name_")
    or column.startswith("Alcohol Type_")
    or column.startswith("Brand Name_")
]

feature_columns = (
    feature_columns +
    encoded_feature_columns
)

X_train = train_encoded[feature_columns]
y_train = train_encoded["Consumed (ml)"]

X_test = test_encoded[feature_columns]
y_test = test_encoded["Consumed (ml)"]

print("\nNumber of model features:")
print(len(feature_columns))

print("\nTraining feature shape:")
print(X_train.shape)

print("\nTesting feature shape:")
print(X_test.shape)

# ============================================================
# 26. TRAIN RANDOM FOREST MODEL
# ============================================================

print("\n" + "=" * 60)
print("TRAINING RANDOM FOREST MODEL")
print("=" * 60)

rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(
    X_train,
    y_train
)

print("\nRandom Forest training completed!")

# ============================================================
# 27. GENERATE TEST PREDICTIONS
# ============================================================

y_pred = rf_model.predict(X_test)

# Demand cannot be negative
y_pred = np.maximum(y_pred, 0)

print("\nFirst 20 predictions:")

# Use the original test data for readable business columns
prediction_results = test_data[
    [
        "Date",
        "Bar Name",
        "Alcohol Type",
        "Brand Name"
    ]
].copy()

prediction_results["Actual Demand (ml)"] = y_test.values
prediction_results["Predicted Demand (ml)"] = y_pred

print(
    prediction_results.head(20).to_string(index=False)
)

# ============================================================
# 28. MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"\nMAE  : {mae:.2f} ml")
print(f"RMSE : {rmse:.2f} ml")

# ============================================================
# 29. SEASONAL NAIVE BASELINE
# ============================================================

baseline_prediction = test_encoded["lag_7"].values

baseline_prediction = np.maximum(
    baseline_prediction,
    0
)

baseline_mae = mean_absolute_error(
    y_test,
    baseline_prediction
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        baseline_prediction
    )
)

print("\n" + "=" * 60)
print("BASELINE PERFORMANCE")
print("=" * 60)

print(f"\nBaseline MAE  : {baseline_mae:.2f} ml")
print(f"Baseline RMSE : {baseline_rmse:.2f} ml")

# ============================================================
# 30. MODEL COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

comparison = pd.DataFrame({
    "Model": [
        "7-Day Seasonal Naive",
        "Random Forest"
    ],
    "MAE (ml)": [
        baseline_mae,
        mae
    ],
    "RMSE (ml)": [
        baseline_rmse,
        rmse
    ]
})

print(
    comparison.to_string(index=False)
)

# ============================================================
# 31. ACTUAL VS PREDICTED DEMAND
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(
    test_encoded["Date"],
    y_test.values,
    label="Actual Demand",
    alpha=0.7
)

plt.plot(
    test_encoded["Date"],
    y_pred,
    label="Predicted Demand",
    alpha=0.7
)

plt.title("Actual vs Predicted Demand")
plt.xlabel("Date")
plt.ylabel("Consumed (ml)")
plt.legend()
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# ============================================================
# 32. INVENTORY RECOMMENDATION
# ============================================================

print("\n" + "=" * 60)
print("INVENTORY RECOMMENDATION")
print("=" * 60)

# Use the test predictions as demand estimates
inventory_data = prediction_results.copy()

# Calculate forecast error
inventory_data["Forecast Error"] = (
    inventory_data["Actual Demand (ml)"]
    - inventory_data["Predicted Demand (ml)"]
)

# Calculate absolute forecast error
inventory_data["Absolute Error"] = (
    inventory_data["Forecast Error"].abs()
)

# Overall forecast error standard deviation
error_std = inventory_data["Forecast Error"].std()

# Assumptions
lead_time_days = 1
service_level_z = 1.65

# Safety stock
safety_stock = (
    service_level_z
    * error_std
    * np.sqrt(lead_time_days)
)

# Average forecast demand
average_forecast = (
    inventory_data["Predicted Demand (ml)"]
    .mean()
)

# Reorder point
reorder_point = (
    average_forecast * lead_time_days
    + safety_stock
)

# PAR level
par_level = (
    average_forecast * 7
    + safety_stock
)

print(f"\nAverage predicted daily demand: "
      f"{average_forecast:.2f} ml")

print(f"Forecast error standard deviation: "
      f"{error_std:.2f} ml")

print(f"Safety stock: "
      f"{safety_stock:.2f} ml")

print(f"Reorder point: "
      f"{reorder_point:.2f} ml")

print(f"7-day PAR level: "
      f"{par_level:.2f} ml")

# ============================================================
# 33. INVENTORY RECOMMENDATION BY BAR / ALCOHOL / BRAND
# ============================================================

print("\n" + "=" * 60)
print("INVENTORY RECOMMENDATION BY BAR / ALCOHOL / BRAND")
print("=" * 60)

# Calculate recommendations separately for each combination
recommendation_data = (
    inventory_data
    .groupby(
        ["Bar Name", "Alcohol Type", "Brand Name"],
        as_index=False
    )
    .agg(
        Average_Demand=("Predicted Demand (ml)", "mean"),
        Forecast_Error_Std=("Forecast Error", "std"),
        Current_Actual_Demand=("Actual Demand (ml)", "last")
    )
)

# Replace missing standard deviation values
# This can happen if a combination has only one prediction.
recommendation_data["Forecast_Error_Std"] = (
    recommendation_data["Forecast_Error_Std"]
    .fillna(0)
)

# Safety stock
recommendation_data["Safety Stock (ml)"] = (
    service_level_z
    * recommendation_data["Forecast_Error_Std"]
    * np.sqrt(lead_time_days)
)

# Reorder point
recommendation_data["Reorder Point (ml)"] = (
    recommendation_data["Average_Demand"]
    * lead_time_days
    + recommendation_data["Safety Stock (ml)"]
)

# 7-day PAR level
recommendation_data["7-Day PAR Level (ml)"] = (
    recommendation_data["Average_Demand"]
    * 7
    + recommendation_data["Safety Stock (ml)"]
)

# Display final recommendation table
recommendation_display = recommendation_data[
    [
        "Bar Name",
        "Alcohol Type",
        "Brand Name",
        "Average_Demand",
        "Safety Stock (ml)",
        "Reorder Point (ml)",
        "7-Day PAR Level (ml)"
    ]
].copy()

# Rename column for readability
recommendation_display = recommendation_display.rename(
    columns={
        "Average_Demand": "Predicted Daily Demand (ml)"
    }
)

# Round numerical values
numeric_columns = [
    "Predicted Daily Demand (ml)",
    "Safety Stock (ml)",
    "Reorder Point (ml)",
    "7-Day PAR Level (ml)"
]

recommendation_display[numeric_columns] = (
    recommendation_display[numeric_columns].round(2)
)

print("\nInventory recommendations:")
print(recommendation_display.to_string(index=False))

print(
    f"\nNumber of recommendations: "
    f"{len(recommendation_display)}"
)

# ============================================================
# 34. RECOMMENDED ORDER QUANTITY
# ============================================================

print("\n" + "=" * 60)
print("RECOMMENDED ORDER QUANTITY")
print("=" * 60)

# Convert original transaction date to datetime
df["Date Time Served"] = pd.to_datetime(
    df["Date Time Served"]
)

# Find the latest inventory record for every
# Bar + Alcohol Type + Brand combination
latest_inventory = (
    df.sort_values("Date Time Served")
    .groupby(
        ["Bar Name", "Alcohol Type", "Brand Name"],
        as_index=False
    )
    .tail(1)
)

# Keep only the information required
latest_inventory = latest_inventory[
    [
        "Bar Name",
        "Alcohol Type",
        "Brand Name",
        "Closing Balance (ml)"
    ]
].copy()

# Merge current inventory with recommendations
final_recommendations = recommendation_display.merge(
    latest_inventory,
    on=[
        "Bar Name",
        "Alcohol Type",
        "Brand Name"
    ],
    how="left"
)

# Calculate recommended order quantity
final_recommendations["Recommended Order (ml)"] = (
    final_recommendations["7-Day PAR Level (ml)"]
    - final_recommendations["Closing Balance (ml)"]
)

# Do not recommend negative orders
final_recommendations["Recommended Order (ml)"] = (
    final_recommendations["Recommended Order (ml)"]
    .clip(lower=0)
)

# Round values
final_recommendations[
    [
        "Closing Balance (ml)",
        "Recommended Order (ml)"
    ]
] = final_recommendations[
    [
        "Closing Balance (ml)",
        "Recommended Order (ml)"
    ]
].round(2)

print("\nFinal inventory recommendations:")

print(
    final_recommendations[
        [
            "Bar Name",
            "Alcohol Type",
            "Brand Name",
            "Predicted Daily Demand (ml)",
            "Closing Balance (ml)",
            "Reorder Point (ml)",
            "7-Day PAR Level (ml)",
            "Recommended Order (ml)"
        ]
    ].to_string(index=False)
)

print(
    f"\nNumber of final recommendations: "
    f"{len(final_recommendations)}"
)

# ============================================================
# 35. SAVE FINAL INVENTORY RECOMMENDATIONS
# ============================================================

final_recommendations.to_csv(
    "final_inventory_recommendations.csv",
    index=False
)

print("\nFinal recommendations saved to:")
print("final_inventory_recommendations.csv")

# ============================================================
# 36. CREATE PROFESSIONAL CHARTS
# ============================================================

import matplotlib.pyplot as plt


# ------------------------------------------------------------
# CHART 1: ACTUAL VS PREDICTED DEMAND
# ------------------------------------------------------------

chart_data = prediction_results.copy()

# Select one representative combination
chart_data = chart_data[
    (chart_data["Bar Name"] == "Anderson's Bar") &
    (chart_data["Alcohol Type"] == "Vodka") &
    (chart_data["Brand Name"] == "Absolut")
].copy()

chart_data = chart_data.sort_values("Date")

plt.figure(figsize=(12, 6))

plt.plot(
    chart_data["Date"],
    chart_data["Actual Demand (ml)"],
    label="Actual Demand"
)

plt.plot(
    chart_data["Date"],
    chart_data["Predicted Demand (ml)"],
    label="Predicted Demand"
)

plt.title("Actual vs Predicted Demand - Anderson's Bar, Absolut")
plt.xlabel("Date")
plt.ylabel("Demand (ml)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    "charts/actual_vs_predicted.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# CHART 2: CONSUMPTION BY ALCOHOL TYPE
# ------------------------------------------------------------

alcohol_consumption = (
    df.groupby("Alcohol Type")["Consumed (ml)"]
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 6))

alcohol_consumption.plot(kind="bar")

plt.title("Total Consumption by Alcohol Type")
plt.xlabel("Alcohol Type")
plt.ylabel("Consumption (ml)")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    "charts/consumption_by_alcohol_type.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# CHART 3: CONSUMPTION BY BAR
# ------------------------------------------------------------

bar_consumption = (
    df.groupby("Bar Name")["Consumed (ml)"]
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 6))

bar_consumption.plot(kind="bar")

plt.title("Total Consumption by Bar")
plt.xlabel("Bar")
plt.ylabel("Consumption (ml)")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    "charts/consumption_by_bar.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# CHART 4: TOP 10 BRANDS BY CONSUMPTION
# ------------------------------------------------------------

brand_consumption = (
    df.groupby("Brand Name")["Consumed (ml)"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10, 6))

brand_consumption.sort_values().plot(kind="barh")

plt.title("Top 10 Brands by Consumption")
plt.xlabel("Consumption (ml)")
plt.ylabel("Brand")
plt.tight_layout()

plt.savefig(
    "charts/top_10_brands.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# CHART 5: RECOMMENDED ORDER QUANTITY
# ------------------------------------------------------------

order_data = (
    final_recommendations[
        final_recommendations["Recommended Order (ml)"] > 0
    ]
    .sort_values(
        "Recommended Order (ml)",
        ascending=False
    )
    .head(15)
)

order_labels = (
    order_data["Bar Name"]
    + " - "
    + order_data["Brand Name"]
)

plt.figure(figsize=(12, 7))

plt.barh(
    order_labels,
    order_data["Recommended Order (ml)"]
)

plt.title("Top Inventory Replenishment Recommendations")
plt.xlabel("Recommended Order (ml)")
plt.ylabel("Bar - Brand")
plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig(
    "charts/recommended_order_quantity.png",
    dpi=300
)

plt.close()


print("\n" + "=" * 60)
print("CHARTS CREATED SUCCESSFULLY")
print("=" * 60)

print("1. charts/actual_vs_predicted.png")
print("2. charts/consumption_by_alcohol_type.png")
print("3. charts/consumption_by_bar.png")
print("4. charts/top_10_brands.png")
print("5. charts/recommended_order_quantity.png")