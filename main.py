import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# Load Data
df = pd.read_csv("tourism_data.csv", encoding="latin1")

# Data Exploration
print(df.head())

print(df.info())

print(df.describe())

print(df.isnull().sum())

#Visualizations 1
# Tourists by Province

tourists_by_region = (
    df.groupby("Province")["Tourists_Number"]
    .sum()
    .sort_values(ascending=False)
)

print(tourists_by_region)

tourists_by_region.plot(kind="bar")

plt.title("Tourists by Province")

plt.ylabel("Number of Tourists")

plt.show()

#Visualizations 2
# Spending by Province
spending_by_region = (
    df.groupby("Province")["Tourists_Spending"]
    .sum()
    .sort_values(ascending=False)
)

print(spending_by_region)

spending_by_region.plot(kind="bar")

plt.title("Tourism Spending by Province")

plt.ylabel("Spending")

plt.show()
#Visualizations 3
# Domestic vs Inbound

tourism_type_spending = (
    df.groupby("Tourism_Type")["Tourists_Spending"]
    .sum()
)

print(tourism_type_spending)
tourism_type_spending.plot(
    kind="pie",
    autopct="%1.1f%%")

plt.title("Domestic vs Inbound Spending")

plt.ylabel("")

plt.show()



# Machine Learning
X = df[
    [
        "Tourists_Number",
        "Overnight_Stay",
        "Avg_Stay",
        "Avg_Spending_Trip",
        "Avg_Spending_Night"
    ]
]

y = df["Tourists_Spending"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
# Train Model
model.fit(X_train, y_train)
# Make Predictions
predictions = model.predict(X_test)

# Model Evaluation
mae = mean_absolute_error(y_test, predictions)

print("MAE:", mae)

comparison = pd.DataFrame({
    "Actual": y_test,
    "Predicted": predictions
})

comparison["Difference"] = (
    comparison["Actual"]
    - comparison["Predicted"]
)

print(comparison.head(10))

# Dashboard Summary


print("\n===== Saudi Tourism Dashboard =====")

print("Total Tourists:",
      df["Tourists_Number"].sum())

print("Total Spending:",
      df["Tourists_Spending"].sum())

print("Top Province By Tourists:",
      df.groupby("Province")["Tourists_Number"]
        .sum()
        .idxmax())

print("Top Province By Spending:",
      df.groupby("Province")["Tourists_Spending"]
        .sum()
        .idxmax())
print("Model MAE:", mae)

# Save comparison results

comparison.to_csv(
    "prediction_results.csv",
    index=False
)



# Save Model

joblib.dump(model, "saudi_tourism_model.pkl")

print("Model Saved Successfully")
