import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load Data
df = pd.read_csv("tourism_data.csv")

# Filters


st.sidebar.title("Filters")

selected_year = st.sidebar.selectbox(
    "Select Year",
    sorted(df["YEARS"].unique())
)

selected_type = st.sidebar.selectbox(
    "Tourism Type",
    ["All", "Domestic", "Inbound"]
)
selected_province = st.sidebar.selectbox(
    "Province",
    ["All"] + sorted(df["Province"].unique().tolist())
)
filtered_df = df[df["YEARS"] == selected_year]

if selected_type != "All":
    filtered_df = filtered_df[
        filtered_df["Tourism_Type"] == selected_type
    ]

if selected_province != "All":
    filtered_df = filtered_df[
        filtered_df["Province"] == selected_province
    ]
    

# Load Model
model = joblib.load("saudi_tourism_model.pkl")

st.title("Saudi Tourism AI Analyst 🇸🇦")
st.caption(
    f"Year: {selected_year} | "
    f"Type: {selected_type} | "
    f"Province: {selected_province}"
)
st.write(
    """
    AI-powered dashboard for analyzing Saudi tourism data,
    exploring tourism trends, and predicting tourism spending.
    """
)

# Dashboard

if not filtered_df.empty:

    col1, col2 = st.columns(2)

    with col1:
        st.info(
            f"Top Province By Tourists: "
            f"{filtered_df.groupby('Province')['Tourists_Number'].sum().idxmax()}"
        )

    with col2:
        st.info(
            f"Top Province By Spending: "
            f"{filtered_df.groupby('Province')['Tourists_Spending'].sum().idxmax()}"
        )

else:
    st.warning("No data available")
# Charts


st.header("Tourism Charts")
# Tourism Type Distribution

st.subheader("Domestic vs Inbound Tourism")

tourism_type_data = (
    df[df["YEARS"] == selected_year]
    .groupby("Tourism_Type")["Tourists_Spending"]
    .sum()
)
if not tourism_type_data.empty:
 fig, ax = plt.subplots()

ax.pie(
    tourism_type_data,
    labels=tourism_type_data.index,
    autopct="%1.1f%%"
)

st.pyplot(fig)
tourists_by_region = (
    filtered_df.groupby("Province")["Tourists_Number"]
      .sum()
      .sort_values(ascending=False)
)
st.subheader("Tourists by Province")
st.bar_chart(tourists_by_region)

spending_by_region = (
    filtered_df.groupby("Province")["Tourists_Spending"]
      .sum()
      .sort_values(ascending=False)
)
st.subheader("Tourism Spending by Province")
st.bar_chart(spending_by_region)

st.header("Tourism Data")

st.dataframe(
    filtered_df,
    use_container_width=True
)
csv = filtered_df.to_csv(index=False)

st.download_button(
    label="Download Filtered Data CSV",
    data=csv,
    file_name="filtered_tourism_data.csv",
    mime="text/csv"
)
# Prediction


st.header("AI Prediction")

tourists_number = st.number_input(
    "Tourists Number",
    min_value=0.0
)

overnight_stay = st.number_input(
    "Overnight Stay",
    min_value=0.0
)

avg_stay = st.number_input(
    "Average Stay",
    min_value=0.0
)

avg_spending_trip = st.number_input(
    "Average Spending Per Trip",
    min_value=0.0
)

avg_spending_night = st.number_input(
    "Average Spending Per Night",
    min_value=0.0
)

if st.button("Predict Spending"):

    data = np.array([[
        tourists_number,
        overnight_stay,
        avg_stay,
        avg_spending_trip,
        avg_spending_night
    ]])

    prediction = model.predict(data)

    st.success(
        f"Predicted Spending: {prediction[0]:,.2f}"
    )