
import plotly.express as px
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from fpdf import FPDF


# Load Data
df = pd.read_csv("tourism_data.csv")
province_coords = {
    "Riyadh": [24.7136, 46.6753],
    "Makkah": [21.3891, 39.8579],
    "Madinah": [24.5247, 39.5692],
    "Eastern_region": [26.4207, 50.0888],
    "Aseer": [18.2164, 42.5053],
    "Tabuk": [28.3998, 36.5715],
    "Hail": [27.5114, 41.7208],
    "Alqassim": [26.2078, 43.4837],
    "Jazan": [16.8892, 42.5511],
    "Najran": [17.5650, 44.2289],
    "Albaha": [20.0129, 41.4677],
    "Jouf": [29.9697, 40.2064],
    "Northern_borders": [30.9753, 41.0381]
}
# Filters


st.sidebar.title("Filters")

selected_year = st.sidebar.selectbox(
    "Select Year",
    sorted(df["YEARS"].unique())
)
compare_year = st.sidebar.selectbox(
    "Compare With Year",
    sorted(df["YEARS"].unique()),
    index=0
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

st.header("Tourism Dashboard")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Tourists",
        f"{filtered_df['Tourists_Number'].sum():,.0f}"
    )

with col2:
    st.metric(
        "Total Spending",
        f"{filtered_df['Tourists_Spending'].sum():,.0f}"
    )

with col3:
    st.metric(
        "Avg Stay",
        f"{filtered_df['Avg_Stay'].mean():.2f}"
    )

with col4:
    st.metric(
        "Avg Trip Spending",
        f"{filtered_df['Avg_Spending_Trip'].mean():,.0f}"
    )
# =========================
# Year-over-Year Comparison
# =========================

st.header("Year-over-Year Comparison")

year1_data = df[df["YEARS"] == selected_year]
year2_data = df[df["YEARS"] == compare_year]

if selected_type != "All":
    year1_data = year1_data[
        year1_data["Tourism_Type"] == selected_type
    ]

    year2_data = year2_data[
        year2_data["Tourism_Type"] == selected_type
    ]

if selected_province != "All":
    year1_data = year1_data[
        year1_data["Province"] == selected_province
    ]

    year2_data = year2_data[
        year2_data["Province"] == selected_province
    ]

tourists_year1 = year1_data["Tourists_Number"].sum()
tourists_year2 = year2_data["Tourists_Number"].sum()

spending_year1 = year1_data["Tourists_Spending"].sum()
spending_year2 = year2_data["Tourists_Spending"].sum()

if tourists_year2 > 0 and spending_year2 > 0:

    tourists_growth = (
        (tourists_year1 - tourists_year2)
        / tourists_year2
    ) * 100

    spending_growth = (
        (spending_year1 - spending_year2)
        / spending_year2
    ) * 100

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Tourists Growth %",
            f"{tourists_growth:.2f}%"
        )

    with col2:
        st.metric(
            "Spending Growth %",
            f"{spending_growth:.2f}%"
        )

    comparison_df = pd.DataFrame(
        {
            "Selected Year": [
                tourists_year1,
                spending_year1
            ],
            "Comparison Year": [
                tourists_year2,
                spending_year2
            ]
        },
        index=[
            "Tourists",
            "Spending"
        ]
    )

    st.subheader("Year Comparison Chart")

    st.bar_chart(comparison_df)

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

#turism insights
st.header("Tourism Insights")

if not filtered_df.empty:

    most_visited = (
        filtered_df.groupby("Province")["Tourists_Number"]
        .sum()
        .idxmax()
    )

    highest_spending = (
        filtered_df.groupby("Province")["Tourists_Spending"]
        .sum()
        .idxmax()
    )

    avg_stay_value = (
        filtered_df["Avg_Stay"]
        .mean()
    )

    st.success(
        f"Most Visited Province: {most_visited}"
    )

    st.success(
        f"Highest Spending Province: {highest_spending}"
    )

    st.info(
        f"Average Stay Duration: {avg_stay_value:.2f} nights"
    )
# =========================
# AI Recommendations
# =========================

st.header("AI Recommendations")

if not filtered_df.empty:

    total_tourists = filtered_df["Tourists_Number"].sum()

    total_spending = filtered_df["Tourists_Spending"].sum()

    avg_trip_spending = (
        filtered_df["Avg_Spending_Trip"].mean()
    )

    if total_spending > 0:
        st.success(
            "Tourism activity shows measurable economic impact."
        )

    if avg_trip_spending > df["Avg_Spending_Trip"].mean():

        st.info(
            "Average trip spending is higher than dataset average."
        )

    else:

        st.info(
            "Average trip spending is below dataset average."
        )

    if selected_province == "All":

        st.warning(
            "Analyze specific provinces for deeper insights."
        )

    else:

        st.success(
            f"{selected_province} shows tourism potential."
        )
# Executive Summary
# =========================

st.header("Executive Summary")

if not filtered_df.empty:

    total_tourists = (
        filtered_df["Tourists_Number"]
        .sum()
    )

    total_spending = (
        filtered_df["Tourists_Spending"]
        .sum()
    )

    avg_stay = (
        filtered_df["Avg_Stay"]
        .mean()
    )

    top_province = (
        filtered_df.groupby("Province")["Tourists_Number"]
        .sum()
        .idxmax()
    )

    st.markdown(f"""
### Tourism Overview

- Total Tourists: **{total_tourists:,.0f}**
- Total Spending: **{total_spending:,.0f}**
- Average Stay: **{avg_stay:.2f} nights**
- Top Province: **{top_province}**

### Key Finding

Tourism activity remains concentrated in **{top_province}**, making it the strongest tourism destination within the selected filters.

### Recommendation

Decision makers should continue investing in tourism infrastructure and visitor experience to increase tourism spending and length of stay.
""")
# =========================
# Download PDF Report
# =========================

def create_pdf_report():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Saudi Tourism AI Analyst Report", ln=True, align="C")
    pdf.ln(10)

    pdf.cell(200, 10, txt=f"Selected Year: {selected_year}", ln=True)
    pdf.cell(200, 10, txt=f"Tourism Type: {selected_type}", ln=True)
    pdf.cell(200, 10, txt=f"Province: {selected_province}", ln=True)
    pdf.ln(10)

    pdf.cell(200, 10, txt=f"Total Tourists: {filtered_df['Tourists_Number'].sum():,.0f}", ln=True)
    pdf.cell(200, 10, txt=f"Total Spending: {filtered_df['Tourists_Spending'].sum():,.0f}", ln=True)
    pdf.cell(200, 10, txt=f"Average Stay: {filtered_df['Avg_Stay'].mean():.2f}", ln=True)
    pdf.cell(200, 10, txt=f"Average Trip Spending: {filtered_df['Avg_Spending_Trip'].mean():,.0f}", ln=True)

    pdf.ln(10)

    if not filtered_df.empty:
        top_province = (
            filtered_df.groupby("Province")["Tourists_Number"]
            .sum()
            .idxmax()
        )

        highest_spending = (
            filtered_df.groupby("Province")["Tourists_Spending"]
            .sum()
            .idxmax()
        )

        pdf.cell(200, 10, txt=f"Most Visited Province: {top_province}", ln=True)
        pdf.cell(200, 10, txt=f"Highest Spending Province: {highest_spending}", ln=True)

    return pdf.output(dest="S").encode("latin1")


pdf_report = create_pdf_report()

st.download_button(
    label="Download PDF Report",
    data=pdf_report,
    file_name="saudi_tourism_report.pdf",
    mime="application/pdf"
)
 #Tourism Charts           
        
st.header("Tourism Charts")

# Tourism Map

st.header("Tourism Map")

map_data = (
    filtered_df.groupby("Province")["Tourists_Number"]
    .sum()
    .reset_index()
)

map_data["Latitude"] = map_data["Province"].map(
    lambda x: province_coords.get(x, [None, None])[0]
)

map_data["Longitude"] = map_data["Province"].map(
    lambda x: province_coords.get(x, [None, None])[1]
)

map_data = map_data.dropna()

fig = px.scatter_geo(
    map_data,
    lat="Latitude",
    lon="Longitude",
    size="Tourists_Number",
    hover_name="Province",
    projection="natural earth",
    title="Tourists Distribution Across Saudi Provinces"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.subheader("Tourism Growth Over Years")

yearly_tourists = (
    df.groupby("YEARS")["Tourists_Number"]
    .sum()
)

st.line_chart(yearly_tourists)
st.subheader("Tourism Spending Over Years")

yearly_spending = (
    df.groupby("YEARS")["Tourists_Spending"]
    .sum()
)

st.line_chart(yearly_spending)
# Tourism Forecast
st.header("Tourism Forecast")

forecast_years = pd.DataFrame({
    "YEARS": [2025, 2026, 2027, 2028, 2029, 2030]
})

historical_yearly = (
    df.groupby("YEARS")["Tourists_Number"]
    .sum()
    .reset_index()
)

from sklearn.linear_model import LinearRegression

forecast_model = LinearRegression()

X_years = historical_yearly[["YEARS"]]
y_tourists = historical_yearly["Tourists_Number"]

forecast_model.fit(X_years, y_tourists)

forecast_values = forecast_model.predict(forecast_years)

forecast_df = pd.DataFrame({
    "Year": forecast_years["YEARS"],
    "Predicted Tourists": forecast_values
})

st.dataframe(forecast_df)

st.line_chart(
    forecast_df.set_index("Year")
)

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
# Tourism Insights
st.subheader("Top 5 Provinces By Tourists")

top5 = (
    filtered_df.groupby("Province")["Tourists_Number"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
)

st.dataframe(top5)
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
st.info(
    "Enter tourism indicators below to estimate expected tourism spending."
)
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