import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from fpdf import FPDF

# =========================
# Page Configuration
# =========================

st.set_page_config(
    page_title="Saudi Tourism Intelligence",
    page_icon="🏜️",
    layout="wide"
)

# =========================
# Load Data & Model
# =========================

@st.cache_data
def load_data():
    data = pd.read_csv("tourism_data.csv")
    return data

@st.cache_resource
def load_model():
    return joblib.load("saudi_tourism_model.pkl")


df = load_data()
model = load_model()

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

province_images = {
    "Riyadh": "https://www.gtreview.com/wp-content/uploads/2025/07/GTR-KSA-2026-Riyadh_600x400.jpg",
    "Makkah": "https://images.trvl-media.com/place/178043/a8ed123e-3a19-4a74-90bd-925ff046c374.jpg",
    "Madinah": "https://islamonline.net/wp-content/uploads/2022/06/Madinah-Mosque.jpg",
    "Aseer": "https://upload.wikimedia.org/wikipedia/commons/4/45/%D8%B1%D8%AC%D8%A7%D9%84_%D8%A3%D9%84%D9%85%D8%B91.jpg",
    "Tabuk": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRKXmVTt1HYfYLlOVzWM26vXUSdM3Tly-yv1xUAWOQTDum04ZPkfvjQusPG&s=10",
    "Eastern_region": "https://www.ithra.com/application/files/cache/thumbnails/f29d9abf311111301475d8013548272b.jpg",
    "All": "https://www.ithra.com/application/files/cache/thumbnails/f29d9abf311111301475d8013548272b.jpg",
}
# =========================
# Top Controls
# =========================

years = sorted(df["YEARS"].unique())
provinces = sorted(df["Province"].unique().tolist())

# =========================
# CSS
# =========================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(212,175,55,0.18), transparent 28%),
        radial-gradient(circle at 90% 5%, rgba(34,197,94,0.16), transparent 30%),
        linear-gradient(135deg, #05070A 0%, #10140F 45%, #172018 100%);
    color: #F8FAFC;
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1320px;
}

/* Hide default sidebar feel if used by Streamlit internally */
section[data-testid="stSidebar"] {
    background: #05070A !important;
}

.hero-tourism {
    position: relative;
    min-height: 520px;
    border-radius: 34px;
    overflow: hidden;
    padding: 48px;
    margin-bottom: 28px;
    border: 1px solid rgba(255,255,255,0.16);
    box-shadow: 0 24px 70px rgba(0,0,0,0.45);
    background-size: cover;
    background-position: center;
    isolation: isolate;
}

.hero-tourism::before {
    content: "";
    position: absolute;
    inset: 0;
    background:
        linear-gradient(90deg, rgba(5,7,10,0.90) 0%, rgba(5,7,10,0.62) 48%, rgba(5,7,10,0.28) 100%),
        radial-gradient(circle at top right, rgba(212,175,55,0.28), transparent 34%);
    z-index: -1;
}

.hero-tourism::after {
    content: "";
    position: absolute;
    inset: -40%;
    background: linear-gradient(120deg, transparent 35%, rgba(255,255,255,0.16) 50%, transparent 65%);
    transform: translateX(-40%);
    animation: shimmer 9s infinite linear;
    z-index: -1;
}

@keyframes shimmer {
    0% { transform: translateX(-55%) rotate(8deg); }
    100% { transform: translateX(55%) rotate(8deg); }
}

.hero-kicker {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    color: #FDE68A;
    background: rgba(212,175,55,0.16);
    border: 1px solid rgba(253,230,138,0.28);
    padding: 10px 16px;
    border-radius: 999px;
    font-size: 14px;
    font-weight: 800;
    margin-bottom: 18px;
}

.hero-title {
    font-size: 62px;
    line-height: 1.03;
    font-weight: 900;
    letter-spacing: -1.6px;
    color: #FFFFFF;
    max-width: 820px;
    margin-bottom: 18px;
}

.hero-text {
    font-size: 19px;
    line-height: 1.8;
    color: #E5E7EB;
    max-width: 760px;
    margin-bottom: 30px;
}

.hero-actions {
    display: flex;
    gap: 14px;
    flex-wrap: wrap;
}

.hero-pill {
    background: rgba(255,255,255,0.11);
    border: 1px solid rgba(255,255,255,0.18);
    color: #FFFFFF;
    padding: 13px 18px;
    border-radius: 16px;
    font-weight: 900;
    backdrop-filter: blur(12px);
}

.filter-panel {
    background: rgba(8,13,10,0.86);
    border: 1px solid rgba(212,175,55,0.23);
    border-radius: 26px;
    padding: 24px;
    box-shadow: 0 18px 48px rgba(0,0,0,0.36);
    margin-bottom: 26px;
}

.filter-title {
    font-size: 18px;
    font-weight: 900;
    color: #FDE68A;
    margin-bottom: 12px;
}

label {
    color: #F8FAFC !important;
    font-weight: 800 !important;
}

/* Inputs */
div[data-baseweb="select"] > div,
.stNumberInput input {
    background: rgba(255,255,255,0.96) !important;
    color: #111827 !important;
    border-radius: 14px !important;
    border: 1px solid rgba(212,175,55,0.30) !important;
    font-weight: 800 !important;
}

div[data-baseweb="select"] span,
input {
    color: #111827 !important;
    font-weight: 800 !important;
}

div[role="listbox"], div[data-baseweb="popover"] {
    background: #FFFFFF !important;
}

div[role="option"] {
    color: #111827 !important;
    background: #FFFFFF !important;
    font-weight: 700 !important;
}

.section-heading {
    margin-top: 34px;
    margin-bottom: 18px;
}

.section-kicker {
    color: #FDE68A;
    font-size: 14px;
    font-weight: 900;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.section-title {
    color: #FFFFFF;
    font-size: 34px;
    line-height: 1.2;
    font-weight: 900;
    letter-spacing: -0.6px;
}

.story-card, .metric-card, .destination-card, .insight-card, .advisor-card, .data-card {
    background: rgba(13,19,15,0.78);
    border: 1px solid rgba(255,255,255,0.13);
    border-radius: 26px;
    box-shadow: 0 16px 45px rgba(0,0,0,0.35);
    backdrop-filter: blur(14px);
}

.metric-card {
    padding: 26px;
    min-height: 150px;
    position: relative;
    overflow: hidden;
}

.metric-card::after {
    content: "";
    position: absolute;
    right: -35px;
    top: -35px;
    width: 120px;
    height: 120px;
    border-radius: 50%;
    background: rgba(212,175,55,0.18);
}

.metric-icon {
    font-size: 30px;
    margin-bottom: 12px;
}

.metric-label {
    color: #D1D5DB;
    font-size: 14px;
    font-weight: 800;
}

.metric-value {
    color: #FFFFFF;
    font-size: 34px;
    font-weight: 900;
    margin-top: 6px;
}

.metric-note {
    color: #A7F3D0;
    font-size: 13px;
    font-weight: 700;
    margin-top: 8px;
}

.destination-card {
    overflow: hidden;
    min-height: 285px;
}

.destination-image {
    height: 135px;
    background-size: cover;
    background-position: center;
    position: relative;
}

.destination-image::after {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(180deg, transparent, rgba(0,0,0,0.70));
}

.destination-content {
    padding: 20px;
}

.destination-title {
    font-size: 23px;
    color: #FFFFFF;
    font-weight: 900;
    margin-bottom: 6px;
}

.destination-subtitle {
    color: #FDE68A;
    font-size: 13px;
    font-weight: 900;
    margin-bottom: 12px;
}

.destination-stat {
    color: #E5E7EB;
    font-size: 14px;
    line-height: 1.7;
}

.insight-card {
    padding: 24px;
    min-height: 190px;
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.insight-card:hover {
    transform: translateY(-6px);
    border-color: rgba(253,230,138,0.48);
}

.insight-icon {
    font-size: 34px;
    margin-bottom: 14px;
}

.insight-title {
    color: #FDE68A;
    font-size: 16px;
    font-weight: 900;
    margin-bottom: 10px;
}

.insight-text {
    color: #E5E7EB;
    font-size: 15px;
    line-height: 1.7;
}

.advisor-card {
    padding: 30px;
    border-color: rgba(34,197,94,0.28);
    background:
        linear-gradient(135deg, rgba(13,19,15,0.88), rgba(20,83,45,0.48));
}

.prediction-result {
    margin-top: 24px;
    padding: 28px;
    border-radius: 26px;
    background: linear-gradient(135deg, rgba(212,175,55,0.16), rgba(34,197,94,0.18));
    border: 1px solid rgba(253,230,138,0.30);
}

.prediction-value {
    color: #FFFFFF;
    font-size: 42px;
    font-weight: 900;
}

.prediction-label {
    color: #FDE68A;
    font-size: 14px;
    text-transform: uppercase;
    font-weight: 900;
    letter-spacing: 1px;
}

.data-card {
    padding: 26px;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button,
button[kind="formSubmit"] {
    background: linear-gradient(135deg, #D4AF37, #22C55E, #0EA5E9) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 16px !important;
    padding: 0.75rem 1.35rem !important;
    font-size: 15px !important;
    font-weight: 900 !important;
    box-shadow: 0 14px 34px rgba(34,197,94,0.25) !important;
}

.stButton > button p,
.stDownloadButton > button p,
button[kind="formSubmit"] p {
    color: #FFFFFF !important;
    font-weight: 900 !important;
}

.stButton > button:hover,
.stDownloadButton > button:hover,
button[kind="formSubmit"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 18px 44px rgba(14,165,233,0.34) !important;
}

/* Plotly transparent feel */
.js-plotly-plot, .plot-container {
    border-radius: 22px;
}

.footer {
    text-align: center;
    color: #D1D5DB;
    padding: 34px;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Horizontal Filters
# =========================

selected_year_default = max(years)
selected_year_index = years.index(selected_year_default)
compare_year_index = 0

filter_col1, filter_col2, filter_col3, filter_col4 = st.columns([1, 1, 1, 1.2])

with filter_col1:
    selected_year = st.selectbox("Explore Year", years, index=selected_year_index)

with filter_col2:
    compare_year = st.selectbox("Compare With", years, index=compare_year_index)

with filter_col3:
    selected_type = st.selectbox("Tourism Type", ["All", "Domestic", "Inbound"])

with filter_col4:
    selected_province = st.selectbox("Destination", ["All"] + provinces)

filtered_df = df[df["YEARS"] == selected_year].copy()

if selected_type != "All":
    filtered_df = filtered_df[filtered_df["Tourism_Type"] == selected_type]

if selected_province != "All":
    filtered_df = filtered_df[filtered_df["Province"] == selected_province]

hero_image = province_images.get(selected_province, province_images["All"])

# =========================
# Hero Experience
# =========================

st.markdown(f"""
<div class="hero-tourism" style="background-image: url('{hero_image}');">
    <div class="hero-kicker">✨ Saudi Tourism Data Story</div>
    <div class="hero-title">Discover Saudi Arabia Through Data</div>
    <div class="hero-text">
        Explore visitor journeys, spending patterns, regional performance and Vision 2030 tourism outlook through an interactive analytics experience.
    </div>
    <div class="hero-actions">
        <div class="hero-pill">📍 Destination: {selected_province}</div>
        <div class="hero-pill">🗓️ Year: {selected_year}</div>
        <div class="hero-pill">🌍 Tourism Type: {selected_type}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# KPI Journey Cards
# =========================

total_tourists = filtered_df["Tourists_Number"].sum() if not filtered_df.empty else 0
total_spending = filtered_df["Tourists_Spending"].sum() if not filtered_df.empty else 0
avg_stay = filtered_df["Avg_Stay"].mean() if not filtered_df.empty else 0
avg_trip_spending = filtered_df["Avg_Spending_Trip"].mean() if not filtered_df.empty else 0

comparison_df = df[df["YEARS"] == compare_year].copy()
if selected_type != "All":
    comparison_df = comparison_df[comparison_df["Tourism_Type"] == selected_type]
if selected_province != "All":
    comparison_df = comparison_df[comparison_df["Province"] == selected_province]

compare_tourists = comparison_df["Tourists_Number"].sum() if not comparison_df.empty else 0
compare_spending = comparison_df["Tourists_Spending"].sum() if not comparison_df.empty else 0

tourist_growth = ((total_tourists - compare_tourists) / compare_tourists * 100) if compare_tourists else 0
spending_growth = ((total_spending - compare_spending) / compare_spending * 100) if compare_spending else 0

st.markdown("""
<div class="section-heading">
    <div class="section-kicker">Tourism Journey</div>
    <div class="section-title">The story behind the numbers</div>
</div>
""", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">👥</div>
        <div class="metric-label">Visitors Journey</div>
        <div class="metric-value">{total_tourists:,.0f}</div>
        <div class="metric-note">Total tourists in selected view</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">💰</div>
        <div class="metric-label">Economic Impact</div>
        <div class="metric-value">{total_spending:,.0f}</div>
        <div class="metric-note">Tourism spending</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">🌙</div>
        <div class="metric-label">Average Stay</div>
        <div class="metric-value">{avg_stay:.2f}</div>
        <div class="metric-note">Nights per trip</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">📈</div>
        <div class="metric-label">Growth Momentum</div>
        <div class="metric-value">{tourist_growth:.1f}%</div>
        <div class="metric-note">vs comparison year</div>
    </div>
    """, unsafe_allow_html=True)

# =========================
# Destination Cards
# =========================

st.markdown("""
<div class="section-heading">
    <div class="section-kicker">Saudi Destinations</div>
    <div class="section-title">Which regions are leading tourism activity?</div>
</div>
""", unsafe_allow_html=True)

province_summary = (
    filtered_df.groupby("Province")[["Tourists_Number", "Tourists_Spending", "Avg_Stay"]]
    .agg({"Tourists_Number": "sum", "Tourists_Spending": "sum", "Avg_Stay": "mean"})
    .sort_values("Tourists_Number", ascending=False)
    .head(5)
    .reset_index()
)

if province_summary.empty:
    st.warning("No data available for the selected filters.")
else:
    dest_cols = st.columns(min(5, len(province_summary)))
    for idx, row in province_summary.iterrows():
        province = row["Province"]
        img = province_images.get(province, province_images["All"])
        with dest_cols[idx]:
            st.markdown(f"""
            <div class="destination-card">
                <div class="destination-image" style="background-image: url('{img}');"></div>
                <div class="destination-content">
                    <div class="destination-title">{province.replace('_', ' ')}</div>
                    <div class="destination-subtitle">Tourism Destination</div>
                    <div class="destination-stat">👥 {row['Tourists_Number']:,.0f} tourists</div>
                    <div class="destination-stat">💰 {row['Tourists_Spending']:,.0f} spending</div>
                    <div class="destination-stat">🌙 {row['Avg_Stay']:.2f} avg stay</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# =========================
# Storytelling Insights
# =========================

if not filtered_df.empty:
    most_visited = filtered_df.groupby("Province")["Tourists_Number"].sum().idxmax()
    highest_spending = filtered_df.groupby("Province")["Tourists_Spending"].sum().idxmax()
else:
    most_visited = "N/A"
    highest_spending = "N/A"

impact_message = "above" if avg_trip_spending > df["Avg_Spending_Trip"].mean() else "below"

st.markdown("""
<div class="section-heading">
    <div class="section-kicker">Data Story</div>
    <div class="section-title">What does the data tell us?</div>
</div>
""", unsafe_allow_html=True)

i1, i2, i3 = st.columns(3)

with i1:
    st.markdown(f"""
    <div class="insight-card">
        <div class="insight-icon">🏆</div>
        <div class="insight-title">Leading Destination</div>
        <div class="insight-text"><b>{most_visited}</b> attracts the highest visitor volume in the selected view.</div>
    </div>
    """, unsafe_allow_html=True)

with i2:
    st.markdown(f"""
    <div class="insight-card">
        <div class="insight-icon">💎</div>
        <div class="insight-title">Spending Power</div>
        <div class="insight-text"><b>{highest_spending}</b> generates the highest tourism spending among selected destinations.</div>
    </div>
    """, unsafe_allow_html=True)

with i3:
    st.markdown(f"""
    <div class="insight-card">
        <div class="insight-icon">🧭</div>
        <div class="insight-title">Visitor Value</div>
        <div class="insight-text">Average trip spending is <b>{impact_message}</b> the national dataset average.</div>
    </div>
    """, unsafe_allow_html=True)

# =========================
# Map & Analytics
# =========================

st.markdown("""
<div class="section-heading">
    <div class="section-kicker">Tourism Intelligence Map</div>
    <div class="section-title">Explore tourism activity across Saudi provinces</div>
</div>
""", unsafe_allow_html=True)

map_data = (
    filtered_df.groupby("Province")[["Tourists_Number", "Tourists_Spending"]]
    .sum()
    .reset_index()
)

map_data["Latitude"] = map_data["Province"].map(lambda x: province_coords.get(x, [None, None])[0])
map_data["Longitude"] = map_data["Province"].map(lambda x: province_coords.get(x, [None, None])[1])
map_data = map_data.dropna()

if not map_data.empty:
    fig_map = px.scatter_mapbox(
        map_data,
        lat="Latitude",
        lon="Longitude",
        size="Tourists_Number",
        color="Tourists_Spending",
        hover_name="Province",
        hover_data={"Tourists_Number": ":,.0f", "Tourists_Spending": ":,.0f"},
        color_continuous_scale="Viridis",
        zoom=3.8,
        height=560,
        mapbox_style="carto-darkmatter",
        title="Tourism Activity Across Saudi Arabia"
    )
    fig_map.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#F8FAFC",
        margin=dict(l=0, r=0, t=50, b=0)
    )
    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.info("No map data available for the selected filters.")

# =========================
# Charts as Story Sections
# =========================

st.markdown("""
<div class="section-heading">
    <div class="section-kicker">Growth Story</div>
    <div class="section-title">Tourism growth and spending patterns over time</div>
</div>
""", unsafe_allow_html=True)

chart_col1, chart_col2 = st.columns(2)

yearly_tourists = df.groupby("YEARS")["Tourists_Number"].sum().reset_index()
yearly_spending = df.groupby("YEARS")["Tourists_Spending"].sum().reset_index()

with chart_col1:
    fig = px.line(
        yearly_tourists,
        x="YEARS",
        y="Tourists_Number",
        markers=True,
        title="Visitors Growth Over Years",
        template="plotly_dark"
    )
    fig.update_traces(line=dict(width=4, color="#D4AF37"), marker=dict(size=9))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC")
    st.plotly_chart(fig, use_container_width=True)

with chart_col2:
    fig = px.line(
        yearly_spending,
        x="YEARS",
        y="Tourists_Spending",
        markers=True,
        title="Tourism Spending Over Years",
        template="plotly_dark"
    )
    fig.update_traces(line=dict(width=4, color="#22C55E"), marker=dict(size=9))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC")
    st.plotly_chart(fig, use_container_width=True)

chart_col3, chart_col4 = st.columns(2)

with chart_col3:
    tourism_type_data = (
        df[df["YEARS"] == selected_year]
        .groupby("Tourism_Type")["Tourists_Spending"]
        .sum()
        .reset_index()
    )
    fig = px.pie(
        tourism_type_data,
        values="Tourists_Spending",
        names="Tourism_Type",
        hole=0.58,
        title="Domestic vs Inbound Spending",
        template="plotly_dark"
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC")
    st.plotly_chart(fig, use_container_width=True)

with chart_col4:
    spending_by_region = (
        filtered_df.groupby("Province")["Tourists_Spending"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    fig = px.bar(
        spending_by_region,
        x="Tourists_Spending",
        y="Province",
        orientation="h",
        title="Top Spending Provinces",
        template="plotly_dark"
    )
    fig.update_traces(marker_color="#0EA5E9")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC")
    st.plotly_chart(fig, use_container_width=True)

# =========================
# Vision 2030 Forecast
# =========================

st.markdown("""
<div class="section-heading">
    <div class="section-kicker">Vision 2030 Outlook</div>
    <div class="section-title">Forecasting future tourism demand</div>
</div>
""", unsafe_allow_html=True)

forecast_years = pd.DataFrame({"YEARS": [2025, 2026, 2027, 2028, 2029, 2030]})
historical_yearly = df.groupby("YEARS")["Tourists_Number"].sum().reset_index()

forecast_model = LinearRegression()
forecast_model.fit(historical_yearly[["YEARS"]], historical_yearly["Tourists_Number"])
forecast_values = forecast_model.predict(forecast_years)

forecast_df = pd.DataFrame({"Year": forecast_years["YEARS"], "Predicted Tourists": forecast_values})

fig_forecast = px.line(
    forecast_df,
    x="Year",
    y="Predicted Tourists",
    markers=True,
    title="Tourism Demand Forecast Toward 2030",
    template="plotly_dark"
)
fig_forecast.update_traces(line=dict(width=5, color="#FDE68A"), marker=dict(size=10))
fig_forecast.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC")
st.plotly_chart(fig_forecast, use_container_width=True)

# =========================
# AI Tourism Advisor
# =========================

st.markdown("""
<div class="section-heading">
    <div class="section-kicker">AI Tourism Advisor</div>
    <div class="section-title">Estimate expected tourism spending</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="advisor-card">
    <div style="color:#FDE68A;font-weight:900;font-size:16px;margin-bottom:10px;">Tourism Spending Simulator</div>
    <div style="color:#E5E7EB;line-height:1.7;margin-bottom:20px;">
        Enter tourism indicators below to estimate expected spending and understand the economic value of a visitor profile.
    </div>
""", unsafe_allow_html=True)

with st.form("tourism_prediction_form"):
    p1, p2, p3 = st.columns(3)
    with p1:
        tourists_number = st.number_input("Tourists Number", min_value=0.0, value=1000.0)
        overnight_stay = st.number_input("Overnight Stay", min_value=0.0, value=5000.0)
    with p2:
        avg_stay_input = st.number_input("Average Stay", min_value=0.0, value=5.0)
        avg_spending_trip = st.number_input("Average Spending Per Trip", min_value=0.0, value=1500.0)
    with p3:
        avg_spending_night = st.number_input("Average Spending Per Night", min_value=0.0, value=300.0)
        submitted_prediction = st.form_submit_button("✨ Estimate Tourism Spending")

st.markdown("</div>", unsafe_allow_html=True)

if submitted_prediction:
    input_data = np.array([[
        tourists_number,
        overnight_stay,
        avg_stay_input,
        avg_spending_trip,
        avg_spending_night
    ]])
    prediction = model.predict(input_data)[0]
    impact_label = "Above Average Impact" if prediction > df["Tourists_Spending"].mean() else "Moderate Impact"

    st.markdown(f"""
    <div class="prediction-result">
        <div class="prediction-label">Expected Visitor Spending</div>
        <div class="prediction-value">{prediction:,.2f}</div>
        <div style="color:#FDE68A;font-size:18px;font-weight:900;margin-top:8px;">{impact_label}</div>
        <div style="color:#E5E7EB;line-height:1.7;margin-top:12px;">
            This tourism profile can help estimate expected contribution to regional tourism revenue and visitor value.
        </div>
    </div>
    """, unsafe_allow_html=True)

# =========================
# Executive Summary & Report
# =========================

st.markdown("""
<div class="section-heading">
    <div class="section-kicker">Executive Summary</div>
    <div class="section-title">Strategic tourism recommendations</div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="insight-card">
    <div class="insight-title">Strategic Finding</div>
    <div class="insight-text">
        Tourism activity is strongest in <b>{most_visited}</b>, while <b>{highest_spending}</b> leads economic contribution in the selected view.
        Decision makers can use these patterns to prioritize infrastructure, visitor experience, and destination development.
    </div>
</div>
""", unsafe_allow_html=True)


def create_pdf_report():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Saudi Tourism Intelligence Report", ln=True, align="C")
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Selected Year: {selected_year}", ln=True)
    pdf.cell(200, 10, txt=f"Tourism Type: {selected_type}", ln=True)
    pdf.cell(200, 10, txt=f"Province: {selected_province}", ln=True)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Total Tourists: {total_tourists:,.0f}", ln=True)
    pdf.cell(200, 10, txt=f"Total Spending: {total_spending:,.0f}", ln=True)
    pdf.cell(200, 10, txt=f"Average Stay: {avg_stay:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"Average Trip Spending: {avg_trip_spending:,.0f}", ln=True)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Most Visited Province: {most_visited}", ln=True)
    pdf.cell(200, 10, txt=f"Highest Spending Province: {highest_spending}", ln=True)
    return pdf.output(dest="S").encode("latin1")

pdf_report = create_pdf_report()

st.download_button(
    label="📥 Download Tourism Intelligence Report",
    data=pdf_report,
    file_name="saudi_tourism_intelligence_report.pdf",
    mime="application/pdf"
)

# =========================
# Data Explorer
# =========================

st.markdown("""
<div class="section-heading">
    <div class="section-kicker">Data Explorer</div>
    <div class="section-title">Filtered tourism records</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="data-card">
    <div style="color:#E5E7EB;line-height:1.7;margin-bottom:14px;">
        Review the filtered dataset used for the analysis. This keeps the analytical foundation visible while the main experience focuses on storytelling and decision support.
    </div>
""", unsafe_allow_html=True)

st.dataframe(filtered_df, use_container_width=True, height=420)

csv = filtered_df.to_csv(index=False)
st.download_button(
    label="Download Filtered Tourism Data",
    data=csv,
    file_name="filtered_tourism_data.csv",
    mime="text/csv"
)

st.markdown("</div>", unsafe_allow_html=True)

# =========================
# Footer
# =========================

st.markdown("""
<div class="footer">
    Saudi Tourism Intelligence Experience © 2026<br>
    Data storytelling, forecasting, and AI-powered tourism decision support.
    
            Developed by Taghreed Mohammed
            </div>
""", unsafe_allow_html=True)
