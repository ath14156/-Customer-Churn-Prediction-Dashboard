import streamlit as st
import plotly.express as px

from utils.data_loader import load_data

st.set_page_config(
    page_title="Dashboard",
    page_icon="📊",
    layout="wide"
)

df = load_data()

st.title("📊 Dashboard")

st.write("Customer churn overview.")

# =============================
# Metrics
# =============================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Customers",
    len(df)
)

col2.metric(
    "Features",
    df.shape[1]
)

col3.metric(
    "Missing Values",
    df.isnull().sum().sum()
)

churn_rate = (df["Churn"] == "Yes").mean() * 100

col4.metric(
    "Churn Rate",
    f"{churn_rate:.1f}%"
)

st.divider()

# =============================
# Churn Distribution
# =============================

churn = df["Churn"].value_counts().reset_index()

churn.columns = [
    "Churn",
    "Customers"
]

fig = px.bar(
    churn,
    x="Churn",
    y="Customers",
    color="Churn",
    text="Customers",
    title="Customer Churn Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)
