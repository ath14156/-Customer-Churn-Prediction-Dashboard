import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_loader import load_data

st.set_page_config(
    page_title="Data Exploration",
    page_icon="📈",
    layout="wide"
)

df = load_data()

st.title("📈 Data Exploration")

st.write(
    """
Explore the customer churn dataset through summary statistics,
feature distributions, and interactive filters.
"""
)

# =====================================
# Sidebar Filters
# =====================================

st.sidebar.header("Filters")

gender = st.sidebar.multiselect(
    "Gender",
    df["gender"].unique(),
    default=df["gender"].unique()
)

contract = st.sidebar.multiselect(
    "Contract",
    df["Contract"].unique(),
    default=df["Contract"].unique()
)

filtered_df = df[
    (df["gender"].isin(gender)) &
    (df["Contract"].isin(contract))
]

# =====================================
# Dataset Preview
# =====================================

st.header("Dataset Preview")

st.dataframe(
    filtered_df,
    use_container_width=True
)

# =====================================
# Dataset Information
# =====================================

st.header("Dataset Information")

info = pd.DataFrame({
    "Column": filtered_df.columns,
    "Data Type": filtered_df.dtypes.astype(str),
    "Missing Values": filtered_df.isnull().sum().values
})

st.dataframe(
    info,
    use_container_width=True
)

# =====================================
# Summary Statistics
# =====================================

st.header("Summary Statistics")

st.dataframe(
    filtered_df.describe(include="all"),
    use_container_width=True
)

# =====================================
# Missing Values
# =====================================

st.header("Missing Values")

missing = pd.DataFrame({
    "Column": filtered_df.columns,
    "Missing": filtered_df.isnull().sum()
})

st.dataframe(
    missing,
    use_container_width=True
)

# =====================================
# Feature Distribution
# =====================================

st.header("Monthly Charges Distribution")

fig = px.histogram(
    filtered_df,
    x="MonthlyCharges",
    nbins=30,
    title="Monthly Charges"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.header("Customer Tenure Distribution")

fig = px.histogram(
    filtered_df,
    x="tenure",
    nbins=30,
    title="Customer Tenure"
)

st.plotly_chart(
    fig,
    use_container_width=True
)
