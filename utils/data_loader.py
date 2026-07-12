import pandas as pd
import streamlit as st


@st.cache_data
def load_data():

    df = pd.read_csv("customer_churn.csv")

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    return df
