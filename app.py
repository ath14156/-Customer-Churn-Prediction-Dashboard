import streamlit as st

st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction Dashboard")

st.markdown("""
# Welcome 👋

This project demonstrates an **end-to-end Machine Learning workflow**
for predicting customer churn.

---

## 📌 Project Features

✅ Interactive Dashboard

✅ Data Exploration

✅ Data Preprocessing

✅ Machine Learning Models

✅ Model Evaluation

✅ Customer Churn Prediction

---

## 📈 Business Problem

Customer churn occurs when customers stop using a company's services.

The goal of this project is to:

- Analyze customer behavior
- Identify churn patterns
- Train machine learning models
- Predict customers likely to leave
- Help businesses improve customer retention

---

### 👈 Use the navigation menu on the left to explore each section.
""")

st.success("Built with Python • Streamlit • Scikit-Learn • Plotly")
