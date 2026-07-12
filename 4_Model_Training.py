import streamlit as st
import pandas as pd

from utils.data_loader import load_data
from models.model_trainer import train_model

st.set_page_config(
    page_title="Model Training",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Machine Learning Model Training")

st.write("""
Train different machine learning models and compare their performance.
""")

df = load_data()

model_name = st.selectbox(
    "Choose a Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting"
    ]
)

if st.button("🚀 Train Model"):

    with st.spinner("Training model..."):

        results = train_model(model_name, df)

    st.success(f"{model_name} trained successfully!")

    col1, col2 = st.columns(2)

    col1.metric(
        "Accuracy",
        f"{results['accuracy']:.3f}"
    )

    col2.metric(
        "F1 Score",
        f"{results['f1']:.3f}"
    )

    col1.metric(
        "Precision",
        f"{results['precision']:.3f}"
    )

    col2.metric(
        "Recall",
        f"{results['recall']:.3f}"
    )

    st.subheader("Confusion Matrix")

    cm = pd.DataFrame(
        results["confusion_matrix"],
        index=["Actual No", "Actual Yes"],
        columns=["Predicted No", "Predicted Yes"]
    )

    st.dataframe(
        cm,
        use_container_width=True
    )
