import streamlit as st
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from utils.data_loader import load_data

st.set_page_config(
    page_title="Data Preprocessing",
    page_icon="🧹",
    layout="wide"
)

st.title("🧹 Data Preprocessing")

st.write("""
This page demonstrates how the raw customer data is cleaned and prepared
before training machine learning models.
""")

# =====================================
# Load Data
# =====================================

df = load_data()

# =====================================
# Original Dataset
# =====================================

st.header("1️⃣ Original Dataset")

st.dataframe(
    df.head(),
    use_container_width=True
)

st.write(f"Rows: {df.shape[0]}")
st.write(f"Columns: {df.shape[1]}")

st.divider()

# =====================================
# Missing Values
# =====================================

st.header("2️⃣ Missing Values")

missing = pd.DataFrame({
    "Column": df.columns,
    "Missing Values": df.isnull().sum()
})

st.dataframe(
    missing,
    use_container_width=True
)

st.divider()

# =====================================
# Data Types
# =====================================

st.header("3️⃣ Data Types")

types = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str)
})

st.dataframe(
    types,
    use_container_width=True
)

st.divider()

# =====================================
# Encode Categorical Features
# =====================================

st.header("4️⃣ Label Encoding")

encoded_df = df.copy()

encoder = LabelEncoder()

for column in encoded_df.columns:

    if encoded_df[column].dtype == "object":

        encoded_df[column] = encoder.fit_transform(
            encoded_df[column].astype(str)
        )

st.success("Categorical features encoded successfully.")

st.dataframe(
    encoded_df.head(),
    use_container_width=True
)

st.divider()

# =====================================
# Feature Matrix
# =====================================

st.header("5️⃣ Features and Target")

X = encoded_df.drop("Churn", axis=1)

y = encoded_df["Churn"]

col1, col2 = st.columns(2)

with col1:

    st.subheader("Features (X)")

    st.write(X.shape)

with col2:

    st.subheader("Target (y)")

    st.write(y.shape)

st.divider()

# =====================================
# Final Dataset
# =====================================

st.header("6️⃣ Ready for Machine Learning")

st.success(
    "The dataset has been cleaned and encoded. It is now ready for model training."
)

st.dataframe(
    encoded_df.head(),
    use_container_width=True
)
