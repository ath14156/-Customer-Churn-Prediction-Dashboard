import streamlit as st
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder

from utils.data_loader import load_data

st.set_page_config(
    page_title="Customer Prediction",
    page_icon="🔮",
    layout="wide"
)

st.title("🔮 Customer Churn Prediction")

st.write("""
Enter customer information to predict whether the customer is likely to churn.
""")

df = load_data()

# -----------------------------
# Encode Training Data
# -----------------------------

data = df.copy()

encoders = {}

for column in data.columns:

    if data[column].dtype == "object":

        encoder = LabelEncoder()

        data[column] = encoder.fit_transform(
            data[column].astype(str)
        )

        encoders[column] = encoder

# -----------------------------
# Load Model
# -----------------------------

try:

    model = joblib.load(
        "models/customer_churn_model.pkl"
    )

except:

    st.error(
        "No trained model found.\n\nPlease train a model first."
    )

    st.stop()

# -----------------------------
# User Input
# -----------------------------

st.header("Customer Information")

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

senior = st.selectbox(
    "Senior Citizen",
    [0,1]
)

partner = st.selectbox(
    "Partner",
    ["Yes","No"]
)

dependents = st.selectbox(
    "Dependents",
    ["Yes","No"]
)

tenure = st.slider(
    "Tenure",
    0,
    72,
    12
)

monthly = st.slider(
    "Monthly Charges",
    20.0,
    120.0,
    70.0
)

contract = st.selectbox(
    "Contract",
    df["Contract"].unique()
)

internet = st.selectbox(
    "Internet Service",
    df["InternetService"].unique()
)

payment = st.selectbox(
    "Payment Method",
    df["PaymentMethod"].unique()
)

# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict"):

    customer = pd.DataFrame({

        "gender":[gender],
        "SeniorCitizen":[senior],
        "Partner":[partner],
        "Dependents":[dependents],
        "tenure":[tenure],
        "PhoneService":["Yes"],
        "MultipleLines":["No"],
        "InternetService":[internet],
        "OnlineSecurity":["No"],
        "OnlineBackup":["No"],
        "DeviceProtection":["No"],
        "TechSupport":["No"],
        "StreamingTV":["No"],
        "StreamingMovies":["No"],
        "Contract":[contract],
        "PaperlessBilling":["Yes"],
        "PaymentMethod":[payment],
        "MonthlyCharges":[monthly],
        "TotalCharges":[monthly * tenure]
    })

    customer["customerID"] = "NEW"

    customer = customer[df.drop("Churn", axis=1).columns]

    for col in customer.columns:

        if col in encoders:

            customer[col] = encoders[col].transform(
                customer[col].astype(str)
            )

    prediction = model.predict(customer)[0]

    probability = model.predict_proba(customer)[0][1]

    st.divider()

    if prediction == 1:

        st.error("⚠️ Customer is likely to churn.")

    else:

        st.success("✅ Customer is likely to stay.")

    st.metric(
        "Churn Probability",
        f"{probability*100:.1f}%"
    )

    st.subheader("Business Recommendation")

    if probability > 0.70:

        st.warning(
            "Offer discounts or long-term contracts to improve customer retention."
        )

    elif probability > 0.40:

        st.info(
            "Customer is medium risk. Consider loyalty rewards."
        )

    else:

        st.success(
            "Customer is low risk."
        )
