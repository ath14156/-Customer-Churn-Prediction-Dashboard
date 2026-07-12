import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)

from utils.data_loader import load_data


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Model Evaluation",
    page_icon="📉",
    layout="wide"
)

st.title("📉 Model Evaluation")

st.write(
    "Evaluate machine learning models using multiple performance metrics."
)

# ==========================================
# Load Dataset
# ==========================================

df = load_data()

data = df.copy()

# Remove customer ID
if "customerID" in data.columns:
    data.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges
if "TotalCharges" in data.columns:
    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"],
        errors="coerce"
    )

# Remove missing values
data.dropna(inplace=True)

# Target
y = data["Churn"]

# Encode target
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Features
X = data.drop("Churn", axis=1)

# One-hot encode categorical features
X = pd.get_dummies(
    X,
    drop_first=True,
    dtype=int
)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================================
# Select Model
# ==========================================

model_name = st.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
    ],
)

if model_name == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)

elif model_name == "Decision Tree":
    model = DecisionTreeClassifier(random_state=42)

elif model_name == "Random Forest":
    model = RandomForestClassifier(random_state=42)

else:
    model = GradientBoostingClassifier(random_state=42)

# ==========================================
# Train Model
# ==========================================

model.fit(X_train, y_train)

predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)[:, 1]

# ==========================================
# Classification Report
# ==========================================

st.header("Classification Report")

report = classification_report(
    y_test,
    predictions,
    output_dict=True,
)

st.dataframe(
    pd.DataFrame(report).transpose(),
    use_container_width=True,
)

st.divider()

# ==========================================
# Confusion Matrix
# ==========================================

st.header("Confusion Matrix")

fig, ax = plt.subplots(figsize=(5, 5))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    ax=ax,
)

st.pyplot(fig)

st.divider()

# ==========================================
# ROC Curve
# ==========================================

st.header("ROC Curve")

fpr, tpr, _ = roc_curve(
    y_test,
    probabilities,
)

auc = roc_auc_score(
    y_test,
    probabilities,
)

fig, ax = plt.subplots(figsize=(6, 5))

ax.plot(
    fpr,
    tpr,
    label=f"AUC = {auc:.3f}",
)

ax.plot([0, 1], [0, 1], "--")

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")

ax.legend()

st.pyplot(fig)

st.metric(
    "ROC AUC Score",
    f"{auc:.3f}",
)
