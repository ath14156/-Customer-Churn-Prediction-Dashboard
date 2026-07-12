import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ==========================================
# Prepare Data
# ==========================================
def prepare_data(df):

    data = df.copy()

    # Remove Customer ID
    if "customerID" in data.columns:
        data = data.drop(columns=["customerID"])

    # Convert TotalCharges to numeric
    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(
            data["TotalCharges"],
            errors="coerce"
        )

    # Remove missing values
    data = data.dropna()

    # Separate target and features
    X = data.drop(columns=["Churn"])
    y = data["Churn"]

    # Encode target (Yes/No → 1/0)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    # One-hot encode categorical features
    X = pd.get_dummies(
        X,
        drop_first=True,
        dtype=int
    )

    # Split dataset
    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )


# ==========================================
# Train Model
# ==========================================
def train_model(model_name, df):

    X_train, X_test, y_train, y_test = prepare_data(df)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")

    model = models[model_name]

    # Train
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Save trained model
    joblib.dump(model, "models/customer_churn_model.pkl")

    # Return evaluation metrics
    return {
        "model": model,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions),
    }


# ==========================================
# Train & Save Best Model
# ==========================================
def train_and_save_best_model(df):
    return train_model("Random Forest", df)
