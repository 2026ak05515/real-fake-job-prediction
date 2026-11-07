import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

st.title("Fake Job Posting Detection - Model Evaluation")

st.markdown("""
Upload a test dataset to evaluate and compare:

- Decision Tree
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Multinomial Naive Bayes
- Support Vector Machine (SVM)

using Accuracy, AUC, Precision, Recall, F1 Score, and MCC.
""")

# Load Models
models = {
    "Decision Tree": "decision_tree_model.pkl",
    "Logistic Regression": "logistic_regression_model.pkl",
    "KNN": "kNN_model.pkl",
    "Naive Bayes": "naive_bayes_multinomial_model.pkl",
    "SVM": "support_vector_machine_model.pkl"
}

# Upload Test File
uploaded_file = st.file_uploader(
    "Upload Test CSV",
    type=["csv"]
)


def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }


if uploaded_file is not None:

    test_df = pd.read_csv(uploaded_file)

    st.success("Test file uploaded successfully.")

    st.write(
    f"Dataset Shape: {test_df.shape[0]} rows × {test_df.shape[1]} columns"
    )

    # Load first model only to retrieve column lists
    first_model = next(iter(models.values()))

    saved = joblib.load(first_model)

    text_cols = saved["text_cols"]
    cat_cols = saved["cat_cols"]
    num_cols = saved["num_cols"]

    # Validate required columns
    required_cols = (
        text_cols +
        cat_cols +
        num_cols +
        ["fraudulent"]
    )

    missing_cols = [
        col for col in required_cols
        if col not in test_df.columns
    ]

    if missing_cols:
        st.error(
            f"Missing required columns: {missing_cols}"
        )
        st.stop()

    # Apply cleaning
    for col in text_cols:
        test_df[col] = (
            test_df[col]
            .fillna("")
            .astype(str)
        )

    for col in cat_cols:
        test_df[col] = (
            test_df[col]
            .fillna("Unspecified")
            .astype(str)
        )

    for col in num_cols:
        test_df[col] = (
            test_df[col]
            .fillna(0)
            .astype(int)
        )

    # Features and Target
    X_test = test_df[
        text_cols + cat_cols + num_cols
    ]

    y_test = test_df["fraudulent"]

    results = []

    # Evaluate all models
    for model_name, model_file in models.items():

        saved = joblib.load(model_file)

        model = saved["model"]

        metrics = evaluate_model(
            model,
            X_test,
            y_test
        )

        metrics["Model"] = model_name

        results.append(metrics)

    # Results DataFrame
    results_df = pd.DataFrame(results)

    results_df = results_df[
        [
            "Model",
            "Accuracy",
            "AUC",
            "Precision",
            "Recall",
            "F1 Score",
            "MCC"
        ]
    ]

    results_df = results_df.sort_values(
        by="F1 Score",
        ascending=False
    )
    
    st.subheader("Model Comparison Results")

    st.dataframe(
        results_df.style.format({
            "Accuracy": "{:.4f}",
            "AUC": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1 Score": "{:.4f}",
            "MCC": "{:.4f}"
        })
    )

    # Best model based on F1 Score
    best_model = results_df.loc[
        results_df["F1 Score"].idxmax(),
        "Model"
    ]

    st.success(
        f"Best Model (Highest F1 Score): {best_model}"
    )
