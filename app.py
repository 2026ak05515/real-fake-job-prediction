import streamlit as st
import pandas as pd
import joblib

#Load Model
models = {
    "Decision Tree": "decision_tree_model.pkl",
    "Logistic Regression": "logistic_regression_model.pkl",
    "KNN": "kNN_model.pkl",
    "Naive Bayes": "naive_bayes_multinomial_model.pkl",
    "SVM": "support_vector_machine_model.pkl"
}

#dropdown list
selected_model = st.selectbox(
    "Select Model",
    list(models.keys())
)


#Upload Test File
uploaded_file = st.file_uploader(
    "Upload Test CSV",
    type=["csv"]
)

#Load Uploaded File
if uploaded_file:

    test_df = pd.read_csv(uploaded_file)

    saved = joblib.load(
        models[selected_model]
    )

    model = saved["model"]

    text_cols = saved["text_cols"]
    cat_cols = saved["cat_cols"]
    num_cols = saved["num_cols"]

#Apply cleaning
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

#Evaluate the Model
X_test = test_df[text_cols + cat_cols + num_cols]

y_pred = model.predict(X_test)

test_df["Prediction"] = y_pred


