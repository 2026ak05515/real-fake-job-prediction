import streamlit as st
import pandas as pd
import joblib

#Load Model
model = joblib.load(
    "decision_tree_model.pkl"
)

#Upload Test File
uploaded_file = st.file_uploader(
    "Upload Test CSV",
    type=["csv"]
)

#Read Uploaded File
if uploaded_file:

    test_df = pd.read_csv(
        uploaded_file
    )

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

predictions = model.predict(X_test)

test_df["Prediction"] = predictions
