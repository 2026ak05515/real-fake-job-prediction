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
