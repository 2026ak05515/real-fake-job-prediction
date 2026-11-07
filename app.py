import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.title("Fake Job Posting Detection - Model Evaluation")

st.markdown("""
Upload your test dataset to evaluate and compare machine learning classification models:
- **Logistic Regression**
- **Decision Tree Classifier**
- **K-Nearest Neighbors (KNN)**
- **Naive Bayes Classifier**
- **Support Vector Machine (SVM)**
""")

# Define Models dictionary (pointing to your saved pkl files)
models = {
    "Logistic Regression": "logistic_regression_model.pkl",
    "Decision Tree": "decision_tree_model.pkl",
    "KNN": "kNN_model.pkl",
    "Naive Bayes": "naive_bayes_multinomial_model.pkl",
    "Support Vector Machine": "support_vector_machine_model.pkl"
}

# Requirement a: Dataset upload option (CSV)
uploaded_file = st.file_uploader("Upload Test CSV (with features and target 'fraudulent')", type=["csv"])

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    st.success("Test file uploaded successfully.")
    st.dataframe(test_df.head())
    st.write(f"**Dataset Shape:** {test_df.shape[0]} rows × {test_df.shape[1]} columns")

    # Load first model to dynamically retrieve required feature columns
    try:
        first_model_file = next(iter(models.values()))
        saved_artifact = joblib.load(first_model_file)
        text_cols = saved_artifact["text_cols"]
        cat_cols = saved_artifact["cat_cols"]
        num_cols = saved_artifact["num_cols"]
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        st.stop()

    # Validate required columns
    required_cols = text_cols + cat_cols + num_cols + ["fraudulent"]
    missing_cols = [col for col in required_cols if col not in test_df.columns]

    if missing_cols:
        st.error(f"Missing required columns in uploaded CSV: {missing_cols}")
        st.stop()

    # Apply preprocessing cleaning logic
    for col in text_cols:
        test_df[col] = test_df[col].replace({None: ""}).fillna("").astype(str)
    for col in cat_cols:
        test_df[col] = test_df[col].replace({None: "Unspecified"}).fillna("Unspecified").astype(str)
    for col in num_cols:
        test_df[col] = test_df[col].fillna(0).astype(int)

    # Features and Target
    X_test = test_df[text_cols + cat_cols + num_cols]
    y_test = test_df["fraudulent"]

    # Evaluate all models for the comparison table
    results = []
    trained_models = {}

    for model_name, model_file in models.items():
        try:
            saved = joblib.load(model_file)
            model = saved["model"]
            trained_models[model_name] = model

            y_pred = model.predict(X_test)
            
            # Handle probabilities safely for AUC
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
                auc_val = roc_auc_score(y_test, y_proba)
            except (AttributeError, IndexError):
                auc_val = float('nan')

            results.append({
                "Model": model_name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "AUC": auc_val,
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1 Score": f1_score(y_test, y_pred, zero_division=0),
                "MCC": matthews_corrcoef(y_test, y_pred)
            })
        except Exception as e:
            st.warning(f"Could not evaluate {model_name}: {e}")

    if results:
        results_df = pd.DataFrame(results).sort_values(by="F1 Score", ascending=False)

        # Best model based on F1 Score
        winner = results_df.loc[
            results_df["F1 Score"].idxmax()
        ]

        st.success(
            f"🏆 Best Model: {winner['Model']} "
            f"(F1 Score = {winner['F1 Score']:.4f})"
        )
        
        st.subheader("📊 Overall Model Comparison")
        st.dataframe(
            results_df.style.format({
                "Accuracy": "{:.4f}",
                "AUC": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1 Score": "{:.4f}",
                "MCC": "{:.4f}"
            }),
            use_container_width=True
        )

        st.markdown("---")

        # Requirement b: Model selection dropdown (if multiple models)
        st.subheader("🔍 Individual Model Inspector")
        selected_model_name = st.selectbox(
            "Select a model to view detailed evaluation metrics and confusion matrix:",
            list(models.keys())
        )

        if selected_model_name in trained_models:
            selected_model = trained_models[selected_model_name]
            y_pred_sel = selected_model.predict(X_test)

            # Requirement c: Display of evaluation metrics
            st.markdown(f"### Performance Metrics for: **{selected_model_name}**")
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            
            sel_metrics = [r for r in results if r["Model"] == selected_model_name][0]
            m1.metric("Accuracy", f"{sel_metrics['Accuracy']:.4f}")
            m2.metric("AUC", f"{sel_metrics['AUC']:.4f}")
            m3.metric("Precision", f"{sel_metrics['Precision']:.4f}")
            m4.metric("Recall", f"{sel_metrics['Recall']:.4f}")
            m5.metric("F1 Score", f"{sel_metrics['F1 Score']:.4f}")
            m6.metric("MCC", f"{sel_metrics['MCC']:.4f}")

            # Requirement d: Confusion matrix or classification report
            col_cm, col_cr = st.columns(2)

            with col_cm:
                st.markdown("#### Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred_sel)
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
                ax.set_xlabel("Predicted Label")
                ax.set_ylabel("True Label")
                st.pyplot(fig)

            with col_cr:
                st.markdown("#### Classification Report")
                report_dict = classification_report(y_test, y_pred_sel, output_dict=True, zero_division=0)
                report_df = pd.DataFrame(report_dict).transpose()
                st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)
