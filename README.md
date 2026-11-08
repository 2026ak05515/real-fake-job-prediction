# real-fake-job-prediction
# a. Problem statement
With the rapid expansion of digital recruitment platforms, online job scams and fraudulent job postings have proliferated, posing severe risks to job seekers ranging from financial fraud and identity theft to compromised personal data. Traditional manual methods are slow, subjective and unable to scale against automated fraudulent postings. This project aims to build an automated, end-to-end machine learning classification system capable of accurately distinguishing between legitimate and fraudulent job postings by analyzing textual descriptions, categorical metadata and binary indicators. By deploying this solution as an interactive web application, the system assists candidates and platform moderators instantly evaluating job authenticity and mitigating recruitment fraud.

# b. Dataset Description

| Item | Value |
| :--- | :--- |
| Source | https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction/ |
| Features | 18 |
| Target | 1 (fraudulent) |
| Instances | 17880 |

[Real or Fake] - Job Description Prediction :
This dataset contains nearly 18K job descriptions out of which about 800 are fake. The data consists of both textual information and meta-information about the jobs. The dataset can be used to create classification models which can learn the job descriptions which are fraudulent.

Feature Engineering & Preprocessing Pipeline: To handle the heterogeneous mix of unstructured text, categorical metadata and binary flags, a custom ColumnTransformer pipeline was implemented.

Text Columns (text_cols): Processed using TfidfVectorizer with custom maximum features and English stop-word removal. Features involved are title, description, requirements, company_profile, benefits.
Categorical Columns (cat_cols): Processed using OneHotEncoder ignoring unknown categories and filtering with min_frequency. Features involved are location, department, employment_type, required_experience, required_education, industry, function.
Numerical/Binary Columns (num_cols): Passed through directly without scaling. Features involved are telecommuting, has_company_logo, has_questions.

Also, below features are dropped as part of pre-processing,

job_id (Unique Identifier): Correctly dropping high-cardinality ID columns prevents the model from memorizing arbitrary row identifiers, avoiding overfitting and noise.
salary_range (High Missingness): An 83.96% missing rate (15,012 out of 17,880 rows) makes imputation unreliable and risky, as imputing that much synthetic data would heavily bias the distribution. If 70-90% of rows are missing, the feature contributes very little to the model. Due to large amount of missing data, it was dropped.

# c. Github Repository Link
https://github.com/2026ak05515/real-fake-job-prediction

Files in the repository:
* app.py: the Streamlit app, which loads the saved models and never retrains
* requirements.txt
* test_data.csv
* model/: All five models .pkl files
* train/: All five models training and split_train_test source code .ipynb files


# d. Models used
Comparison Table with the evaluation metrics calculated for all the 5 models is shown below:
| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.9815 | 0.9873 | 0.7725 | 0.9006 | 0.8316 | 0.8246 |
| Decision Tree | 0.9636 | 0.9169 | 0.6123 | 0.7680 | 0.6814 | 0.6671 |
| KNN | 0.9326 | 0.9611 | 0.4246 | 0.9337 | 0.5838 | 0.6038 |
| Naive Bayes - Multinomial | 0.9497 | 0.9796 | 0.5016 | 0.8895 | 0.6414 | 0.6460 |
| Support Vector Machine | 0.9835 | 0.9626 | 0.8466 | 0.8232 | 0.8347 | 0.8261 |


Observations on the performance of each model:
| ML Model Name | Observation about model performance |
| :--- | :--- |
| Logistic Regression | Achieved a very high AUC (0.9873) and strong recall (0.9006), proving exceptionally effective at catching fraudulent postings. However, its lower precision (0.7725) indicates a higher rate of false positives on legitimate listings. |
| Decision Tree | Demonstrated solid performance overall (F1 Score: 0.6814, MCC: 0.6671). While it captures basic decision boundaries, it generalizes less effectively on high-dimensional sparse TF-IDF and categorical features compared to linear models. |
| KNN | Achieved the highest recall (0.9337) in the experiment, capturing almost all fraudulent instances. However, its lower precision (0.4246) and F1-Score (0.5838) show that distance-based metrics struggle in high-dimensional text vector spaces, leading to increased false alarms. |
| Naive Bayes - Multinomial | Delivered a high AUC (0.9796) and strong recall (0.8895), but suffered from lower precision (0.5016) and F1 Score (0.6414), indicating susceptibility to class imbalance and feature frequency variations in job descriptions. |
| Support Vector Machine | Best balanced overall performer. Achieved the highest F1 Score (0.8347), highest accuracy (0.9835), highest precision (0.8466), and highest MCC (0.8261) combined with robust recall (0.8232), proving ideal for separating high-dimensional text boundaries. |
| Overall, Winner for your dataset? | Support Vector Machine (SVM), closely followed by Logistic Regression. SVM takes the top spot due to its superior balance of precision and recall (highest F1-score and MCC), successfully minimizing false alarms while reliably catching fraudulent job postings. |
