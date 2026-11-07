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

[Real or Fake] : Job Description Prediction
This dataset contains nearly 18K job descriptions out of which about 800 are fake. The data consists of both textual information and meta-information about the jobs. The dataset can be used to create classification models which can learn the job descriptions which are fraudulent.

# c. Github Repository Link
https://github.com/2026ak05515/real-fake-job-prediction

Files in the repository:
* app.py: the Streamlit app, which loads the saved models and never retrains
* requirements.txt
* test_data.csv
* model/: All five models saved as .pkl pipelines

# d. Models used
| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Support Vector Machine | 0.9832214 | 0.9626188 | 0.8418079 | 0.8232044 | 0.8324022 | 0.8236290 |
| Logistic Regression | 0.9815436 | 0.987316 | 0.7725118 | 0.9005524 | 0.8316326 | 0.8246355 |
| Decision Tree | 0.9678411 | 0.9141628 | 0.6601941 | 0.7513812 | 0.7028423 | 0.6875225 |
| Naive Bayes | 0.9499440 | 0.9796190 | 0.503125 | 0.8895027 | 0.6427145 | 0.6471412 |
| KNN | 0.9312080 | 0.9610851 | 0.4189526 | 0.9281767| 0.5773195 | 0.597151 |
