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
| Support Vector Machine | 0.9832214[cite: 2] | 0.9626188[cite: 2] | 0.8418079[cite: 2] | 0.8232044[cite: 2] | 0.8324022[cite: 2] | 0.8236290[cite: 2] |
| Logistic Regression[cite: 2] | 0.9815436[cite: 2] | 0.987316[cite: 2] | 0.7725118[cite: 2] | 0.9005524[cite: 2] | 0.8316326[cite: 2] | 0.8246355[cite: 2] |
| Decision Tree[cite: 2] | 0.9678411[cite: 2] | 0.9141628[cite: 2] | 0.6601941[cite: 2] | 0.7513812[cite: 2] | 0.7028423[cite: 2] | 0.6875225[cite: 2] |
| Naive Bayes[cite: 2] | 0.9499440[cite: 2] | 0.9796190[cite: 2] | 0.503125[cite: 2] | 0.8895027[cite: 2] | 0.6427145[cite: 2] | 0.6471412[cite: 2] |
| KNN[cite: 2] | 0.9312080[cite: 2] | 0.9610851[cite: 2] | 0.4189526[cite: 2] | 0.9281767[cite: 2] | 0.5773195[cite: 2] | 0.597151[cite: 2] |
