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
Comparison Table with the evaluation metrics calculated for all the 5 models is shown below:
| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.9815 | 0.9873 | 0.7725 | 0.9006 | 0.8316 | 0.8246 |
| Decision Tree | 0.9678 | 0.9142 | 0.6602 | 0.7514 | 0.7028 | 0.6875 |
| KNN | 0.9312 | 0.9611 | 0.4190 | 0.9282 | 0.5773 | 0.5972 |
| Naive Bayes | 0.9499 | 0.9796 | 0.5031 | 0.8895 | 0.6427 | 0.6471 |
| Support Vector Machine | 0.9832 | 0.9626 | 0.8418 | 0.8232 | 0.8324 | 0.8236 |

Observations on the performance of each model:

| ML Model Name | Observation about model performance |
| :--- | :--- |
| Logistic Regression | Achieved a very high AUC (0.9873) and strong recall (0.9006), proving exceptionally effective at catching fraudulent postings. However, its lower precision (0.7725) indicates a higher rate of false positives on legitimate listings. |
| Decision Tree | Demonstrated moderate performance across all metrics (F1 Score: 0.7028, MCC: 0.6875). While it captures basic rules, it struggles with generalization on high-dimensional sparse TF-IDF and categorical features compared to linear models. |
| KNN | Achieved the highest recall (0.9282) in the experiment, capturing almost all fraudulent instances. However, its poor precision (0.4190) and lowest F1-Score (0.5773) show that distance-based metrics suffer severely in high-dimensional text vector spaces, creating massive false alarms. |
| Naive Bayes | Delivered a high AUC (0.9796) and strong recall (0.8895), but suffered from very low precision (0.5031) and lower F1 Score (0.6427), indicating sensitivity to word frequency imbalances in job descriptions. |
| Support Vector Machine | Best balanced overall performer. Achieved the highest F1 Score (0.8324), highest accuracy (0.9832), and strong precision (0.8418) combined with high recall (0.8232) and MCC (0.8236), proving ideal for separating high-dimensional text boundaries. |
| Overall, Winner for your dataset? | Support Vector Machine (SVM), closely followed by Logistic Regression. SVM wins due to its superior balance of precision and recall (highest F1-score and MCC), successfully minimizing false alarms while reliably catching fraudulent job postings. |
