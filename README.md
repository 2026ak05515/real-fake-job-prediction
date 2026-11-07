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

