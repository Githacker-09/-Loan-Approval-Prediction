# Loan Approval Prediction API with Risk Scoring

A machine learning-based web application that predicts loan approval outcomes using applicant information. The system uses Logistic Regression to generate predictions, estimate approval probability, and calculate a corresponding risk score.

**Live Application:** https://loan-approval-prediction-otjd.onrender.com

**GitHub Repository:** https://github.com/Githacker-09/-Loan-Approval-Prediction

---

## 1. Project Overview

Loan approval decisions depend on multiple applicant characteristics, including income, credit history, education, employment status, and property location.

This project develops a machine learning classification system that predicts whether a loan application is likely to be approved or rejected.

The model is integrated into a Flask REST API and a simple web interface, allowing users to enter applicant details and receive predictions with risk assessments.

## 2. Objectives

- Develop a supervised machine learning classification model.
- Perform exploratory data analysis and data preprocessing.
- Predict loan approval using Logistic Regression.
- Calculate approval probability and a corresponding risk score.
- Develop a Flask-based REST API.
- Provide a simple web interface for user interaction.
- Implement automated testing using GitHub Actions.
- Containerize the application using Docker.
- Deploy the application using Render.

## 3. Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Data Visualization | Matplotlib |
| Machine Learning | Scikit-learn |
| Algorithm | Logistic Regression |
| Backend | Flask |
| Frontend | HTML, CSS, JavaScript |
| Model Persistence | Joblib |
| API Testing | Postman |
| Automated Testing | Pytest |
| Version Control | Git, GitHub |
| CI | GitHub Actions |
| Containerization | Docker |
| Deployment | Render |

## 4. Machine Learning Workflow

1. Dataset loading.
2. Exploratory Data Analysis (EDA).
3. Missing-value identification and handling.
4. Categorical encoding.
5. Numerical feature scaling.
6. Stratified train-test split (80:20).
7. Five-fold stratified cross-validation.
8. Logistic Regression model training.
9. Model evaluation.
10. Model serialization using Joblib.
11. Flask API integration.
12. Risk score calculation.

### Data Preprocessing

- Numerical missing values are handled using median imputation.
- Categorical missing values are handled using most-frequent imputation.
- Categorical variables are encoded using One-Hot Encoding.
- Numerical variables are standardized using StandardScaler.
- Loan_ID is excluded because it is an identifier.
- Gender is excluded from the selected model features as a demographic attribute.

Preprocessing and model training are combined into a Scikit-learn Pipeline to maintain consistent transformations.

## 5. Model Performance

The model was evaluated using a stratified 80:20 train-test split and five-fold stratified cross-validation on the training data.

| Evaluation Metric | Result |
|---|---:|
| Cross-validation Accuracy | 79.83% |
| Test Accuracy | 86.18% |
| Precision (Approved Class) | 84% |
| Recall (Approved Class) | 98.82% |
| F1-score (Approved Class) | 90.81% |

These results are based on the selected dataset split. Performance may vary on new applicant data.

## 6. Risk Scoring

The project calculates a model-defined risk score using:

**Risk Score = 100 − (Approval Probability × 100)**

The resulting score is classified into three categories:

| Risk Score | Risk Level |
|---|---|
| 0–30 | Low |
| 31–60 | Medium |
| 61–100 | High |

The risk score is an illustrative project indicator derived from the model's approval probability. It is not an official banking credit score or a validated financial risk assessment.

## 7. Application Features

- Applicant information input form.
- Loan approval prediction.
- Approval probability calculation.
- Risk score generation.
- Risk-level classification.
- REST API integration.
- Input validation and error handling.
- Responsive web interface.

## 8. API Documentation

### Endpoint

`POST /predict`

### Example Request

```json
{
  "Married": "Yes",
  "Dependents": "0",
  "Education": "Graduate",
  "Self_Employed": "No",
  "ApplicantIncome": 5000,
  "CoapplicantIncome": 2000,
  "LoanAmount": 150,
  "Loan_Amount_Term": 360,
  "Credit_History": 1,
  "Property_Area": "Urban"
}
```

### Example Response

```json
{
  "approval_probability": 79.8,
  "loan_prediction": "Approved",
  "risk_level": "Low",
  "risk_score": 20.2
}
```

The API returns HTTP 400 when required input fields are missing.

## 9. Project Structure

```text
Loan-Approval-Prediction/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── dataset/
│   └── train.csv
│
├── outputs/
│   ├── eda_graphs/
│   └── loan_approval_model.pkl
│
├── static/
│   ├── style.css
│   └── script.js
│
├── templates/
│   └── index.html
│
├── app.py
├── train_model.py
├── test_app.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md
```

## 10. Installation and Execution

### Prerequisites

- Python
- Git
- pip

### Clone the Repository

```bash
git clone https://github.com/Githacker-09/-Loan-Approval-Prediction.git
```

```bash
cd -Loan-Approval-Prediction
```

### Create a Virtual Environment

Windows:

```powershell
python -m venv venv
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
python app.py
```

Open:

`http://127.0.0.1:5000`

The saved model must be available at `outputs/loan_approval_model.pkl`.

## 11. Automated Testing

The project uses Pytest to test the Flask application.

Run:

```bash
pytest test_app.py -v
```

The automated tests cover:

1. Homepage availability.
2. Successful loan prediction.
3. Missing input validation.

All three tests passed locally and the GitHub Actions CI workflow completed successfully.

## 12. Continuous Integration

GitHub Actions is configured to automatically run tests when code is pushed to the main branch or when a pull request targets main.

The workflow:
- Checks out the repository.
- Sets up Python.
- Installs dependencies.
- Executes automated tests using Pytest.

## 13. Docker

The application can also be containerized using Docker.

Build the image:

```bash
docker build -t loan-approval-api .
```

Run the container:

```bash
docker run -d -p 5000:5000 --name loan-approval-container loan-approval-api
```

Access the application at:

`http://localhost:5000`

## 14. Cloud Deployment

The application is deployed using Render and connected to the GitHub repository.

**Live URL:** https://loan-approval-prediction-otjd.onrender.com

The deployment automatically installs the required Python dependencies and starts the Flask application.

Note: The free hosting instance may spin down after inactivity, resulting in a delay on the first request.

## 15. Limitations

- The dataset is relatively small.
- Predictions depend on the quality and distribution of the training data.
- Model performance may vary for applicants outside the training data distribution.
- The calculated risk score is a project-defined indicator and should not be used for actual lending decisions.
- The application is an academic demonstration, not a production banking system.

## 16. Future Scope

- Evaluate additional classification algorithms.
- Improve model generalization using larger datasets.
- Introduce model monitoring and retraining.
- Add explainability techniques for prediction interpretation.
- Improve input validation and application security.
- Integrate a database for storing application records.

## 17. Author

**Krishna Yadav**

B.E. Computer Science and Engineering

Chandigarh University

---

*Developed as an academic Machine Learning and MLOps mini-project.*