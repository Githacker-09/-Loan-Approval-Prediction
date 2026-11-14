import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# Load dataset

df = pd.read_csv("dataset/train.csv")

print("Dataset Loaded Successfully!")

# Display first five rows

print("\nFirst 5 Rows:")
print(df.head())

# Dataset dimensions

print("\nDataset Shape:")
print(df.shape)

# Dataset information

print("\nDataset Information:")
print(df.info())

# Statistical description

print("\nStatistical Description:")
print(df.describe())

# Separate independent and dependent variables

X = df.drop(['Loan_ID', 'Gender', 'Loan_Status'], axis=1)
y = df['Loan_Status']

print("\nIndependent Variables (X):")
print(X.head())

print("\nDependent Variable (y):")
print(y.head())

# Identify numerical and categorical columns

numerical_features = X.select_dtypes(
    include=['int64', 'float64']
).columns

categorical_features = X.select_dtypes(
    include=['object', 'str']
).columns

print("\nNumerical Features:")
print(list(numerical_features))

print("\nCategorical Features:")
print(list(categorical_features))

# Check missing values

print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate records

print("\nDuplicate Records:")
print(df.duplicated().sum())

plt.figure(figsize=(6, 4))

sns.countplot(data=df, x='Loan_Status')

plt.title("Loan Approval Distribution")
plt.xlabel("Loan Status")
plt.ylabel("Number of Applicants")

plt.savefig("outputs/eda_graphs/loan_status.png")

plt.close()

plt.figure(figsize=(7, 4))

sns.histplot(df['ApplicantIncome'], bins=30, kde=True)

plt.title("Applicant Income Distribution")
plt.xlabel("Applicant Income")
plt.ylabel("Frequency")

plt.savefig("outputs/eda_graphs/income_distribution.png")

plt.close()

plt.figure(figsize=(7, 4))

sns.histplot(df['LoanAmount'].dropna(), bins=30, kde=True)

plt.title("Loan Amount Distribution")
plt.xlabel("Loan Amount")
plt.ylabel("Frequency")

plt.savefig("outputs/eda_graphs/loan_amount_distribution.png")

plt.close()

plt.figure(figsize=(8, 5))

correlation = df.select_dtypes(include='number').corr()

sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')

plt.title("Feature Correlation Heatmap")

plt.savefig("outputs/eda_graphs/correlation_heatmap.png")

plt.close()

# ==============================
# EXPLORATORY DATA ANALYSIS (EDA)
# ==============================

# 1. Loan Approval Distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Loan_Status')
plt.title("Loan Approval Distribution")
plt.xlabel("Loan Status (Y = Approved, N = Rejected)")
plt.ylabel("Number of Applicants")
plt.savefig("outputs/eda_graphs/loan_status.png")
plt.close()
plt.close()


# 2. Applicant Income Distribution
plt.figure(figsize=(7, 4))
sns.histplot(df['ApplicantIncome'], bins=30, kde=True)
plt.title("Applicant Income Distribution")
plt.xlabel("Applicant Income")
plt.ylabel("Frequency")
plt.savefig("outputs/eda_graphs/income_distribution.png")
plt.close()
plt.close()


# 3. Loan Amount Distribution
plt.figure(figsize=(7, 4))
sns.histplot(df['LoanAmount'].dropna(), bins=30, kde=True)
plt.title("Loan Amount Distribution")
plt.xlabel("Loan Amount")
plt.ylabel("Frequency")
plt.savefig("outputs/eda_graphs/loan_amount_distribution.png")
plt.close()
plt.close()


# 4. Correlation Heatmap
plt.figure(figsize=(8, 5))
correlation = df.select_dtypes(include='number').corr()

sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.savefig("outputs/eda_graphs/correlation_heatmap.png")
plt.close()
plt.close()


print("\nEDA completed successfully!")
print("All graphs saved in outputs/eda_graphs/")

# Separate numerical and categorical features
numerical_features = X.select_dtypes(include='number').columns
categorical_features = X.select_dtypes(include=['object', 'str']).columns

# Numerical preprocessing
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical preprocessing
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine both preprocessing pipelines
preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features)
])

print("\nPreprocessing pipeline created successfully!")

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining and Testing Data:")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# Create complete machine learning pipeline
model = Pipeline([
    ('preprocessing', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Train the model
model.fit(X_train, y_train)

print("\nLogistic Regression model trained successfully!")

# 5-Fold Stratified Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    model, X_train, y_train,
    cv=cv,
    scoring='accuracy'
)

print("\nCross-Validation Results:")
print("Individual Fold Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate model
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, pos_label='Y'))
print("Recall:", recall_score(y_test, y_pred, pos_label='Y'))
print("F1 Score:", f1_score(y_test, y_pred, pos_label='Y'))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=['N', 'Y']))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Calculate approval probabilities
approval_probabilities = model.predict_proba(X_test)

# Get the probability corresponding to approved class Y
approved_index = list(model.classes_).index('Y')
approval_probabilities = approval_probabilities[:, approved_index]

# Calculate risk scores
risk_scores = 100 - (approval_probabilities * 100)

# Display sample results
print("\nRisk Score Examples:")

for i in range(5):
    if risk_scores[i] <= 30:
        risk_level = "Low"
    elif risk_scores[i] <= 60:
        risk_level = "Medium"
    else:
        risk_level = "High"

    print(f"Approval Probability: {approval_probabilities[i]:.2%}, "
          f"Risk Score: {risk_scores[i]:.2f}, "
          f"Risk Level: {risk_level}")

    # Save the trained model
joblib.dump(model, "outputs/loan_approval_model.pkl")

print("\nModel saved successfully!")