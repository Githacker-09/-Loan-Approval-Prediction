from flask import Flask, request, jsonify
from flask import render_template
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load the trained ML pipeline
model = joblib.load("outputs/loan_approval_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():

    data = request.get_json()

    # Check whether all required fields are provided
    required_fields = [
        'Married', 'Dependents', 'Education', 'Self_Employed',
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
        'Loan_Amount_Term', 'Credit_History', 'Property_Area'
    ]

    if not data or not all(field in data for field in required_fields):
        return jsonify({'error': 'Missing required input fields'}), 400

    # Convert input into a DataFrame
    input_data = pd.DataFrame([data])[required_fields]

    # Predict loan status
    prediction = model.predict(input_data)[0]

    # Calculate approval probability
    approved_index = list(model.classes_).index('Y')
    probability = model.predict_proba(input_data)[0][approved_index]

    # Calculate risk score
    risk_score = 100 - (probability * 100)

    if risk_score <= 30:
        risk_level = "Low"
    elif risk_score <= 60:
        risk_level = "Medium"
    else:
        risk_level = "High"

    return jsonify({
        'loan_prediction': 'Approved' if prediction == 'Y' else 'Rejected',
        'approval_probability': round(probability * 100, 2),
        'risk_score': round(risk_score, 2),
        'risk_level': risk_level
    })


if __name__ == '__main__':
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )