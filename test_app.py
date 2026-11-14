from app import app

# Create Flask test client
client = app.test_client()


# Test 1: Check homepage
def test_homepage():
    response = client.get('/')
    assert response.status_code == 200


# Test 2: Check successful loan prediction
def test_loan_prediction():
    data = {
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

    response = client.post('/predict', json=data)

    assert response.status_code == 200

    result = response.get_json()

    assert "loan_prediction" in result
    assert "approval_probability" in result
    assert "risk_score" in result
    assert "risk_level" in result


# Test 3: Check missing input validation
def test_missing_fields():
    data = {
        "Married": "Yes",
        "ApplicantIncome": 5000
    }

    response = client.post('/predict', json=data)

    assert response.status_code == 400