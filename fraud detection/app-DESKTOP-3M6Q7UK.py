from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("logistic_regression_model.pkl")

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for handling the form submission
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        type_val = request.form['type']
        amount = float(request.form['amount'])
        oldbalanceOrg = float(request.form['oldbalanceOrg'])
        newbalanceOrig = float(request.form['newbalanceOrig'])

        # Convert 'type_val' to lowercase and explicitly handle 'numeric' conversion
        type_val_numeric = 1 if type_val.lower() in ['transfer', 'cash_out'] else 0

        # Convert form data to a NumPy array with correct data types
        input_data = np.array([[type_val_numeric, amount, oldbalanceOrg, newbalanceOrig]])

        # Make prediction
        prediction_prob = model.predict_proba(input_data)[:, 1]

        # Assuming 0.5 as a threshold for binary classification
        threshold = 0.5
        is_fraud = prediction_prob > threshold

        # Map the result to 'Fraud' or 'No Fraud'
        result = 'Fraud' if is_fraud else 'No Fraud'

        # Render the result page with the prediction
        return render_template('result.html', type=type_val, prediction=result)

    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
