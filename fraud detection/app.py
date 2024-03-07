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
        oldbalanceorig = float(request.form['oldbalanceorig'])
        newbalanceorig = float(request.form['newbalanceorig'])

        # Convert form data to a NumPy array
        input_data = np.array([[type_val, amount, oldbalanceorig, newbalanceorig]])

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Render the result page with the prediction
        return render_template('result.html', type=type_val, prediction=prediction)

    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug = True)