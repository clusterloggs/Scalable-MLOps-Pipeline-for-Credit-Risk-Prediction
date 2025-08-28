# app.py
import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        data = request.json
        # Process the data and make a prediction
        return jsonify(message="Prediction result")
    else:
        return jsonify(message="Send a POST request to make a prediction")


if __name__ == '__main__':
    # The port is specified by Gunicorn or the deployment environment
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
