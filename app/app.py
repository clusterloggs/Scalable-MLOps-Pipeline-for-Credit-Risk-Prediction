# app.py
import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# --- Model Loading ---
# Load the trained model pipeline once when the application starts.
MODEL_PATH = 'models/model.joblib'
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    model = None
    print(f"Error: Model file not found at {MODEL_PATH}")

@app.route('/predict', methods=['POST'])
def predict():
    """Receives loan application data as JSON and returns a prediction."""
    if model is None:
        return jsonify({"error": "Model is not loaded"}), 500

    # Get data from the POST request
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided"}), 400

    # Convert the JSON data into a pandas DataFrame
    df = pd.DataFrame([data])

    # Make prediction
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1] # Probability of class '1' (default)

    # Return the response
    return jsonify({
        'prediction': int(prediction),
        'probability_of_default': float(probability)
    })

@app.route('/health', methods=['GET'])
def health_check():
    """A simple health check endpoint."""
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    # The port is specified by Gunicorn or the deployment environment
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))