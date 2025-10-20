import requests
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# The URL of your running FastAPI application
API_URL = "http://localhost:8000/predict"

# The data for the loan application, matching the Pydantic model in app/api.py
payload = {
    "income": 40000,
    "emp_years": 12,
    "loan_amount": 15000,
    "loan_int_rate": 12.5,
    "loan_percent_income": 0.5,
    "credit_history": 3,
    "home_status": "MORTGAGE",
    "loan_intent": "HOMEIMPROVEMENT",
    "loan_grade": "D"
}

logger.info(f"Sending prediction request to {API_URL} with payload:\n{json.dumps(payload, indent=2)}")

try:
    # Send the POST request to the API
    response = requests.post(API_URL, json=payload)
    response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

    # Print the prediction result
    prediction_result = response.json()
    logger.info(f"API Response:\n{json.dumps(prediction_result, indent=2)}")

except requests.exceptions.RequestException as e:
    logger.error(f"An error occurred while calling the API: {e}")
