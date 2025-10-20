# app/api.py
import os
import joblib
import pandas as pd
import yaml
from fastapi import FastAPI, Response, status
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- App and Model Loading ---
app = FastAPI(
    title="Credit Risk Prediction API",
    description="An API to predict loan default risk based on applicant data.",
    version="1.0.0"
)

# --- Monitoring Setup ---
Instrumentator().instrument(app).expose(app)

# Construct paths relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

MODEL_PATH = os.path.join(ROOT_DIR, 'models', 'model.joblib')
PARAMS_PATH = os.path.join(ROOT_DIR, 'params.yaml')

try:
    model = joblib.load(MODEL_PATH)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
    with open(PARAMS_PATH, 'r') as f:
        params = yaml.safe_load(f)
    logger.info("Parameters loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model or params: {e}")
    model = None
    params = {}

# --- Pydantic Model for Input Validation ---
# This defines the structure and types for the incoming request data.
# FastAPI will automatically validate against this.
class LoanApplication(BaseModel):
    income: float = Field(..., example=60000, description="Annual income of the applicant")
    emp_years: int = Field(..., example=5, description="Employment length in years")
    loan_amount: float = Field(..., example=10000, description="Requested loan amount")
    loan_int_rate: float = Field(..., example=7.5, description="Loan interest rate")
    loan_percent_income: float = Field(..., example=0.17, description="Loan amount as a percentage of income")
    credit_history: int = Field(..., example=6, description="Credit history length in years")
    home_status: str = Field(..., example="RENT", description="Applicant's home ownership status")
    loan_intent: str = Field(..., example="DEBTCONSOLIDATION", description="The intent of the loan")
    loan_grade: str = Field(..., example="B", description="The grade assigned to the loan")

@app.get("/health", status_code=status.HTTP_200_OK, tags=["Health"])
def health_check():
    """
    Health check to verify that the API is running and the model is loaded.
    """
    if model and params:
        return {"status": "ok", "message": "API is healthy and model is loaded."}
    else:
        # Return a 503 Service Unavailable status if the model isn't ready
        return Response(
            content='{"status": "unhealthy", "message": "Model or parameters not loaded."}',
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            media_type="application/json"
        )

@app.post("/predict")
def predict(application: LoanApplication):
    """
    Predicts the probability of loan default for a single applicant.
    """
    if not model or not params:
        return {"error": "Model or parameters not loaded. API is not operational."}
    
    # Convert Pydantic model to a dictionary
    input_dict = application.model_dump()

    # Get the feature order from params.yaml to ensure consistency
    feature_params = params.get('features', {})
    feature_names = feature_params.get('numerical', []) + feature_params.get('categorical', [])

    # Create a DataFrame with columns in the correct order
    input_data = pd.DataFrame([input_dict])[feature_names]

    # Predict probability and apply the decision threshold
    prediction_proba = model.predict_proba(input_data)[:, 1]
    decision_threshold = params.get('train', {}).get('decision_threshold', 0.65)
    prediction = (prediction_proba >= decision_threshold).astype(int)[0]

    return {
        "prediction": "Default" if prediction == 1 else "No Default",
        "probability_of_default": float(prediction_proba[0])
    }
