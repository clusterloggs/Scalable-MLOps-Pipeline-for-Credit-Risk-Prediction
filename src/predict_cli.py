# src/predict_cli.py
import argparse
import pandas as pd
import joblib
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def predict_single(sample_json, model_path):
    """Predicts the class for a single data sample provided as a JSON object."""
    logger.info("Loading model...")
    model = joblib.load(model_path)

    logger.info("Preparing sample data...")
    # Wrap dict in a list to create a single-row DataFrame
    df = pd.DataFrame([sample_json])

    logger.info("Making prediction ...")
    prediction = model.predict(df)[0]

    logger.info(f"Prediction: {prediction} (1 = Default, 0 = No Default)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="models/model.joblib", help="Path to the saved model")
    parser.add_argument("--sample_json", required=True, help="Path to a single JSON file for prediction")

    args = parser.parse_args()

    try:
        with open(args.sample_json, "r") as f:
            sample_data = json.load(f)
        predict_single(sample_data, args.model_path)
    except FileNotFoundError:
        logger.error(f"Error: The file '{args.sample_json}' was not found.")
    except json.JSONDecodeError:
        logger.error(f"Error: The file '{args.sample_json}' is not a valid JSON.")
