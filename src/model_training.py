# src/train.py
import argparse, os, joblib, pandas as pd, yaml
from utils import load_data, load_config
from feature_engineering import build_pipeline
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def train_model(model_type, output_path):
    """
    Loads training data, trains a model pipeline, and saves it.
    Configuration is loaded from params.yaml.
    """
    logger.info("--- Starting Model Training ---")
    # ---- load params.yaml ----
    P = load_config()

    # Allow CLI to override YAML, with robust fallbacks from params.yaml using .get()
    train_params = P.get("train", {})
    model_type = model_type or train_params.get("model_type", "logistic")
    
    # Pipeline parameters
    smote_rnd = train_params.get("smote_random_state", 42)
    label_col = train_params.get("label_col", "default")

    # --- features ---
    feature_params = P.get("features", {})
    cat_cols = feature_params.get("categorical", [])
    num_cols = feature_params.get("numerical", [])

    # --- model params from YAML ---
    model_params = P.get(model_type, {})

    # --- data paths ---
    data_params = P.get("data", {})
    train_data_path = data_params.get("train_path", "data/processed/train.parquet")

    # --- Load training data ---
    logger.info(f"Loading training data from {train_data_path}")
    df = load_data(train_data_path)
    X_train = df[cat_cols + num_cols]
    y_train = df[label_col].astype(int)

    # --- Build and Train Pipeline ---
    logger.info(f"Building pipeline for '{model_type}' model...")
    pipeline = build_pipeline(model_type, model_params, cat_cols, num_cols, smote_rnd)
    
    logger.info(f"Training model with params: {model_params}")
    pipeline.fit(X_train, y_train)

    # --- Save model artifact ---
    logger.info(f"Saving model to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(pipeline, output_path)
    logger.info("Model training complete.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_type", default=None, help="Type of model to train. Overrides params.yaml.")
    ap.add_argument("--output_path", default="models/model.joblib", help="Path to save the trained model.")
    args = ap.parse_args()
    train_model(args.model_type, args.output_path)