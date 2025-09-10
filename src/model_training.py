# src/model_training.py
import argparse, os, joblib, pandas as pd, yaml, mlflow
from utils import load_data, load_config
from feature_engineering import build_pipeline
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def train_model(model_type, output_path):
    """
    Loads training data, trains a model pipeline, saves it, and registers with MLflow.
    Configuration is loaded from params.yaml.
    """
    logger.info("--- Starting Model Training and Registration ---")
    # ---- load params.yaml ----
    P = load_config()

    # Get all parameter sections
    mlflow_params = P.get('mlflow', {})
    train_params = P.get("train", {})
    feature_params = P.get("features", {})
    
    # Determine model type, allowing CLI override
    model_type = model_type or train_params.get("model_type", "logistic")
    model_params = P.get(model_type, {})

    # --- MLflow Setup ---
    if mlflow_params.get('tracking_uri'):
        mlflow.set_tracking_uri(mlflow_params['tracking_uri'])
    mlflow.set_experiment(mlflow_params.get('experiment_name', 'DefaultExperiment'))

    # --- data paths ---
    data_params = P.get("data", {})
    train_data_path = data_params.get("train_path", "data/processed/train.parquet")

    # --- Load training data ---
    logger.info(f"Loading training data from {train_data_path}")
    df = load_data(train_data_path)

    label_col = train_params.get("label_col", "default")
    cat_cols = feature_params.get("categorical", [])
    num_cols = feature_params.get("numerical", [])
    X_train = df[cat_cols + num_cols]
    y_train = df[label_col].astype(int)

    # --- Start an MLflow Run ---
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"MLflow Run ID: {run_id}")

        # Log parameters to MLflow
        logger.info("Logging parameters to MLflow...")
        mlflow.log_params(train_params)
        mlflow.log_params(model_params)
        mlflow.log_param("model_type", model_type)

        # --- Build and Train Pipeline ---
        logger.info(f"Building pipeline for '{model_type}' model...")
        smote_rnd = train_params.get("smote_random_state", 42)
        pipeline = build_pipeline(model_type, model_params, cat_cols, num_cols, smote_rnd)
        
        logger.info(f"Training model with params: {model_params}")
        pipeline.fit(X_train, y_train)

        # --- Save model artifact for DVC ---
        logger.info(f"Saving model to {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump(pipeline, output_path)

        # --- Log and Register Model in MLflow Registry ---
        logger.info("Logging model artifact to MLflow...")
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model"
        )

        # Attempt to register the model. If the backend doesn't support it (like DagsHub's
        # programmatic model creation), log a warning and continue without failing.
        try:
            logger.info(f"Attempting to register model '{mlflow_params.get('registered_model_name')}'...")
            mlflow.register_model(
                model_uri=model_info.model_uri,
                name=mlflow_params.get('registered_model_name')
            )
            logger.info("Model registration successful.")
        except mlflow.exceptions.MlflowException as e:
            logger.warning(
                f"Could not register model. The MLflow backend may not support this "
                f"operation, or the model may need to be created in the UI first. "
                f"Skipping registration. Error: {e}"
            )
        
        # --- Save run_id for the next DVC stage ---
        run_id_path = "models/run_id.txt"
        logger.info(f"Saving MLflow Run ID to {run_id_path}")
        with open(run_id_path, "w") as f:
            f.write(run_id)
            
        logger.info("Model training and MLflow registration complete.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_type", default=None, help="Type of model to train. Overrides params.yaml.")
    ap.add_argument("--output_path", default="models/model.joblib", help="Path to save the trained model.")
    args = ap.parse_args()
    train_model(args.model_type, args.output_path)