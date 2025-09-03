# src/train.py
import argparse, os, joblib, pandas as pd, yaml, json
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from utils import load_data
from pipeline import build_pipeline  # if import issues, use: from src.pipeline import build_pipeline
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def train_model(data_path, model_type, test_split_ratio, output_path, metrics_path):
    # ---- load params.yaml ----
    with open("params.yaml") as f:
        P = yaml.safe_load(f)

    # Allow CLI to override YAML, with robust fallbacks from params.yaml using .get()
    # This prevents crashes if a key is missing from the YAML file.
    model_type = model_type or P["train"].get("model_type", "logistic")
    test_split_ratio = test_split_ratio if test_split_ratio is not None else P["train"].get("test_split_ratio", 0.2)
    rnd = P["train"].get("random_state", 42)
    thr = P["train"].get("decision_threshold", 0.68)           
    smote_rnd = P["train"].get("smote_random_state", 42)      
    label_col = P["train"].get("label_col", "default")       
    do_stratify = P["train"].get("stratify", True)            

    # --- features ---
    cat_cols = P["features"]["categorical"]
    num_cols = P["features"]["numerical"]

    # --- model params from YAML ---
    model_params = P["logistic"] if model_type == "logistic" else P["random_forest"]

    # --- data paths for outputs ---
    train_data_path = P["data"].get("train_path", "data/processed/train.parquet")
    test_data_path = P["data"].get("test_path", "data/processed/test.parquet")

    # --- data ---
    df = load_data(data_path)
    X = df[cat_cols + num_cols]
    # The label column is already preprocessed to be 0s and 1s.
    y = df[label_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split_ratio, random_state=rnd, stratify=y if do_stratify else None
    )

    # --- Save split data ---
    logging.info(f"Saving train and test sets...")
    os.makedirs(os.path.dirname(train_data_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_data_path), exist_ok=True)

    # Combine features and labels before saving
    train_df = X_train.copy()
    train_df[label_col] = y_train
    train_df.to_parquet(train_data_path, index=False)
    logging.info(f"Train data saved to {train_data_path}")

    test_df = X_test.copy()
    test_df[label_col] = y_test
    test_df.to_parquet(test_data_path, index=False)
    logging.info(f"Test data saved to {test_data_path}")

    # Start an MLflow run
    with mlflow.start_run():
        logging.info("MLflow run started.")
        
        # Log parameters
        mlflow.log_param("model_type", model_type)
        mlflow.log_params(model_params)
        mlflow.log_param("decision_threshold", thr)
        mlflow.log_param("smote_random_state", smote_rnd)

        print(f"Training {model_type} model with params: {model_params}")
        pipeline = build_pipeline(model_type, model_params, cat_cols, num_cols, smote_rnd)
        pipeline.fit(X_train, y_train)

        # --- Save model artifact ---
        logging.info(f"Saving model to {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump(pipeline, output_path)

        # logging.info("Logging model to MLflow...")
        # mlflow.sklearn.log_model(
        #     sk_model=pipeline,
        #     artifact_path="model",
        #     # Add this line to provide an example and create a signature
        #     input_example=X_train.head(), 
        #     registered_model_name=f"{model_type}-model"
        # )
        # logging.info("Model logged to MLflow.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="data/df_clean.parquet")
    ap.add_argument("--model_type")
    ap.add_argument("--test_split_ratio", type=float)
    ap.add_argument("--output_path", default="models/model.joblib")
    ap.add_argument("--metrics_path", default="reports/metrics.json")
    args = ap.parse_args()
    train_model(args.data_path, args.model_type, args.test_split_ratio, args.output_path, args.metrics_path)