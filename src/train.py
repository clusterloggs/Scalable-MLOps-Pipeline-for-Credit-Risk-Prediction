# src/train.py
import argparse, os, joblib, pandas as pd, yaml, json
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from utils import load_data
from pipeline import build_pipeline  # if import issues, use: from src.pipeline import build_pipeline
import logging

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

    # --- data ---
    df = load_data(data_path)
    X = df[cat_cols + num_cols]
    # The label column is already preprocessed to be 0s and 1s.
    y = df[label_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split_ratio, random_state=rnd, stratify=y if do_stratify else None
    )

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

        print("Evaluating...")
        metrics = {}

        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(X_test)[:, 1]
            y_pred = (proba >= thr).astype(int)   # threshold set in the params
            metrics['roc_auc'] = roc_auc_score(y_test, proba)
        else:
            y_pred = pipeline.predict(X_test)

        # Generate classification report and extract key metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics['accuracy'] = accuracy_score(y_test, y_pred)
        # Extract metrics for the positive class (1), using .get() for safety
        metrics['precision'] = report.get('1', {}).get('precision', 0)
        metrics['recall'] = report.get('1', {}).get('recall', 0)
        metrics['f1_score'] = report.get('1', {}).get('f1-score', 0)

        # Log metrics to MLflow
        mlflow.log_metrics(metrics)
        logging.info(f"Metrics logged to MLflow: {metrics}")

        # Print all collected metrics
        if 'roc_auc' in metrics:
            print("ROC AUC:", metrics['roc_auc'])
            print("Accuracy:", metrics['accuracy'])
            print("Recall:", metrics['recall'])
            print("Precision:", metrics['precision'])
            print("Classification Report:\n", classification_report(y_test, y_pred))
            print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

        # Save metrics for DVC
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to {metrics_path}")

        # Log the model to MLflow and save it for DVC
        mlflow.sklearn.log_model(pipeline, "model")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump(pipeline, output_path)
        print(f"Model saved to {output_path} and logged to MLflow.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="data/df_clean.parquet")
    ap.add_argument("--model_type")
    ap.add_argument("--test_split_ratio", type=float)
    ap.add_argument("--output_path", default="models/model.joblib")
    ap.add_argument("--metrics_path", default="reports/metrics.json")
    args = ap.parse_args()
    train_model(args.data_path, args.model_type, args.test_split_ratio, args.output_path, args.metrics_path)
