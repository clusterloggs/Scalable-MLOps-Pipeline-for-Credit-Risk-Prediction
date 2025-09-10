# src/model_evaluation.py
import argparse
import json
import joblib
import pandas as pd
import yaml
import os
import logging
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from utils import load_data, load_config
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def evaluate_and_promote(model_path, report_path, metrics_path):
    """
    Evaluates the model, logs metrics to the existing MLflow run,
    and promotes the model to 'Staging' if it meets performance criteria.
    """
    logger.info("--- Starting Model Evaluation and Promotion ---")
    
    # Load params from params.yaml
    P = load_config()
    mlflow_params = P.get('mlflow', {})
    train_params = P.get('train', {})
    data_params = P.get("data", {})

    # --- MLflow Setup ---
    # Set tracking URI for local or remote tracking
    if mlflow_params.get('tracking_uri'):
        mlflow.set_tracking_uri(mlflow_params['tracking_uri'])
    
    client = MlflowClient()

    # --- Load test data and model ---
    test_data_path = data_params.get("test_path")
    logger.info(f"Loading test data from {test_data_path}")
    df = load_data(test_data_path)
    
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    
    label_col = train_params.get("label_col", "default")
    X_test = df.drop(label_col, axis=1)
    y_test = df[label_col].astype(int)

    # --- Get Run ID from the training stage ---
    run_id_path = "models/run_id.txt"
    try:
        with open(run_id_path, "r") as f:
            run_id = f.read().strip()
        logger.info(f"Evaluating model from MLflow Run ID: {run_id}")
    except FileNotFoundError:
        logger.error(f"Could not find {run_id_path}. Cannot link to training run.")
        raise

    # --- Re-open the training run to log evaluation results ---
    with mlflow.start_run(run_id=run_id):
        logger.info("Making predictions on the test set...")
        decision_threshold = train_params.get('decision_threshold', 0.5)
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= decision_threshold).astype(int)

        # --- Calculate metrics ---
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="binary")
        recall = recall_score(y_test, y_pred, average="binary")
        f1 = f1_score(y_test, y_pred, average="binary")
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        metrics = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc_score": roc_auc,
        }
        logger.info(f"Evaluation Metrics: {metrics}")

        # --- Log metrics and artifacts to MLflow ---
        logger.info("Logging evaluation results to MLflow...")
        mlflow.log_metrics(metrics)
        
        # Save classification report for DVC and log to MLflow
        report_str = classification_report(y_test, y_pred)
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            f.write(report_str)
        mlflow.log_artifact(report_path, "evaluation_reports")
        logger.info(f"Classification report saved to {report_path}")

        # Save metrics JSON for DVC
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Metrics saved to {metrics_path}")

        # Save confusion matrix plot for DVC and log to MLflow
        cm_path = "reports/confusion_matrix.png"
        plt.figure(figsize=(6, 4))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.title("Confusion Matrix")
        os.makedirs(os.path.dirname(cm_path), exist_ok=True)
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path, "evaluation_plots")
        plt.close()
        logger.info(f"Confusion matrix plot saved to {cm_path}")

        # --- Model Promotion Logic ---
        promotion_metric = train_params.get('promotion_metric')
        promotion_threshold = train_params.get('promotion_threshold')
        
        if promotion_metric and promotion_threshold is not None:
            if metrics.get(promotion_metric, 0) >= promotion_threshold:
                logger.info(f"Model met promotion threshold: {promotion_metric} ({metrics[promotion_metric]:.4f}) >= {promotion_threshold}.")
                logger.info("Promoting model to 'Staging' in MLflow Registry...")
                
                registered_model_name = mlflow_params.get('registered_model_name')
                
                # Find the model version created by this run
                model_versions = client.search_model_versions(f"run_id='{run_id}'")
                
                if model_versions:
                    version_to_promote = model_versions[0].version
                    client.transition_model_version_stage(
                        name=registered_model_name,
                        version=version_to_promote,
                        stage="Staging",
                        archive_existing_versions=True
                    )
                    logger.info(f"Successfully promoted model version {version_to_promote} of '{registered_model_name}' to 'Staging'.")
                else:
                    logger.warning(f"Could not find a model version for run_id '{run_id}' to promote.")
            else:
                logger.info(f"Model did not meet promotion threshold: {promotion_metric} ({metrics.get(promotion_metric, 0):.4f}) < {promotion_threshold}.")
        else:
            logger.warning("Skipping model promotion: 'promotion_metric' or 'promotion_threshold' not defined in params.yaml.")

    logger.info("--- Model Evaluation and Promotion Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model and handle promotion in MLflow Registry.")
    parser.add_argument(
        "--model_path", default="models/model.joblib", help="Path to the saved model"
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="Flag to run evaluation (for DVC cmd compatibility)."
    )
    parser.add_argument(
        "--report_path",
        default="reports/classification_report.txt",
        help="Path to save the evaluation report.",
    )
    parser.add_argument(
        "--metrics_path",
        default="reports/metrics.json",
        help="Path to save the metrics JSON file.",
    )
    args = parser.parse_args()

    # The --evaluate flag is mainly for DVC to have a clear command,
    # the script's purpose is evaluation.
    if args.evaluate:
        evaluate_and_promote(args.model_path, args.report_path, args.metrics_path)
    else:
        logger.warning("Skipping evaluation. Use the --evaluate flag to run.")
