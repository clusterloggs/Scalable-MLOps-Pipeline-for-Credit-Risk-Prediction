# src/evaluation.py
import argparse, pandas as pd, joblib, json, yaml, os
from urllib.parse import urlparse
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
import logging
import mlflow
import mlflow.sklearn
import dagshub
from utils import load_data, load_config
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def evaluate(model_path, report_path, metrics_path):
    """
    Evaluate the model on the test dataset specified in params.yaml.

    Args:
        model_path (str): Path to the saved model (.joblib).
        report_path (str): Path to save human-readable classification report.
        metrics_path (str): Path to save machine-readable metrics (JSON).
    """
    # Load params to get label column and mapping
    P = load_config()

    data_params = P.get("data", {})
    test_data_path = data_params.get("test_path")
    if not test_data_path:
        logger.error("Test data path not found in params.yaml under data.test_path")
        raise ValueError("Test data path not found in params.yaml")

    train_params = P.get("train", {})
    label_col = train_params.get("label_col", "default")
    # Get model_type to load the correct hyperparameters for logging
    model_type = train_params.get("model_type", "logistic")
    model_params = P.get(model_type, {})

    logger.info(f"Loading test data from {test_data_path}")
    df = load_data(test_data_path)

    # Separate features and labels
    y_true = df[label_col].astype(int)
    X_test = df.drop(label_col, axis=1)

    try:
        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)
    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading the model: {e}")
        raise

    logger.info("Generating predictions on the test set...")
    predictions = model.predict(X_test)

    logger.info("Evaluating predictions...")
    acc = accuracy_score(y_true, predictions)
    precision = precision_score(y_true, predictions, average="binary")
    recall = recall_score(y_true, predictions, average="binary")
    f1 = f1_score(y_true, predictions, average="binary")

    # For binary classification default, but handle multi-class too
    try:
        roc_auc = roc_auc_score(y_true, predictions)
    except ValueError:
        roc_auc = roc_auc_score(y_true, model.predict_proba(X_test), multi_class="ovr")

    report_str = classification_report(y_true, predictions)
    conf_matrix = confusion_matrix(y_true, predictions)

    # Log metrics
    logger.info(f"Accuracy: {acc}")
    logger.info(f"Precision: {precision}")
    logger.info(f"Recall: {recall}")
    logger.info(f"F1-score: {f1}")
    logger.info(f"ROC AUC Score: {roc_auc}")
    logger.info(f"Confusion Matrix:\n{conf_matrix}")
    logger.info(f"Classification Report:\n{report_str}")

    # --- MLflow Logging ---
    logger.info("Logging metrics and artifacts to MLflow...")
    mlflow.log_params(train_params)  # Logs general params like test_split_ratio
    mlflow.log_params(model_params)  # Logs model-specific hyperparameters
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc_score", roc_auc)

    # Save classification report
    if report_path:
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            f.write(f"Classification Report:\n{report_str}\n")
        logger.info(f"Evaluation report saved to {report_path}")
        mlflow.log_artifact(report_path, "reports")

    # Save metrics JSON (for DVC)
    if metrics_path:
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        metrics = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc_score": roc_auc,
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Metrics saved to {metrics_path}")
        mlflow.log_artifact(metrics_path, "metrics")

    # Save confusion matrix plot
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    cm_path = "reports/confusion_matrix.png"
    os.makedirs(os.path.dirname(cm_path), exist_ok=True)
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path, "plots")
    plt.close()

    # --- Model Logging and Registration ---
    # Workaround for a recurring MLflow client/server incompatibility on Dagshub.
    # Instead of using mlflow.sklearn.log_model, which is causing an 'unsupported endpoint' error,
    # we log the raw .joblib file as a generic artifact. This is more resilient to version mismatches.
    logger.info(f"Logging model file '{model_path}' as an artifact...")
    mlflow.log_artifact(model_path, artifact_path="model")
    logger.info("Model artifact logged successfully.")

    # Since we are not using log_model, we cannot easily get the model_uri for automatic registration.
    # We will skip this step to resolve the blocking error. The model can be registered from the UI.
    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
    if tracking_url_type_store != "file":
        model_type = train_params.get("model_type", "default_model")
        model_name = f"CreditRiskModel-{model_type}"
        logger.warning(f"Skipping automatic model registration for '{model_name}'. "
                       "Please register the model manually from the Dagshub UI's 'Experiments' tab if needed.")

    logger.info("Evaluation and logging complete.")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", default="models/model.joblib", help="Path to the saved model"
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="Evaluate model on the test dataset."
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

    # Initialize Dagshub and MLflow
    dagshub.init(
        repo_owner="azkintonde",
        repo_name="Scalable-MLOps-Pipeline-for-Credit-Risk-Prediction",
        mlflow=True,
    )

    if args.evaluate:
        with mlflow.start_run():
            logger.info("MLflow run started for evaluation.")
            mlflow.set_tag("mlflow.source.name", "evaluation_stage")
            evaluate(args.model_path, args.report_path, args.metrics_path)
            logger.info("MLflow run finished.")
    else:
        logger.warning(
            "No action requested. Use --evaluate to run model evaluation."
        )
