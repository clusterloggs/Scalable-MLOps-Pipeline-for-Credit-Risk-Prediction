# src/predict.py
import argparse, pandas as pd, joblib, json, yaml, os
from utils import load_data
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import logging
import mlflow
import mlflow.sklearn

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def evaluate(model_path, report_path=None):
    """
    Evaluate the model on the test dataset specified in params.yaml.
    """
    # Load params to get label column and mapping, needed for evaluation
    with open("params.yaml") as f:
        P = yaml.safe_load(f)

    data_params = P.get("data", {})
    test_data_path = data_params.get("test_path")
    if not test_data_path:
        logger.error("Test data path not found in params.yaml under data.test_path")
        raise ValueError("Test data path not found in params.yaml")

    label_col = P["train"].get("label_col", "default")

    logger.info(f"Loading test data from {test_data_path}")
    df = load_data(test_data_path)

    # Map true labels to numeric format to match model output
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
    roc_auc = roc_auc_score(y_true, predictions)
    report_str = classification_report(y_true, predictions)
    conf_matrix = confusion_matrix(y_true, predictions)

    logger.info(f"Accuracy: {acc}")
    logger.info(f"ROC AUC Score: {roc_auc}")
    logger.info(f"Confusion Matrix:\n {conf_matrix}")
    logger.info(f"Classification Report:\n {report_str}")

    if report_path:
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(f"Classification Report:\n{report_str}\n")
        logger.info(f"Evaluation report saved to {report_path}")


def predict_single(sample_json, model_path):
    logger.info("Loading model...")
    model = joblib.load(model_path)

    logger.info("Preparing sample data...")
    df = pd.DataFrame([sample_json])  # wrap dict into list to get a single-row DataFrame

    logger.info("Making prediction ...")
    prediction = model.predict(df)[0]

    logger.info(f"Prediction: {prediction} (1 = Default, 0 = No Default)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="models/model.joblib", help="Path to the saved model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model on the test dataset.")
    parser.add_argument("--report_path", default=None, help="Path to save the evaluation report.")
    parser.add_argument("--sample_json", default=None, help="Path to a single JSON file (for individual prediction)")

    args = parser.parse_args()

    if args.sample_json:
        with open(args.sample_json, "r") as f:
            sample = json.load(f)
        predict_single(sample, args.model_path)
    elif args.evaluate:
        evaluate(args.model_path, args.report_path)
    else:
        logger.warning("Please provide either --evaluate to evaluate the model or --sample_json for a single prediction.")
