# src/train.py
import argparse, os, joblib, pandas as pd, yaml, json
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from utils import load_data
from pipeline import build_pipeline  # if import issues, use: from src.pipeline import build_pipeline

def train_model(data_path, model_type, test_split_ratio, output_path, metrics_path):
    # ---- load params.yaml ----
    with open("params.yaml") as f:
        P = yaml.safe_load(f)

    # Allow CLI to override YAML, with robust fallbacks from params.yaml using .get()
    # This prevents crashes if a key is missing from the YAML file.
    model_type = model_type or P["train"].get("model_type", "logistic")
    test_split_ratio = test_split_ratio if test_split_ratio is not None else P["train"].get("test_split_ratio", 0.2)
    rnd = P["train"].get("random_state", 42)                  # For train_test_split
    thr = P["train"].get("decision_threshold", 0.5)           # For converting probabilities to classes
    smote_rnd = P["train"].get("smote_random_state", 42)      # For SMOTE reproducibility
    label_col = P["train"].get("label_col", "default")        # Target column name
    do_stratify = P["train"].get("stratify", True)            # Whether to stratify the train/test split
    label_mapping = P["train"].get("label_mapping", {'Y': 1, 'N': 0}) # For target variable

    # --- features ---
    cat_cols = P["features"]["categorical"]
    num_cols = P["features"]["numerical"]

    # --- model params from YAML ---
    model_params = P["logistic"] if model_type == "logistic" else P["random_forest"]

    # --- data ---
    df = load_data(data_path)
    X = df[cat_cols + num_cols]
    # Map labels consistently to 0/1 using the mapping from params.yaml
    y = df[label_col].map(label_mapping).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split_ratio, random_state=rnd, stratify=y if do_stratify else None
    )

    print(f"Training {model_type} model with params: {model_params}")
    pipeline = build_pipeline(model_type, model_params, cat_cols, num_cols, smote_rnd)
    pipeline.fit(X_train, y_train)

    print("Evaluating...")
    metrics = {}
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (proba >= thr).astype(int)   # use the same threshold you used in the notebook
        # Dynamically create target names from the mapping, sorted by value (0, 1, etc.)
        target_names = [k for k, v in sorted(label_mapping.items(), key=lambda item: item[1])]
        metrics['accuracy'] = accuracy_score(y_test, y_pred)
        metrics['roc_auc'] = roc_auc_score(y_test, proba)
        print("Accuracy:", metrics['accuracy'])
        print("Classification Report:\n", classification_report(y_test, y_pred, target_names=target_names))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("ROC AUC:", metrics['roc_auc'])
    else:
        y_pred = pipeline.predict(X_test)
        # Dynamically create target names from the mapping
        target_names = [k for k, v in sorted(label_mapping.items(), key=lambda item: item[1])]
        metrics['accuracy'] = accuracy_score(y_test, y_pred)
        print("Accuracy:", metrics['accuracy'])
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred, target_names=target_names))

    # Save metrics
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(pipeline, output_path)
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="data/df_clean.parquet")
    ap.add_argument("--model_type")
    ap.add_argument("--test_split_ratio", type=float)
    ap.add_argument("--output_path", default="models/model.joblib")
    ap.add_argument("--metrics_path", default="reports/metrics.json")
    args = ap.parse_args()
    train_model(args.data_path, args.model_type, args.test_split_ratio, args.output_path, args.metrics_path)
