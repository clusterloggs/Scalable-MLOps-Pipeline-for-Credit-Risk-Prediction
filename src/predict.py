# src/predict.py

import argparse, pandas as pd, joblib, json
from utils import load_data
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def predict_batch(data_path, model_path, has_labels=False):
    print(f"Loading data from {data_path}")
    df = load_data(data_path)

    if has_labels:
        y_true = df["default"]
        df = df.drop("default", axis=1)
    else:
        y_true = None

    print(f"Loading model from {model_path}")
    model = joblib.load(model_path)

    print("Generating batch predictions...")
    predictions = model.predict(df)

    if has_labels:
        print("Evaluating predictions...")
        print("Accuracy:", accuracy_score(y_true, predictions))
        print("Confusion Matrix:\n", confusion_matrix(y_true, predictions))
        print("Classification Report:\n", classification_report(y_true, predictions))
    else:
        print("Predictions:\n", predictions)


def predict_single(sample_json, model_path):
    print("Loading model...")
    model = joblib.load(model_path)

    print("Preparing sample data...")
    df = pd.DataFrame([sample_json])  # wrap dict into list to get a single-row DataFrame

    print("Making prediction ...")
    prediction = model.predict(df)[0]

    print(f"Prediction: {prediction} (Y = Default, N = No Default)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="models/model.joblib", help="Path to the saved model")
    parser.add_argument("--has_labels", action="store_true", help="Specify this flag if the data contains labels")
    parser.add_argument("--data_path", default="data/df_clean.parquet", help="Path to the new batch data (.parquet)")
    parser.add_argument("--sample_json", default="data/sample.json", help="Path to a single JSON file (for individual prediction)")


    args = parser.parse_args()

    if args.sample_json:
        with open(args.sample_json, "r") as f:
            sample = json.load(f)
        predict_single(sample, args.model_path)
    elif args.data_path:
        predict_batch(args.data_path, args.model_path, args.has_labels)
    else:
        print(" Please provide either --data_path for batch prediction or --sample_json for a single prediction.")
