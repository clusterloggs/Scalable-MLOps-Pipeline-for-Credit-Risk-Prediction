# src/train.py

import argparse
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from utils import load_data, print_evaluation

from pipeline import build_pipeline

def train_model(data_path, model_type, output_path):
    df = load_data(data_path)

    cat_cols = ['home_status', 'loan_intent', 'loan_grade']
    num_cols = ['income', 'emp_years', 'loan_amount',
                'loan_int_rate', 'loan_status', 'loan_percent_income',
                'credit_history']
    
    X = df[cat_cols + num_cols]
    y = df["default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Training {model_type} model...")
    pipeline = build_pipeline(model_type)
    pipeline.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = pipeline.predict(X_test)
    print_evaluation(y_test, y_pred)


    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(pipeline, output_path)
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/df_clean.parquet")
    parser.add_argument("--model_type", default="logistic", choices=["logistic", "random_forest"])
    parser.add_argument("--output_path", default="models/model.joblib")
    args = parser.parse_args()

    train_model(args.data_path, args.model_type, args.output_path)
