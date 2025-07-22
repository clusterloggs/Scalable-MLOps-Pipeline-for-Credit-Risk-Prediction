# src/utils.py

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def load_data(path):
    """Load cleaned dataset."""
    df = pd.read_parquet(path)
    df.columns = df.columns.str.strip()
    return df

def print_evaluation(y_true, y_pred):
    """Print evaluation metrics."""
    print("Accuracy Score:", accuracy_score(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
