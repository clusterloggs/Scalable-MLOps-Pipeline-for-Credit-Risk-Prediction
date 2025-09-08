# src/process.py
import argparse
import os
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from utils import load_data, load_config
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def process_and_split_data(config_path, data_path):
    """
    Loads the clean dataset, splits it into training and testing sets based on
    parameters in `params.yaml`, and saves them as parquet files.
    """
    logger.info("--- Starting Data Processing and Splitting ---")
    
    logger.info(f"Loading configuration from {config_path}")    
    P = load_config(config_path)

    data_params = P.get("data", {})
    train_params = P.get("train", {})
    feature_params = P.get("features", {})

    # Parameters for splitting from the 'train' section of params.yaml
    test_split_ratio = train_params.get("test_split_ratio", 0.2)
    rnd = train_params.get("random_state", 42)
    label_col = train_params.get("label_col", "default")
    do_stratify = train_params.get("stratify", True)

    # Feature and label columns
    cat_cols = feature_params.get("categorical", [])
    num_cols = feature_params.get("numerical", [])

    # Output paths
    train_data_path = data_params.get("train_path", "data/processed/train.parquet")
    test_data_path = data_params.get("test_path", "data/processed/test.parquet")

    logger.info(f"Loading data from {data_path}")
    df = load_data(data_path)

    X = df[cat_cols + num_cols]
    y = df[label_col].astype(int)

    logger.info(f"Splitting data with test ratio: {test_split_ratio}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split_ratio, random_state=rnd, stratify=y if do_stratify else None
    )

    logger.info("Saving train and test sets...")
    os.makedirs(os.path.dirname(train_data_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_data_path), exist_ok=True)

    # Combine features and labels before saving
    train_df = X_train.copy()
    train_df[label_col] = y_train
    train_df.to_parquet(train_data_path, index=False)
    logger.info(f"Train data saved to {train_data_path}")

    test_df = X_test.copy()
    test_df[label_col] = y_test
    test_df.to_parquet(test_data_path, index=False)
    logger.info(f"Test data saved to {test_data_path}")
    logger.info("--- Data Processing and Splitting Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml", help="Path to configuration file")
    parser.add_argument("--data_path", default="data/df_clean.parquet", help="Path to the clean dataset to be split")
    args = parser.parse_args()
    
    process_and_split_data(config_path=args.config, data_path=args.data_path)