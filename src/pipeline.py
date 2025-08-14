# src/pipeline.py
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from typing import List, Dict, Any

def build_pipeline(
    model_type: str = "logistic",
    model_params: Dict[str, Any] = {},
    cat_cols: List[str] = [],
    num_cols: List[str] = [],
    smote_random_state: int = 42
) -> Pipeline:
    """
    Builds a machine learning pipeline with preprocessing, oversampling, and a classifier.

    Args:
        model_type (str): The type of model to use ('logistic' or 'random_forest').
        model_params (Dict[str, Any]): Hyperparameters for the selected model.
        cat_cols (List[str]): A list of names for the categorical columns.
        num_cols (List[str]): A list of names for the numerical columns.
        smote_random_state (int): The random state for the SMOTE oversampler for reproducibility.

    Raises:
        ValueError: If an unsupported model_type is provided.

    Returns:
        Pipeline: A scikit-learn compatible pipeline object.
    """
    # Define preprocessing
    preprocessor = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(drop='first',
                              handle_unknown='ignore'),
                              cat_cols),
        ('num', StandardScaler(), num_cols),
    ])

    # Select model
    if model_type == "logistic":
        model = LogisticRegression(**model_params)
    elif model_type == "random_forest":
        model = RandomForestClassifier(**model_params)
    else:
        raise ValueError(f"Invalid model type: '{model_type}'. Supported types are 'logistic' and 'random_forest'.")

    # Build pipeline
    pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("smote", SMOTE(random_state=smote_random_state)),
        ("classifier", model)
    ])

    return pipeline
