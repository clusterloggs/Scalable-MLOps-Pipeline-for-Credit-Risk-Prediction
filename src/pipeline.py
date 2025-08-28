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
    pipeline with preprocessing, ohe, oversampling, and a classifier.

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

    # Define a model registry for easy extension
    model_registry = {
        "logistic": LogisticRegression,
        "random_forest": RandomForestClassifier,
    }

    if model_type not in model_registry:
        raise ValueError(
            f"Invalid model type: '{model_type}'. Supported types are {list(model_registry.keys())}."
        )
    model = model_registry[model_type](**model_params)

    # Build pipeline
    pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("smote", SMOTE(random_state=smote_random_state)),
        ("classifier", model)
    ])

    return pipeline
