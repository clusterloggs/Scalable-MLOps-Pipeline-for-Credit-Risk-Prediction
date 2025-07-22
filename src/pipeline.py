# src/pipeline.py
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

def build_pipeline(model_type="logistic"):
    # Define preprocessing
    cat_cols = ['home_status', 'loan_intent', 'loan_grade']
    num_cols = ['income', 'emp_years', 'loan_amount',
                'loan_int_rate', 'loan_status', 'loan_percent_income',
                'credit_history']
    
    preprocessor = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(drop='first',
                              handle_unknown='ignore'),
                              cat_cols),
        ('num', StandardScaler(), num_cols)
    ])

    # Select model
    if model_type == "logistic":
        model = LogisticRegression(
            solver="liblinear",
            penalty="l2",
            class_weight=None,
            C=10,
            max_iter=1000
        )
    elif model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=1000, random_state=42)
    else:
        raise ValueError("Invalid model type")

    # Build pipeline
    pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", model)
    ])
    
    return pipeline
