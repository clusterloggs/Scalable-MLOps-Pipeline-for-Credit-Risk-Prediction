# Scalable MLOps Pipeline for Credit Risk Prediction

This project implements an end-to-end machine learning pipeline to predict credit risk. It demonstrates a full MLOps workflow, including data versioning, experiment tracking, automated model training, and deployment of both an interactive UI and a REST API.

## Features

- **Reproducible Pipeline:** Uses DVC to define and execute a version-controlled ML pipeline (preprocessing, training, evaluation).
- **Experiment Tracking:** Integrates with MLflow and Dagshub for logging parameters, metrics, and model artifacts.
- **Dual Deployment:**
  - **Interactive UI:** A Dash dashboard for visual data analysis and single-prediction queries.
  - **REST API:** A Flask API for programmatic access to the model.
- **CI/CD Ready:** Structured for easy integration with CI/CD tools like GitHub Actions for automated testing and deployment.
- **Clean Code:** Follows best practices with modular, well-documented Python scripts.

## Tech Stack

- **ML & Data:** Python, Scikit-learn, Pandas, imbalanced-learn
- **Pipeline & Versioning:** DVC, Git
- **Experiment Tracking:** MLflow, Dagshub
- **Web Serving:**
  - **Dash & Plotly:** For the interactive UI dashboard.
  - **Flask:** For the REST API.
- **Containerization:** Docker (Dockerfile ready for setup)

## Project Structure

```
credit-risk-mlops/
|
├── notebooks/              # EDA and prototyping notebooks
|   |── Credit_Prediction.ipynb
|   |── rf_hyperpara.ipynb
|   |── lr_hyperpara.ipynb
|   |── xgboost_model.ipynb
|
├── src/                    # Core ML logic
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_evaluation.py
|   ├── model_training.py
│   ├── predict_cli.py
│   └── utils.py
|
├── app/                    # Web app (Flask)
│   └── main.py
|
├── models/                 # Trained model artifacts
│   └── model.joblib
|
├── Dockerfile              # For containerization
├── requirements.txt        # Python dependencies
├── .github/workflows/      # CI/CD pipelines
│   └── deploy.yml
├── .gitignore
└── README.md
```
