# Scalable-MLOps-Pipeline-for-Credit-Risk-Prediction


An end-to-end machine learning web application for credit risk classification.
It predicts the likelihood of a borrower defaulting on a loan using supervised learning and includes full MLOps capabilities, including model training, deployment (using Docker and Heroku), CI/CD (GitHub Actions), and automated monitoring/retraining.

## Features

- Data preprocessing and feature engineering
- Model training and evaluation (Random Forest, SVM, Logistic Regression)
- Modular pipeline with sklearn.pipeline
- Model versioning with joblib
- Containerized with Docker
- CI/CD enabled via GitHub Actions
- Deployed on Heroku
- Model monitoring & auto-retraining support

## Tech Stack

- Python (3.12)
- Pandas, NumPy, Scikit-Learn
- Streamlit or Flask (for UI)
- Docker & Heroku (Deployment)
- GitHub Actions (CI/CD)
- MLflow (Optional: Model registry)

## Project Structure
```
credit-risk-mlops/
|
├── notebooks/              # EDA and prototyping notebooks
|   |── Credit_Prediction.ipynb
|   |── rf_hyperpara.ipynb
├── src/                    # Core ML logic (training, pipeline, utils)
│   ├── pipeline.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
|
├── app/                    # Web app (Streamlit or Flask)
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
