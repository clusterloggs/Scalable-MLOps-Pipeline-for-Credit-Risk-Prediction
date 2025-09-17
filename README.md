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

## How to Run

### 1. Setup

First, clone the repository and set up the Python environment:

```bash
# Clone the repo
git clone https://github.com/azkintonde/Scalable-MLOps-Pipeline-for-Credit-Risk-Prediction.git
cd Scalable-MLOps-Pipeline-for-Credit-Risk-Prediction

# Create and activate a virtual environment
py -3.12 -m venv venv (Python 3.12 is recommend)
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the ML Pipeline

Use DVC to reproduce the pipeline and generate the model artifact. You will need to configure your MLflow credentials first (e.g., by running `dagshub login`).

```bash
dvc repro
```
You can run this command ```dvc dag``` to see the pipeline flow diagram
### 3. Run the Web Application

*   **On Windows**, use `waitress`:
    ```bash
    waitress-serve --host=0.0.0.0 --port=8080 app.main:server
    ```
*   On **Linux or macOS** (and in Docker), use `gunicorn`:
    ```bash
    gunicorn --bind 0.0.0.0:8080 app.main:server
    ```

### 4. Run with Docker

The project includes a `Dockerfile` for containerizing the web application. This is the recommended way to run the application in production.

1.  **Build the Docker image:**
    ```bash
    docker build -t credit-risk-app .
    ```
2.  **Run the Docker container:**
    ```bash
    docker run -p 8080:8080 credit-risk-app
    ```
    You can then access the application at `http://localhost:8080`.

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
