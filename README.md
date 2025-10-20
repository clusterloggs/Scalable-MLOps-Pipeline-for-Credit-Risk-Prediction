# Scalable MLOps Pipeline for Credit Risk Prediction

This project provides a complete, end-to-end MLOps pipeline for training and deploying a credit risk prediction model. It is designed to be scalable, reproducible, and production-ready, demonstrating best practices in machine learning engineering.

The primary purpose is to predict the likelihood of a loan applicant defaulting on their credit. The project automates the entire ML lifecycle, from data processing and experiment tracking to model deployment as both a REST API and an interactive web dashboard.

## Key Features

-   **Reproducible ML Pipeline**: Uses DVC to version-control data and define a multi-stage pipeline (preprocessing, training, evaluation), ensuring full reproducibility.
-   **Comprehensive Experiment Tracking**: Integrates with MLflow and DagsHub to log parameters, metrics, and model artifacts for every experiment.
-   **Dual Deployment Options**:
    -   **Interactive UI**: A Dash dashboard for visual data exploration and single-prediction queries.
    -   **REST API**: A Flask API for programmatic, low-latency predictions.
-   **Automated Model Promotion**: Models are automatically evaluated and can be promoted based on performance thresholds defined in `params.yaml`.
-   **Containerized Application**: A `Dockerfile` is included for easy, consistent deployment in any environment.
-   **CI/CD Integration**: The project structure and scripts are designed for easy integration with CI/CD workflows (e.g., GitHub Actions) for automated testing and deployment.
-   **Modular & Clean Code**: Follows modern software engineering principles with a clear separation of concerns.

## Tech Stack

| Category                | Technologies                                                              |
| ----------------------- | ------------------------------------------------------------------------- |
| **ML & Data Science**   | `Python`, `Scikit-learn`, `Pandas`, `imbalanced-learn`                      |
| **Pipeline & Versioning** | `DVC`, `Git`                                                              |
| **Experiment Tracking** | `MLflow`, `DagsHub`                                                       |
| **Web Serving & API**   | `Flask`, `Dash`, `Plotly`, `Gunicorn`, `Waitress`                           |
| **Containerization**    | `Docker`                                                                  |

## MLOps Pipeline Architecture

The project follows a standard MLOps workflow managed by DVC and Git.

1.  **Data Versioning**: Raw and processed data are tracked by DVC, ensuring that any version of the data can be checked out and used.
2.  **DVC Pipeline (`dvc.yaml`)**: Defines the stages of the ML workflow:
    -   `prepare_data`: Splits the raw data into training and testing sets.
    -   `train_model`: Preprocesses data, handles class imbalance with SMOTE, trains the model (Logistic Regression or Random Forest), and logs experiments to MLflow.
    -   `evaluate_model`: Evaluates the trained model on the test set and logs performance metrics.
3.  **Experiment Tracking**: MLflow, connected to a DagsHub remote, logs all parameters, metrics (like F1-score, AUC), and the trained model artifact for each run.
4.  **Deployment**: The final model is served via a Flask/Dash application, which can be run locally or as a Docker container.

You can visualize the pipeline dependencies by running:
```bash
dvc dag
```

## Installation & Setup

### Prerequisites

-   Git
-   Python (3.12 is recommended)
-   DVC
-   Docker (for containerized deployment)

### Step-by-Step Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/azkintonde/Scalable-MLOps-Pipeline-for-Credit-Risk-Prediction.git
    cd Scalable-MLOps-Pipeline-for-Credit-Risk-Prediction
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # On Windows
    py -3.12 -m venv venv
    venv\Scripts\activate

    # On macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure DagsHub for MLflow Tracking:**
    This allows DVC and MLflow to push artifacts and metrics to your DagsHub remote.
    ```bash
    dagshub login
    ```
    Follow the prompts to authenticate. This will configure your local machine to communicate with the MLflow tracking server specified in `params.yaml`.

## Usage Guide

### 1. Run the ML Pipeline

To execute the entire pipeline from data preparation to model evaluation, run the following DVC command. DVC will automatically run each stage in the correct order.

```bash
dvc repro
```

This command will:
-   Split the data.
-   Train a new model using the parameters in `params.yaml`.
-   Log all results to MLflow.
-   Save the trained model to `models/model.joblib`.

### 2. Run the Web Application

After the pipeline has been run and `models/model.joblib` is generated, you can start the web server. The application provides both a Dash UI and a Flask API.

-   **On Windows (using Waitress):**
    ```bash
    waitress-serve --host=0.0.0.0 --port=8080 app.main:server
    ```

-   **On macOS/Linux (using Gunicorn):**
    ```bash
    gunicorn --bind 0.0.0.0:8080 app.main:server
    ```

Once running, access the interactive dashboard at `http://localhost:8080`.

## Project Structure

```
credit-risk-mlops/
├── .github/workflows/      # CI/CD pipeline definitions (e.g., for automated deployment)
│   └── deploy.yml
├── app/                    # Web application source code
│   └── main.py             # Contains both the Flask API and the Dash UI
├── data/                   # Data files (tracked by DVC)
│   ├── df_clean.parquet    # Raw input data
│   └── processed/          # Processed data splits
│       ├── train.parquet
│       └── test.parquet
├── models/                 # Trained model artifacts (tracked by DVC)
│   └── model.joblib
├── notebooks/              # Jupyter notebooks for exploration and prototyping
├── src/                    # Core ML pipeline scripts
│   ├── data_preprocessing.py # Data splitting logic
│   ├── feature_engineering.py# Feature transformation logic (if any)
│   ├── model_evaluation.py   # Model evaluation and metric calculation
│   ├── model_training.py     # Model training, SMOTE, and MLflow logging
│   ├── predict_cli.py        # CLI for making single predictions
│   └── utils.py              # Utility functions (e.g., loading/saving objects)
├── Dockerfile              # Defines the container for the web application
├── dvc.yaml                # DVC pipeline definition file
├── LICENSE                 # Project license file
├── params.yaml             # Configuration for the entire project
├── README.md               # This file
└── requirements.txt        # Python dependencies
```

## Configuration

All project configurations are centralized in `params.yaml`. This allows you to change parameters without modifying the source code.

```yaml
# params.yaml

mlflow:
  experiment_name: "Credit Risk Prediction"
  registered_model_name: "CreditRiskModel"
  tracking_uri: "https://dagshub.com/azkintonde/Scalable-MLOps-Pipeline-for-Credit-Risk-Prediction.mlflow"

data:
  raw_path: "data/df_clean.parquet"
  train_path: "data/processed/train.parquet"
  test_path: "data/processed/test.parquet"

train:
  model_type: logistic   # 'logistic' or 'random_forest'
  test_split_ratio: 0.2
  random_state: 42
  stratify: true
  smote_random_state: 42
  decision_threshold: 0.65
  promotion_metric: 'f1_score' # Metric to check for promotion
  promotion_threshold: 0.85      # Threshold for the metric
  label_col: default

features:
  categorical:
    - home_status
    - loan_intent
    - loan_grade
  numerical:
    - income
    - emp_years
    - loan_amount
    - loan_int_rate
    - loan_percent_income
    - credit_history

# --- Model Hyperparameters ---
logistic:
  solver: liblinear
  penalty: l2
  C: 0.01
  max_iter: 1200
  class_weight: balanced

random_forest:
  max_features: 5
  n_estimators: 405
  class_weight: balanced
  max_depth: 8
  min_samples_split: 2
  min_samples_leaf: 1
```

## API Documentation

The Flask application exposes a REST API for predictions.

### Endpoint: `/predict`

-   **Method**: `POST`
-   **Description**: Submits a single applicant's data and returns a credit risk prediction.
-   **Content-Type**: `application/json`

#### Request Body

A JSON object containing the features for a single prediction. All features defined in `params.yaml` under `features` are required.

**Example Request:**
```bash
curl -X POST http://localhost:8080/predict \
-H "Content-Type: application/json" \
-d '{
    "home_status": "RENT",
    "loan_intent": "PERSONAL",
    "loan_grade": "D",
    "income": 59000,
    "emp_years": 3,
    "loan_amount": 35000,
    "loan_int_rate": 16.02,
    "loan_percent_income": 0.59,
    "credit_history": 3
}'
```

#### Response Body

A JSON object containing the prediction result and the probability of default.

**Example Success Response:**
```json
{
    "prediction": "Default",
    "probability": 0.88
}
```

## Deployment

The recommended method for deployment is using the provided `Dockerfile`.

1.  **Ensure DVC is configured for production:**
    Before building the image, ensure your DVC remote is accessible from your deployment environment (e.g., by setting up credentials). Pull the required model artifacts:
    ```bash
    dvc pull models/model.joblib
    ```

2.  **Build the Docker image:**
    ```bash
    docker build -t credit-risk-app .
    ```

3.  **Run the Docker container:**
    This command maps the container's port 8080 to the host's port 8080.
    ```bash
    docker run -p 8080:8080 credit-risk-app
    ```

The application will be available at `http://localhost:8080`.

### Production Considerations

-   **Web Server**: The `Dockerfile` uses `gunicorn` as the production WSGI server, which is more robust than Flask's built-in server.
-   **Environment Variables**: For more complex deployments, consider parameterizing settings like model paths or MLflow URIs using environment variables instead of hardcoding them in `params.yaml`.
-   **CI/CD**: The `.github/workflows/deploy.yml` file serves as a template for setting up automated deployment to a cloud service or container registry.

## Development

### Contribution Guidelines

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix: `git checkout -b feature/my-new-feature`.
3.  Make your changes. Ensure you follow the existing code style.
4.  If you change the pipeline (`dvc.yaml`) or add new data/models, update the DVC tracking accordingly.
5.  Commit your changes and push to your branch: `git push origin feature/my-new-feature`.
6.  Open a pull request.

### Testing

While this project does not currently include a dedicated test suite (`pytest`), contributions to add unit and integration tests are welcome. Key areas to test include:
-   Data preprocessing logic in `src/model_training.py`.
-   Prediction logic in `app/main.py`.
-   API endpoint request/response validation.

## License

This project is licensed under the **Apache License 2.0**. See the LICENSE file for more details.
