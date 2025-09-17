# Stage 1: Builder - Install dependencies in a virtual environment
FROM python:3.12-slim as builder

WORKDIR /app

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Create and activate a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install dependencies into the venv
COPY requirements.txt requirements-serving.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-serving.txt

# Stage 2: Final Image - Setup the runtime environment
FROM python:3.12-slim

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create a non-root user for security
RUN useradd --create-home appuser
USER appuser

# Copy application code and necessary data/model files
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser models/ ./models/
COPY --chown=appuser:appuser data/df_clean.parquet ./data/df_clean.parquet
COPY --chown=appuser:appuser params.yaml ./params.yaml

# Expose the port the app runs on
EXPOSE 8080

# Set the command to run the application using Gunicorn (the production server for Linux)
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app.main:server"]