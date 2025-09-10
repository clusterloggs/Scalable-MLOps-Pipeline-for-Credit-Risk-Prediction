
FROM python:3.12 AS builder

WORKDIR /app


COPY requirements-serving.txt .

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements-serving.txt


FROM python:3.12-slim

WORKDIR /app

# Install curl for health checks, then clean up the apt cache to keep the image small
RUN apt-get update && \
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd --system app && useradd --system --gid app app

COPY --from=builder /opt/venv /opt/venv

# Copy only the necessary application code, data, and models for serving
COPY --chown=app:app app/ /app/app
COPY --chown=app:app models/ /app/models
COPY --chown=app:app data/df_clean.parquet /app/data/df_clean.parquet

ENV PATH="/opt/venv/bin:$PATH"
USER app

EXPOSE 8000

# Add a health check to ensure the application is responsive.
# It gives the app 30s to start, then checks every 30s.
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/ || exit 1

CMD ["gunicorn", "--workers=4", "--bind", "0.0.0.0:8000", "app.main:app"]