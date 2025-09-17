# Use a smaller, more secure base image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Create a non-root user and group
RUN groupadd --system app && useradd --system --gid app app

# Copy just the requirements file to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt && rm -f requirements.txt

# Copy the rest of the application code and set permissions
COPY --chown=app:app . /app

# Switch to the non-root user
USER app

# Expose the port the app runs on
EXPOSE $PORT

# The command to run the application
CMD ["gunicorn", "--workers=4", "--bind", "0.0.0.0:$PORT", "app:app"]