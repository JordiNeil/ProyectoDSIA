FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and saved models
COPY app2.py .
COPY saved_models/ ./saved_models/

# Create data directory
RUN mkdir -p data

# Set environment variables
ENV PORT=8080

# Expose the port
EXPOSE 8080

# Command to run the application
CMD gunicorn --bind 0.0.0.0:$PORT app2:server 