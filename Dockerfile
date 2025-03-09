FROM python:3.10-slim

WORKDIR /app

# Install git and dependencies for AWS CLI
RUN apt-get update && \
    apt-get install -y git curl unzip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Upgrade pyopenssl for better security
RUN pip install --upgrade pyopenssl

# Install DVC with S3 support
RUN pip install dvc[s3]

# Install AWS CLI v2 using curl
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf aws awscliv2.zip

# Copy the application code and saved models
COPY app2.py .
COPY saved_models/ ./saved_models/

# Copy and set permissions for entrypoint script
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Create data directory
RUN mkdir -p data

# Set environment variables
ENV PORT=8080

# Expose the port
EXPOSE 8080

# Use entrypoint script
ENTRYPOINT ["./entrypoint.sh"] 