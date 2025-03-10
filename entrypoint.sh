#!/bin/bash
set -e

# Clone the repository
echo "Cloning repository..."
git clone https://github.com/JordiNeil/ProyectoDSIA.git /tmp/repo

# Configure AWS credentials if provided as environment variables
if [ -n "$AWS_ACCESS_KEY_ID" ] && [ -n "$AWS_SECRET_ACCESS_KEY" ]; then
    echo "Configuring AWS credentials..."
    mkdir -p ~/.aws
    
    # Check if session token is provided
    if [ -n "$AWS_SESSION_TOKEN" ]; then
        echo "Using AWS session token..."
        cat > ~/.aws/credentials << EOF
[default]
aws_access_key_id = $AWS_ACCESS_KEY_ID
aws_secret_access_key = $AWS_SECRET_ACCESS_KEY
aws_session_token = $AWS_SESSION_TOKEN
EOF
    else
        cat > ~/.aws/credentials << EOF
[default]
aws_access_key_id = $AWS_ACCESS_KEY_ID
aws_secret_access_key = $AWS_SECRET_ACCESS_KEY
EOF
    fi
    
    if [ -n "$AWS_REGION" ]; then
        cat > ~/.aws/config << EOF
[default]
region = $AWS_REGION
EOF
    fi
    
    # Test AWS credentials
    echo "Testing AWS credentials..."
    aws s3 ls s3://proyecto-dsia/ || echo "Warning: Could not list S3 bucket. Check your credentials."
fi

# Pull data using DVC from the cloned repository
echo "Pulling data from DVC..."
cd /tmp/repo
dvc pull

# If DVC pull fails, try direct S3 download
if [ $? -ne 0 ]; then
    echo "DVC pull failed, trying direct S3 download..."
    mkdir -p data
    aws s3 cp s3://proyecto-dsia/data/ data/ --recursive || echo "Warning: Direct S3 download failed."
fi

# Copy data to the application directory
echo "Copying data to application directory..."
if [ -d "/tmp/repo/data" ] && [ "$(ls -A /tmp/repo/data 2>/dev/null)" ]; then
    mkdir -p /app/data
    cp -r /tmp/repo/data/* /app/data/
    echo "Data copied successfully."
else
    echo "WARNING: Data directory not found or empty in the repository."
fi

# Return to the application directory
cd /app

# Start the application
echo "Starting the application..."
exec python app2.py
