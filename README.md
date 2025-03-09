# ProyectoDSIA

Este es el proyecto de ciclo de la asignatura Proyecto - Desarrollo de soluciones. 

# Autores
Victor Perez

Jordi Sanchez

Simon Aristizabal

Maryi Alejandra Carvajal

# House Price Prediction Dashboard

A Dash application for predicting house prices using machine learning models.

## Deployment on Railway

This project is configured for deployment on Railway using Docker.

### Deployment Steps

1. Push this repository to GitHub
2. Connect your GitHub repository to Railway
3. Set the required environment variables (see below)
4. Railway will automatically detect the Dockerfile and deploy the application

### Environment Variables

The following environment variables are required for DVC to pull data from S3:

- `AWS_ACCESS_KEY_ID`: Your AWS access key
- `AWS_SECRET_ACCESS_KEY`: Your AWS secret key
- `AWS_SESSION_TOKEN`: Your AWS session token (required for temporary credentials)
- `AWS_REGION`: (Optional) Your AWS region (e.g., us-east-1)

Additional optional environment variables:

- `PORT`: The port on which the application will run (default: 8080)

### Data Version Control (DVC)

This project uses DVC to access data files stored in an S3 bucket. The data will be automatically downloaded during container startup if the AWS credentials are provided.

#### How DVC Works in This Project

1. During container startup, the repository is cloned to a temporary directory
2. AWS credentials are configured, including the session token if provided
3. DVC pull is executed in the cloned repository
4. If DVC pull fails, a direct S3 download is attempted as a fallback
5. The data is copied from the cloned repository to the application directory
6. The application uses the data from the `data/` directory

#### Troubleshooting DVC

If you encounter issues with DVC data access:

1. Verify your AWS credentials are correct
2. Ensure you've provided the AWS session token if using temporary credentials
3. Check that the S3 bucket `proyecto-dsia` exists and is accessible
4. Ensure you have the necessary permissions to access the S3 bucket
5. Verify that the repository contains the proper DVC configuration

### Application Execution

The application is run directly with Python instead of using Gunicorn. The entrypoint script automatically configures the application to listen on the specified port and host (0.0.0.0).

### Local Development

To run the application locally:

1. Clone the repository:
   ```
   git clone https://github.com/JordiNeil/ProyectoDSIA.git
   cd ProyectoDSIA
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Install DVC and pull the data:
   ```
   pip install dvc dvc[s3]
   
   # Configure AWS credentials
   export AWS_ACCESS_KEY_ID=your_access_key
   export AWS_SECRET_ACCESS_KEY=your_secret_key
   export AWS_SESSION_TOKEN=your_session_token  # If using temporary credentials
   export AWS_REGION=your_region
   
   dvc pull
   ```

4. Run the application:
   ```
   python app2.py
   ```

### Docker

To build and run the Docker container locally:

```bash
docker build -t house-price-dashboard .
docker run -p 8080:8080 \
  -e AWS_ACCESS_KEY_ID=your_access_key \
  -e AWS_SECRET_ACCESS_KEY=your_secret_key \
  -e AWS_SESSION_TOKEN=your_session_token \
  -e AWS_REGION=your_region \
  house-price-dashboard
```

Then access the application at http://localhost:8080
