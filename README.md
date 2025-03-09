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
3. Railway will automatically detect the Dockerfile and deploy the application

### Environment Variables

No specific environment variables are required, but you can set:

- `PORT`: The port on which the application will run (default: 8080)

### Local Development

To run the application locally:

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the application:
   ```
   python app2.py
   ```

### Docker

To build and run the Docker container locally:

```bash
docker build -t house-price-dashboard .
docker run -p 8080:8080 house-price-dashboard
```

Then access the application at http://localhost:8080
