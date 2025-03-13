# ProyectoDSIA

Este es el proyecto de ciclo de la asignatura Proyecto - Desarrollo de soluciones. 

# Autores
Victor Perez

Jordi Sanchez

Simon Aristizabal

Maryi Alejandra Carvajal

# Panel de Predicción de Precios de Viviendas

Una aplicación Dash para predecir precios de viviendas utilizando modelos de aprendizaje automático.

# 🛠 Manual de instalación

## Despliegue en Railway

Este proyecto está configurado para ser desplegado en Railway utilizando Docker.

### Pasos para el Despliegue

1. Sube este repositorio a GitHub
2. Conecta tu repositorio de GitHub a Railway
3. Configura las variables de entorno requeridas (ver abajo)
4. Railway detectará automáticamente el Dockerfile y desplegará la aplicación

### Variables de Entorno

Las siguientes variables de entorno son necesarias para que DVC extraiga datos de S3:

- `AWS_ACCESS_KEY_ID`: Tu clave de acceso AWS
- `AWS_SECRET_ACCESS_KEY`: Tu clave secreta AWS
- `AWS_SESSION_TOKEN`: Tu token de sesión AWS (requerido para credenciales temporales)
- `AWS_REGION`: (Opcional) Tu región AWS (ej., us-east-1)

Variables de entorno opcionales adicionales:

- `PORT`: El puerto en el que se ejecutará la aplicación (predeterminado: 8080)

### Control de Versiones de Datos (DVC)

Este proyecto utiliza DVC para acceder a archivos de datos almacenados en un bucket S3. Los datos se descargarán automáticamente durante el inicio del contenedor si se proporcionan las credenciales AWS.

#### Cómo Funciona DVC en Este Proyecto

1. Durante el inicio del contenedor, el repositorio se clona en un directorio temporal
2. Se configuran las credenciales AWS, incluyendo el token de sesión si se proporciona
3. Se ejecuta DVC pull en el repositorio clonado
4. Si DVC pull falla, se intenta una descarga directa de S3 como alternativa
5. Los datos se copian del repositorio clonado al directorio de la aplicación
6. La aplicación utiliza los datos del directorio `data/`

#### Solución de Problemas con DVC

Si encuentras problemas con el acceso a datos de DVC:

1. Verifica que tus credenciales AWS sean correctas
2. Asegúrate de haber proporcionado el token de sesión AWS si estás usando credenciales temporales
3. Comprueba que el bucket S3 `proyecto-dsia` exista y sea accesible
4. Asegúrate de tener los permisos necesarios para acceder al bucket S3
5. Verifica que el repositorio contenga la configuración DVC adecuada

### Ejecución de la Aplicación

La aplicación se ejecuta directamente con Python en lugar de usar Gunicorn. El script de entrada configura automáticamente la aplicación para escuchar en el puerto especificado y host (0.0.0.0).

### Desarrollo Local

Para ejecutar la aplicación localmente:

1. Clona el repositorio:
   ```
   git clone https://github.com/JordiNeil/ProyectoDSIA.git
   cd ProyectoDSIA
   ```

2. Instala las dependencias:
   ```
   pip install -r requirements.txt
   ```

3. Instala DVC y extrae los datos:
   ```
   pip install dvc dvc[s3]
   
   # Configura las credenciales AWS
   export AWS_ACCESS_KEY_ID=tu_clave_de_acceso
   export AWS_SECRET_ACCESS_KEY=tu_clave_secreta
   export AWS_SESSION_TOKEN=tu_token_de_sesion  # Si usas credenciales temporales
   export AWS_REGION=tu_region
   
   dvc pull
   ```

4. Ejecuta la aplicación:
   ```
   python app2.py
   ```

### Docker

Para construir y ejecutar el contenedor Docker localmente:

```bash
docker build -t house-price-dashboard .
docker run -p 8080:8080 \
  -e AWS_ACCESS_KEY_ID=tu_clave_de_acceso \
  -e AWS_SECRET_ACCESS_KEY=tu_clave_secreta \
  -e AWS_SESSION_TOKEN=tu_token_de_sesion \
  -e AWS_REGION=tu_region \
  house-price-dashboard
```

Luego accede a la aplicación en http://localhost:8080

# 📖 Manual de Usuario

## 1. Introducción

Este dashboard está diseñado para ayudar a los usuarios a analizar y predecir precios de viviendas utilizando un modelo de **Bayesian Ridge Regressor**. Proporciona información clave sobre el rendimiento del modelo, la importancia de las variables, las correlaciones entre características y permite realizar predicciones interactivas.

### 1.1 Objetivo del Dashboard
El objetivo principal es facilitar la interpretación de los precios de viviendas y brindar herramientas visuales para comprender los factores más influyentes en la predicción.
 
## 2. Secciones del Dashboard

### 2.1 Métricas del Modelo
Esta sección presenta las principales métricas del modelo:
- **Error Cuadrático Medio (MSE):** Mide la diferencia entre los valores predichos y los valores reales.
- **Coeficiente de Determinación (R²):** Indica la precisión del modelo en la predicción de precios.
- **Errores Absolutos Medios (MAE y Mediana AE):** Representan el error medio entre la predicción y el precio real.

### 2.2 Correlación con Precio
Esta gráfica muestra la correlación entre las variables y el precio de la vivienda. Factores con alta correlación pueden indicar fuertes relaciones predictivas.

### 2.3 Importancia de Características
Presenta las variables más influyentes en la predicción de precios. 
- **Ejemplo:** Si "Superficie" es la variable más importante, indica que el tamaño de la vivienda tiene un impacto significativo en el precio.

### 2.4 Precio Real vs. Predicho
Gráfico de dispersión que compara los valores predichos con los valores reales. Una línea de tendencia sugiere la precisión del modelo.

### 2.5 Precio por Ciudad / Vecindario
Visualización de los precios promedio en diferentes ciudades o vecindarios. Permite identificar zonas con precios más altos o bajos.

### 2.6 Correlación entre Variables Clave
Un mapa de calor que muestra las relaciones entre variables como "Tamaño", "Número de habitaciones" y "Precio".

### 2.7 Predicción Interactiva
Permite al usuario ingresar valores personalizados (ejemplo: número de habitaciones, calidad del vecindario) para obtener una predicción del precio con unas características predeterminadas.

## 3. Conclusión

El dashboard de predicción de precios de viviendas es una herramienta poderosa para analizar tendencias en el mercado inmobiliario y tomar decisiones informadas. Gracias a su enfoque visual y a la integración de modelos de aprendizaje automático, permite a usuarios de distintos niveles de experiencia comprender los factores que influyen en el precio de una propiedad.

El modelo Bayesian Ridge Regressor ofrece un equilibrio entre precisión y robustez, asegurando predicciones confiables. Sin embargo, es importante recordar que ningún modelo es perfecto, y los resultados deben ser interpretados en conjunto con otros factores del mercado inmobiliario.

Este manual ha sido diseñado para proporcionar una guía clara sobre cómo usar el dashboard y sacarle el máximo provecho. Se recomienda a los usuarios experimentar con las funcionalidades interactivas para obtener una comprensión más profunda de las dinámicas de precios en sus regiones de interés.