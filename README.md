# ProyectoDSIA

Este es el proyecto de ciclo de la asignatura Proyecto - Desarrollo de soluciones. 

# Autores
Victor Perez

Jordi Sanchez

Simon Aristizabal

Maryi Alejandra Carvajal

# Panel de Predicción de Precios de Viviendas

Una aplicación Dash para predecir precios de viviendas utilizando modelos de aprendizaje automático.

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
