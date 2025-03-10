# ProyectoDSIA

Este es el proyecto de ciclo de la asignatura Proyecto - Desarrollo de soluciones. 

# Autores
Victor Perez

Jordi Sanchez

Simon Aristizabal

Maryi Alejandra Carvajal


# 📌 Análisis de Precios de Viviendas - Manual de Usuario e Instalación

Este documento proporciona una guía completa sobre el uso y la instalación del tablero interactivo de Análisis de Precios de Viviendas.

---

## 📖 Manual de Usuario

### Introducción

El tablero interactivo permite analizar los precios de viviendas utilizando un modelo de Machine Learning (XGBoost). Proporciona visualizaciones clave, información estadística y una herramienta de predicción interactiva para estimar el precio de una vivienda en función de sus características.

### Secciones del Tablero

1. **Métricas del Modelo** 📊

   - Muestra indicadores de precisión del modelo, incluyendo RMSE y R² Score.

2. **Estadísticas de Precios** 💰

   - Incluye el precio mínimo, máximo, promedio y mediano de las viviendas.

3. **Correlación con Precio** 🔗

   - Presenta un gráfico de barras con las variables más influyentes en el precio de la vivienda.

4. **Visualizaciones Clave** 📈

   - **Importancia de Características:** Muestra cuáles son las variables más relevantes.
   - **Precio por Calidad General:** Gráfico del precio promedio según la calidad de la vivienda.
   - **Precio Real vs. Predicho:** Comparación de los valores reales y predichos.
   - **Precio Promedio por Vecindario:** Análisis de los 10 vecindarios más costosos.
   - **Correlación entre Variables Clave:** Mapa de calor con la relación entre variables.
   - **Relación de Variables con Precio:** Gráfico interactivo que permite analizar diferentes variables en relación con el precio de la vivienda.

5. **Predicción Interactiva** 🎯

   - Permite ingresar valores para características clave de una vivienda y obtener una estimación del precio.
   - Características ajustables:
     - Cantidad de autos en el garaje.
     - Calidad general de la vivienda.
     - Área habitable.
     - Superficie total del sótano.
     - Tamaño del lote.

---

## 🛠 Manual de Instalación

### Requisitos Previos

Antes de instalar y ejecutar el tablero, asegúrate de cumplir con los siguientes requisitos:

- Tener **Python 3.8** o superior instalado.
- Tener `pip` instalado (viene por defecto con Python).
- Tener `git` instalado para clonar el repositorio.

### Pasos de Instalación

1. **Clonar el Repositorio**

   ```bash
   git clone https://github.com/JordiNeil/ProyectoDSIA.git
   cd repositorio
   ```

2. **Crear y Activar un Entorno Virtual (Opcional pero Recomendado)**

   - En Windows:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
   - En macOS/Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Instalar las Dependencias**

   ```bash
   pip install -r requirements.txt
   ```

4. **Ejecutar el Tablero**

   ```bash
   python app.py
   ```

5. **Abrir el Tablero en el Navegador**

   - Una vez que el servidor esté corriendo, abre tu navegador y ve a:
     ```
     http://127.0.0.1:8050
     ```

### Solución de Problemas

- Si `pip install -r requirements.txt` da error, intenta actualizar `pip`:
  ```bash
  python -m pip install --upgrade pip
  ```
- Si el puerto `8050` está ocupado, puedes cambiarlo ejecutando:
  ```bash
  python app.py --port 8080
  ```
- Si tienes problemas con `xgboost`, instálalo manualmente:
  ```bash
  pip install xgboost
  ```

---

