# -*- coding: utf-8 -*-
import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Cargar los datos
file_path = "data/train.csv"
df = pd.read_csv(file_path)

# Seleccionar columnas numéricas y categóricas relevantes
selected_num_columns = ['LotFrontage', 'LotArea', 'TotalBsmtSF', 'GrLivArea', 'SalePrice']
categorical_columns_filtered = [
    'Street', 'LandContour', 'LandSlope', 'Utilities', 'Neighborhood', 'Condition1',
    'Condition2', 'HouseStyle', 'BldgType', 'OverallQual', 'OverallCond', 'RoofStyle',
    'Exterior1st', 'ExterCond', 'BsmtCond', 'BsmtFinType1', 'CentralAir', 'Heating',
    'KitchenQual', 'TotRmsAbvGrd', 'GarageType', 'GarageCond', 'PavedDrive',
    'SaleType', 'SaleCondition', 'Fireplaces', 'GarageCars'
]

df = df[selected_num_columns + categorical_columns_filtered]

# Manejo de valores nulos
df.fillna(df.mean(numeric_only=True), inplace=True)
for col in df.select_dtypes(include=['object']).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Convertir variables categóricas en dummies
df_encoded = pd.get_dummies(df, drop_first=True)

# Separar features y target
X = df_encoded.drop(['SalePrice'], axis=1)
y = df_encoded['SalePrice']

# Dividir y escalar datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar modelo XGBoost con los mejores parámetros
best_params = {
    'colsample_bytree': 1.0,
    'learning_rate': 0.1,
    'max_depth': 3,
    'min_child_weight': 3,
    'n_estimators': 200,
    'subsample': 1.0
}
model = xgb.XGBRegressor(**best_params, random_state=42)
model.fit(X_train_scaled, y_train)

# Obtener predicciones
y_pred = model.predict(X_test_scaled)

# Evaluar modelo
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Obtener importancia de las características
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False).head(10)

# Crear la aplicación Dash
app = dash.Dash(__name__)

# Definir estilos
COLORS = {
    'background': '#f8f9fa',
    'text': '#2c3e50',
    'accent': '#3498db',
    'light-accent': '#ebf5fb',
    'border': '#e9ecef'
}

# Layout de la aplicación
app.layout = html.Div([
    html.Div([
        html.H1("Análisis de Precios de Viviendas", style={'textAlign': 'center', 'color': COLORS['text']})
    ], style={'backgroundColor': COLORS['light-accent'], 'padding': '20px'}),

    # Sección de visualización de métricas
    html.Div([
        html.H3("Métricas del Modelo", style={'textAlign': 'center', 'color': COLORS['text']}),
        html.P(f"RMSE: ${rmse:,.2f}", style={'textAlign': 'center'}),
        html.P(f"R2 Score: {r2:.4f}", style={'textAlign': 'center'}),
    ], style={'padding': '20px', 'backgroundColor': COLORS['background']}),

    # Gráfico de Importancia de Características
    html.Div([
        html.H3("Top 10 Características Más Importantes", style={'textAlign': 'center', 'color': COLORS['text']}),
        dcc.Graph(
            figure=px.bar(feature_importance, x='importance', y='feature', orientation='h',
                          title="Top 10 Most Important Features")
        )
    ], style={'padding': '20px', 'backgroundColor': COLORS['background']}),

    # Gráfico de Predicción vs Real
    html.Div([
        html.H3("Actual vs Predicted Sale Prices", style={'textAlign': 'center', 'color': COLORS['text']}),
        dcc.Graph(
            figure=px.scatter(x=y_test, y=y_pred,
                              labels={'x': 'Actual Sale Price', 'y': 'Predicted Sale Price'},
                              title="Actual vs Predicted Sale Prices")
        )
    ], style={'padding': '20px', 'backgroundColor': COLORS['background']}),

    # Sección de Predicción Interactiva
    html.Div([
        html.H3("Predicción Interactiva", style={'textAlign': 'center', 'color': COLORS['text']}),

        # Dropdowns y sliders para input
        html.Div([
            html.Label("Garage Cars"),
            dcc.Slider(min=0, max=5, step=1, value=2, id='garage-cars-slider'),

            html.Label("Overall Quality"),
            dcc.Slider(min=1, max=10, step=1, value=5, id='overall-quality-slider'),

            html.Label("GrLivArea (Above Ground Living Area)"),
            dcc.Input(type='number', value=1500, id='grlivarea-input'),

            html.Label("TotalBsmtSF (Total Basement SF)"),
            dcc.Input(type='number', value=1000, id='totalbsmt-input'),

            html.Label("Lot Area"),
            dcc.Input(type='number', value=8000, id='lotarea-input')
        ], style={'padding': '20px', 'display': 'grid', 'grid-template-columns': 'repeat(2, 1fr)', 'gap': '20px'}),

        # Output de Predicción
        html.H3("Precio Predicho:", style={'textAlign': 'center', 'color': COLORS['text']}),
        html.Div(id='predicted-price', style={'textAlign': 'center', 'fontSize': '2em', 'color': COLORS['accent']})
    ], style={'padding': '20px', 'backgroundColor': COLORS['background']})
])


# Callback para actualizar la predicción basada en los inputs
@app.callback(
    Output('predicted-price', 'children'),
    [Input('garage-cars-slider', 'value'),
     Input('overall-quality-slider', 'value'),
     Input('grlivarea-input', 'value'),
     Input('totalbsmt-input', 'value'),
     Input('lotarea-input', 'value')]
)
def predict_price(garage_cars, overall_quality, grlivarea, totalbsmt, lotarea):
    # Crear dataframe de entrada para el modelo
    input_data = pd.DataFrame(columns=X.columns, data=np.zeros((1, len(X.columns))))
    input_data['GarageCars'] = garage_cars
    input_data['OverallQual'] = overall_quality
    input_data['GrLivArea'] = grlivarea
    input_data['TotalBsmtSF'] = totalbsmt
    input_data['LotArea'] = lotarea

    # Normalizar con el mismo scaler
    input_scaled = scaler.transform(input_data)

    # Predecir precio
    predicted_price = model.predict(input_scaled)[0]
    return f"${predicted_price:,.2f}"


# Ejecutar la aplicación Dash
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)