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
    'background': '#ffffff',
    'secondary-bg': '#f8f9fa',
    'text': '#2c3e50',
    'accent': '#3498db',
    'accent-dark': '#2980b9',
    'light-accent': '#e3f2fd',
    'border': '#e9ecef',
    'success': '#2ecc71',
    'card-shadow': '0px 4px 12px rgba(0, 0, 0, 0.1)'
}

# Estilos comunes
card_style = {
    'backgroundColor': COLORS['background'],
    'borderRadius': '8px',
    'padding': '20px',
    'marginBottom': '20px',
    'boxShadow': COLORS['card-shadow'],
    'border': f'1px solid {COLORS["border"]}'
}

title_style = {
    'color': COLORS['text'],
    'marginBottom': '20px',
    'fontWeight': 'bold',
    'textAlign': 'center'
}

# Layout de la aplicación
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Análisis de Precios de Viviendas", 
                style={'textAlign': 'center', 'color': 'white', 'fontWeight': 'bold', 'marginBottom': '0'})
    ], style={'backgroundColor': COLORS['accent-dark'], 'padding': '30px 20px', 'borderRadius': '0 0 10px 10px', 'boxShadow': COLORS['card-shadow']}),
    
    # Contenedor principal
    html.Div([
        # Fila superior - Métricas y Características importantes
        html.Div([
            # Tarjeta de métricas
            html.Div([
                html.H3("Métricas del Modelo", style=title_style),
                html.Div([
                    html.Div([
                        html.H4("RMSE", style={'color': COLORS['text'], 'marginBottom': '5px', 'textAlign': 'center'}),
                        html.P(f"${rmse:,.2f}", style={'textAlign': 'center', 'fontSize': '1.5em', 'fontWeight': 'bold', 'color': COLORS['accent-dark']}),
                    ], style={'flex': '1', 'padding': '10px', 'backgroundColor': COLORS['light-accent'], 'borderRadius': '5px'}),
                    
                    html.Div([
                        html.H4("R² Score", style={'color': COLORS['text'], 'marginBottom': '5px', 'textAlign': 'center'}),
                        html.P(f"{r2:.4f}", style={'textAlign': 'center', 'fontSize': '1.5em', 'fontWeight': 'bold', 'color': COLORS['accent-dark']}),
                    ], style={'flex': '1', 'padding': '10px', 'backgroundColor': COLORS['light-accent'], 'borderRadius': '5px'}),
                ], style={'display': 'flex', 'gap': '20px'})
            ], style={**card_style, 'flex': '1'}),
            
            # Gráfico de Importancia de Características
            html.Div([
                html.H3("Top 10 Características Más Importantes", style=title_style),
                dcc.Graph(
                    figure=px.bar(
                        feature_importance, 
                        x='importance', 
                        y='feature', 
                        orientation='h',
                        color='importance',
                        color_continuous_scale=['#74b9ff', '#0984e3', '#2980b9'],
                        labels={'importance': 'Importancia', 'feature': 'Característica'}
                    ).update_layout(
                        plot_bgcolor=COLORS['background'],
                        paper_bgcolor=COLORS['background'],
                        margin=dict(l=10, r=10, t=10, b=10),
                        coloraxis_showscale=False
                    )
                )
            ], style={**card_style, 'flex': '2'})
        ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px'}),
        
        # Fila media - Gráfico de Predicción vs Real
        html.Div([
            html.H3("Precios Reales vs. Predichos", style=title_style),
            dcc.Graph(
                figure=px.scatter(
                    x=y_test, 
                    y=y_pred,
                    labels={'x': 'Precio Real ($)', 'y': 'Precio Predicho ($)'},
                    opacity=0.7
                ).update_layout(
                    plot_bgcolor=COLORS['background'],
                    paper_bgcolor=COLORS['background'],
                    margin=dict(l=10, r=10, t=10, b=10)
                ).add_shape(
                    type="line", line=dict(dash="dash", color=COLORS['accent']),
                    x0=y_test.min(), y0=y_test.min(),
                    x1=y_test.max(), y1=y_test.max()
                )
            )
        ], style=card_style),
        
        # Sección de Predicción Interactiva
        html.Div([
            html.H3("Predicción Interactiva", style=title_style),
            
            # Contenedor para inputs y resultado
            html.Div([
                # Columna de inputs
                html.Div([
                    # Garage Cars
                    html.Div([
                        html.Label("Garage Cars", style={'fontWeight': 'bold', 'color': COLORS['text']}),
                        dcc.Slider(
                            min=0, max=5, step=1, value=2, id='garage-cars-slider',
                            marks={i: str(i) for i in range(6)},
                            className='custom-slider'
                        ),
                    ], style={'marginBottom': '20px'}),
                    
                    # Overall Quality
                    html.Div([
                        html.Label("Overall Quality", style={'fontWeight': 'bold', 'color': COLORS['text']}),
                        dcc.Slider(
                            min=1, max=10, step=1, value=5, id='overall-quality-slider',
                            marks={i: str(i) for i in range(1, 11)},
                            className='custom-slider'
                        ),
                    ], style={'marginBottom': '20px'}),
                    
                    # GrLivArea
                    html.Div([
                        html.Label("GrLivArea (Above Ground Living Area)", style={'fontWeight': 'bold', 'color': COLORS['text']}),
                        dcc.Input(
                            type='number', value=1500, id='grlivarea-input',
                            style={'width': '100%', 'padding': '8px', 'borderRadius': '4px', 'border': f'1px solid {COLORS["border"]}'}
                        ),
                    ], style={'marginBottom': '20px'}),
                    
                    # TotalBsmtSF
                    html.Div([
                        html.Label("TotalBsmtSF (Total Basement SF)", style={'fontWeight': 'bold', 'color': COLORS['text']}),
                        dcc.Input(
                            type='number', value=1000, id='totalbsmt-input',
                            style={'width': '100%', 'padding': '8px', 'borderRadius': '4px', 'border': f'1px solid {COLORS["border"]}'}
                        ),
                    ], style={'marginBottom': '20px'}),
                    
                    # Lot Area
                    html.Div([
                        html.Label("Lot Area", style={'fontWeight': 'bold', 'color': COLORS['text']}),
                        dcc.Input(
                            type='number', value=8000, id='lotarea-input',
                            style={'width': '100%', 'padding': '8px', 'borderRadius': '4px', 'border': f'1px solid {COLORS["border"]}'}
                        ),
                    ]),
                ], style={'flex': '3', 'padding': '10px'}),
                
                # Columna de resultado
                html.Div([
                    html.Div([
                        html.H3("Precio Predicho", style={'textAlign': 'center', 'color': COLORS['text'], 'marginBottom': '20px'}),
                        html.Div(
                            id='predicted-price', 
                            style={
                                'textAlign': 'center', 
                                'fontSize': '2.5em', 
                                'fontWeight': 'bold',
                                'color': COLORS['success'],
                                'padding': '30px 0'
                            }
                        ),
                        html.P("Basado en las características seleccionadas", style={'textAlign': 'center', 'color': COLORS['text'], 'opacity': '0.7'})
                    ], style={
                        'backgroundColor': COLORS['light-accent'], 
                        'padding': '20px', 
                        'borderRadius': '8px',
                        'height': '100%',
                        'display': 'flex',
                        'flexDirection': 'column',
                        'justifyContent': 'center'
                    })
                ], style={'flex': '2', 'padding': '10px'})
            ], style={'display': 'flex'})
        ], style=card_style),
        
        # Footer
        html.Div([
            html.P("Dashboard de Análisis de Precios de Viviendas | Desarrollado con Dash y XGBoost", 
                   style={'textAlign': 'center', 'color': COLORS['text'], 'opacity': '0.7'})
        ], style={'marginTop': '20px'})
        
    ], style={'maxWidth': '1200px', 'margin': '20px auto', 'padding': '0 20px'})
], style={'backgroundColor': COLORS['secondary-bg'], 'minHeight': '100vh', 'fontFamily': 'Arial, sans-serif'})


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