# -*- coding: utf-8 -*-
import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectFromModel
import pickle
import os

# Cargar los datos
file_path = "data/train.csv"
df = pd.read_csv(file_path)

# df_encoded = pd.get_dummies(df, drop_first=True)
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Identify categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

# Separar features y target
X = df.drop(['SalePrice'], axis=1)
y = df['SalePrice']

# Aplicar transformación logarítmica al precio de venta para evaluación
y_log = np.log1p(y)

# Dividir datos para evaluación
X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

# Cargar el modelo pre-entrenado
model_path = os.path.join('saved_models', 'BR_regressor.pkl')
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    print("Model loaded successfully with the following components:")
    if hasattr(model, 'named_steps'):
        for step_name in model.named_steps:
            print(f"- {step_name}: {type(model.named_steps[step_name]).__name__}")
    
except Exception as e:
    print(f"Error loading model: {e}")

# Obtener parámetros del modelo para visualización
best_params = model['br'].get_params() if hasattr(model['br'], 'get_params') else {}

# Obtener predicciones
y_pred_log = model.predict(X_test)
# Convertir de vuelta a la escala original
y_pred = np.expm1(y_pred_log)
y_test_original = np.expm1(y_test)

# Evaluar modelo
mse = mean_squared_error(y_test_original, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_original, y_pred)

# Obtener importancia de las características
if 'feature_importance' in globals():
    print("Using existing feature importance")
else:
    print("Extracting feature importance...")
    if hasattr(model, 'named_steps') and 'feature_selection' in model.named_steps and 'br' in model.named_steps:
        # Obtener el selector y el modelo
        selector = model.named_steps['feature_selection']
        br_model = model.named_steps['br']
        preprocessor = model.named_steps['preprocessor']
        
        # Obtener features seleccionadas
        support = selector.get_support()
        
        cat_ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_feature_names = cat_ohe.get_feature_names_out(categorical_columns)

        # Combine numeric and categorical feature names.
        all_feature_names = numeric_columns + list(cat_feature_names)
        print("All features:", len(all_feature_names))
        selected_features = [name for name, selected in zip(all_feature_names, support) if selected]

        print("Selected features:")
        print(selected_features)
        
        
        # Crear DataFrame con importancia
        test_importance = pd.DataFrame({
            'feature': selected_features,
            'importance': np.abs(br_model.coef_)
        }).sort_values('importance', ascending=False)
        
        print("Todas als caracteristicas seleccionadas:")
        print(test_importance)

        feature_importance = test_importance.head(10)
        print("Top 10 características más importantes:")
        print(feature_importance)
    else:
        raise ValueError("Model structure doesn't match expected pipeline")

# Preparar datos adicionales para visualizaciones
# Distribución de precios
price_stats = {
    'min': int(df['SalePrice'].min()),
    'max': int(df['SalePrice'].max()),
    'mean': int(df['SalePrice'].mean()),
    'median': int(df['SalePrice'].median())
}

# Precio promedio por calidad general
avg_price_by_quality = df.groupby('OverallQual')['SalePrice'].mean().reset_index()

# Precio promedio por vecindario (top 10)
avg_price_by_neighborhood = df.groupby('Neighborhood')['SalePrice'].mean().sort_values(ascending=False).reset_index().head(10)

# Correlación entre variables numéricas y precio
corr_with_price = df[numeric_columns].corr()['SalePrice'].sort_values(ascending=False).drop('SalePrice')

# Variables más correlacionadas con el precio
# Check which features are available in the dataset
available_corr_features = ["SalePrice", "OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF"]
# Remove YearBuilt since it's not in the filtered dataset
corr_df = df[available_corr_features].corr()

# Crear la aplicación Dash
app = dash.Dash(__name__, meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])

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
    'warning': '#f39c12',
    'danger': '#e74c3c',
    'card-shadow': '0px 4px 12px rgba(0, 0, 0, 0.1)'
}

# Estilos comunes
card_style = {
    'backgroundColor': COLORS['background'],
    'borderRadius': '8px',
    'padding': '15px',
    'marginBottom': '15px',
    'boxShadow': COLORS['card-shadow'],
    'border': f'1px solid {COLORS["border"]}',
    'height': '100%'
}

title_style = {
    'color': COLORS['text'],
    'marginBottom': '15px',
    'fontWeight': 'bold',
    'textAlign': 'center',
    'fontSize': '1.2rem'
}

# Layout de la aplicación
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Análisis de Precios de Viviendas", 
                style={'textAlign': 'center', 'color': 'white', 'fontWeight': 'bold', 'marginBottom': '0'})
    ], style={'backgroundColor': COLORS['accent-dark'], 'padding': '20px', 'borderRadius': '0 0 10px 10px', 'boxShadow': COLORS['card-shadow']}),
    
    # Contenedor principal
    html.Div([
        # Fila de KPIs
        html.Div([
            # Tarjeta de métricas del modelo
            html.Div([
                html.H3("Métricas del Modelo", style=title_style),
                html.Div([
                    html.Div([
                        html.H4("RMSE", style={'color': COLORS['text'], 'marginBottom': '5px', 'textAlign': 'center', 'fontSize': '0.9rem'}),
                        html.P(f"${rmse:,.2f}", style={'textAlign': 'center', 'fontSize': '1.2em', 'fontWeight': 'bold', 'color': COLORS['accent-dark']}),
                    ], style={'flex': '1', 'padding': '10px', 'backgroundColor': COLORS['light-accent'], 'borderRadius': '5px'}),
                    
                    html.Div([
                        html.H4("R² Score", style={'color': COLORS['text'], 'marginBottom': '5px', 'textAlign': 'center', 'fontSize': '0.9rem'}),
                        html.P(f"{r2:.4f}", style={'textAlign': 'center', 'fontSize': '1.2em', 'fontWeight': 'bold', 'color': COLORS['accent-dark']}),
                    ], style={'flex': '1', 'padding': '10px', 'backgroundColor': COLORS['light-accent'], 'borderRadius': '5px'}),
                ], style={'display': 'flex', 'gap': '10px'})
            ], style={**card_style, 'flex': '1'}),
            
            # Tarjeta de estadísticas de precios
            html.Div([
                html.H3("Estadísticas de Precios", style=title_style),
                html.Div([
                    html.Div([
                        html.H4("Precio Mínimo", style={'color': COLORS['text'], 'marginBottom': '5px', 'textAlign': 'center', 'fontSize': '0.9rem'}),
                        html.P(f"${price_stats['min']:,}", style={'textAlign': 'center', 'fontSize': '1.2em', 'fontWeight': 'bold', 'color': COLORS['danger']}),
                    ], style={'flex': '1', 'padding': '10px', 'backgroundColor': COLORS['light-accent'], 'borderRadius': '5px'}),
                    
                    html.Div([
                        html.H4("Precio Máximo", style={'color': COLORS['text'], 'marginBottom': '5px', 'textAlign': 'center', 'fontSize': '0.9rem'}),
                        html.P(f"${price_stats['max']:,}", style={'textAlign': 'center', 'fontSize': '1.2em', 'fontWeight': 'bold', 'color': COLORS['success']}),
                    ], style={'flex': '1', 'padding': '10px', 'backgroundColor': COLORS['light-accent'], 'borderRadius': '5px'}),
                ], style={'display': 'flex', 'gap': '10px', 'marginBottom': '10px'}),
                
                html.Div([
                    html.Div([
                        html.H4("Precio Medio", style={'color': COLORS['text'], 'marginBottom': '5px', 'textAlign': 'center', 'fontSize': '0.9rem'}),
                        html.P(f"${price_stats['mean']:,}", style={'textAlign': 'center', 'fontSize': '1.2em', 'fontWeight': 'bold', 'color': COLORS['warning']}),
                    ], style={'flex': '1', 'padding': '10px', 'backgroundColor': COLORS['light-accent'], 'borderRadius': '5px'}),
                    
                    html.Div([
                        html.H4("Precio Mediano", style={'color': COLORS['text'], 'marginBottom': '5px', 'textAlign': 'center', 'fontSize': '0.9rem'}),
                        html.P(f"${price_stats['median']:,}", style={'textAlign': 'center', 'fontSize': '1.2em', 'fontWeight': 'bold', 'color': COLORS['warning']}),
                    ], style={'flex': '1', 'padding': '10px', 'backgroundColor': COLORS['light-accent'], 'borderRadius': '5px'}),
                ], style={'display': 'flex', 'gap': '10px'})
            ], style={**card_style, 'flex': '1'}),
            
            # Tarjeta de correlaciones
            html.Div([
                html.H3("Correlación con Precio", style=title_style),
                html.Div([
                    dcc.Graph(
                        figure=px.bar(
                            y=corr_with_price.index, 
                            x=corr_with_price.values,
                            orientation='h',
                            labels={'x': 'Correlación', 'y': ''},
                            color=corr_with_price.values,
                            color_continuous_scale=['#74b9ff', '#0984e3', '#2980b9'],
                        ).update_layout(
                            plot_bgcolor=COLORS['background'],
                            paper_bgcolor=COLORS['background'],
                            margin=dict(l=10, r=10, t=10, b=10),
                            height=180,
                            coloraxis_showscale=False,
                            xaxis=dict(title='', tickformat='.2f'),
                            yaxis=dict(title='')
                        )
                    )
                ])
            ], style={**card_style, 'flex': '1'}),
        ], style={'display': 'flex', 'gap': '15px', 'marginBottom': '15px'}),
        
        # Fila de gráficos principales - 2 columnas
        html.Div([
            # Columna izquierda
            html.Div([
                # Gráfico de importancia de características
                html.Div([
                    html.H3("Importancia de Características", style=title_style),
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
                            height=300,
                            coloraxis_showscale=False,
                            yaxis=dict(autorange="reversed")
                        )
                    )
                ], style=card_style),
                
                # Gráfico de precio por calidad
                html.Div([
                    html.H3("Precio por Calidad General", style=title_style),
                    dcc.Graph(
                        figure=px.bar(
                            avg_price_by_quality, 
                            x='OverallQual', 
                            y='SalePrice',
                            color='SalePrice',
                            color_continuous_scale=['#74b9ff', '#0984e3', '#2980b9'],
                            labels={'OverallQual': 'Calidad General', 'SalePrice': 'Precio Promedio'}
                        ).update_layout(
                            plot_bgcolor=COLORS['background'],
                            paper_bgcolor=COLORS['background'],
                            margin=dict(l=10, r=10, t=10, b=10),
                            height=300,
                            coloraxis_showscale=False,
                            yaxis=dict(title='Precio Promedio', tickprefix='$', tickformat=','),
                            xaxis=dict(title='Calidad General (1-10)')
                        )
                    )
                ], style=card_style),
            ], style={'flex': '1', 'display': 'flex', 'flexDirection': 'column', 'gap': '15px'}),
            
            # Columna derecha
            html.Div([
                # Gráfico de precio real vs predicho
                html.Div([
                    html.H3("Precio Real vs Predicho", style=title_style),
                    dcc.Graph(
                        figure=px.scatter(
                            x=y_test_original, 
                            y=y_pred,
                            labels={'x': 'Precio Real', 'y': 'Precio Predicho'},
                            opacity=0.7,
                            color_discrete_sequence=['#3498db']
                        ).update_layout(
                            plot_bgcolor=COLORS['background'],
                            paper_bgcolor=COLORS['background'],
                            margin=dict(l=10, r=10, t=10, b=10),
                            height=300,
                            xaxis=dict(title='Precio Real', tickprefix='$', tickformat=','),
                            yaxis=dict(title='Precio Predicho', tickprefix='$', tickformat=',')
                        ).add_shape(
                            type="line",
                            x0=y_test_original.min(),
                            y0=y_test_original.min(),
                            x1=y_test_original.max(),
                            y1=y_test_original.max(),
                            line=dict(
                                color="#e74c3c",  # Changed to a bright red color
                                width=3,          # Increased line width
                                dash="solid"      # Changed from dotted to solid
                            )
                        )
                    )
                ], style=card_style),
                
                # Gráfico de precio por vecindario
                html.Div([
                    html.H3("Precio Promedio por Vecindario (Top 10)", style=title_style),
                    dcc.Graph(
                        figure=px.bar(
                            avg_price_by_neighborhood, 
                            x='Neighborhood', 
                            y='SalePrice',
                            color='SalePrice',
                            color_continuous_scale=['#74b9ff', '#0984e3', '#2980b9'],
                            labels={'Neighborhood': 'Vecindario', 'SalePrice': 'Precio Promedio'}
                        ).update_layout(
                            plot_bgcolor=COLORS['background'],
                            paper_bgcolor=COLORS['background'],
                            margin=dict(l=10, r=10, t=10, b=10),
                            height=300,
                            coloraxis_showscale=False,
                            yaxis=dict(title='Precio Promedio', tickprefix='$', tickformat=','),
                            xaxis=dict(title='Vecindario', tickangle=45)
                        )
                    )
                ], style=card_style),
            ], style={'flex': '1', 'display': 'flex', 'flexDirection': 'column', 'gap': '15px'}),
        ], style={'display': 'flex', 'gap': '15px', 'marginBottom': '15px'}),
        
        # Nueva fila para visualizaciones adicionales
        html.Div([
            # Columna izquierda - Mapa de calor de correlaciones
            html.Div([
                html.Div([
                    html.H3("Correlación entre Variables Clave", style=title_style),
                    dcc.Graph(
                        figure=px.imshow(
                            corr_df, 
                            text_auto=True,
                            color_continuous_scale=['#74b9ff', '#0984e3', '#2980b9'],
                            labels={'color': 'Correlación'}
                        ).update_layout(
                            plot_bgcolor=COLORS['background'],
                            paper_bgcolor=COLORS['background'],
                            margin=dict(l=10, r=10, t=10, b=10),
                            height=300,
                            coloraxis_showscale=True
                        )
                    )
                ], style=card_style)
            ], style={'flex': '1', 'display': 'flex', 'flexDirection': 'column'}),
            
            # Columna derecha - Scatter plot interactivo
            html.Div([
                html.Div([
                    html.H3("Relación de Variables con Precio", style=title_style),
                    html.Div([
                        dcc.Dropdown(
                            id='scatter-variable',
                            options=[
                                {"label": "Calidad General", "value": "OverallQual"},
                                {"label": "Área Habitable", "value": "GrLivArea"},
                                {"label": "Garaje (Capacidad Autos)", "value": "GarageCars"},
                                {"label": "Año de Construcción", "value": "YearBuilt"},
                            ],
                            value="OverallQual",
                            clearable=False,
                            style={
                                'width': '100%',
                                'marginBottom': '15px',
                                'backgroundColor': COLORS['background']
                            }
                        ),
                    ]),
                    dcc.Graph(id='scatter-plot')
                ], style=card_style)
            ], style={'flex': '1', 'display': 'flex', 'flexDirection': 'column'}),
        ], style={'display': 'flex', 'gap': '15px', 'marginBottom': '15px'}),
        
        # Sección de Predicción Interactiva
        html.Div([
            html.H3("Predicción Interactiva", style={**title_style, 'fontSize': '1.4rem', 'marginBottom': '20px'}),
            
            # Contenedor para inputs y resultado
            html.Div([
                # Input column
                html.Div([
                    # Primera fila de inputs
                    html.Div([
                        # Garage Cars
                        html.Div([
                            html.Label("Garage Cars", style={'fontWeight': 'bold', 'color': COLORS['text']}),
                            dcc.Slider(
                                min=0, max=5, step=1, value=2, id='garage-cars-slider',
                                marks={i: str(i) for i in range(6)},
                                className='custom-slider'
                            ),
                        ], style={'flex': '1'}),
                        
                        # Overall Quality
                        html.Div([
                            html.Label("Overall Quality", style={'fontWeight': 'bold', 'color': COLORS['text']}),
                            dcc.Slider(
                                min=1, max=10, step=1, value=5, id='overall-quality-slider',
                                marks={i: str(i) for i in range(1, 11)},
                                className='custom-slider'
                            ),
                        ], style={'flex': '1'}),
                    ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px'}),
                    
                    # Segunda fila de inputs
                    html.Div([
                        # GrLivArea
                        html.Div([
                            html.Label("GrLivArea (Above Ground Living Area)", style={'fontWeight': 'bold', 'color': COLORS['text']}),
                            dcc.Input(
                                type='number', value=1500, id='grlivarea-input',
                                style={'width': '100%', 'padding': '8px', 'borderRadius': '4px', 'border': f'1px solid {COLORS["border"]}'}
                            ),
                        ], style={'flex': '1'}),
                        
                        # Neighborhood dropdown (replacing TotalBsmtSF)
                        html.Div([
                            html.Label("Neighborhood", style={'fontWeight': 'bold', 'color': COLORS['text']}),
                            dcc.Dropdown(
                                id='neighborhood-dropdown',
                                options=[
                                    {'label': 'Blueste', 'value': 'Blueste'},
                                    {'label': 'BrDale', 'value': 'BrDale'},
                                    {'label': 'BrkSide', 'value': 'BrkSide'},
                                    {'label': 'ClearCr', 'value': 'ClearCr'},
                                    {'label': 'CollgCr', 'value': 'CollgCr'},
                                    {'label': 'Crawfor', 'value': 'Crawfor'},
                                    {'label': 'Edwards', 'value': 'Edwards'},
                                    {'label': 'Gilbert', 'value': 'Gilbert'},
                                    {'label': 'IDOTRR', 'value': 'IDOTRR'},
                                    {'label': 'MeadowV', 'value': 'MeadowV'},
                                    {'label': 'Mitchel', 'value': 'Mitchel'},
                                    {'label': 'NAmes', 'value': 'NAmes'},
                                    {'label': 'NPkVill', 'value': 'NPkVill'},
                                    {'label': 'NWAmes', 'value': 'NWAmes'},
                                    {'label': 'NoRidge', 'value': 'NoRidge'},
                                    {'label': 'NridgHt', 'value': 'NridgHt'},
                                    {'label': 'OldTown', 'value': 'OldTown'},
                                    {'label': 'SWISU', 'value': 'SWISU'},
                                    {'label': 'Sawyer', 'value': 'Sawyer'},
                                    {'label': 'SawyerW', 'value': 'SawyerW'},
                                    {'label': 'Somerst', 'value': 'Somerst'},
                                    {'label': 'StoneBr', 'value': 'StoneBr'},
                                    {'label': 'Timber', 'value': 'Timber'},
                                    {'label': 'Veenker', 'value': 'Veenker'}
                                ],
                                value='NAmes',  # Default value
                                style={'width': '100%', 'borderRadius': '4px'}
                            ),
                        ], style={'flex': '1'}),
                    ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px'}),
                    
                    # Tercera fila de inputs
                    html.Div([
                        # Lot Area
                        html.Div([
                            html.Label("Lot Area", style={'fontWeight': 'bold', 'color': COLORS['text']}),
                            dcc.Input(
                                type='number', value=8000, id='lotarea-input',
                                style={'width': '100%', 'padding': '8px', 'borderRadius': '4px', 'border': f'1px solid {COLORS["border"]}'}
                            ),
                        ], style={'flex': '1'}),
                    ], style={'display': 'flex', 'gap': '20px'}),
                ], style={'flex': '3', 'padding': '10px', 'minWidth': '0'}),
                
                # Result column
                html.Div([
                    html.Div([
                        html.H3("Precio Predicho", style={'textAlign': 'center', 'color': COLORS['text'], 'marginBottom': '15px'}),
                        html.Div(
                            id='predicted-price', 
                            style={
                                'textAlign': 'center', 
                                'fontSize': '2em',
                                'fontWeight': 'bold',
                                'color': COLORS['success'],
                                'padding': '10px 5px',
                                'wordBreak': 'break-word',
                                'overflowWrap': 'break-word',
                                'maxWidth': '100%'
                            }
                        ),
                        html.P("Basado en las características seleccionadas", style={'textAlign': 'center', 'color': COLORS['text'], 'opacity': '0.7'})
                    ], style={
                        'backgroundColor': COLORS['light-accent'], 
                        'padding': '15px',
                        'borderRadius': '8px',
                        'display': 'flex',
                        'flexDirection': 'column',
                        'justifyContent': 'center',
                        'overflow': 'hidden',
                        'boxSizing': 'border-box',
                        'width': '100%'
                    })
                ], style={'flex': '2', 'padding': '10px', 'minWidth': '0', 'maxWidth': '100%'})
            ], style={'display': 'flex', 'flexWrap': 'wrap'})
        ], style={**card_style, 'marginBottom': '15px'}),
        
        # Información adicional
        html.Div([
            html.H3("Información del Modelo", style=title_style),
            html.Div([
                html.P([
                    "Este dashboard utiliza un modelo Bayesian Ridge para predecir precios de viviendas basado en datos de Ames, Iowa."
                ]),
                
                # Bayesian Ridge model information
                html.P([
                    "El modelo fue desarrollado utilizando MLflow para seguimiento de experimentos. ",
                    "Este modelo aplica una transformación logarítmica al precio de venta para mejorar la distribución."
                ]),
                html.P([
                    "El modelo incluye selección de características utilizando SelectFromModel con un umbral basado en la mediana ",
                    "de los coeficientes absolutos."
                ]),
                html.P([
                    "Los hiperparámetros fueron optimizados mediante GridSearchCV con validación cruzada de 5 pliegues, ",
                    "evaluando métricas como error cuadrático medio negativo y R²."
                ])
            ], style={'backgroundColor': COLORS['light-accent'], 'padding': '15px', 'borderRadius': '5px'})
        ], style={**card_style, 'marginBottom': '15px'}),
        
        # Footer
        html.Div([
            html.P("Dashboard de Análisis de Precios de Viviendas | Desarrollado con Dash y Bayesian Ridge", 
                   style={'textAlign': 'center', 'color': COLORS['text'], 'opacity': '0.7'})
        ], style={'marginTop': '10px'})
        
    ], style={'maxWidth': '1200px', 'margin': '20px auto', 'padding': '0 20px'})
], style={'backgroundColor': COLORS['secondary-bg'], 'minHeight': '100vh', 'fontFamily': 'Arial, sans-serif'})


# Callback para actualizar la predicción basada en los inputs
@app.callback(
    Output('predicted-price', 'children'),
    [Input('garage-cars-slider', 'value'),
     Input('overall-quality-slider', 'value'),
     Input('grlivarea-input', 'value'),
     Input('neighborhood-dropdown', 'value'),
     Input('lotarea-input', 'value')]
)
def predict_price(garage_cars, overall_quality, grlivarea, neighborhood, lotarea):
    # Create a dataframe with default values: median for numeric, mode for categorical
    input_data = pd.DataFrame(index=[0], columns=X.columns)
    
    # Fill with median for numeric columns
    for col in X.select_dtypes(include=['int64', 'float64']).columns:
        input_data[col] = X[col].median()
    
    # Fill with mode for categorical columns
    for col in X.select_dtypes(include=['object']).columns:
        input_data[col] = X[col].mode()[0]
    
    # Update with user inputs
    input_data['GarageCars'] = garage_cars
    input_data['OverallQual'] = overall_quality
    input_data['GrLivArea'] = grlivarea
    input_data['Neighborhood'] = neighborhood # Neighborhood
    input_data['LotArea'] = lotarea
    
    # Predict price using the model
    predicted_price_log = model.predict(input_data)[0]
    
    # Convert back to original scale
    predicted_price = np.expm1(predicted_price_log)
    
    return f"${predicted_price:,.2f}"

# Add this callback for the interactive scatter plot
@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('scatter-variable', 'value')]
)
def update_scatter(selected_var):
    fig = px.scatter(
        df, 
        x=selected_var, 
        y="SalePrice",
        opacity=0.7,
        color_discrete_sequence=[COLORS['accent']]
    )
    fig.update_layout(
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        margin=dict(l=10, r=10, t=10, b=10),
        height=250,
        xaxis=dict(title=selected_var),
        yaxis=dict(title='Precio', tickprefix='$', tickformat=',')
    )
    return fig

# Ejecutar la aplicación Dash
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8080)