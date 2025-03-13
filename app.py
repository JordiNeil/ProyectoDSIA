# -*- coding: utf-8 -*-
import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Cargar los datos
file_path = "data/train.csv"
df = pd.read_csv(file_path)

# Aplicar transformación logarítmica a SalePrice
df["LogSalePrice"] = np.log1p(df["SalePrice"])

# Variables más correlacionadas
corr_features = ["SalePrice", "OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "YearBuilt"]
corr_df = df[corr_features].corr()

selected_num_columns = ['LotFrontage','LotArea','TotalBsmtSF','GrLivArea','SalePrice']
##Se selecciona porque garage cars porque tiene mejor correlacion con saleprice y ambas estaban muy correlacionadas
categorical_columns_filtered =  ['Street','LandContour',
                    'LandSlope','Utilities','Neighborhood','Condition1',
                    'Condition2','HouseStyle','BldgType','OverallQual',
                    'OverallCond','RoofStyle','Exterior1st',
                    'ExterCond', 'BsmtCond','BsmtFinType1','CentralAir',
                    'Heating','KitchenQual','TotRmsAbvGrd', 'GarageType',
                    'GarageCond','PavedDrive',
                    'SaleType','SaleCondition','Fireplaces',
                    'GarageCars',
                    ]

df = df[selected_num_columns + categorical_columns_filtered]

# Función para crear el pairplot
def create_pairplot(df, columns, hue_column):
    # Calcular el número de variables
    n_vars = len(columns)
    
    # Crear subplots
    fig = make_subplots(rows=n_vars, cols=n_vars,
                        subplot_titles=[f"{col1} vs {col2}" for col1 in columns for col2 in columns])
    
    # Cambiar a una paleta de colores diferente
    unique_hues = df[hue_column].unique()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'][:len(unique_hues)]
    hue_color_map = dict(zip(unique_hues, colors))
    
    # Llenar la matriz de gráficos
    for i, col1 in enumerate(columns, 1):
        for j, col2 in enumerate(columns, 1):
            for hue_val in unique_hues:
                mask = df[hue_column] == hue_val
                if col1 == col2:  # Diagonal: histogramas
                    fig.add_trace(
                        go.Histogram(
                            x=df[mask][col1],
                            name=f'{hue_column}={hue_val}',
                            showlegend=True if (i==1 and j==1) else False,
                            marker_color=hue_color_map[hue_val]
                        ),
                        row=i, col=j
                    )
                else:  # Fuera de la diagonal: scatter plots
                    fig.add_trace(
                        go.Scatter(
                            x=df[mask][col2],
                            y=df[mask][col1],
                            mode='markers',
                            name=f'{hue_column}={hue_val}',
                            showlegend=False,
                            marker_color=hue_color_map[hue_val]
                        ),
                        row=i, col=j
                    )
    
    # Actualizar el diseño con dimensiones más grandes
    fig.update_layout(
        height=1200,
        width=1500,
        title=f'Pairplot colored by {hue_column}',
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=100, b=50),
        grid=dict(rows=n_vars, columns=n_vars, pattern='independent')
    )
    
    return fig

# Crear el pairplot
pairplot_fig = create_pairplot(df, selected_num_columns, 'OverallCond')

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

app.layout = html.Div([
    # Header con estilo
    html.Div([
        html.H1("Análisis de Precios de Viviendas",
                style={
                    'textAlign': 'center',
                    'color': COLORS['text'],
                    'padding': '40px 0',
                    'borderBottom': f'4px solid {COLORS["accent"]}',
                    'marginBottom': '30px',
                    'fontSize': '2.5em',
                    'fontWeight': 'bold'
                })
    ], style={'backgroundColor': COLORS['light-accent']}),
    
    # Descripción de las variables con nuevo estilo
    html.Div([
        html.H3("Variables Analizadas:", 
                style={
                    'textAlign': 'center',
                    'color': COLORS['text'],
                    'marginBottom': '20px',
                    'fontSize': '1.8em'
                }),
        html.Div([
            html.Ul([
                html.Li([
                    html.Strong("LotFrontage: "),
                    "Metros lineales de calle conectados a la propiedad"
                ]),
                html.Li([
                    html.Strong("LotArea: "),
                    "Tamaño del lote en pies cuadrados"
                ]),
                html.Li([
                    html.Strong("TotalBsmtSF: "),
                    "Total de pies cuadrados del área del sótano"
                ]),
                html.Li([
                    html.Strong("GrLivArea: "),
                    "Área habitable por encima del nivel del suelo en pies cuadrados"
                ]),
                html.Li([
                    html.Strong("SalePrice: "),
                    "Precio de venta de la propiedad en dólares (variable objetivo)"
                ]),
            ], style={
                'listStyleType': 'none',
                'padding': '20px',
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'margin': 'auto',
                'width': '80%',
                'lineHeight': '2'
            })
        ], style={'backgroundColor': COLORS['background'], 'padding': '20px'})
    ]),
    
    # Sección de visualizaciones
    html.Div([
        # Pairplot
        html.Div([
            html.H3("Pairplot de Variables", 
                    style={
                        'textAlign': 'center',
                        'color': COLORS['text'],
                        'marginTop': '40px',
                        'marginBottom': '20px',
                        'fontSize': '1.8em'
                    }),
            html.Div([  # Contenedor adicional para centrar
                dcc.Graph(
                    id='pairplot',
                    figure=pairplot_fig.update_layout(
                        width=1600,  # Ancho fijo
                        height=1200, # Alto fijo
                        margin=dict(l=50, r=50, t=100, b=50)
                    ),
                    style={
                        'margin': '0 auto',  # Centrar horizontalmente
                        'backgroundColor': 'white',
                        'borderRadius': '10px',
                        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                        'padding': '20px',
                        'display': 'flex',
                        'justifyContent': 'center'
                    }
                )
            ], style={
                'display': 'flex',
                'justifyContent': 'center',
                'width': '100%'
            })
        ], style={'marginBottom': '40px'}),
        
        # Distribución de precios
        html.Div([
            html.H3("Distribución de Precios de Viviendas",
                    style={
                        'textAlign': 'center',
                        'color': COLORS['text'],
                        'marginBottom': '20px',
                        'fontSize': '1.8em'
                    }),
            dcc.Graph(
                figure=px.histogram(df, x="LogSalePrice", nbins=30, 
                                  title="Distribución de Log(SalePrice)")
                .update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                ),
                style={
                    'backgroundColor': 'white',
                    'borderRadius': '10px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                    'padding': '20px'
                }
            )
        ], style={'marginBottom': '40px'}),
        
        # Correlaciones
        html.Div([
            html.H3("Correlación entre Variables Clave",
                    style={
                        'textAlign': 'center',
                        'color': COLORS['text'],
                        'marginBottom': '20px',
                        'fontSize': '1.8em'
                    }),
            dcc.Graph(
                figure=px.imshow(corr_df, text_auto=True,
                               title="Mapa de Calor de Correlaciones")
                .update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                ),
                style={
                    'backgroundColor': 'white',
                    'borderRadius': '10px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                    'padding': '20px'
                }
            )
        ], style={'marginBottom': '40px'}),
        
        # Scatter plot interactivo
        html.Div([
            html.H3("Relación de Variables con SalePrice",
                    style={
                        'textAlign': 'center',
                        'color': COLORS['text'],
                        'marginBottom': '20px',
                        'fontSize': '1.8em'
                    }),
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
                        'width': '50%',
                        'margin': '20px auto',
                        'backgroundColor': 'white'
                    }
                ),
            ], style={'textAlign': 'center'}),
            dcc.Graph(
                id='scatter-plot',
                style={
                    'backgroundColor': 'white',
                    'borderRadius': '10px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                    'padding': '20px'
                }
            )
        ]),
        
        # Boxplot de precios por vecindario
        html.Div([
            html.H3("Comparación de Precios por Vecindario",
                    style={
                        'textAlign': 'center',
                        'color': COLORS['text'],
                        'marginTop': '40px',
                        'marginBottom': '20px',
                        'fontSize': '1.8em'
                    }),
            dcc.Graph(
                figure=px.box(df, x="Neighborhood", y="SalePrice", 
                            title="Distribución de Precios por Vecindario")
                .update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font={'color': COLORS['text']},
                    xaxis={'tickangle': 45}
                ),
                style={
                    'backgroundColor': 'white',
                    'borderRadius': '10px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                    'padding': '20px'
                }
            )
        ], style={'marginBottom': '40px'})
    ], style={'backgroundColor': COLORS['background'], 'padding': '20px'})
], style={'backgroundColor': COLORS['background']})

@app.callback(
    dash.dependencies.Output('scatter-plot', 'figure'),
    [dash.dependencies.Input('scatter-variable', 'value')]
)
def update_scatter(selected_var):
    fig = px.scatter(df, x=selected_var, y="SalePrice", 
                    title=f"SalePrice vs {selected_var}")
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'color': COLORS['text']},
    )
    return fig

# Ejecutar el servidor
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
