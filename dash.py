import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
import numpy as np

# Cargar los datos
file_path = "train(1).csv"
df = pd.read_csv(file_path)

# Aplicar transformación logarítmica a SalePrice
df["LogSalePrice"] = np.log1p(df["SalePrice"])

# Variables más correlacionadas
corr_features = ["SalePrice", "OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "YearBuilt"]
corr_df = df[corr_features].corr()

# Crear la aplicación Dash
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Análisis de Precios de Viviendas"),
    
    # Histograma de SalePrice
    html.H3("Distribución de Precios de Viviendas"),
    dcc.Graph(
        figure=px.histogram(df, x="LogSalePrice", nbins=30, title="Distribución de Log(SalePrice)")
    ),
    
    # Heatmap de correlaciones
    html.H3("Correlación entre Variables Clave"),
    dcc.Graph(
        figure=px.imshow(corr_df, text_auto=True, title="Mapa de Calor de Correlaciones")
    ),
    
    # Dispersión de variables clave contra SalePrice
    html.H3("Relación de Variables con SalePrice"),
    dcc.Dropdown(
        id='scatter-variable',
        options=[
            {"label": "Calidad General", "value": "OverallQual"},
            {"label": "Área Habitable", "value": "GrLivArea"},
            {"label": "Garaje (Capacidad Autos)", "value": "GarageCars"},
            {"label": "Año de Construcción", "value": "YearBuilt"},
        ],
        value="OverallQual",
        clearable=False
    ),
    dcc.Graph(id='scatter-plot'),
    
    # Comparación por vecindario
    html.H3("Comparación de Precios por Vecindario"),
    dcc.Graph(
        figure=px.box(df, x="Neighborhood", y="SalePrice", title="Distribución de Precios por Vecindario")
    )
])

@app.callback(
    dash.dependencies.Output('scatter-plot', 'figure'),
    [dash.dependencies.Input('scatter-variable', 'value')]
)
def update_scatter(selected_var):
    fig = px.scatter(df, x=selected_var, y="SalePrice", title=f"SalePrice vs {selected_var}")
    return fig

# Ejecutar el servidor
if __name__ == '__main__':
    app.run_server(debug=True)
