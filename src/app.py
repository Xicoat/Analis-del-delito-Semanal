from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import pandas as pd

# 1. Importar tus funciones o módulos existentes
# from analisis import procesar_datos, tu_grafica

# 2. Inicializar la app
app = Dash(__name__)

# 3. Definir la interfaz (Layout)
app.layout = html.Div([
    html.H1("Panel de Análisis", style={"textAlign": "center", "color": "#2563EB"}),
    
    html.Div([
        html.Label("Seleccionar variable:"),
        dcc.Dropdown(
            id="filtro-categoria",
            options=[
                {"label": "Opción A", "value": "A"},
                {"label": "Opción B", "value": "B"}
            ],
            value="A",
            clearable=False
        )
    ], style={"width": "30%", "margin": "0 auto 20px"}),

    dcc.Graph(id="grafica-principal")
])

# 4. Definir la reactividad (Callbacks)
@app.callback(
    Output("grafica-principal", "figure"),
    Input("filtro-categoria", "value")
)
def actualizar_grafico(valor_seleccionado):
    # Aquí puedes llamar tus funciones de analisis.py
    # Por ejemplo, armar una figura con Plotly
    df_ejemplo = pd.DataFrame({
        "Eje X": [1, 2, 3, 4],
        "Eje Y": [10, 15, 7, 12] if valor_seleccionado == "A" else [12, 9, 14, 5]
    })
    
    fig = px.line(df_ejemplo, x="Eje X", y="Eje Y", title=f"Visualización: {valor_seleccionado}")
    return fig

# 5. Ejecutar el servidor
if __name__ == "__main__":
    app.run(
        debug=True,
        dev_tools_hot_reload=True,
        dev_tools_hot_reload_interval=1000  # Intervalo en milisegundos para revisar cambios
    )