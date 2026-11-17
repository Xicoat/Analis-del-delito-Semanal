# %%
# [CONFIG] Imports, logging y parámetros globales
from pathlib import Path
import sys
from typing import Union
import logging
import pandas as pd
import networkx as nx
import powerlaw
import warnings
from scipy.optimize import OptimizeWarning
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import re
import plotly.graph_objects as go
from matplotlib.lines import Line2D

# Configuración del sistema de logging
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - [%(funcName)s] - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger("Delitos")

# Constantes
DATA_PATH = Path("data/delitos/csv")
START_DATE = pd.Timestamp("2016-01-01")
END_DATE = pd.Timestamp("2024-06-30")
DATE_COL = "fecha_hecho" # Columna de FECHA
# TARGET_CRIME = "HOMICIDIO POR ARMA DE FUEGO"
# TARGET_CRIME = "FEMINICIDIO"
# TARGET_CRIME = "ROBO DE VEHICULO DE SERVICIO PARTICULAR SIN VIOLENCIA"
# TARGET_CRIME = "ROBO A CASA HABITACION CON VIOLENCIA"
# TARGET_CRIME = "ROBO A CASA HABITACION SIN VIOLENCIA"
TARGET_CRIME = "ROBO A TRANSEUNTE SALIENDO DEL CAJERO CON VIOLENCIA"
# TARGET_CRIME = "ROBO A TRANSEUNTE SALIENDO DEL BANCO CON VIOLENCIA"

logger.info("Constantes cargadas con éxito.")

# %% [markdown]
# <details>
# <summary><b>Función: <code>extract_crime_data(path, target_crime)</code></b></summary>
# 
# <br>
# 
# Extrae y consolida en un único `DataFrame` los registros asociados a una categoría específica de delito a partir de los archivos CSV presentes en un directorio.
# 
# #### Parámetros
# * **`path`** (`str` | `Path`): Ruta al directorio que contiene los archivos CSV de origen.
# * **`target_crime`** (`str`): Etiqueta exacta del delito a filtrar (columna `delito`).
# 
# #### Retorno
# * **`pd.DataFrame`**: Registros consolidados. Si no hay coincidencias, retorna un `DataFrame` vacío.
# 
# #### Excepciones
# * **`FileNotFoundError`**: Si el directorio no existe o no contiene archivos `.csv`.
# 
# </details>

# %%
# [FUNCION] extract_crime_data(path, target_crime)

def extract_crime_data(path: Union[str, Path], target_crime: str) -> pd.DataFrame:
    data_dir = Path(path)

    logger.info("Revisando directorio.")
    if not data_dir.exists() or not data_dir.is_dir():
        logger.error(f"Directorio inválido o inexistente: {data_dir}")
        raise FileNotFoundError(f"Directorio no encontrado: {data_dir}")

    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        logger.error(f"Sin archivos CSV en: {data_dir}")
        raise FileNotFoundError(f"No se encontraron CSV en: {data_dir}")

    logger.info(f"Buscando {target_crime}")
    logger.info("Extrayendo registros")
    records = []

    for file_path in csv_files:
        try:
            df_temp = pd.read_csv(file_path, low_memory=False)
            if "delito" in df_temp.columns:
                filtered = df_temp[df_temp["delito"] == target_crime]
                if not filtered.empty:
                    records.append(filtered)
        except Exception as e:
            logger.warning(f"No se pudo procesar {file_path.name}: {e}")

    if not records:
        logger.warning(f"No se encontraron registros para: {target_crime}")
        return pd.DataFrame()

    logger.info("Generando dataframe")
    df_raw = pd.concat(records, ignore_index=True)

    logger.info(f"EXITO. Se extrajeron {len(df_raw):,} registros.")
    return df_raw

# %% [markdown]
# <details>
# <summary><b>Ejecución: Extracción de datos iniciales</b></summary>
# 
# <br>
# 
# Dispara el proceso de extracción para consolidar en memoria el conjunto de datos crudos (`df_raw`) asociado al delito configurado en `TARGET_CRIME`.
# 
# #### Flujo de ejecución
# 1. **Llamada:** Ejecuta `extract_crime_data(DATA_PATH, TARGET_CRIME)`.
# 2. **Asignación:** Almacena el resultado consolidado en la variable global `df_raw`.
# 3. **Control de excepciones:** Captura fallos de ruta (`FileNotFoundError`) y los reporta en el logger sin interrumpir abruptamente el kernel.
# 
# #### Variables generadas
# * **`df_raw`** (`pd.DataFrame`): Conjunto de datos crudo con los registros coincidentes.
# 
# </details>

# %%
# [EJECUCIÓN] Extracción de datos iniciales
# INPUT:  DATA_PATH (Path), TARGET_CRIME (str)
# OUTPUT: df_raw (pd.DataFrame)

try:
    df_raw = extract_crime_data(DATA_PATH, TARGET_CRIME)
except FileNotFoundError as err:
    logger.error(f"Error de ejecución: {err}")

# %% [markdown]
# <details>
# <summary><b>Helpers: Filtrado de datos RAW por intervalo de tiempo</b></summary>
# 
# <br>
# 
# Conjunto de funciones auxiliares para normalización de tipos temporales, extracción de series de fecha y filtrado por rangos cronológicos.
# 
# #### Funciones incluidas
# * **`ensure_datetime_type(df, column)`**: Convierte la columna a `datetime64` con coerción de errores si no tiene el tipo correcto.
# * **`extract_dates(df, date_column)`**: Extrae una proyección de la columna de fecha asegurando su tipado.
# * **`filter_by_date_range(df, date_column, start, end)`**: Aplica máscara booleana para acotar registros dentro del intervalo `[start, end]`.
# 
# #### Parámetros comunes
# * `df` (`pd.DataFrame`): DataFrame a procesar.
# * `date_column` (`str`): Nombre de la columna que contiene los datos temporales.
# * `start`, `end` (`pd.Timestamp`): Límites inferior y superior del rango de análisis.
# 
# </details>

# %%
# [HELPERS] Filtrado de datos RAW por intervalo de tiempo
# INPUT:  df (pd.DataFrame), date_column (str), start/end (pd.Timestamp)
# OUTPUT: pd.DataFrame


def ensure_datetime_type(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Verifica si la columna es datetime; si no lo es, realiza la conversión."""
    if not pd.api.types.is_datetime64_any_dtype(df[column]):
        logger.info(f"Convirtiendo tiempo en columna '{column}'")
        df[column] = pd.to_datetime(df[column], errors="coerce")
    return df


def extract_dates(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """Extrae la columna como DataFrame y garantiza que el tipo sea datetime."""
    logger.info("Extrayendo datos")
    df_dates = df[[date_column]].copy()
    df_dates = ensure_datetime_type(df_dates, date_column)
    logger.info("EXITO.")
    return df_dates


def filter_by_date_range(df: pd.DataFrame, date_column: str, start: pd.Timestamp, end: pd.Timestamp,) -> pd.DataFrame:
    """Filtra el DataFrame reteniendo estrictamente los registros dentro del periodo definido."""
    logger.info("Filtrando registros")
    mask = (df[date_column] >= start) & (df[date_column] <= end)
    df_filtered = df[mask].copy()
    logger.info("EXITO.")
    return df_filtered

# %% [markdown]
# <details>
# <summary><b>Ejecución: Extracción y filtrado temporal de registros</b></summary>
# 
# <br>
# 
# Procesa la serie cronológica del conjunto de datos crudos, asegurando el tipado correcto y delimitando las observaciones al intervalo de análisis configurado.
# 
# #### Flujo de ejecución
# 1. **Extracción y normalización:** Ejecuta `extract_dates(df_raw, DATE_COL)` para aislar la columna temporal y garantizar el tipo de dato `datetime64`.
# 2. **Filtrado cronológico:** Aplica `filter_by_date_range(df_dates, DATE_COL, START_DATE, END_DATE)` para retener exclusivamente los registros comprendidos entre `START_DATE` y `END_DATE`.
# 3. **Control de excepciones:** Captura errores de columnas ausentes (`KeyError`) o excepciones generales de procesamiento e informa al logger.
# 
# #### Variables generadas
# * **`df_dates`** (`pd.DataFrame`): Serie temporal extraída y casteada a tipo fecha.
# * **`df_filtered_range`** (`pd.DataFrame`): Registros comprendidos estrictamente dentro de la ventana de tiempo definida.
# 
# </details>

# %%
# [EJECUCIÓN] Extracción y filtrado temporal de registros
# INPUT:  df_raw (pd.DataFrame), START_DATE (pd.Timestamp), END_DATE (pd.Timestamp)
# OUTPUT: df_dates (pd.DataFrame), df_filtered_range (pd.DataFrame)

try:
    # 1. Extracción y conversión a fecha
    df_dates = extract_dates(df_raw, DATE_COL)

    # 2. Filtrado por rango temporal
    df_filtered_range = filter_by_date_range(df_dates, DATE_COL, START_DATE, END_DATE)

except KeyError as err:
    logger.error(f"Columna no encontrada en el DataFrame: {err}")
except Exception as err:
    logger.error(f"Error durante el procesamiento temporal: {err}")

# %% [markdown]
# <details>
# <summary><b>Helpers: Construcción del Grafo de Visibilidad</b></summary>
# 
# <br>
# 
# Funciones para el remuestreo de la serie temporal en bloques de 7 días, indexación topológica de nodos y construcción del Grafo de Visibilidad Natural (NVG).
# 
# #### Funciones incluidas
# * **`events_by_week(df, date_column, origin_date)`**: Agrupa temporalmente en ventanas de 168 horas ancladas al origen y normaliza la fecha a formato `YYYY-MM-DD`.
# * **`assign_network_nodes(df)`**: Indexa secuencialmente cada intervalo temporal con un identificador de nodo (`nodo`) entero ascendente.
# * **`build_natural_visibility_graph(df)`**: Construye el objeto `nx.Graph` evaluando la visibilidad geométrica de pendientes máximas continuas entre observaciones.
# 
# </details>

# %%
# [HELPERS] Construcción del Grafo de Visibilidad.
# INPUT:  df (pd.DataFrame), date_column (str), origin_date (pd.Timestamp)
# OUTPUT: nx.Graph


def events_by_week(
    df: pd.DataFrame,
    date_column: str = "fecha_hecho",
    origin_date: pd.Timestamp = START_DATE,
) -> pd.DataFrame:
    """Agrupa en bloques de 7 días absolutos (168 horas) anclados a origin_date."""
    logger.info("Agrupando serie temporal en bloques semanales")
    weekly_series = (
        df.resample("168h", on=date_column, origin=origin_date)
        .size()
        .reset_index(name="cantidad")
    )
    weekly_series[date_column] = weekly_series[date_column].dt.normalize()
    logger.info("Exito.")
    return weekly_series


def assign_network_nodes(df: pd.DataFrame) -> pd.DataFrame:
    """Asigna un identificador secuencial (nodo) a cada bloque temporal."""
    logger.info("Asignando identificadores topologicos a nodos")
    df_nodes = df.copy()
    df_nodes["nodo"] = range(len(df_nodes))
    logger.info("Exito.")
    return df_nodes[["nodo", "fecha_hecho", "cantidad"]]


def build_natural_visibility_graph(df: pd.DataFrame) -> nx.Graph:
    """Construye el Grafo de Visibilidad Natural (NVG) a partir de una serie temporal."""
    logger.info("Construyendo Grafo de Visibilidad Natural")
    g = nx.Graph()

    # 1. Asignar nodos con sus atributos
    for _, row in df.iterrows():
        g.add_node(
            int(row["nodo"]),
            fecha_hecho=row["fecha_hecho"],
            cantidad=float(row["cantidad"]),
        )

    y = df["cantidad"].to_numpy()
    n = len(y)

    # 2. Evaluación de visibilidad por pendientes
    for i in range(n - 1):
        g.add_edge(i, i + 1)
        max_slope = y[i + 1] - y[i]

        for j in range(i + 2, n):
            slope = (y[j] - y[i]) / (j - i)
            if slope > max_slope:
                g.add_edge(i, j)
                max_slope = slope

    logger.info(f"Exito. Grafo generado (|V|={g.number_of_nodes()}, |E|={g.number_of_edges()}).")
    return g

# %% [markdown]
# <details>
# <summary><b>Ejecución: Construye el grafo y verifica la red</b></summary>
# 
# <br>
# 
# Ejecuta secuencialmente la agregación semanal, la asignación de nodos y la generación del grafo `visibility_graph`.
# 
# #### Flujo de ejecución
# 1. **Agregación:** Resampleo de `df_filtered_range` cada 168 horas a partir de `START_DATE`.
# 2. **Indexación:** Enumeración continua de nodos temporales en `df_serie_semanal`.
# 3. **Mapeo geométrico:** Construcción del grafo de visibilidad `visibility_graph` (`nx.Graph`).
# 4. **Validación:** Emisión al logger del orden $\vert{}V\vert{}$ y tamaño $\vert{}E\vert{}$ del grafo resultante.
# 
# #### Variables generadas
# * **`df_weekly`** (`pd.DataFrame`): Conteos semanales agregados.
# * **`df_serie_semanal`** (`pd.DataFrame`): Serie temporal con índice formal de nodos.
# * **`visibility_graph`** (`nx.Graph`): Red compleja resultante del NVG.
# 
# </details>

# %%
# [EJECUCIÓN] Construcción de red compleja (NVG)
# INPUT:  df_filtered_range (pd.DataFrame), DATE_COL (str), START_DATE (pd.Timestamp)
# OUTPUT: df_weekly (pd.DataFrame), df_serie_semanal (pd.DataFrame), visibility_graph (nx.Graph)

try:
    # 1. Agrupación semanal
    df_weekly = events_by_week(df_filtered_range, DATE_COL, START_DATE)

    # 2. Indexación de nodos
    df_serie_semanal = assign_network_nodes(df_weekly)

    # 3. Construcción del grafo
    visibility_graph = build_natural_visibility_graph(df_serie_semanal)

    # 4. Verificación topológica
    logger.info(f"Total de nodos (|V|): {visibility_graph.number_of_nodes()}")
    logger.info(f"Total de aristas (|E|): {visibility_graph.number_of_edges()}")

except KeyError as err:
    logger.error(f"Error en columna requerida: {err}")
except Exception as err:
    logger.error(f"Fallo en la construcción del grafo: {err}")

# %% [markdown]
# <details>
# <summary><b>Visualización: Histograma</b></summary>
# 
# <br>
# 
# Genera el gráfico interactivo de barras y marcadores discretos extrayendo la topología y atributos temporales almacenados directamente en los nodos del grafo de visibilidad (`nx.Graph`).
# 
# #### Parámetros
# * **`graph`** (`nx.Graph`): Red compleja con atributos `fecha_hecho` y `cantidad` en cada nodo.
# * **`crime_name`** (`str`): Etiqueta del delito para el título descriptivo.
# 
# #### Retorno
# * **`go.Figure`**: Objeto gráfico interactivo de Plotly renderizado.
# 
# </details>

# %%
# [HELPERS] Histograma
# INPUT:  graph (nx.Graph), crime_name (str)
# OUTPUT: go.Figure

def plot_weekly_network_nodes(graph: nx.Graph, crime_name: str) -> go.Figure:
    """Renderiza el histograma semanal y la asignación de nodos leyendo desde el grafo."""
    logger.info("Extrayendo atributos de nodos para renderizado")
    if graph.number_of_nodes() == 0:
        logger.warning("El grafo recibido no contiene nodos.")
        return go.Figure()

    # 1. Extracción ordenada de atributos desde los nodos del grafo
    nodes = sorted(graph.nodes())
    fechas = [graph.nodes[n]["fecha_hecho"] for n in nodes]
    cantidades = [graph.nodes[n]["cantidad"] for n in nodes]

    max_val = max(cantidades) if cantidades else 0
    offset = max_val * 0.05 if max_val > 0 else 1

    fig = go.Figure()

    # 2. Barras de delitos por semana
    fig.add_trace(
        go.Bar(
            x=fechas,
            y=cantidades,
            marker=dict(color="white", line=dict(color="black", width=1)),
            name="Delitos",
        )
    )

    # 3. Marcadores de nodos sobre cada barra
    y_markers = [val + offset for val in cantidades]
    text_labels = [str(n) for n in nodes]

    fig.add_trace(
        go.Scatter(
            x=fechas,
            y=y_markers,
            mode="markers+text",
            marker=dict(
                symbol="circle",
                color="white",
                size=14,
                line=dict(color="black", width=1),
            ),
            text=text_labels,
            textfont=dict(color="black", size=8),
            textposition="middle center",
            name="Nodos",
        )
    )

    # 4. Configuración de títulos y ejes
    formatted_crime = str(crime_name).title()
    min_date = min(fechas)
    max_date = max(fechas)

    title_str = (
        f"Asignación de Nodos en la Serie Temporal Semanal de<br><b>{formatted_crime}</b><br>"
        f"<sub>Periodo: {min_date.strftime('%Y-%m-%d')} a {max_date.strftime('%Y-%m-%d')}</sub>"
    )

    # Etiquetas visibles en el eje X (cada 16 semanas)
    visible_ticks = fechas[::16]

    fig.update_layout(
        title=dict(
            text=title_str,
            x=0.5,
            xanchor="center",
            font=dict(size=16, color="black"),
        ),
        width=1300,
        height=750,
        plot_bgcolor="white",
        paper_bgcolor="white",
        bargap=0.1,
        margin=dict(l=70, r=50, t=110, b=100),
        showlegend=False,
        xaxis=dict(
            title="Inicio de Semana (Bloques de 168h)",
            color="black",
            showline=True,
            linecolor="black",
            linewidth=1.5,
            tickangle=45,
            tickmode="array",
            tickvals=visible_ticks,
            tickformat="%Y-%m-%d",
        ),
        yaxis=dict(
            title="Cantidad de delitos",
            color="black",
            showline=True,
            linecolor="black",
            linewidth=1.5,
            showgrid=True,
            gridcolor="lightgrey",
            gridwidth=0.2,
            griddash="dash",
            range=[0, max_val + (offset * 3)],
        ),
    )

    logger.info("Exito.")
    return fig

# %% [markdown]
# <details>
# <summary><b>Ejecución: Histograma</b></summary>
# 
# <br>
# 
# Invoca la función de renderizado `plot_weekly_network_nodes` pasando como origen de datos `visibility_graph` y despliega la figura interactiva en el notebook.
# 
# #### Variables de entrada
# * **`visibility_graph`** (`nx.Graph`): Red compleja generada previamente.
# * **`TARGET_CRIME`** (`str`): Constante global del delito analizado.
# 
# </details>

# %%
# [EJECUCIÓN] Renderizado gráfico de la red
# INPUT:  visibility_graph (nx.Graph), TARGET_CRIME (str)
# OUTPUT: None (despliegue en output)

try:
    fig = plot_weekly_network_nodes(visibility_graph, TARGET_CRIME)
    fig.show()
except Exception as err:
    logger.error(f"Error al renderizar el gráfico: {err}")

# %% [markdown]
# <details>
# <summary><b>Visualización: Red de Visibilidad Natural (NVG)</b></summary>
# 
# <br>
# 
# Renderiza la serie temporal discreta junto con la topología de la Red de Visibilidad Natural (NVG), trazando las aristas entre las cúspides de las observaciones y superponiendo los identificadores de cada nodo.
# 
# #### Parámetros
# * **`graph`** (`nx.Graph`): Red compleja con coordenadas y atributos temporales (`fecha_hecho`, `cantidad`).
# * **`crime_name`** (`str`, opcional): Nombre del delito para la cabecera del gráfico.
# 
# #### Retorno
# * **`go.Figure`**: Figura interactiva de Plotly con las 4 trazas superpuestas (barras, aristas de visibilidad, cúspides y etiquetas de nodo).
# 
# </details>

# %%
# [HELPERS] Visualización completa de la red NVG
# INPUT:  graph (nx.Graph), crime_name (str)
# OUTPUT: go.Figure

def plot_visibility_graph_from_network(
    graph: nx.Graph, crime_name: str = ""
) -> go.Figure:
    """Renderiza la serie temporal, aristas de visibilidad y nodos usando únicamente el grafo."""
    logger.info("Extrayendo topología y coordenadas de la red NVG")
    if graph.number_of_nodes() == 0:
        logger.warning("El grafo recibido no contiene nodos.")
        return go.Figure()

    # 1. Coordenadas de los nodos
    nodes = sorted(graph.nodes())
    x_bars = [graph.nodes[i]["fecha_hecho"] for i in nodes]
    y_bars = [graph.nodes[i]["cantidad"] for i in nodes]

    max_val = max(y_bars) if y_bars else 0
    offset = max_val * 0.05 if max_val > 0 else 1

    # 2. Segmentos de aristas de visibilidad (separados por None para trazo continuo)
    edge_x = []
    edge_y = []
    for u, v in graph.edges():
        edge_x.extend([graph.nodes[u]["fecha_hecho"], graph.nodes[v]["fecha_hecho"], None])
        edge_y.extend([graph.nodes[u]["cantidad"], graph.nodes[v]["cantidad"], None])

    fig = go.Figure()

    # Traza 1: Barras de la serie temporal
    fig.add_trace(
        go.Bar(
            x=x_bars,
            y=y_bars,
            marker=dict(color="white", line=dict(color="black", width=1)),
            name="Serie Temporal",
        )
    )

    # Traza 2: Líneas de visibilidad (aristas de la red)
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(color="red", width=0.8),
            hoverinfo="none",
            name="Visibilidad",
        )
    )

    # Traza 3: Marcadores en la cúspide
    fig.add_trace(
        go.Scatter(
            x=x_bars,
            y=y_bars,
            mode="markers",
            marker=dict(size=4, color="black"),
            hoverinfo="skip",
            name="Cúspides",
        )
    )

    # Traza 4: Nodos numerados sobre las barras
    fig.add_trace(
        go.Scatter(
            x=x_bars,
            y=[y + offset for y in y_bars],
            mode="markers+text",
            marker=dict(
                symbol="circle",
                color="white",
                size=14,
                line=dict(color="black", width=1),
            ),
            text=[str(i) for i in nodes],
            textfont=dict(color="black", size=8),
            textposition="middle center",
            name="Nodos",
        )
    )

    # Configuración de layout
    title_text = "Red de Visibilidad Natural (NVG)"
    if crime_name:
        title_text += f"<br><b>{crime_name.title()}</b>"

    visible_ticks = x_bars[::16]

    fig.update_layout(
        title=dict(
            text=title_text,
            x=0.5,
            xanchor="center",
            font=dict(size=16, color="black"),
        ),
        width=1300,
        height=750,
        plot_bgcolor="white",
        paper_bgcolor="white",
        bargap=0.1,
        margin=dict(l=70, r=50, t=110, b=100),
        showlegend=False,
        xaxis=dict(
            title="Inicio de Semana (Bloques de 168h)",
            color="black",
            showline=True,
            linecolor="black",
            linewidth=1.5,
            tickangle=45,
            tickmode="array",
            tickvals=visible_ticks,
            tickformat="%Y-%m-%d",
        ),
        yaxis=dict(
            title="Cantidad de delitos",
            color="black",
            showline=True,
            linecolor="black",
            linewidth=1.5,
            showgrid=True,
            gridcolor="lightgrey",
            gridwidth=0.2,
            griddash="dash",
            range=[0, max_val + (offset * 3)],
        ),
    )

    logger.info("Exito.")
    return fig

# %% [markdown]
# <details>
# <summary><b>Ejecución: Visualización del Grafo de Visibilidad Natural</b></summary>
# 
# <br>
# 
# Invoca `plot_visibility_graph_from_network` sobre el grafo compilado `visibility_graph` para proyectar las interconexiones de visibilidad sobre la serie cronológica.
# 
# #### Variables de entrada
# * **`visibility_graph`** (`nx.Graph`): Red compleja generada con aristas y atributos.
# * **`TARGET_CRIME`** (`str`): Constante global con el nombre del delito en estudio.
# 
# </details>

# %%
# [EJECUCIÓN] Renderizado del Grafo de Visibilidad Natural (NVG)
# INPUT:  visibility_graph (nx.Graph), TARGET_CRIME (str)
# OUTPUT: None (despliegue en output)

try:
    fig_nvg = plot_visibility_graph_from_network(visibility_graph, TARGET_CRIME)
    fig_nvg.show()
except Exception as err:
    logger.error(f"Error al renderizar el grafo de visibilidad: {err}")

# %% [markdown]
# <details>
# <summary><b>Stats: Métricas Topológicas Globales</b></summary>
# 
# <br>
# 
# Calcula las propiedades estructurales y de distancia geodésica global del grafo de visibilidad natural, identificando las rutas críticas correspondientes al diámetro y radio de la red.
# 
# #### Parámetros
# * **`grafo`** (`nx.Graph`): Grafo de visibilidad construido.
# 
# #### Retorno
# * **`dict`**: Diccionario con métricas escalares y subestructuras:
#   * `densidad_empirica`: Proporción de aristas existentes respecto al total teórico posible.
#   * `radio` y `diametro`: Excentricidad mínima y máxima de la red conexa.
#   * `centro`: Conjunto de nodos cuya excentricidad iguala al radio.
#   * `nodos_diametro` / `aristas_diametro`: Ruta geodésica de longitud máxima.
#   * `nodos_radio` / `aristas_radio`: Ruta geodésica representativa desde el centro.
# 
# </details>

# %%
# [STATS] Métricas topológicas globales de la red
# INPUT:  grafo (nx.Graph)
# OUTPUT: dict

def compute_graph_metrics(grafo: nx.Graph) -> dict:
    """Calcula densidad, radio, diámetro, centro y rutas geodésicas del grafo."""
    logger.info("Calculando métricas topológicas globales")
    densidad_empirica = nx.density(grafo)

    radio, diametro, centro = None, None, []
    aristas_diametro, aristas_radio = [], []
    nodos_diametro, nodos_radio = [], []

    if not nx.is_connected(grafo):
        logger.warning("El grafo no es conexo. Métricas basadas en distancias no aplican.")
        return {
            "densidad_empirica": densidad_empirica,
            "radio": radio,
            "diametro": diametro,
            "centro": centro,
            "nodos_diametro": nodos_diametro,
            "aristas_diametro": aristas_diametro,
            "nodos_radio": nodos_radio,
            "aristas_radio": aristas_radio,
        }

    radio = nx.radius(grafo)
    diametro = nx.diameter(grafo)
    centro = nx.center(grafo)

    path_lengths = dict(nx.shortest_path_length(grafo))

    # Identificación del camino diametral
    n1_diam, n2_diam = None, None
    for u in grafo.nodes():
        for v in grafo.nodes():
            if path_lengths[u][v] == diametro:
                n1_diam, n2_diam = u, v
                break
        if n1_diam is not None:
            break

    if n1_diam is not None:
        nodos_diametro = nx.shortest_path(grafo, source=n1_diam, target=n2_diam)
        aristas_diametro = list(zip(nodos_diametro, nodos_diametro[1:]))

    # Identificación del camino radial desde el centro
    if centro:
        nodo_centro = centro[0]
        for target_node in grafo.nodes():
            if path_lengths[nodo_centro][target_node] == radio:
                nodos_radio = nx.shortest_path(
                    grafo, source=nodo_centro, target=target_node
                )
                aristas_radio = list(zip(nodos_radio, nodos_radio[1:]))
                break

    logger.info("EXITO.")
    return {
        "densidad_empirica": densidad_empirica,
        "radio": radio,
        "diametro": diametro,
        "centro": centro,
        "nodos_diametro": nodos_diametro,
        "aristas_diametro": aristas_diametro,
        "nodos_radio": nodos_radio,
        "aristas_radio": aristas_radio,
    }

# %% [markdown]
# <details>
# <summary><b>Ejecución: Extracción de métricas topológicas</b></summary>
# 
# <br>
# 
# Ejecuta el cómputo de propiedades estructurales sobre `visibility_graph` y reporta en consola los valores escalares calculados.
# 
# #### Variables generadas
# * **`topologia_global`** (`dict`): Diccionario con los parámetros calculados y rutas geodésicas.
# 
# </details>

# %%
# [EJECUCIÓN] Extracción de métricas topológicas globales
# INPUT:  visibility_graph (nx.Graph)
# OUTPUT: topologia_global (dict)

try:
    graph_metrics = compute_graph_metrics(visibility_graph)

    logger.info(f"Densidad empírica: {graph_metrics['densidad_empirica']:.6f}")
    logger.info(f"Radio: {graph_metrics['radio']}")
    logger.info(f"Diámetro: {graph_metrics['diametro']}")
    logger.info(f"Centro: {graph_metrics['centro']}")
    logger.info(f"Nodos en camino diametral: {graph_metrics['nodos_diametro']}")
    logger.info(f"Nodos en camino radial: {graph_metrics['nodos_radio']}")
except Exception as err:
    logger.error(f"Fallo al calcular métricas topológicas: {err}")

# %% [markdown]
# <details>
# <summary><b>Visualización: Métricas del grafo</b></summary>
# 
# <br>
# 
# Renderiza la topología del grafo en el espacio bidimensional mediante el algoritmo de fuerzas `spring_layout` de NetworkX y Matplotlib, resaltando las subestructuras críticas del diámetro (rojo), radio (verde) y centro (amarillo).
# 
# #### Parámetros
# * **`grafo`** (`nx.Graph`): Red compleja a graficar con atributos de fecha en nodos.
# * **`metricas`** (`dict`): Diccionario de métricas globales precalculadas (`densidad_empirica`, `centro`, rutas diametrales y radiales).
# * **`crime_name`** (`str`): Delito analizado para el encabezado del lienzo.
# 
# #### Retorno
# * **`None`**: Despliegue directo del gráfico en el frontend de Matplotlib.
# 
# </details>

# %%
# [HELPERS] Visualización métricas del grafo
# INPUT:  grafo (nx.Graph), metricas (dict), crime_name (str)
# OUTPUT: None

def plot_network(grafo: nx.Graph, metricas: dict, crime_name: str) -> None:
    """Renderiza la red de visibilidad estática usando Matplotlib con métricas precalculadas."""
    logger.info("Preparando renderizado del grafo topológico")
    if grafo.number_of_nodes() == 0:
        logger.warning("El grafo recibido está vacío.")
        return

    # 1. Rango temporal para el título
    fechas = [
        pd.to_datetime(data["fecha_hecho"])
        for _, data in grafo.nodes(data=True)
        if "fecha_hecho" in data
    ]
    rango_fechas = (
        f"({min(fechas).strftime('%Y-%m-%d')} – {max(fechas).strftime('%Y-%m-%d')})"
        if fechas
        else ""
    )
    titulo_grafico = (
        f"Red de Visibilidad Semanal con Métricas Clave\n"
        f"{crime_name.title()}\n{rango_fechas}"
    )

    # 2. Lienzo y disposición espacial
    fig, ax = plt.subplots(figsize=(15, 8))
    pos = nx.spring_layout(grafo, seed=42)

    # 3. Capas base
    nx.draw_networkx_edges(
        grafo, pos, edge_color="lightgray", width=1.0, alpha=0.6, ax=ax
    )
    nx.draw_networkx_nodes(
        grafo,
        pos,
        node_color="white",
        node_size=180,
        edgecolors="black",
        linewidths=0.6,
        ax=ax,
    )

    # 4. Capas para métricas extremas (diámetro, radio, centro)
    if metricas.get("aristas_diametro"):
        nx.draw_networkx_edges(
            grafo,
            pos,
            edgelist=metricas["aristas_diametro"],
            edge_color="red",
            width=2.5,
            ax=ax,
        )
    if metricas.get("aristas_radio"):
        nx.draw_networkx_edges(
            grafo,
            pos,
            edgelist=metricas["aristas_radio"],
            edge_color="green",
            width=2.5,
            ax=ax,
        )
    if metricas.get("centro"):
        nx.draw_networkx_nodes(
            grafo,
            pos,
            nodelist=metricas["centro"],
            node_color="yellow",
            node_size=220,
            edgecolors="black",
            ax=ax,
        )

    nx.draw_networkx_labels(grafo, pos, font_size=5.5, font_weight="bold", ax=ax)

    # 5. Construcción de leyenda
    elementos_leyenda = [
        Line2D(
            [0],
            [0],
            color="w",
            label=f"Densidad Empírica: {metricas.get('densidad_empirica', 0):.4f}",
        )
    ]
    if metricas.get("centro"):
        elementos_leyenda.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=f"Centro: {metricas['centro']}",
                markerfacecolor="yellow",
                markersize=10,
                markeredgecolor="black",
            )
        )
    if metricas.get("diametro") is not None:
        elementos_leyenda.append(
            Line2D(
                [0],
                [0],
                color="red",
                lw=3,
                label=f"Diámetro: {metricas['diametro']}",
            )
        )
    if metricas.get("radio") is not None:
        elementos_leyenda.append(
            Line2D(
                [0],
                [0],
                color="green",
                lw=3,
                label=f"Radio: {metricas['radio']}",
            )
        )

    ax.legend(
        handles=elementos_leyenda,
        loc="best",
        facecolor="white",
        framealpha=0.8,
        fontsize=10,
    )

    plt.title(titulo_grafico, fontsize=16, pad=15)
    plt.axis("off")
    plt.show()
    logger.info("Exito.")

# %% [markdown]
# <details>
# <summary><b>Ejecución: Visualización de la Red de Visibilidad Topológica</b></summary>
# 
# <br>
# 
# Ejecuta el trazado de la red global utilizando las métricas contenidas en `topologia_global` sobre `visibility_graph`.
# 
# #### Variables de entrada
# * **`visibility_graph`** (`nx.Graph`): Red compleja del delito seleccionado.
# * **`topologia_global`** (`dict`): Diccionario con rutas geodésicas y métricas de centralidad calculadas.
# * **`TARGET_CRIME`** (`str`): Constante del delito analizado.
# 
# </details>

# %%
# [EJECUCIÓN] Renderizado de red topológica con métricas
# INPUT:  visibility_graph (nx.Graph), graph_metrics (dict), TARGET_CRIME (str)
# OUTPUT: None (despliegue en output)

try:
    plot_network(visibility_graph, graph_metrics, TARGET_CRIME)
except Exception as err:
    logger.error(f"Fallo al renderizar el gráfico topológico: {err}")

# %% [markdown]
# <details>
# <summary><b>Stats: Métricas y Centralidades por Nodo</b></summary>
# 
# <br>
# 
# Calcula las métricas estructurales locales para cada vértice de la red (grado, intermediación, cercanía, agrupamiento y excentricidad) integrándolas con los atributos temporales y de frecuencia de eventos.
# 
# #### Parámetros
# * **`grafo`** (`nx.Graph`): Red compleja con atributos `fecha_hecho` y `cantidad` asignados a cada nodo.
# 
# #### Retorno
# * **`pd.DataFrame`**: Tabla consolidada con las métricas locales indexadas por nodo.
# 
# </details>

# %%
# [HELPERS] Métricas y centralidades por nodo
# INPUT:  grafo (nx.Graph)
# OUTPUT: pd.DataFrame

def compute_node_metrics(grafo: nx.Graph) -> pd.DataFrame:
    """Calcula métricas topológicas locales por nodo y devuelve un DataFrame consolidado."""
    logger.info("Calculando métricas y centralidades por nodo")
    if grafo.number_of_nodes() == 0:
        logger.warning("El grafo recibido no contiene nodos.")
        return pd.DataFrame()

    # 1. Medidas de conectividad y centralidad
    grados = dict(grafo.degree())
    intermediacion = nx.betweenness_centrality(grafo)
    cercania = nx.closeness_centrality(grafo)
    agrupamiento = nx.clustering(grafo)

    if nx.is_connected(grafo):
        excentricidad = nx.eccentricity(grafo)
    else:
        excentricidad = {n: "N/A" for n in grafo.nodes()}

    # 2. Construcción tabular con atributos del nodo
    registros = []
    for nodo, datos in sorted(grafo.nodes(data=True)):
        fecha_dt = pd.to_datetime(datos.get("fecha_hecho", pd.NaT))
        fecha_str = fecha_dt.strftime("%Y-%m-%d") if pd.notna(fecha_dt) else "N/A"

        registros.append(
            {
                "Nodo": nodo,
                "Inicio de Semana(fecha_hecho)": fecha_str,
                "Delitos(cantidad)": datos.get("cantidad", 0),
                "Grado (Conexiones)": grados.get(nodo, 0),
                "Intermediación": round(intermediacion.get(nodo, 0.0), 4),
                "Cercanía": round(cercania.get(nodo, 0.0), 4),
                "Agrupamiento": round(agrupamiento.get(nodo, 0.0), 4),
                "Excentricidad": excentricidad.get(nodo),
            }
        )

    logger.info(f"Exito. Métricas generadas para {len(registros)} nodos.")
    return pd.DataFrame(registros)

# %% [markdown]
# <details>
# <summary><b>Ejecución: Cómputo de métricas por nodo</b></summary>
# 
# <br>
# 
# Ejecuta `compute_node_metrics` sobre el grafo compilado `visibility_graph` para construir la tabla de propiedades nodales `df_node_metrics`.
# 
# #### Variables generadas
# * **`df_node_metrics`** (`pd.DataFrame`): Conjunto estructurado con las métricas locales de cada nodo temporal.
# 
# </details>

# %%
# [EJECUCIÓN] Cómputo de métricas por nodo
# INPUT:  visibility_graph (nx.Graph)
# OUTPUT: df_node_metrics (pd.DataFrame)

try:
    df_node_metrics = compute_node_metrics(visibility_graph)
    logger.info(f"Dimensiones de df_node_metrics: {df_node_metrics.shape}")
    display(df_node_metrics.head())
except Exception as err:
    logger.error(f"Fallo al calcular las métricas por nodo: {err}")

# %% [markdown]
# <details>
# <summary><b>Reportes: Resumen Analítico de Métricas Topológicas</b></summary>
# 
# <br>
# 
# Genera el reporte tabular consolidado del análisis de la red de visibilidad, consumiendo las propiedades globales de `graph_metrics` y las métricas nodales precalculadas en `df_node_metrics`.
# 
# #### Parámetros
# * **`grafo`** (`nx.Graph`): Red compleja evaluada.
# * **`graph_metrics`** (`dict`): Diccionario con métricas globales (diámetro, radio, centro).
# * **`df_nodes`** (`pd.DataFrame`): DataFrame con las métricas y centralidades por nodo.
# 
# #### Retorno
# * **`None`**: Emisión del reporte analítico formateado en consola.
# 
# </details>

# %%
# [HELPERS] Generación de reporte analítico consolidado
# INPUT:  grafo (nx.Graph), graph_metrics (dict), df_nodes (pd.DataFrame)
# OUTPUT: None (impresión en consola)

def print_network_metrics_summary(
    grafo: nx.Graph, graph_metrics: dict, df_nodes: pd.DataFrame
) -> None:
    """Imprime un reporte analítico exhaustivo consumiendo métricas precalculadas."""
    logger.info("Generando resumen analítico de métricas")
    if grafo.number_of_nodes() == 0 or df_nodes.empty:
        logger.warning("Grafo o DataFrame de métricas vacío.")
        return

    print("======================================================")
    print("   RESUMEN DE MÉTRICAS DE LA RED DE VISIBILIDAD SEMANAL ")
    print("======================================================")

    # 1. Métricas Generales
    print("\n--- 1. MÉTRICAS GENERALES DE LA RED ---")
    print(f"- Número total de Nodos (Semanas): {grafo.number_of_nodes()}")
    print(f"- Número total de Enlaces: {grafo.number_of_edges()}")
    print(f"- Componentes Conectados: {nx.number_connected_components(grafo)}")
    print(f"- Densidad Empírica: {graph_metrics.get('densidad_empirica', 0):.6f}")

    # 2. Métricas de Distancia
    print("\n--- 2. MÉTRICAS DE DISTANCIA ---")
    if nx.is_connected(grafo):
        print(f"- Diámetro de la Red: {graph_metrics.get('diametro')}")
        print(f"- Radio de la Red: {graph_metrics.get('radio')}")
        print(f"- Centro de la Red (Nodos): {graph_metrics.get('centro')}")

        top_centricos = df_nodes.sort_values(
            by=["Excentricidad", "Nodo"], ascending=[True, True]
        ).head(5)
        print("\n  - TOP 5 Nodos con Menor Excentricidad (más céntricos):")
        for _, r in top_centricos.iterrows():
            print(
                f"    - Nodo {r['Nodo']} (Semana {r['Inicio de Semana(fecha_hecho)']}): "
                f"Excentricidad = {r['Excentricidad']}"
            )
    else:
        print(
            "- El grafo no es conexo, no se pueden calcular Diámetro, Radio, Centro ni Excentricidad."
        )

    # 3. Análisis de Nodos Clave
    print("\n--- 3. ANÁLISIS DE NODOS CLAVE ---")
    nodo_pico_row = df_nodes.loc[df_nodes["Delitos(cantidad)"].idxmax()]
    print("\n  - Nodo Pico (Semana con Valor Máximo):")
    print(
        f"    - Nodo {int(nodo_pico_row['Nodo'])} "
        f"(Inicio {nodo_pico_row['Inicio de Semana(fecha_hecho)']}) con "
        f"{nodo_pico_row['Delitos(cantidad)']} delitos"
    )
    print("    - Métricas Específicas del Nodo Pico:")
    print(f"      - Grado (Conexiones): {nodo_pico_row['Grado (Conexiones)']}")
    print(f"      - Excentricidad: {nodo_pico_row['Excentricidad']}")
    print(
        f"      - Centralidad de Intermediación: {nodo_pico_row['Intermediación']:.4f}"
    )
    print(f"      - Centralidad de Cercanía: {nodo_pico_row['Cercanía']:.4f}")
    print(
        f"      - Coeficiente de Agrupamiento: {nodo_pico_row['Agrupamiento']:.4f}"
    )

    centros = graph_metrics.get("centro", [])
    if centros:
        print(f"\n  - Métricas de los Nodos del Centro Geodésico ({centros}):")
        df_centros = df_nodes[df_nodes["Nodo"].isin(centros)]
        for _, r in df_centros.iterrows():
            print(
                f"    - Análisis del Nodo {int(r['Nodo'])} (Semana {r['Inicio de Semana(fecha_hecho)']}):"
            )
            print(f"      - Grado (Conexiones): {r['Grado (Conexiones)']}")
            print(
                f"      - Centralidad de Intermediación: {r['Intermediación']:.4f}"
            )
            print(f"      - Centralidad de Cercanía: {r['Cercanía']:.4f}")
            print(
                f"      - Coeficiente de Agrupamiento: {r['Agrupamiento']:.4f}"
            )

    # 4. TOP 5 Nodos por Métrica
    print("\n--- 4. TOP 5 NODOS POR MÉTRICA ---")

    print("\n  - TOP 5 Nodos con Mayor Grado (más conectados):")
    for _, r in (
        df_nodes.sort_values(by="Grado (Conexiones)", ascending=False)
        .head(5)
        .iterrows()
    ):
        print(
            f"    - Nodo {int(r['Nodo'])} ({r['Inicio de Semana(fecha_hecho)']}): "
            f"{r['Grado (Conexiones)']} conexiones"
        )

    print("\n  - TOP 5 Nodos con Mayor Centralidad de Intermediación (puentes):")
    for _, r in (
        df_nodes.sort_values(by="Intermediación", ascending=False)
        .head(5)
        .iterrows()
    ):
        print(
            f"    - Nodo {int(r['Nodo'])} ({r['Inicio de Semana(fecha_hecho)']}): "
            f"{r['Intermediación']:.4f}"
        )

    print("\n  - TOP 5 Nodos con Mayor Centralidad de Cercanía (mejor acceso):")
    for _, r in (
        df_nodes.sort_values(by="Cercanía", ascending=False).head(5).iterrows()
    ):
        print(
            f"    - Nodo {int(r['Nodo'])} ({r['Inicio de Semana(fecha_hecho)']}): "
            f"{r['Cercanía']:.4f}"
        )

    # 5. Coeficiente de Agrupamiento
    print("\n--- 5. COEFICIENTE DE AGRUPAMIENTO (CLUSTERING) ---")
    print("\n  - Nodos con Mayor Coeficiente (en grupos más densos):")
    for _, r in (
        df_nodes.sort_values(by="Agrupamiento", ascending=False)
        .head(5)
        .iterrows()
    ):
        print(
            f"    - Nodo {int(r['Nodo'])} ({r['Inicio de Semana(fecha_hecho)']}): "
            f"{r['Agrupamiento']:.4f}"
        )

    print("\n  - Nodos con Menor Coeficiente (>0):")
    df_clustering_pos = df_nodes[df_nodes["Agrupamiento"] > 0]
    nodos_menor = (
        df_clustering_pos.sort_values(by="Agrupamiento", ascending=True).head(5)
        if not df_clustering_pos.empty
        else df_nodes.sort_values(by="Agrupamiento", ascending=True).head(5)
    )

    for _, r in nodos_menor.iterrows():
        print(
            f"    - Nodo {int(r['Nodo'])} ({r['Inicio de Semana(fecha_hecho)']}): "
            f"{r['Agrupamiento']:.4f}"
        )

    print("\n======================================================")
    print("     Análisis de Métricas Semanales Finalizado        ")
    print("======================================================")
    logger.info("Exito.")

# %% [markdown]
# <details>
# <summary><b>Ejecución: Reporte consolidado de métricas de red</b></summary>
# 
# <br>
# 
# Invoca `print_network_metrics_summary` para formatear y desplegar el diagnóstico estructural completo en stdout sin recalcular métricas costosas.
# 
# #### Variables de entrada
# * **`visibility_graph`** (`nx.Graph`): Red analizada.
# * **`graph_metrics`** (`dict`): Métricas globales precalculadas.
# * **`df_node_metrics`** (`pd.DataFrame`): Métricas por nodo precalculadas.
# 
# </details>

# %%
# [EJECUCIÓN] Reporte consolidado de métricas
# INPUT:  visibility_graph (nx.Graph), graph_metrics (dict), df_node_metrics (pd.DataFrame)
# OUTPUT: None (impresión en stdout)

try:
    print_network_metrics_summary(
        visibility_graph, graph_metrics, df_node_metrics
    )
except Exception as err:
    logger.error(f"Fallo al emitir el resumen analítico: {err}")

# %% [markdown]
# <details>
# <summary><b>Visualización: Distribución de Grados</b></summary>
# 
# <br>
# 
# Genera el diagrama de barras de la secuencia ordenada de grados (conectividad) de la red, permitiendo evaluar de forma preliminar la heterogeneidad en el número de enlaces y la presencia de hubs temporales.
# 
# #### Parámetros
# * **`grafo`** (`nx.Graph`): Grafo de visibilidad con atributos temporales en los nodos.
# * **`crime_name`** (`str`): Delito analizado para el encabezado del gráfico.
# 
# #### Retorno
# * **`None`**: Despliegue directo del gráfico en el frontend de Matplotlib.
# 
# </details>

# %%
# [HELPERS] Visualización de la distribución de grados ordenada
# INPUT:  grafo (nx.Graph), crime_name (str)
# OUTPUT: None

def plot_degree_distribution(grafo: nx.Graph, crime_name: str) -> None:
    """Genera el gráfico de barras de la distribución ordenada de grados."""
    logger.info("Preparando gráfico de distribución de grados")
    if grafo.number_of_nodes() == 0:
        logger.warning("El grafo recibido está vacío.")
        return

    # 1. Grados ordenados descendentemente
    grados_ordenados = sorted((grado for _, grado in grafo.degree()), reverse=True)

    # 2. Rango de fechas para el título
    fechas = [
        pd.to_datetime(data["fecha_hecho"])
        for _, data in grafo.nodes(data=True)
        if "fecha_hecho" in data
    ]
    rango_fechas = (
        f"({min(fechas).strftime('%Y-%m-%d')} – {max(fechas).strftime('%Y-%m-%d')})"
        if fechas
        else "(Periodo Desconocido)"
    )

    titulo = (
        f"Distribución de Grados en la Red de Visibilidad Semanal\n"
        f"{crime_name.title()}\n{rango_fechas}"
    )

    # 3. Lienzo
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=100)
    fig.patch.set_facecolor("white")
    ax.patch.set_facecolor("white")

    # 4. Barras
    x_pos = np.arange(len(grados_ordenados))
    ax.bar(
        x_pos,
        grados_ordenados,
        color="white",
        edgecolor="black",
        linewidth=1.2,
    )

    # 5. Estilos y formato
    ax.set_title(titulo, fontsize=13, pad=15, color="black")
    ax.set_ylabel("Grado del nodo (Conexiones)", fontsize=11, color="black")
    ax.set_xlabel(
        "Nodos (Semanas) ordenados por conectividad", fontsize=11, color="black"
    )

    gmax = max(grados_ordenados) if grados_ordenados else 0
    ax.set_ylim(0, gmax + max(1, int(gmax * 0.08)))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")

    ax.tick_params(axis="both", direction="out", colors="black", labelsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, color="gray")

    plt.tight_layout()
    plt.show()
    logger.info("Exito.")

# %% [markdown]
# <details>
# <summary><b>Ejecución: Renderizado de la distribución de conectividad</b></summary>
# 
# <br>
# 
# Invoca `plot_degree_distribution` para graficar la secuencia decreciente de conexiones sobre `visibility_graph`.
# 
# #### Variables de entrada
# * **`visibility_graph`** (`nx.Graph`): Red compleja generada.
# * **`TARGET_CRIME`** (`str`): Constante global del delito analizado.
# 
# </details>

# %%
# [EJECUCIÓN] Renderizado de distribución de grados
# INPUT:  visibility_graph (nx.Graph), TARGET_CRIME (str)
# OUTPUT: None (despliegue en output)

try:
    plot_degree_distribution(visibility_graph, TARGET_CRIME)
except Exception as err:
    logger.error(f"Fallo al graficar la distribución de grados: {err}")

# %% [markdown]
# <details>
# <summary><b>Análisis Estadístico: Ajuste de Ley de Potencia</b></summary>
# 
# <br>
# 
# Ajusta el modelo estadístico de Ley de Potencia ($P(k) \sim k^{-\gamma}$) a la distribución empírica de grados mediante el formalismo de Clauset, Shalizi y Newman (CSN) vía la librería `powerlaw`.
# 
# #### Métodos y pruebas incluidas
# * **Estimación de parámetros:** Cálculo del umbral óptimo $x_{\min}$ y del exponente de escala $\gamma$ ($\alpha$) por máxima verosimilitud.
# * **Test de razón de verosimilitud ($R$):** Comparación directa entre la hipótesis de Ley de Potencia frente al modelo alternativo Lognormal.
# * **Visualización:** Gráfico en escala logarítmica ($\log$-$\log$) con simulación de dispersión en el exponente $\gamma$.
# 
# #### Parámetros
# * **`grafo`** (`nx.Graph`): Red compleja evaluada.
# * **`crime_name`** (`str`): Delito analizado para el título de la figura.
# 
# #### Retorno
# * **`None`**: Despliegue directo del gráfico en el frontend de Matplotlib.
# 
# </details>

# %%
# [HELPERS] Ajuste estadístico y gráfico de Ley de Potencia
# INPUT:  grafo (nx.Graph), crime_name (str)
# OUTPUT: None

def plot_power_law_fit(grafo: nx.Graph, crime_name: str) -> None:
    """Calcula y grafica el ajuste de Ley de Potencia (P(k)) para la red."""
    logger.info("Iniciando ajuste estadístico de Ley de Potencia")
    if grafo.number_of_nodes() == 0:
        logger.warning("El grafo recibido está vacío.")
        return

    # 1. Preparación de grados (filtrar nodos con k = 0)
    data = np.array([d for _, d in grafo.degree() if d > 0])
    if len(data) == 0:
        logger.warning("El grafo no contiene nodos con grado positivo.")
        return

    # 2. Ajuste matemático (Clauset et al.) aislando warnings numéricos de SciPy
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=OptimizeWarning)
        fit = powerlaw.Fit(data, discrete=True, verbose=False)
        R, p_value_PL = fit.distribution_compare(
            "power_law", "lognormal", normalized_ratio=True
        )

    res_log = "Power Law" if R > 0 else "Alternativa"

    sigma_val = fit.power_law.sigma
    xmin_val = fit.power_law.xmin
    gamma_val = fit.power_law.alpha
    nodos_cumplen = np.sum(data >= xmin_val)

    # 3. Rango de fechas para el título
    fechas = [
        pd.to_datetime(d["fecha_hecho"])
        for _, d in grafo.nodes(data=True)
        if "fecha_hecho" in d
    ]
    rango_fechas = (
        f"({min(fechas).strftime('%Y-%m-%d')} – {max(fechas).strftime('%Y-%m-%d')})"
        if fechas
        else "(Periodo Desconocido)"
    )

    titulo = (
        f"Distribución de Grado P(k) y Ajuste de Ley de Potencia (CSN)\n"
        f"Red de Visibilidad Semanal - {crime_name.title()}\n"
        f"{rango_fechas}"
    )

    # 4. Construcción del gráfico
    plt.figure(figsize=(7, 5), dpi=100)

    # Simulación de dispersión de gamma
    x_space = np.linspace(xmin_val, data.max(), 100)
    for s in np.linspace(-2, 2, 50):
        gamma_var = gamma_val + (s * sigma_val)
        y_var = (
            (gamma_var - 1)
            * (xmin_val ** (gamma_var - 1))
            * (x_space**-gamma_var)
        )
        plt.plot(
            x_space,
            y_var,
            color="royalblue",
            alpha=0.15,
            linewidth=0.6,
            zorder=1,
        )

    # Curva empírica y ajuste óptimo
    fit.plot_pdf(
        color="black",
        marker="o",
        linestyle="None",
        markersize=5,
        label="Datos Empíricos",
    )
    fit.power_law.plot_pdf(
        color="red",
        linestyle="--",
        linewidth=2,
        label="Ajuste Óptimo Power Law",
    )

    # 5. Caja de texto estadístico
    texto_stats = (
        r"$\mathbf{Métricas\ del\ Ajuste:}$" + "\n"
        f"Gamma ($\\gamma$): {gamma_val:.4f}\n"
        f"Xmin: {xmin_val:.2f}\n"
        f"Nodos (k >= Xmin): {nodos_cumplen} / {len(data)}\n\n"
        r"$\mathbf{Bondad\ de\ Ajuste:}$" + "\n"
        f"p-value (PL): {p_value_PL:.4f}\n\n"
        r"$\mathbf{Comparación\ (R):}$" + "\n"
        f"R: {R:.4f} ({res_log})"
    )

    plt.text(
        0.05,
        0.05,
        texto_stats,
        transform=plt.gca().transAxes,
        fontsize=8,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.title(titulo, fontsize=11, pad=15)
    plt.xlabel("Grado (k)", fontsize=10)
    plt.ylabel("P(k)", fontsize=10)
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()
    logger.info("Exito.")

# %% [markdown]
# <details>
# <summary><b>Ejecución: Estimación y ajuste de Ley de Potencia</b></summary>
# 
# <br>
# 
# Invoca `plot_power_law_fit` para evaluar si la conectividad de `visibility_graph` exhibe invariancia de escala frente a la alternativa lognormal.
# 
# #### Variables de entrada
# * **`visibility_graph`** (`nx.Graph`): Red compleja analizada.
# * **`TARGET_CRIME`** (`str`): Delito bajo estudio.
# 
# </details>

# %%
# [EJECUCIÓN] Ajuste y evaluación de Ley de Potencia
# INPUT:  visibility_graph (nx.Graph), TARGET_CRIME (str)
# OUTPUT: None (despliegue en output)

try:
    plot_power_law_fit(visibility_graph, TARGET_CRIME)
except Exception as err:
    logger.error(f"Fallo en el ajuste de Ley de Potencia: {err}")

# %% [markdown]
# <details>
# <summary><b>Análisis Estadístico: Comparación Multimodelo y Distancia Euclidiana</b></summary>
# 
# <br>
# 
# Ajusta y compara de forma multimodelo las distribuciones teóricas alternativas (Ley de Potencia, Log-Normal y Exponencial) sobre el régimen de cola ($k \geq x_{\min}$), evaluando la bondad de ajuste mediante la distancia euclidiana discreta respecto a las frecuencias empíricas.
# 
# #### Parámetros
# * **`grafo`** (`nx.Graph`): Red compleja evaluada.
# * **`crime_name`** (`str`): Delito analizado para el título del gráfico.
# 
# #### Retorno
# * **`pd.DataFrame`**: Tabla resumen de distancias euclidianas ordenadas descendentemente por proximidad al dato empírico.
# 
# </details>

# %%
# [HELPERS] Comparación multimodelo y distancia euclidiana
# INPUT:  grafo (nx.Graph), crime_name (str)
# OUTPUT: pd.DataFrame

def plot_model_comparison(grafo: nx.Graph, crime_name: str) -> pd.DataFrame:
    """Compara modelos teóricos frente a la distribución empírica y grafica las curvas."""
    logger.info("Iniciando comparacion multimodelo y calculo de distancias")
    if grafo.number_of_nodes() == 0:
        logger.warning("El grafo recibido está vacío.")
        return pd.DataFrame()

    # 1. Grados positivos
    data = np.array([d for _, d in grafo.degree() if d > 0])
    if len(data) == 0:
        logger.warning("El grafo no contiene nodos con grado positivo.")
        return pd.DataFrame()

    # 2. Ajuste multimodelo aislando warnings de convergencia
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=OptimizeWarning)
        fit = powerlaw.Fit(data, discrete=True, verbose=False)

    xmin_val = fit.xmin

    # 3. Filtrado de cola (k >= xmin) y frecuencias empíricas
    data_xmin = data[data >= xmin_val]
    if len(data_xmin) == 0:
        logger.warning("Muestra insuficiente para k >= xmin.")
        return pd.DataFrame()

    k_eval, counts = np.unique(data_xmin, return_counts=True)
    y_emp_eval = counts / len(data_xmin)

    # 4. Evaluación de funciones de densidad teóricas (PDF)
    y_pl = fit.power_law.pdf(k_eval)
    y_lognorm = fit.lognormal.pdf(k_eval)
    y_exp = fit.exponential.pdf(k_eval)

    # 5. Distancia euclidiana vectorial: ||y_emp - y_teorica||_2
    dist_pl = float(np.linalg.norm(y_emp_eval - y_pl))
    dist_lognorm = float(np.linalg.norm(y_emp_eval - y_lognorm))
    dist_exp = float(np.linalg.norm(y_emp_eval - y_exp))

    df_distancias = pd.DataFrame(
        {
            "Modelo": ["Ley de Potencia", "Log-Normal", "Exponencial"],
            "Distancia Euclidiana": [dist_pl, dist_lognorm, dist_exp],
        }
    ).sort_values(by="Distancia Euclidiana").reset_index(drop=True)

    # 6. Rango temporal para título
    fechas = [
        pd.to_datetime(d["fecha_hecho"])
        for _, d in grafo.nodes(data=True)
        if "fecha_hecho" in d
    ]
    rango_fechas = (
        f"({min(fechas).strftime('%Y-%m-%d')} – {max(fechas).strftime('%Y-%m-%d')})"
        if fechas
        else "(Periodo Desconocido)"
    )

    titulo = (
        f"Comparación de Modelos Teóricos y Distancia Euclidiana\n"
        f"Red de Visibilidad Semanal - {crime_name.title()}\n"
        f"{rango_fechas}"
    )

    # 7. Renderizado en escala log-log
    plt.figure(figsize=(8, 6), dpi=100)
    plt.plot(
        k_eval,
        y_emp_eval,
        "ko",
        label=r"Datos Empíricos ($k \geq x_{min}$)",
        markersize=5,
    )
    plt.plot(
        k_eval,
        y_pl,
        "r--",
        linewidth=2,
        label=f"Power Law (D = {dist_pl:.5f})",
    )
    plt.plot(
        k_eval,
        y_lognorm,
        "b-.",
        linewidth=2,
        label=f"Log-Normal (D = {dist_lognorm:.5f})",
    )
    plt.plot(
        k_eval,
        y_exp,
        "g:",
        linewidth=2,
        label=f"Exponencial (D = {dist_exp:.5f})",
    )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Grado (k)", fontsize=10)
    plt.ylabel("P(k)", fontsize=10)
    plt.title(titulo, fontsize=11, pad=12)
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    plt.show()

    logger.info("EXITO.")
    return df_distancias

# %% [markdown]
# <details>
# <summary><b>Ejecución: Comparación multimodelo</b></summary>
# 
# <br>
# 
# Ejecuta `plot_model_comparison` para proyectar las curvas de distribución teóricas y genera la tabla comparativa de distancias euclidianas `df_model_distances`.
# 
# #### Variables generadas
# * **`df_model_distances`** (`pd.DataFrame`): Métricas de bondad de ajuste por distancia euclidiana para cada familia de distribución evaluada.
# 
# </details>

# %%
# [EJECUCIÓN] Comparativa de modelos teóricos de grado
# INPUT:  visibility_graph (nx.Graph), TARGET_CRIME (str)
# OUTPUT: df_model_distances (pd.DataFrame)

try:
    df_model_distances = plot_model_comparison(visibility_graph, TARGET_CRIME)
    display(df_model_distances)
except Exception as err:
    logger.error(f"Fallo en la comparación de modelos: {err}")

# %% [markdown]
# <details>
# <summary><b>Análisis Estadístico: Validación Formal CSN (Clauset-Shalizi-Newman)</b></summary>
# 
# <br>
# 
# Aplica el protocolo riguroso de Clauset, Shalizi y Newman (2009) para contrastar la hipótesis de libre de escala en la red de visibilidad.
# 
# #### Parámetros estadísticos calculados
# * **Exponente de escala ($\gamma$):** Estimador de máxima verosimilitud con desviación estándar asintótica ($\sigma$).
# * **Punto de corte óptimo ($X_{\min}$):** Minimización de la distancia de Kolmogorov-Smirnov ($D_{KS}$).
# * **Validación absoluta:** Distancia $D$ respecto a la distribución acumulada empírica.
# * **Test de Razón de Verosimilitud Normalizado ($R$):** Contraste frente a la alternativa lognormal con criterio de significancia ($p < 0.05$).
# 
# #### Parámetros de entrada
# * **`grafo`** (`nx.Graph`): Red compleja evaluada.
# * **`crime_name`** (`str`): Delito bajo estudio para el encabezado del gráfico.
# 
# #### Retorno
# * **`dict`**: Diccionario estructurado con los estimadores analíticos y el veredicto del test de razón de verosimilitud.
# 
# </details>

# %%
# [HELPERS] Validación analítica de Ley de Potencia bajo metodología CSN
# INPUT:  grafo (nx.Graph), crime_name (str)
# OUTPUT: dict

def plot_csn_analysis(grafo: nx.Graph, crime_name: str) -> dict:
    """Calcula y grafica el análisis de Ley de Potencia bajo metodología CSN."""
    logger.info("Iniciando análisis riguroso CSN")
    if grafo.number_of_nodes() == 0:
        logger.warning("El grafo recibido está vacío.")
        return {}

    # 1. Extracción de grados positivos
    data = np.array([d for _, d in grafo.degree() if d > 0])
    if len(data) == 0:
        logger.warning("El grafo no contiene nodos con grado positivo.")
        return {}

    # 2. Ajuste matemático con supresión de advertencias de convergencia
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=OptimizeWarning)
        fit = powerlaw.Fit(data, discrete=True, verbose=False)
        R_log, p_comp = fit.distribution_compare(
            "power_law", "lognormal", normalized_ratio=True
        )

    ks_distancia = float(fit.power_law.D)
    gamma_val = float(fit.power_law.alpha)
    sigma_val = float(fit.power_law.sigma)
    xmin_val = float(fit.power_law.xmin)
    total_nodos = len(data)
    nodos_cumplen = int(np.sum(data >= xmin_val))

    # Criterio de verosimilitud relativa
    if p_comp < 0.05:
        veredicto = "Power Law" if R_log > 0 else "Lognormal"
    else:
        veredicto = "Inconclusivo (Ambas)"

    # 3. Formateo de fechas para el título
    fechas = [
        pd.to_datetime(d["fecha_hecho"])
        for _, d in grafo.nodes(data=True)
        if "fecha_hecho" in d
    ]
    rango_fechas = (
        f"({min(fechas).strftime('%Y-%m-%d')} – {max(fechas).strftime('%Y-%m-%d')})"
        if fechas
        else "(Periodo Desconocido)"
    )

    titulo_grafico = (
        f"Distribución de Grado P(k) y Ajuste de Ley de Potencia (CSN)\n"
        f"Red de Visibilidad Semanal - {crime_name.title()}\n"
        f"{rango_fechas}"
    )

    # 4. Renderizado del gráfico
    plt.figure(figsize=(7, 5), dpi=100)
    fit.plot_pdf(
        color="black",
        marker="o",
        linestyle="None",
        markersize=5,
        label="Datos Empíricos",
    )
    fit.power_law.plot_pdf(
        color="red",
        linestyle="--",
        linewidth=2,
        label="Ajuste Óptimo Power Law",
    )

    # 5. Anotaciones de métricas
    texto_stats = (
        r"$\mathbf{Métricas\ del\ Ajuste:}$" + "\n"
        f"Exponente Gamma ($\\gamma$): {gamma_val:.4f} $\\pm$ {sigma_val:.4f}\n"
        f"Corte Inferior ($X_{{min}}$): {xmin_val:.2f}\n"
        f"Nodos ($k \\geq X_{{min}}$): {nodos_cumplen} / {total_nodos} "
        f"({nodos_cumplen / total_nodos * 100:.1f}%)\n\n"
        r"$\mathbf{Validación\ Absoluta:}$" + "\n"
        f"Distancia KS (D): {ks_distancia:.4f}\n\n"
        r"$\mathbf{Comparación\ Relativa:}$" + "\n"
        f"R (Log-Likelihood): {R_log:.4f}\n"
        f"p-value (Comp): {p_comp:.4f}\n"
        f"Ganador: {veredicto}"
    )

    plt.text(
        0.05,
        0.05,
        texto_stats,
        transform=plt.gca().transAxes,
        fontsize=8,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    plt.title(titulo_grafico, fontsize=11, pad=15)
    plt.xlabel("Grado (k)", fontsize=10)
    plt.ylabel("P(k)", fontsize=10)
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()

    logger.info("EXITO.")
    return {
        "gamma": gamma_val,
        "sigma": sigma_val,
        "xmin": xmin_val,
        "ks_distancia": ks_distancia,
        "R_log": R_log,
        "p_value": p_comp,
        "veredicto": veredicto,
    }

# %% [markdown]
# <details>
# <summary><b>Ejecución: Protocolo estadístico CSN</b></summary>
# 
# <br>
# 
# Ejecuta `plot_csn_analysis` sobre `visibility_graph` para estimar los parámetros de máxima verosimilitud de la red y almacena los resultados en `csn_metrics`.
# 
# #### Variables generadas
# * **`csn_metrics`** (`dict`): Diccionario con los parámetros formales $\gamma$, $x_{\min}$, $D_{KS}$ y el resultado del test relativo.
# 
# </details>

# %%
# [EJECUCIÓN] Evaluación estadística bajo protocolo CSN
# INPUT:  visibility_graph (nx.Graph), TARGET_CRIME (str)
# OUTPUT: csn_metrics (dict)

try:
    csn_metrics = plot_csn_analysis(visibility_graph, TARGET_CRIME)
    logger.info(f"Veredicto CSN: {csn_metrics.get('veredicto')}")
except Exception as err:
    logger.error(f"Fallo en la evaluación CSN: {err}")

# %% [markdown]
# <details>
# <summary><b>Análisis de Resiliencia: Ataques Dirigidos vs. Fallas Aleatorias</b></summary>
# 
# <br>
# 
# Evalúa la robustez y percolación de la red de visibilidad comparándola con familias de redes teóricas y sintéticas (Erdős-Rényi, Barabási-Albert, Watts-Strogatz, redes regulares y deterministas) ante dos estrategias de desmantelamiento secuencial:
# 1. **Ataque Dirigido:** Eliminación priorizada de hubs (nodos ordenados descendentemente por grado).
# 2. **Falla Aleatoria:** Remoción estocástica uniforme de nodos.
# 
# #### Métricas observadas
# * Evolución del número de componentes conexos en función de la fracción de nodos removidos.
# * Cuantificación de parámetros base: tamaño $N$, volumen $M$, grado medio $\langle k \rangle$ y densidad $p$.
# 
# #### Parámetros de entrada
# * **`grafo`** (`nx.Graph`): Red compleja evaluada.
# * **`crime_name`** (`str`): Delito bajo estudio.
# 
# #### Retorno
# * **`pd.DataFrame`**: Tabla consolidada con los parámetros estructurales base de la red empírica.
# 
# </details>

# %%
# [HELPERS] Simulación de resiliencia 
# INPUT:  grafo (nx.Graph), crime_name (str)
# OUTPUT: pd.DataFrame

def evaluate_topological_resilience(
    grafo: nx.Graph, crime_name: str
) -> pd.DataFrame:
    """Evalúa la resiliencia de la red frente a ataques dirigidos y fallas aleatorias."""
    logger.info("Iniciando evaluación de resiliencia topológica comparada")
    if getattr(grafo, "number_of_nodes", lambda: 0)() == 0:
        logger.warning("El grafo recibido está vacío.")
        return pd.DataFrame()

    # 1. Parámetros empíricos
    n_nodes = grafo.number_of_nodes()
    n_edges = grafo.number_of_edges()
    density_val = (
        (2 * n_edges) / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0
    )
    k_avg = (
        sum(dict(grafo.degree()).values()) / n_nodes if n_nodes > 0 else 0.0
    )

    df_params = pd.DataFrame(
        {
            "Métrica": [
                "Nodos (Semanas)",
                "Enlaces (m)",
                "Grado Promedio (k)",
                "Densidad (p)",
            ],
            "Valor": [
                n_nodes,
                n_edges,
                round(k_avg, 4),
                round(density_val, 4),
            ],
        }
    )

    # 2. Generación de familias de modelos teóricos
    def enlace_igualitario(n: int, k: int) -> nx.Graph:
        g = nx.Graph()
        g.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i + 1, min(i + k + 1, n)):
                g.add_edge(i, j)
        return g

    sqrt_nodes = int(np.sqrt(n_nodes))
    modelos = {
        "Red Empírica Semanal": grafo,
        f"Erdos Renyi (p={density_val:.4f})": nx.erdos_renyi_graph(
            n_nodes, density_val, seed=42
        ),
        "Barabasi Albert (m=2)": (
            nx.barabasi_albert_graph(n_nodes, 2, seed=42)
            if n_nodes > 2
            else None
        ),
        "Barabasi Albert (m=3)": (
            nx.barabasi_albert_graph(n_nodes, 3, seed=42)
            if n_nodes > 3
            else None
        ),
        "Barabasi Albert (m=4)": (
            nx.barabasi_albert_graph(n_nodes, 4, seed=42)
            if n_nodes > 4
            else None
        ),
        f"Watts Strogatz (p={density_val:.4f})": (
            nx.watts_strogatz_graph(n_nodes, 4, density_val, seed=42)
            if n_nodes >= 4
            else None
        ),
        "Random Regular (k=2)": (
            nx.random_regular_graph(2, n_nodes, seed=42)
            if n_nodes > 2
            else None
        ),
        "Enlace Igualitario 2": enlace_igualitario(n_nodes, 2),
        "Enlace Igualitario 1": enlace_igualitario(n_nodes, 1),
        "Star": nx.star_graph(n_nodes - 1) if n_nodes > 0 else None,
        "Grid 2D": (
            nx.grid_2d_graph(sqrt_nodes, sqrt_nodes)
            if sqrt_nodes > 0
            else None
        ),
        "Ladder": nx.ladder_graph(n_nodes // 2) if n_nodes >= 2 else None,
        "Hypercube": (
            nx.hypercube_graph(int(np.log2(n_nodes)))
            if (n_nodes > 0 and (n_nodes & (n_nodes - 1)) == 0)
            else None
        ),
        "Complete": nx.complete_graph(n_nodes),
        "Path": nx.path_graph(n_nodes),
        "Cycle": nx.cycle_graph(n_nodes),
        "Circular Ladder": (
            nx.circular_ladder_graph(n_nodes // 2) if n_nodes >= 2 else None
        ),
    }

    modelos_validos = {k: v for k, v in modelos.items() if v is not None}

    # 3. Formateo de fechas para el título
    fechas = [
        pd.to_datetime(d["fecha_hecho"])
        for _, d in grafo.nodes(data=True)
        if "fecha_hecho" in d
    ]
    rango_fechas = (
        f"({min(fechas).strftime('%Y-%m-%d')} – {max(fechas).strftime('%Y-%m-%d')})"
        if fechas
        else "(Periodo Desconocido)"
    )

    # 4. Estrategias de desmantelamiento
    rng = random.Random(42)
    estrategias = [
        (
            "Ataque Dirigido (Nodos de Mayor Grado)",
            lambda g: sorted(g.nodes(), key=lambda x: g.degree(x), reverse=True),
        ),
        (
            "Falla Aleatoria (Nodos al Azar)",
            lambda g: rng.sample(list(g.nodes()), g.number_of_nodes()),
        ),
    ]

    # 5. Simulación y renderizado comparativo
    for nombre_estrategia, func_ordenamiento in estrategias:
        filas = math.ceil(len(modelos_validos) / 5)
        fig, axes = plt.subplots(filas, 5, figsize=(18, 3.5 * filas))
        axes = axes.flatten()

        for i, (nombre_red, red_teorica) in enumerate(modelos_validos.items()):
            g_temp = red_teorica.copy()
            nodos_a_remover = func_ordenamiento(g_temp)

            componentes = []
            for nodo in nodos_a_remover:
                g_temp.remove_node(nodo)
                componentes.append(nx.number_connected_components(g_temp))

            axes[i].plot(componentes, color="royalblue")
            axes[i].set_title(nombre_red, fontsize=9)
            axes[i].set_xlabel("Nodos removidos", fontsize=8)
            axes[i].set_ylabel("Componentes conexos", fontsize=8)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(
            f"{nombre_estrategia}\n"
            f"Red de Visibilidad Semanal - {crime_name.title()}\n"
            f"{rango_fechas}",
            fontsize=14,
            y=1.02,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()

    logger.info("EXITO.")
    return df_params

# %% [markdown]
# <details>
# <summary><b>Ejecución: Simulación de resiliencia </b></summary>
# 
# <br>
# 
# Ejecuta `evaluate_topological_resilience` sobre `visibility_graph` para generar las curvas comparativas de fragmentación estructural y extrae la tabla de parámetros base `df_network_params`.
# 
# #### Variables generadas
# * **`df_network_params`** (`pd.DataFrame`): Métricas estructurales agregadas de la red semanal evaluada.
# 
# </details>

# %%
# [EJECUCIÓN] Simulación de robustez y resiliencia
# INPUT:  visibility_graph (nx.Graph), TARGET_CRIME (str)
# OUTPUT: df_network_params (pd.DataFrame)

try:
    df_network_params = evaluate_topological_resilience(
        visibility_graph, TARGET_CRIME
    )
    display(df_network_params)
except Exception as err:
    logger.error(f"Fallo en la evaluación de resiliencia topológica: {err}")

# %% [markdown]
# <details>
# <summary><b>Análisis de Resiliencia: Modelo Sintético Más Cercano (Distancia de Percolación)</b></summary>
# 
# <br>
# 
# Identifica y contrasta gráficamente la familia de red teórica cuya trayectoria de desmantelamiento minimiza la distancia euclidiana normalizada respecto a la red empírica, evaluada bajo Ataque Dirigido y Falla Aleatoria.
# 
# #### Métricas y formalismo
# * **Distancia Normalizada ($D$):** Se calcula como la norma $L_2$ entre las curvas normalizadas de componentes conexos:
#   $$D = \sqrt{\sum_{i} \left(\frac{y_{\text{emp}}(i)}{\max(y_{\text{emp}})} - \frac{y_{\text{teo}}(i)}{\max(y_{\text{teo}})}\right)^2}$$
# * **Extracción de Parámetros:** Parsea mediante expresiones regulares los hiperparámetros intrínsecos ($k$, $p$) del modelo óptimo identificado.
# 
# #### Parámetros de entrada
# * **`grafo`** (`nx.Graph`): Red empírica de visibilidad.
# * **`crime_name`** (`str`): Delito bajo estudio.
# 
# #### Retorno
# * **`pd.DataFrame`**: Tabla consolidada con el modelo teórico más cercano por estrategia, su distancia euclidiana y sus parámetros asociados.
# 
# </details>

# %%
# [HELPERS] Búsqueda y contraste del modelo sintético de resiliencia más cercano
# INPUT:  grafo (nx.Graph), crime_name (str)
# OUTPUT: pd.DataFrame

def plot_closest_resilience_models(
    grafo: nx.Graph, crime_name: str
) -> pd.DataFrame:
    """Encuentra y grafica el modelo teórico más cercano a la red empírica bajo ataques y fallas."""
    logger.info("Identificando modelos sintéticos más cercanos por distancia estructural")
    if getattr(grafo, "number_of_nodes", lambda: 0)() == 0:
        logger.warning("El grafo recibido está vacío.")
        return pd.DataFrame()

    # 1. Parámetros base y rango temporal
    n_nodes = grafo.number_of_nodes()
    n_edges = grafo.number_of_edges()
    p_density = (
        (2 * n_edges) / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0
    )
    sqrt_nodes = int(np.sqrt(n_nodes))

    fechas = [
        pd.to_datetime(d["fecha_hecho"])
        for _, d in grafo.nodes(data=True)
        if "fecha_hecho" in d
    ]
    rango_fechas = (
        f"({min(fechas).strftime('%Y-%m-%d')} – {max(fechas).strftime('%Y-%m-%d')})"
        if fechas
        else "(Periodo Desconocido)"
    )

    # 2. Generación de modelos teóricos
    def enlace_igualitario(n: int, k: int) -> nx.Graph:
        g = nx.Graph()
        g.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i + 1, min(i + k + 1, n)):
                g.add_edge(i, j)
        return g

    modelos = {
        f"Erdos Renyi (p={p_density:.4f})": nx.erdos_renyi_graph(
            n_nodes, p_density, seed=42
        ),
        "Barabasi Albert (m=2)": (
            nx.barabasi_albert_graph(n_nodes, 2, seed=42)
            if n_nodes > 2
            else None
        ),
        "Barabasi Albert (m=3)": (
            nx.barabasi_albert_graph(n_nodes, 3, seed=42)
            if n_nodes > 3
            else None
        ),
        "Barabasi Albert (m=4)": (
            nx.barabasi_albert_graph(n_nodes, 4, seed=42)
            if n_nodes > 4
            else None
        ),
        f"Watts Strogatz (p={p_density:.4f})": (
            nx.watts_strogatz_graph(n_nodes, 4, p_density, seed=42)
            if n_nodes >= 4
            else None
        ),
        "Random Regular (k=2)": (
            nx.random_regular_graph(2, n_nodes, seed=42)
            if n_nodes > 2
            else None
        ),
        "Enlace Igualitario (k=2)": enlace_igualitario(n_nodes, 2),
        "Enlace Igualitario (k=1)": enlace_igualitario(n_nodes, 1),
        "Star": nx.star_graph(n_nodes - 1) if n_nodes > 0 else None,
        "Grid 2D": (
            nx.grid_2d_graph(sqrt_nodes, sqrt_nodes)
            if sqrt_nodes > 0
            else None
        ),
        "Ladder": nx.ladder_graph(n_nodes // 2) if n_nodes >= 2 else None,
        "Hypercube": (
            nx.hypercube_graph(int(np.log2(n_nodes)))
            if (n_nodes > 0 and (n_nodes & (n_nodes - 1)) == 0)
            else None
        ),
        "Complete": nx.complete_graph(n_nodes),
        "Path": nx.path_graph(n_nodes),
        "Cycle": nx.cycle_graph(n_nodes),
        "Circular Ladder": (
            nx.circular_ladder_graph(n_nodes // 2) if n_nodes >= 2 else None
        ),
    }
    modelos_validos = {k: v for k, v in modelos.items() if v is not None}

    # 3. Funciones de simulación y distancia
    def simular_remocion(g_sim: nx.Graph, ordenamiento) -> np.ndarray:
        g_copy = g_sim.copy()
        nodos = ordenamiento(g_copy)
        valores = []
        for nodo in nodos:
            if nodo in g_copy:
                g_copy.remove_node(nodo)
            valores.append(nx.number_connected_components(g_copy))
        return np.array(valores)

    def distancia_normalizada(y1: np.ndarray, y2: np.ndarray) -> float:
        min_len = min(len(y1), len(y2))
        if min_len == 0:
            return float(np.nan)
        y1_trim, y2_trim = y1[:min_len], y2[:min_len]
        max_y1 = max(y1_trim) if len(y1_trim) > 0 else 0
        max_y2 = max(y2_trim) if len(y2_trim) > 0 else 0
        y1_norm = y1_trim / max_y1 if max_y1 > 0 else y1_trim
        y2_norm = y2_trim / max_y2 if max_y2 > 0 else y2_trim
        return float(np.sqrt(np.sum((y1_norm - y2_norm) ** 2)))

    rng = random.Random(42)
    estrategias = {
        "Ataque Dirigido (Grado Alto)": lambda g: sorted(
            g.nodes(), key=lambda x: g.degree(x), reverse=True
        ),
        "Falla Aleatoria": lambda g: rng.sample(
            list(g.nodes()), g.number_of_nodes()
        ),
    }

    # 4. Procesamiento y evaluación
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=100)
    resultados = []

    for idx, (nombre_estrategia, func_ordenamiento) in enumerate(
        estrategias.items()
    ):
        curva_base = simular_remocion(grafo, func_ordenamiento)
        distancias = []

        for nombre_modelo, g_teorico in modelos_validos.items():
            curva_teorica = simular_remocion(g_teorico, func_ordenamiento)
            d = distancia_normalizada(curva_base, curva_teorica)
            distancias.append((nombre_modelo, curva_teorica, d))

        mejor_nombre, mejor_curva, mejor_d = sorted(
            distancias, key=lambda x: x[2]
        )[0]

        # Extracción regex de parámetros
        k_match = re.search(r"[kK]=(\d+)", mejor_nombre)
        p_match = re.search(r"[pP]=([\d\.]+)", mejor_nombre)

        resultados.append(
            {
                "Estrategia": nombre_estrategia,
                "Modelo": mejor_nombre,
                "Distancia (D)": round(mejor_d, 4),
                "Parametro k": int(k_match.group(1)) if k_match else "N/A",
                "Parametro p": float(p_match.group(1)) if p_match else "N/A",
            }
        )

        ax = axes[idx]
        ax.plot(
            curva_base,
            label=f"Red Empírica: {crime_name.title()}",
            linewidth=3,
            color="blue",
            zorder=3,
        )
        ax.plot(
            mejor_curva,
            label=(
                f"Modelo más cercano: {mejor_nombre.split('(')[0].strip()}\n"
                f"(D={mejor_d:.4f})"
            ),
            alpha=0.9,
            linestyle="--",
            color="red",
            linewidth=2,
            zorder=2,
        )

        ax.set_title(f"{nombre_estrategia}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Nodos eliminados (Semanas)", fontsize=10)
        ax.set_ylabel("Componentes conectados", fontsize=10)
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, linewidth=0.3, linestyle="-", alpha=0.5)

    # 5. Configuración de lienzo
    plt.suptitle(
        f"Comparativa de Distancia Estructural (Red de Visibilidad Semanal)\n"
        f"en {crime_name.title()}\n"
        f"{rango_fechas}",
        fontsize=14,
        y=0.98,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.82)
    plt.show()

    logger.info("EXITO.")
    return pd.DataFrame(resultados)

# %% [markdown]
# <details>
# <summary><b>Ejecución: Identificación de modelos teóricos de resiliencia más cercanos</b></summary>
# 
# <br>
# 
# Invoca `plot_closest_resilience_models` para identificar los modelos con menor distancia estructural euclidiana respecto a `visibility_graph` y almacena los resultados en `df_closest_resilience_models`.
# 
# #### Variables generadas
# * **`df_closest_resilience_models`** (`pd.DataFrame`): Cuadro comparativo con los modelos óptimos por estrategia de falla y sus parámetros identificados.
# 
# </details>

# %%
# [EJECUCIÓN] Búsqueda de modelos teóricos más cercanos
# INPUT:  visibility_graph (nx.Graph), TARGET_CRIME (str)
# OUTPUT: df_closest_resilience_models (pd.DataFrame)

try:
    df_closest_resilience_models = plot_closest_resilience_models(
        visibility_graph, TARGET_CRIME
    )
    display(df_closest_resilience_models)
except Exception as err:
    logger.error(f"Fallo al comparar modelos de resiliencia más cercanos: {err}")


