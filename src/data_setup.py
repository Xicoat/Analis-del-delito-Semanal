# region  Setup
import logging
from pathlib import Path
from typing import Union
import pandas as pd
import pickle
from .config import START_DATE

import networkx as nx

logger = logging.getLogger("Delitos")
# endregion





# region extract_csv_data
def extract_csv_data(path: Union[str, Path], target_crime: str,) -> dict:
    """Extrae registros de un delito y devuelve su nombre junto al DataFrame."""
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
            logger.warning(
                f"No se pudo procesar {file_path.name}: {e}"
            )
    if not records:
        logger.warning(
            f"No se encontraron registros para: {target_crime}"
        )
        return {"crime_name": target_crime, "data": pd.DataFrame(),}
    logger.info("Generando dataframe")
    df_raw = pd.concat(records, ignore_index=True)

    logger.info(f"Se extrajeron {len(df_raw):,} registros.")
    logger.info("EXITO ✅")
    return {"crime_name": target_crime,"data": df_raw,}
# endregion







# region ensure_datetime_type
def ensure_datetime_type(df: pd.DataFrame, column: str,) -> pd.DataFrame:
    """Convierte la columna indicada a datetime, conservando las demás."""
    if column not in df.columns:
        raise ValueError(f"No existe la columna '{column}'.")
    if not pd.api.types.is_datetime64_any_dtype(df[column]):
        logger.info(f"Convirtiendo tiempo en columna '{column}'")
        df[column] = pd.to_datetime(df[column], errors="coerce")
    return df
# endregion







# region extract_dates
def extract_dates(crime_data: dict, date_column: str, crime_column: str,) -> dict:
    """Extrae las fechas del delito y conserva la identidad del conjunto."""
    logger.info("Seleccionando delito y extrayendo fechas")
    df = crime_data["data"]
    crime_name = crime_data["crime_name"]
    if df.empty:
        raise ValueError("El DataFrame de entrada está vacío.")
    columnas = [date_column, crime_column]
    faltantes = set(columnas) - set(df.columns)

    if faltantes:
        raise ValueError(
            f"Faltan columnas requeridas: {sorted(faltantes)}"
        )

    coincidencias = df[crime_column].eq(crime_name).fillna(False)
    descartados = int((~coincidencias).sum())

    if descartados:
        logger.warning(
            "Se descartan %s registros cuyo delito no coincide con '%s'.",
            descartados,
            crime_name,
        )

    df_dates = df.loc[coincidencias, columnas].copy()

    if df_dates.empty:
        raise ValueError(
            f"No hay registros para el delito '{crime_name}'."
        )

    df_dates = ensure_datetime_type(df_dates, date_column)

    logger.info(
        "EXITO. %s registros conservados de '%s'.",
        len(df_dates),
        crime_name,
    )

    return {"crime_name": crime_name, "data": df_dates,}
# endregion






# region filter_by_date_range
def filter_by_date_range(crime_data: dict, date_column: str, start: pd.Timestamp, end: pd.Timestamp,) -> dict:
    """Filtra los registros por fecha y conserva la identidad del conjunto."""
    logger.info("Filtrando registros")
    df = crime_data["data"]
    mask = (df[date_column] >= start) & (df[date_column] <= end)
    df_filtered = df[mask].copy()
    logger.info("EXITO ✅")
    return {"crime_name": crime_data["crime_name"], "data": df_filtered,}
# endregion






# region compute_events_by_week
def compute_events_by_week(
    crime_data: dict,
    date_column: str = "fecha_hecho",
    origin_date: pd.Timestamp = START_DATE,
) -> dict:
    """Aggregate event records into 168-hour weekly bins.

    Resamples the raw event timestamps into fixed 7-day intervals anchored at
    ``origin_date`` and normalizes the resulting timestamps to midnight.

    Parameters
    ----------
    crime_data : dict
        Dictionary containing the following items:

        crime_name : str
            Name of the analyzed crime.
        data : pd.DataFrame
            Non-empty DataFrame containing at least ``date_column`` with
            ``datetime64`` values.
    date_column : str, default="fecha_hecho"
        Name of the timestamp column in ``crime_data['data']`` used for
        resampling.
    origin_date : pd.Timestamp, default=START_DATE
        Reference timestamp anchoring the start of the 168-hour bins.

    Returns
    -------
    dict
        Dictionary containing the following items:

        series_name : str
            Name of the analyzed time series (copied from ``'crime_name'``).
        data : pd.DataFrame
            DataFrame with columns ``['timestamp', 'frequency']``, where
            ``'timestamp'`` is the normalized start date of each week and
            ``'frequency'`` is the weekly event count.

    Raises
    ------
    ValueError
        If ``crime_data['data']`` is empty.
    """
    logger.info("Agrupando serie temporal en bloques semanales")
    df = crime_data["data"]
    if df.empty:
        raise ValueError("El DataFrame de entrada está vacío.")

    weekly_series = (
        df.resample("168h", on=date_column, origin=origin_date)
        .size()
        .reset_index(name="frequency")
        .rename(columns={date_column: "timestamp"})
    )
    weekly_series["timestamp"] = weekly_series["timestamp"].dt.normalize()

    logger.info("EXITO ✅")
    return {
        "series_name": crime_data["crime_name"],
        "data": weekly_series,
    }
# endregion






# region assign_network_nodes
def assign_network_nodes(series_data: dict) -> dict:
    """Assign sequential node identifiers to each weekly record in the time series.

    Parameters
    ----------
    series_data : dict
        Dictionary containing the following items:

        series_name : str
            Name of the analyzed time series.
        data : pd.DataFrame
            DataFrame containing at least the columns ``['timestamp', 'frequency']``.

    Returns
    -------
    dict
        Dictionary containing the following items:

        series_name : str
            Name of the analyzed time series.
        data : pd.DataFrame
            DataFrame with columns ``['node_id', 'timestamp', 'frequency']``, where
            ``'node_id'`` is a zero-based sequential integer identifier.
    """
    df_nodes = series_data["data"].copy()
    df_nodes["node_id"] = range(len(df_nodes))
    logger.info("SUCCESS ✅")
    return {
        "series_name": series_data["series_name"],
        "data": df_nodes[["node_id", "timestamp", "frequency"]],
    }
# endregion







# region prepare_series_nodes
def prepare_series_nodes(crime_name: str, data_path: str | Path,
    start_date: pd.Timestamp, end_date: pd.Timestamp,) -> dict:

    """Extract, filter, and prepare the weekly time series for a crime, assigning node identifiers.

    Parameters
    ----------
    crime_name : str
        Name of the crime to extract from the dataset.
    data_path : str | Path
        Path to the CSV file containing the incident records.
    start_date : pd.Timestamp
        Start date for filtering the data.
    end_date : pd.Timestamp
        End date for filtering the data.

    Returns
    -------
    dict
        Dictionary containing the following items:

        crime_name : str
            Name of the analyzed crime.
        data : pd.DataFrame
            Columns ``['nodo', 'fecha_hecho', 'cantidad']``.
        start_date : pd.Timestamp
            Start date used for filtering.
        end_date : pd.Timestamp
            End date used for filtering.

    Raises
    ------
    ValueError
        If the filtered DataFrame is empty for the specified crime and date range.
    """

    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    crime_raw = extract_csv_data(path=data_path, target_crime=crime_name)
    crime_dates = extract_dates(crime_data=crime_raw, date_column="fecha_hecho", crime_column="delito")
    crime_filtered = filter_by_date_range(crime_data=crime_dates, date_column="fecha_hecho", start=start_date, end=end_date)
    if crime_filtered["data"].empty:
        raise ValueError(f"No hay registros de '{crime_name}' en el periodo indicado.")
    events_by_week = compute_events_by_week(crime_data=crime_filtered, date_column="fecha_hecho", origin_date=start_date)

    # assign_network_nodes returns a dictionary with two keys: 'series_name' and 'data'
    series_nodes = assign_network_nodes(series_data=events_by_week)
    series_nodes["start_date"] = start_date
    series_nodes["end_date"] = end_date
    return series_nodes
# endregion






# region load_series_nodes
def load_series_nodes(series_nodes_path: str | Path) -> dict:
    """Load the prepared weekly time series and metadata from a Pickle file.

    Only load trusted Pickle files generated by this pipeline.

    Parameters
    ----------
    series_nodes_path : str | Path
        Path to the serialized ``.pkl`` file containing the prepared series
        dictionary (with keys ``'series_name'`` and ``'data'``).

    Returns
    -------
    dict
        Dictionary containing ``'series_name'`` and the weekly ``'data'``
        DataFrame with columns ``['node_id', 'timestamp', 'frequency']``.

    Raises
    ------
    FileNotFoundError
        If ``series_nodes_path`` does not exist or is not a file.
    """
    path = Path(series_nodes_path)

    if not path.is_file():
        raise FileNotFoundError(f"No existe el archivo: {path}")

    with path.open("rb") as f_in:
        series_nodes = pickle.load(f_in)

    logger.info("EXITO ✅. Serie cargada desde: %s", path.resolve())
    return series_nodes
# endregion






# region load_rolling_experiments
def load_rolling_experiments(series_folder_path: str | Path) -> list[dict]:
    """Load all persisted rolling-window experiments from a series directory.

    Reads all ``.pkl`` experiment files in ``series_folder_path``, verifies that
    every file belongs to the same ``'series_name'`` and that no duplicate
    ``(window_size, step)`` combinations exist, and returns the experiments
    sorted in ascending order by ``(window_size, step)``.

    Only load trusted Pickle files generated by this pipeline.

    Parameters
    ----------
    series_folder_path : str | Path
        Path to the directory containing the ``.pkl`` experiment files for a
        single series (e.g., ``'reports/FEMINICIDIO'``).

    Returns
    -------
    list[dict]
        List of experiment dictionaries sorted by ``(window_size, step)``.

    Raises
    ------
    FileNotFoundError
        If ``series_folder_path`` does not exist or contains no ``.pkl`` files.
    ValueError
        If duplicate ``(window_size, step)`` pairs are detected or if files
        belong to more than one ``'series_name'``.
    """
    folder = Path(series_folder_path)

    if not folder.is_dir():
        raise FileNotFoundError(f"No existe la carpeta: {folder}")

    files = sorted(folder.glob("*.pkl"))

    if not files:
        raise FileNotFoundError(f"No hay archivos .pkl en: {folder}")

    experiments = []
    combinations = {}
    series_names = set()

    for file_path in files:
        logger.info("Leyendo experimento: %s", file_path.name)

        with file_path.open("rb") as f_in:
            experiment = pickle.load(f_in)

        combo = (
            experiment["window_size"],
            experiment["step"],
        )

        if combo in combinations:
            raise ValueError(
                f"Hay dos experimentos para W={combo[0]}, "
                f"S={combo[1]}: "
                f"'{combinations[combo]}' y '{file_path.name}'."
            )

        combinations[combo] = file_path.name
        series_names.add(experiment["series_name"])
        experiments.append(experiment)

    if len(series_names) != 1:
        raise ValueError("La carpeta contiene experimentos de distintas series.")

    experiments.sort(
        key=lambda exp: (
            exp["window_size"],
            exp["step"],
        )
    )

    logger.info("EXITO ✅. %s experimentos cargados.", len(experiments))
    return experiments
# endregion






# region load_rolling_experiment
def load_rolling_experiment(experiment_selector: str | Path) -> dict:
    """Load a serialized rolling-window experiment from the selected file path.

    Opens the target binary pickle file and deserializes the stored experiment
    dictionary.

    Parameters
    ----------
    experiment_selector : str | Path
        Path or widget-selected file path pointing to the serialized pickle
        experiment file.

    Returns
    -------
    dict
        Deserialized experiment dictionary loaded from disk.
    """
    experiment_path = Path(experiment_selector)

    with experiment_path.open("rb") as file:
        experiment = pickle.load(file)

    logger.info("EXITO ✅. Experimento cargado: %s", experiment_path.name)
    return experiment
# endregion






# region load_global_visibility_graph
def load_global_visibility_graph(global_graph_path: str | Path) -> nx.Graph:
    """Carga el grafo completo desde un archivo Pickle de confianza.

    Devuelve el grafo con sus nodos, atributos, aristas y metadatos.
    """

    ruta = Path(global_graph_path)

    with ruta.open("rb") as archivo:
        grafo = pickle.load(archivo)

    if not isinstance(grafo, nx.Graph):
        raise TypeError(f"El archivo no contiene un grafo NetworkX: {ruta}")

    logger.info("EXITO. Grafo completo cargado: %s", ruta.name)
    return grafo
# endregion






# region graph_to_nodes
def graph_to_nodes(grafo: nx.Graph) -> dict:
    """Recupera la serie semanal y sus identificadores desde el grafo.

    Devuelve el formato crime_nodes: nombre del delito y DataFrame
    con las columnas nodo, fecha_hecho y cantidad.

    Conserva los identificadores originales, ordenados por nodo.
    """

    registros = [
        {
            "nodo": nodo,
            "fecha_hecho": datos["fecha_hecho"],
            "cantidad": datos["cantidad"],
        }
        for nodo, datos in sorted(grafo.nodes(data=True))
    ]

    return {
        "crime_name": grafo.graph["crime_name"],
        "data": pd.DataFrame(
            registros,
            columns=["nodo", "fecha_hecho", "cantidad"],
        ),
    }
# endregion






# region get_graph_by_window
def get_graph_by_window(rolling_experiment: dict, window_id: int) -> dict:
    """Recupera el grafo empírico y los metadatos de una ventana.

    Busca por window_id, independientemente de la posición en la lista.
    Devuelve una referencia al grafo guardado, sin copiarlo ni modificarlo.
    """

    ventanas = rolling_experiment["rolling_resilience_results"]["windows"]
    coincidencias = [
        ventana for ventana in ventanas
        if ventana["window_id"] == window_id
    ]

    if not coincidencias:
        raise ValueError(f"No existe una ventana con window_id={window_id}.")

    if len(coincidencias) > 1:
        raise ValueError(f"Hay ventanas duplicadas con window_id={window_id}.")

    ventana = coincidencias[0]
    analisis = ventana["resilience_results"]

    if not analisis:
        raise ValueError(
            f"La ventana {window_id} no contiene resultados de resiliencia."
        )

    # Conservar los metadatos sin duplicar las ramas del experimento.
    resultado = {
        clave: valor
        for clave, valor in rolling_experiment.items()
        if clave not in ("rolling_metrics", "rolling_resilience_results")
    }

    resultado.update({
        "window_id": ventana["window_id"],
        "start_idx": ventana["start_idx"],
        "end_idx": ventana["end_idx"],
        "start_date": ventana["start_date"],
        "end_date": ventana["end_date"],
        "grafo": analisis["red_empirica"]["grafo"],
    })

    return resultado
# endregion












