# region Libraries
import pandas as pd
from IPython.display import display 
from numbers import Real
import logging

logger = logging.getLogger("Delitos")
# endregion




# region print_rolling_metrics
def print_rolling_metrics(rolling_metrics: dict) -> None:
    """Muestra un resumen de métricas del grafo y distribuciones por ventana."""

    filas = []

    for ventana in rolling_metrics["windows"]:
        grafo = ventana["graph_metrics"]
        distribuciones = ventana["distribution_metrics"]

        filas.append({
            "window_id": ventana["window_id"],
            "start_date": ventana["start_date"],
            "end_date": ventana["end_date"],
            "nodos": grafo["nodos"],
            "enlaces": grafo["enlaces"],
            "grado_promedio": grafo["grado_promedio"],
            "densidad": grafo["densidad_empirica"],
            "clustering": grafo["clustering"],
            "camino_medio": grafo["camino_medio"],
            "diametro": grafo["diametro"],
            "radio": grafo["radio"],
            "entropia_grado": distribuciones["entropia_grado"],
            "gamma_powerlaw": distribuciones["gamma_powerlaw"],
            "sigma": distribuciones["sigma"],
            "xmin": distribuciones["xmin"],
            "ks_distancia": distribuciones["ks_distancia"],
            "dist_eucl_pl": distribuciones["dist_eucl_pl"],
            "dist_eucl_lognormal": distribuciones["dist_eucl_lognormal"],
            "dist_eucl_exp": distribuciones["dist_eucl_exp"],
            "R_log": distribuciones["R_log"],
            "p_comp": distribuciones["p_comp"],
            "veredicto": distribuciones["veredicto"],
            "estado_distribuciones": distribuciones["estado"],
            "error_distribuciones": distribuciones["error"],
        })

    df_resumen = pd.DataFrame(filas)

    print(f"Delito: {rolling_metrics['crime_name']}")
    print(
        f"Ventana: {rolling_metrics['window_size']} | "
        f"Paso: {rolling_metrics['step']}"
    )
    print(f"Ventanas procesadas: {len(rolling_metrics['windows'])}")

    if df_resumen.empty:
        print("No hay métricas por ventana para mostrar.")
        return

    display(df_resumen)
# endregion






# region print_rolling_resilience_results
def print_rolling_resilience_results(
    rolling_resilience_results: dict,
) -> None:
    """Muestra los modelos ganadores por ventana y estrategia."""

    filas = []

    for ventana in rolling_resilience_results["windows"]:
        analisis = ventana["resilience_results"]

        if not analisis:
            continue

        for estrategia in ("ataque", "aleatorio"):
            ganador = analisis["mejores_modelos"][estrategia]

            filas.append({
                "window_id": ventana["window_id"],
                "start_date": ventana["start_date"],
                "end_date": ventana["end_date"],
                "estrategia": estrategia,
                "modelos_evaluados": len(analisis["modelos"]),
                "modelo_ganador": (
                    ganador["modelo"] if ganador else None
                ),
                "distancia": (
                    ganador["distancia"] if ganador else None
                ),
                "notas": analisis["notas"][estrategia],
            })

    df_resumen = pd.DataFrame(filas)

    print(f"Delito: {rolling_resilience_results['crime_name']}")
    print(f"Experimento: {rolling_resilience_results['experiment_id']}")
    print(
        f"Ventana: {rolling_resilience_results['window_size']} | "
        f"Paso: {rolling_resilience_results['step']}"
    )
    print(
        f"Ventanas procesadas: "
        f"{len(rolling_resilience_results['windows'])}"
    )

    if df_resumen.empty:
        print("No hay resultados de resiliencia para mostrar.")
        return

    display(df_resumen)
# endregion






# region extract_rolling_resilience_rankings
def extract_rolling_resilience_rankings(experiment: dict) -> dict:
    """Extract theoretical model resilience rankings across all rolling windows.

    Returns experiment metadata alongside a tidy DataFrame under ``'data'``
    with one row per ``(window_id, strategy, model)``.

    The central date corresponds to an actual weekly observation; when a
    window has an even number of observations, the earlier central observation
    is selected. Preserves tie ordering and unrounded curve distances without
    running simulations or mutating the input experiment.

    Parameters
    ----------
    experiment : dict
        Persisted experiment dictionary containing ``'rolling_resilience_results'``
        and experiment-level metadata keys.

    Returns
    -------
    dict
        Dictionary containing experiment metadata (``'experiment_id'``,
        ``'executed_at'``, ``'series_name'``, ``'window_size'``, ``'step'``,
        ``'removal_seed'``) and ``'data'`` as a ``pd.DataFrame`` with columns:
        ``['window_id', 'start_idx', 'end_idx', 'center_idx', 'start_date',
        'end_date', 'center_date', 'strategy', 'model', 'rank', 'distance']``.
    """
    res_results = experiment["rolling_resilience_results"]

    columns = [
        "window_id",
        "start_idx",
        "end_idx",
        "center_idx",
        "start_date",
        "end_date",
        "center_date",
        "strategy",
        "model",
        "rank",
        "distance",
    ]

    records = []

    for window in res_results["windows"]:
        analysis = window["resilience_results"]

        if not analysis:
            continue

        # 1. Identify the central observation in the original series.
        start_idx = window["start_idx"]
        end_idx = window["end_idx"]
        center_idx = (start_idx + end_idx) // 2

        # Window graph nodes are zero-indexed locally.
        center_node = center_idx - start_idx
        g = analysis["empirical_network"]["graph"]

        center_date = pd.Timestamp(
            g.nodes[center_node]["timestamp"]
        )

        # 2. Extract pre-sorted rankings for both removal strategies.
        for strategy in ("attack", "random"):
            ranking = analysis["sorted_distances"][strategy]

            for rank, item in enumerate(ranking, start=1):
                records.append({
                    "window_id": window["window_id"],
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "center_idx": center_idx,
                    "start_date": pd.Timestamp(window["start_date"]),
                    "end_date": pd.Timestamp(window["end_date"]),
                    "center_date": center_date,
                    "strategy": strategy,
                    "model": item["model"],
                    "rank": rank,
                    "distance": item["distance"],
                })

    # 3. Preserve experiment identity and configuration alongside the table.
    return {
        "experiment_id": experiment["experiment_id"],
        "executed_at": experiment["executed_at"],
        "series_name": experiment["series_name"],
        "window_size": experiment["window_size"],
        "step": experiment["step"],
        "removal_seed": experiment.get("removal_seed"),
        "data": pd.DataFrame(records, columns=columns),
    }
# endregion






# region merge_resilience_graph
def merge_resilience_graph(
    node_metrics: dict,
    rolling_resilience_rankings: dict,
) -> dict:
    """Link rolling resilience rankings to the full-series node metrics table.

    Matches each window's ``'center_date'`` in ``rolling_resilience_rankings``
    against the weekly ``'timestamp'`` in ``node_metrics``, attaching the
    corresponding ``'node_id'`` to every ranking row.

    Preserves all experiment metadata from ``rolling_resilience_rankings`` and
    returns two DataFrames: ``'nodes'`` (full series) and ``'resilience'``
    (linked rankings). Does not mutate inputs, compute metrics, or assign
    visual styling.

    Parameters
    ----------
    node_metrics : dict
        Dictionary returned by ``metrics.compute_node_metrics``, containing
        ``'graph_name'`` (or ``'series_name'``) and ``'data'`` with columns
        ``['node_id', 'timestamp', ...]``.
    rolling_resilience_rankings : dict
        Dictionary returned by ``extract_rolling_resilience_rankings``,
        containing ``'series_name'``, experiment metadata, and ``'data'``
        with a ``'center_date'`` column.

    Returns
    -------
    dict
        Dictionary preserving experiment metadata along with:

        nodes : pd.DataFrame
            Copy of the full-series node metrics table with parsed timestamps.
        resilience : pd.DataFrame
            Copy of the resilience rankings table enriched with ``'node_id'``.

    Raises
    ------
    ValueError
        If series names do not match, either DataFrame is empty, timestamps
        are missing or duplicated in the full series, or any window's
        ``'center_date'`` fails to match a node in the full series.
    """
    node_series = node_metrics.get(
        "graph_name",
        node_metrics.get("series_name", node_metrics.get("crime_name")),
    )
    ranking_series = rolling_resilience_rankings["series_name"]

    if node_series != ranking_series:
        raise ValueError(
            "El grafo completo y las clasificaciones corresponden "
            "a distintas series."
        )

    df_nodes = node_metrics["data"].copy()
    df_resilience = rolling_resilience_rankings["data"].copy()

    date_col = "timestamp"

    if df_nodes.empty:
        raise ValueError("El reporte de nodos está vacío.")

    if df_resilience.empty:
        raise ValueError("La tabla de clasificaciones está vacía.")

    # 1. Normalize timestamps for exact relational join.
    df_nodes[date_col] = pd.to_datetime(
        df_nodes[date_col],
        errors="raise",
    )

    df_resilience["center_date"] = pd.to_datetime(
        df_resilience["center_date"],
        errors="raise",
    )

    if (
        df_nodes[date_col].isna().any()
        or df_resilience["center_date"].isna().any()
    ):
        raise ValueError("Hay fechas faltantes para realizar la relación.")

    if df_nodes[date_col].duplicated().any():
        raise ValueError(
            "Hay fechas duplicadas en la serie completa; "
            "no se puede asignar un único nodo a cada fecha central."
        )

    # 2. Build timestamp-to-node_id lookup table.
    node_lookup = df_nodes[["node_id", date_col]].rename(
        columns={date_col: "center_date"}
    )

    # Multiple ranking rows can map to the same central node.
    df_merged = df_resilience.merge(
        node_lookup,
        on="center_date",
        how="left",
        sort=False,
        validate="many_to_one",
        indicator=True,
    )

    unmatched = df_merged["_merge"].eq("left_only")

    if unmatched.any():
        missing_windows = (
            df_merged.loc[unmatched, "window_id"]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            "Las fechas centrales de estas ventanas no aparecen "
            f"en el grafo completo: {missing_windows}"
        )

    df_merged = df_merged.drop(columns="_merge")

    # 3. Preserve experiment metadata without duplicating the input table.
    results = {
        key: value
        for key, value in rolling_resilience_rankings.items()
        if key != "data"
    }

    results["nodes"] = df_nodes
    results["resilience"] = df_merged

    return results
# endregion





# region print_distribution_distances
def print_distribution_distances(distribution_metrics: dict) -> None:
    """Display precomputed Euclidean distances sorted from smallest to largest.

    Builds a summary DataFrame ordering the fitted candidate models by their
    Euclidean distance to the empirical tail distribution, keeping missing
    values at the bottom and printing the series name and analysis status. Does
    not perform new distribution fits or distance calculations. All rendered
    column names and printed labels are formatted in Spanish.

    Parameters
    ----------
    distribution_metrics : dict
        Dictionary returned by ``distributions.compute_distribution_metrics``,
        containing ``'series_name'``, ``'status'``, ``'dist_eucl_pl'``,
        ``'dist_eucl_lognormal'``, and ``'dist_eucl_exp'``.

    Returns
    -------
    None
        Prints series metadata and displays the sorted DataFrame via
        ``display()``.
    """
    df_distances = pd.DataFrame({
        "Modelo": ["Ley de Potencia", "Log-Normal", "Exponencial"],
        "Distancia Euclidiana": [
            distribution_metrics["dist_eucl_pl"],
            distribution_metrics["dist_eucl_lognormal"],
            distribution_metrics["dist_eucl_exp"],
        ],
    }).sort_values(
        "Distancia Euclidiana", na_position="last", kind="stable"
    ).reset_index(drop=True)

    print(f"Delito: {distribution_metrics['series_name']}")
    print(f"Estado: {distribution_metrics['status']}")
    display(df_distances)
# endregion





# region extract_metrics_by_window
# CHECK IF ITS USEFUL
def extract_metrics_by_window(rolling_experiment: dict) -> dict:
    """Join graph, distribution, and resilience distance metrics per window.

    Builds a consolidated dictionary preserving experiment-level metadata and a
    tabular ``pd.DataFrame`` under ``'data'`` with one row per window,
    node-removal strategy, and theoretical reference model. Retains numeric
    and missing metric values while excluding graphs, arrays, lists, and fit
    objects. Degree-distribution metrics are prefixed with
    ``'distribution_'``. Does not recompute metrics, distances, or rankings.

    Parameters
    ----------
    rolling_experiment : dict
        Consolidated rolling-window experiment dictionary returned by
        ``run_rolling_experiment``, containing ``'rolling_metrics'``,
        ``'rolling_resilience_results'``, and experiment-level metadata keys.

    Returns
    -------
    dict
        Dictionary containing experiment-level metadata (excluding
        ``'rolling_metrics'`` and ``'rolling_resilience_results'``) plus:

        n_windows : int
            Total number of windows present in
            ``rolling_resilience_results['windows']``.
        data : pd.DataFrame
            Long-format DataFrame with columns ``'window_id'``,
            ``'start_date'``, ``'end_date'``, numeric ``graph_metrics``
            fields, ``'distribution_*'`` fields, ``'strategy'``, ``'model'``,
            and ``'resilience_distance'``.

    Raises
    ------
    ValueError
        If duplicate ``window_id`` values exist in either ``'rolling_metrics'``
        or ``'rolling_resilience_results'``, or if the two branches do not
        contain the exact same set of ``window_id`` values.
    """
    rolling_metrics = rolling_experiment["rolling_metrics"]
    rolling_resilience = rolling_experiment["rolling_resilience_results"]

    # 1. Indexar las métricas mediante el identificador de ventana.
    metrics_by_id = {}

    for window in rolling_metrics["windows"]:
        window_id = window["window_id"]

        if window_id in metrics_by_id:
            raise ValueError(
                f"window_id duplicado en las métricas: {window_id}"
            )

        metrics_by_id[window_id] = window

    resilience_windows = rolling_resilience["windows"]
    resilience_ids = [
        window["window_id"] for window in resilience_windows
    ]

    if len(set(resilience_ids)) != len(resilience_ids):
        raise ValueError("Hay window_id duplicados en resiliencia.")

    if set(metrics_by_id) != set(resilience_ids):
        raise ValueError(
            "Las ramas de métricas y resiliencia no contienen "
            "los mismos identificadores de ventana."
        )

    # 2. Seleccionar valores numéricos y ausentes.
    def extract_numeric(metrics: dict) -> dict:
        return {
            name: value
            for name, value in metrics.items()
            if value is None
            or (isinstance(value, Real) and not isinstance(value, bool))
        }

    records = []

    # 3. Relacionar cada ventana con todas sus comparaciones.
    for window in sorted(
        resilience_windows, key=lambda item: item["window_id"]
    ):
        window_id = window["window_id"]
        analysis = window["resilience_results"]

        if not analysis:
            logger.warning(
                "Se omite la ventana %s: no tiene análisis de resiliencia.",
                window_id,
            )
            continue

        window_metrics = metrics_by_id[window_id]

        graph_metrics = extract_numeric(
            window_metrics["graph_metrics"]
        )

        distribution_metrics = {
            f"distribution_{name}": value
            for name, value in extract_numeric(
                window_metrics["distribution_metrics"]
            ).items()
        }

        window_data = {
            "window_id": window_id,
            "start_date": window["start_date"],
            "end_date": window["end_date"],
            **graph_metrics,
            **distribution_metrics,
        }

        for strategy in ("attack", "random"):
            ranking = analysis["sorted_distances"][strategy]

            for result in ranking:
                records.append({
                    **window_data,
                    "strategy": strategy,
                    "model": result["model"],
                    "resilience_distance": result["distance"],
                })

    # 4. Conservar los metadatos del experimento.
    results = {
        key: value
        for key, value in rolling_experiment.items()
        if key not in ("rolling_metrics", "rolling_resilience_results")
    }

    results["n_windows"] = len(resilience_windows)
    results["data"] = pd.DataFrame(records)

    logger.info(
        "EXITO ✅. Datos relacionados: %s ventanas y %s comparaciones.",
        results["n_windows"],
        len(records),
    )

    return results
# endregion







# region get_graph_by_window
def get_graph_by_window(rolling_experiment: dict, window_id: int) -> dict:
    """Retrieve the empirical visibility graph and metadata for a specific window.

    Searches ``rolling_experiment['rolling_resilience_results']['windows']`` by
    ``window_id`` regardless of list position, returning a direct reference to
    the stored empirical ``nx.Graph`` without copying or mutating it, alongside
    experiment-level and window-boundary metadata.

    Parameters
    ----------
    rolling_experiment : dict
        Consolidated rolling-window experiment dictionary returned by
        ``run_rolling_experiment``, containing ``'rolling_resilience_results'``
        and experiment-level metadata keys.
    window_id : int
        Zero-based sequential identifier of the target window.

    Returns
    -------
    dict
        Dictionary containing experiment-level metadata (excluding
        ``'rolling_metrics'`` and ``'rolling_resilience_results'``) merged with
        window boundary fields (``'window_id'``, ``'start_idx'``,
        ``'end_idx'``, ``'start_date'``, ``'end_date'``) and ``'graph'``
        (nx.Graph).

    Raises
    ------
    ValueError
        If no window matches ``window_id``, if duplicate ``window_id`` entries
        exist, or if the matched window lacks ``'resilience_results'``.
    """
    windows = rolling_experiment["rolling_resilience_results"]["windows"]
    matches = [
        window for window in windows
        if window["window_id"] == window_id
    ]

    if not matches:
        raise ValueError(f"No existe una ventana con window_id={window_id}.")

    if len(matches) > 1:
        raise ValueError(f"Hay ventanas duplicadas con window_id={window_id}.")

    window = matches[0]
    analysis = window["resilience_results"]

    if not analysis:
        raise ValueError(
            f"La ventana {window_id} no contiene resultados de resiliencia."
        )

    # Conservar los metadatos sin duplicar las ramas del experimento.
    result = {
        key: value
        for key, value in rolling_experiment.items()
        if key not in ("rolling_metrics", "rolling_resilience_results")
    }

    result.update({
        "window_id": window["window_id"],
        "start_idx": window["start_idx"],
        "end_idx": window["end_idx"],
        "start_date": window["start_date"],
        "end_date": window["end_date"],
        "graph": analysis["empirical_network"]["graph"],
    })

    return result
# endregion






# region get_graph_metrics_by_window
def get_graph_metrics_by_window(
    rolling_experiment: dict,
    window_id: int,
) -> dict:
    """Retrieve stored structural graph metrics and metadata for a specific window.

    Searches ``rolling_experiment['rolling_metrics']['windows']`` by
    ``window_id`` regardless of list position, returning a direct reference to
    the stored ``'graph_metrics'`` dictionary without recomputing or mutating
    results, alongside experiment-level and window-boundary metadata.

    Parameters
    ----------
    rolling_experiment : dict
        Consolidated rolling-window experiment dictionary returned by
        ``run_rolling_experiment``, containing ``'rolling_metrics'`` and
        experiment-level metadata keys.
    window_id : int
        Zero-based sequential identifier of the target window.

    Returns
    -------
    dict
        Dictionary containing experiment-level metadata (excluding
        ``'rolling_metrics'`` and ``'rolling_resilience_results'``) merged with
        window boundary fields (``'window_id'``, ``'start_idx'``,
        ``'end_idx'``, ``'start_date'``, ``'end_date'``) and
        ``'graph_metrics'`` (dict).

    Raises
    ------
    ValueError
        If no window matches ``window_id`` or if duplicate ``window_id`` entries
        exist.
    """
    windows = rolling_experiment["rolling_metrics"]["windows"]
    matches = [
        window for window in windows
        if window["window_id"] == window_id
    ]

    if not matches:
        raise ValueError(f"No existe una ventana con window_id={window_id}.")

    if len(matches) > 1:
        raise ValueError(f"Hay ventanas duplicadas con window_id={window_id}.")

    window = matches[0]

    # Conservar los metadatos sin duplicar las ramas del experimento.
    result = {
        key: value
        for key, value in rolling_experiment.items()
        if key not in ("rolling_metrics", "rolling_resilience_results")
    }

    result.update({
        "window_id": window["window_id"],
        "start_idx": window["start_idx"],
        "end_idx": window["end_idx"],
        "start_date": window["start_date"],
        "end_date": window["end_date"],
        "graph_metrics": window["graph_metrics"],
    })

    return result
# endregion






# region get_distribution_metrics_by_window
def get_distribution_metrics_by_window(
    rolling_experiment: dict, window_id: int
) -> dict:
    """Recupera las métricas de distribución y metadatos de una ventana.

    Busca por window_id sin depender de la posición en la lista.
    Conserva los arreglos y objetos de ajuste previamente calculados.
    No recalcula métricas ni modifica los resultados originales.
    """

    ventanas = rolling_experiment["rolling_metrics"]["windows"]
    coincidencias = [
        ventana for ventana in ventanas
        if ventana["window_id"] == window_id
    ]

    if not coincidencias:
        raise ValueError(f"No existe una ventana con window_id={window_id}.")

    if len(coincidencias) > 1:
        raise ValueError(f"Hay ventanas duplicadas con window_id={window_id}.")

    ventana = coincidencias[0]

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
        "distribution_metrics": ventana["distribution_metrics"],
    })

    return resultado
# endregion





# region get_resilience_results_by_window
def get_resilience_results_by_window(
    rolling_experiment: dict,
    windows_id: int,
) -> dict:
    """Retrieve stored resilience analysis results and metadata for a specific window.

    Searches ``rolling_experiment['rolling_resilience_results']['windows']`` by
    the integer stored under ``'window_id'``, returning a direct reference to
    ``'resilience_results'`` (preserving graphs, curves, distances, rankings,
    and tie notes) alongside experiment-level and window-boundary metadata.
    Does not recompute or mutate original results.

    Parameters
    ----------
    rolling_experiment : dict
        Consolidated rolling-window experiment dictionary returned by
        ``run_rolling_experiment``, containing ``'rolling_resilience_results'``
        and experiment-level metadata keys.
    windows_id : int
        Zero-based sequential identifier matched against each window's
        ``'window_id'`` field.

    Returns
    -------
    dict
        Dictionary containing experiment-level metadata (excluding
        ``'rolling_metrics'`` and ``'rolling_resilience_results'``) merged with
        window boundary fields (``'window_id'``, ``'start_idx'``,
        ``'end_idx'``, ``'start_date'``, ``'end_date'``) and
        ``'resilience_results'`` (dict).

    Raises
    ------
    ValueError
        If no window matches ``windows_id`` or if duplicate ``window_id``
        entries exist.
    """
    windows = rolling_experiment["rolling_resilience_results"]["windows"]
    matches = [
        window for window in windows
        if window["window_id"] == windows_id
    ]

    if not matches:
        raise ValueError(f"No existe una ventana con window_id={windows_id}.")

    if len(matches) > 1:
        raise ValueError(f"Hay ventanas duplicadas con window_id={windows_id}.")

    window = matches[0]

    # Conservar los metadatos sin duplicar las ramas del experimento.
    result = {
        key: value
        for key, value in rolling_experiment.items()
        if key not in ("rolling_metrics", "rolling_resilience_results")
    }

    result.update({
        "window_id": window["window_id"],
        "start_idx": window["start_idx"],
        "end_idx": window["end_idx"],
        "start_date": window["start_date"],
        "end_date": window["end_date"],
        "resilience_results": window["resilience_results"],
    })

    return result
# endregion


