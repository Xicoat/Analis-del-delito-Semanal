# region Libraries
import logging

import networkx as nx
import numpy as np
import pandas as pd

from . import graph
from .distributions import compute_distribution_metrics


logger = logging.getLogger("Delitos")
# endregion






# region compute_node_metrics
def compute_node_metrics(g: nx.Graph) -> dict:
    """Compute structural centrality and local metrics for each node in a graph.

    Parameters
    ----------
    g : nx.Graph
        Undirected NetworkX graph. Nodes are expected to have attributes
        ``'timestamp'`` (pd.Timestamp) and ``'frequency'`` (float).

    Returns
    -------
    dict
        Dictionary containing the following items:

        graph_name : str
            Name of the graph retrieved via ``graph.get_graph_name(g)``.
        data : pd.DataFrame
            DataFrame ordered sequentially by ``'node_id'`` with columns:
            ``['node_id', 'timestamp', 'frequency', 'degree', 'betweenness',
            'closeness', 'clustering', 'eccentricity']``. Returns an empty
            DataFrame if the graph has no nodes.
    """
    logger.info("Calculando métricas y centralidades por nodo")
    graph_name = graph.get_graph_name(g)

    if graph.get_node_count(g) == 0:
        logger.warning("El grafo recibido no contiene nodos.")
        return {
            "graph_name": graph_name,
            "data": pd.DataFrame(),
        }

    # 1. Compute connectivity and centrality measures.
    degrees = dict(g.degree())
    betweenness = nx.betweenness_centrality(g)
    closeness = nx.closeness_centrality(g)
    clustering = nx.clustering(g)

    if nx.is_connected(g):
        eccentricity = nx.eccentricity(g)
    else:
        eccentricity = {n: np.nan for n in g.nodes()}

    # 2. Build tabular structure with node attributes.
    records = []
    for node_id, attrs in sorted(g.nodes(data=True)):
        timestamp_dt = pd.to_datetime(attrs.get("timestamp", pd.NaT))
        records.append({
            "node_id": node_id,
            "timestamp": timestamp_dt,
            "frequency": attrs.get("frequency", 0.0),
            "degree": degrees.get(node_id, 0),
            "betweenness": betweenness.get(node_id, 0.0),
            "closeness": closeness.get(node_id, 0.0),
            "clustering": clustering.get(node_id, 0.0),
            "eccentricity": eccentricity.get(node_id, np.nan),
        })

    logger.info("EXITO ✅. Métricas generadas para %s nodos.", len(records))
    return {
        "graph_name": graph_name,
        "data": pd.DataFrame(records),
    }
# endregion





# region compute_graph_metrics
def compute_graph_metrics(g: nx.Graph) -> dict:
    """Compute a global connectivity and path-distance summary for a visibility graph.

    Extracts graph identity, node/edge counts, average degree, empirical
    density, average clustering coefficient, and the node degree sequence. When
    the graph is non-empty and connected, also computes radius, diameter,
    center nodes, average shortest path length, and representative diametral
    and radial paths.

    If the graph is empty or disconnected, distance-based metrics remain at
    their default ``None`` / ``np.nan`` values and path lists remain empty.

    Parameters
    ----------
    g : nx.Graph
        Undirected Natural Visibility Graph to analyze.

    Returns
    -------
    dict
        Dictionary containing global structural and path metrics:

        graph_name : str
            Name of the graph retrieved via ``graph.get_graph_name(g)``.
        node_count : int
            Total number of nodes in the graph.
        edge_count : int
            Total number of edges in the graph.
        average_degree : float
            Mean node degree across the graph.
        density : float
            Graph edge density.
        average_clustering : float
            Average local clustering coefficient (``np.nan`` if empty).
        average_shortest_path_length : float
            Average shortest path length (``np.nan`` if empty or disconnected).
        radius : int | None
            Graph radius (``None`` if empty or disconnected).
        diameter : int | None
            Graph diameter (``None`` if empty or disconnected).
        center : list[int]
            List of central nodes whose eccentricity equals the radius.
        diameter_nodes : list[int]
            Sequence of nodes forming a shortest path of length ``diameter``.
        diameter_edges : list[tuple[int, int]]
            Sequence of edges forming the representative diametral path.
        radius_nodes : list[int]
            Sequence of nodes from ``center[0]`` to a node at distance ``radius``.
        radius_edges : list[tuple[int, int]]
            Sequence of edges forming the representative radial path.
        degrees : tuple[int, ...]
            Ordered tuple of node degrees.
    """
    logger.info("Calculando métricas globales del grafo")

    # Size and connectivity metrics using standardized graph helpers.
    graph_name = graph.get_graph_name(g)
    node_count = graph.get_node_count(g)
    edge_count = graph.get_edge_count(g)

    degrees = [deg for _, deg in g.degree()]
    average_degree = graph.get_average_degree(g)

    density = nx.density(g)
    average_clustering = (
        float(nx.average_clustering(g))
        if node_count > 0
        else np.nan
    )

    radius, diameter, center = None, None, []
    avg_path_length = np.nan
    diameter_edges, radius_edges = [], []
    diameter_nodes, radius_nodes = [], []

    # Default schema returned even for empty or disconnected graphs.
    results = {
        "graph_name": graph_name,
        "node_count": node_count,
        "edge_count": edge_count,
        "average_degree": average_degree,
        "density": density,
        "average_clustering": average_clustering,
        "average_shortest_path_length": avg_path_length,
        "radius": radius,
        "diameter": diameter,
        "center": center,
        "diameter_nodes": diameter_nodes,
        "diameter_edges": diameter_edges,
        "radius_nodes": radius_nodes,
        "radius_edges": radius_edges,
        "degrees": tuple(degrees),
    }

    if node_count == 0:
        logger.warning("El grafo recibido está vacío.")
        return results

    if not nx.is_connected(g):
        logger.warning(
            "El grafo no es conexo. "
            "No se calculan métricas globales basadas en distancias."
        )
        return results

    radius = nx.radius(g)
    diameter = nx.diameter(g)
    center = nx.center(g)
    avg_path_length = float(nx.average_shortest_path_length(g))

    path_lengths = dict(nx.shortest_path_length(g))

    # Identify a representative diametral path.
    n1_diam, n2_diam = None, None

    for u in g.nodes():
        for v in g.nodes():
            if path_lengths[u][v] == diameter:
                n1_diam, n2_diam = u, v
                break

        if n1_diam is not None:
            break

    if n1_diam is not None:
        diameter_nodes = nx.shortest_path(
            g,
            source=n1_diam,
            target=n2_diam,
        )
        diameter_edges = list(
            zip(diameter_nodes, diameter_nodes[1:])
        )

    # Identify a representative radial path starting from the center.
    if center:
        center_node = center[0]

        for target_node in g.nodes():
            if path_lengths[center_node][target_node] == radius:
                radius_nodes = nx.shortest_path(
                    g,
                    source=center_node,
                    target=target_node,
                )
                radius_edges = list(
                    zip(radius_nodes, radius_nodes[1:])
                )
                break

    results.update({
        "average_shortest_path_length": avg_path_length,
        "radius": radius,
        "diameter": diameter,
        "center": center,
        "diameter_nodes": diameter_nodes,
        "diameter_edges": diameter_edges,
        "radius_nodes": radius_nodes,
        "radius_edges": radius_edges,
    })

    logger.info("EXITO ✅.")
    return results
# endregion






# region compute_rolling_metrics
def compute_rolling_metrics(rolling_graphs: dict) -> dict:
    """Compute graph-level and degree-distribution metrics for each rolling window.

    Iterates through the rolling visibility graphs produced by
    ``graph.build_rolling_graphs``, evaluates global graph metrics and degree
    distributions per window, and preserves window boundaries and series
    metadata without mutating or duplicating the input graphs.

    Parameters
    ----------
    rolling_graphs : dict
        Dictionary returned by ``graph.build_rolling_graphs``, containing:

        series_name : str
            Name of the analyzed time series (e.g., ``'FEMINICIDIO'``).
        window_size : int
            Window length used in the rolling partition.
        step : int
            Step size between consecutive windows.
        windows : list[dict]
            List of window graph dictionaries, each containing ``'window_id'``,
            ``'start_idx'``, ``'end_idx'``, ``'start_date'``, ``'end_date'``,
            and ``'graph'`` (nx.Graph).

    Returns
    -------
    dict
        Dictionary containing rolling metrics metadata and per-window results:

        series_name : str
            Name of the analyzed time series.
        window_size : int
            Window length used in the rolling partition.
        step : int
            Step size between consecutive windows.
        windows : list[dict]
            List of window metric dictionaries (empty if input ``'windows'``
            is empty), each containing:

            window_id : int
                Zero-based sequential window identifier.
            start_idx : int
                Inclusive starting row index in the original series.
            end_idx : int
                Inclusive ending row index in the original series.
            start_date : pd.Timestamp
                Timestamp of the first node in the window.
            end_date : pd.Timestamp
                Timestamp of the last node in the window.
            graph_metrics : dict
                Global graph metrics returned by ``compute_graph_metrics``.
            distribution_metrics : dict
                Degree-distribution fit and summary metrics returned by
                ``compute_distribution_metrics``.
    """
    logger.info(
        "Iniciando cálculo de métricas sobre %s ventanas",
        len(rolling_graphs["windows"]),
    )

    results = {
        "series_name": rolling_graphs["series_name"],
        "window_size": rolling_graphs["window_size"],
        "step": rolling_graphs["step"],
        "windows": [],
    }

    if not rolling_graphs["windows"]:
        logger.warning("No hay grafos por ventana para analizar.")
        return results

    for window_item in rolling_graphs["windows"]:
        logger.info(
            "Calculando métricas de la ventana %s",
            window_item["window_id"],
        )

        # 1. Compute global structural metrics for the window graph.
        graph_metrics = compute_graph_metrics(
            g=window_item["graph"],
        )

        # 2. Reuse degree sequence to fit and summarize degree distributions.
        distribution_metrics = compute_distribution_metrics(
            graph_metrics=graph_metrics,
            min_positive_degrees=10,
        )

        # 3. Preserve metadata and metrics without duplicating the graph object.
        results["windows"].append({
            "window_id": window_item["window_id"],
            "start_idx": window_item["start_idx"],
            "end_idx": window_item["end_idx"],
            "start_date": window_item["start_date"],
            "end_date": window_item["end_date"],
            "graph_metrics": graph_metrics,
            "distribution_metrics": distribution_metrics,
        })

    logger.info(
        "EXITO ✅. Métricas calculadas para %s ventanas.",
        len(results["windows"]),
    )

    return results
# endregion







# region extract_window_metrics
def extract_window_metrics(
    grafo: nx.Graph,
    window_id: int,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> dict:
    """Genera el resumen de métricas del grafo de una ventana temporal."""

    metricas = compute_graph_metrics(grafo)

    distribuciones = compute_distribution_metrics(
        metricas_grafo=metricas,
        min_grados_positivos=10,
    )

    # Mantener NaN cuando las métricas de distancia no están definidas.
    diametro = (
        metricas["diametro"]
        if metricas["diametro"] is not None
        else np.nan
    )
    radio = (
        metricas["radio"]
        if metricas["radio"] is not None
        else np.nan
    )

    return {
        "window_id": window_id,
        "start_date": start_date,
        "end_date": end_date,
        "fecha_corte": end_date,
        "nodos": metricas["nodos"],
        "enlaces": metricas["enlaces"],
        "grado_promedio": round(metricas["grado_promedio"], 4),
        "densidad": round(metricas["densidad_empirica"], 5),
        "clustering": round(metricas["clustering"], 4),
        "camino_medio": round(metricas["camino_medio"], 4),
        "diametro": diametro,
        "radio": radio,
        "entropia_grado": round(
            distribuciones["entropia_grado"], 4
        ),
        "gamma_powerlaw": round(
            distribuciones["gamma_powerlaw"], 4
        ),
        "dist_eucl_pl": round(
            distribuciones["dist_eucl_pl"], 5
        ),
        "dist_eucl_exp": round(
            distribuciones["dist_eucl_exp"], 5
        ),
    }
# endregion







# region compute_metrics_distance_correlations
# DEPRECATED
def compute_metrics_distance_correlations(
    metrics_by_window: dict,
    strategy: str = "ataque",
    metricas: list[str] | None = None,
) -> dict:
    """Calcula correlaciones de Spearman entre métricas y distancias.

    Calcula cada asociación a través de las ventanas, por modelo.
    Conserva los nombres completos de los modelos.

    Requiere al menos tres pares válidos por asociación.
    Las correlaciones con variables constantes quedan como NaN.
    Devuelve las correlaciones, cantidades de pares y metadatos.
    """

    if strategy not in ("ataque", "aleatorio"):
        raise ValueError("strategy debe ser 'ataque' o 'aleatorio'.")

    df = metrics_by_window["data"]

    if df.empty:
        raise ValueError("La tabla de métricas por ventana está vacía.")

    df = df.loc[df["estrategia"].eq(strategy)].copy()

    if df.empty:
        raise ValueError(f"No hay datos para la estrategia '{strategy}'.")

    if metricas is None:
        metricas = [
            "nodos",
            "enlaces",
            "grado_promedio",
            "densidad_empirica",
            "clustering",
            "camino_medio",
            "radio",
            "diametro",
        ]

    if not metricas or len(set(metricas)) != len(metricas):
        raise ValueError("Selecciona métricas sin nombres repetidos.")

    faltantes = set(metricas) - set(df.columns)

    if faltantes:
        raise ValueError(f"No existen estas métricas: {sorted(faltantes)}")

    if df.duplicated(["window_id", "modelo"]).any():
        raise ValueError("Hay comparaciones duplicadas por ventana y modelo.")

    columnas_numericas = metricas + ["distancia_resiliencia"]

    for columna in columnas_numericas:
        df[columna] = pd.to_numeric(df[columna], errors="raise")

    df[columnas_numericas] = df[columnas_numericas].replace(
        [np.inf, -np.inf], np.nan
    )

    modelos = sorted(df["modelo"].unique())

    correlaciones = pd.DataFrame(
        np.nan, index=modelos, columns=metricas, dtype=float
    )
    pares_validos = pd.DataFrame(
        0, index=modelos, columns=metricas, dtype=int
    )

    # Identificar constantes usando una observación por ventana.
    df_ventanas = df.drop_duplicates("window_id")
    constantes = {}

    for metrica in metricas:
        valores = df_ventanas[metrica].dropna().unique()

        if len(valores) == 1:
            constantes[metrica] = valores[0]

    for modelo, grupo in df.groupby("modelo", sort=False):
        for metrica in metricas:
            pares = grupo[
                [metrica, "distancia_resiliencia"]
            ].dropna()

            pares_validos.loc[modelo, metrica] = len(pares)

            if len(pares) < 3:
                continue

            if (
                pares[metrica].nunique() < 2
                or pares["distancia_resiliencia"].nunique() < 2
            ):
                continue

            correlaciones.loc[modelo, metrica] = pares[metrica].corr(
                pares["distancia_resiliencia"], method="spearman"
            )

    return {
        "crime_name": metrics_by_window["crime_name"],
        "window_size": metrics_by_window["window_size"],
        "step": metrics_by_window["step"],
        "n_windows": df["window_id"].nunique(),
        "strategy": strategy,
        "constantes": constantes,
        "correlaciones": correlaciones,
        "pares_validos": pares_validos,
    }
# endregion