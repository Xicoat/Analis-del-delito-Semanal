# region Libraries
import logging

import networkx as nx
import pandas as pd

from .config import WINDOW_SIZE, STEP_SIZE
from .windows import generate_rolling_windows

logger = logging.getLogger("Delitos")
# endregion






# region build_visibility_graph
def build_nvg(series_data: dict, messages: bool = True) -> nx.Graph:
    """Build a Natural Visibility Graph (NVG) from a weekly time series.

    Parameters
    ----------
    series_data : dict
        Dictionary containing the following items:

        series_name : str
            Name of the analyzed time series.
        data : pd.DataFrame
            DataFrame containing at least the columns
            ``['node_id', 'timestamp', 'frequency']``.
    messages : bool, default=True
        Whether to emit progress log messages during graph construction.

    Returns
    -------
    nx.Graph
        Undirected NetworkX graph with the following structure:

        Nodes
            Integer identifiers from ``'node_id'``, with node attributes
            ``'timestamp'`` and ``'frequency'`` (float).
        Graph attributes (``g.graph``)
            Dictionary containing ``'graph_name'`` (str), ``'node_count'`` (int),
            ``'edge_count'`` (int), ``'average_degree'`` (float), and
            ``'density'`` (float).
    """
    if messages:
        logger.info("Build NVG")
    df = series_data["data"]
    g = nx.Graph()
    g.graph["graph_name"] = series_data["series_name"]

    # 1. Assign nodes with their attributes.
    for _, row in df.iterrows():
        g.add_node(
            int(row["node_id"]),
            timestamp=row["timestamp"],
            frequency=float(row["frequency"]),
        )
    y = df["frequency"].to_numpy()
    n = len(y)

    # 2. Evaluate the geometric visibility criterion.
    for i in range(n - 1):
        g.add_edge(i, i + 1)
        max_slope = y[i + 1] - y[i]

        for j in range(i + 2, n):
            slope = (y[j] - y[i]) / (j - i)
            if slope > max_slope:
                g.add_edge(i, j)
                max_slope = slope

    # 3. Store summary metrics of the constructed graph.
    node_count = g.number_of_nodes()
    edge_count = g.number_of_edges()
    g.graph["node_count"] = node_count
    g.graph["edge_count"] = edge_count
    g.graph["average_degree"] = 2 * edge_count / node_count if node_count else 0.0
    g.graph["density"] = float(nx.density(g))

    # 4. Log completion summary.
    if messages:
        logger.info(
            f"EXITO ✅"
            f"(|V|={g.number_of_nodes()}, |E|={g.number_of_edges()})."
        )
    return g
# endregion






# region build_rolling_graphs
def build_rolling_graphs(rolling_windows: dict) -> dict:
    """Build a Natural Visibility Graph (NVG) for each rolling window.

    Iterates through all window slices produced by
    ``windows.generate_rolling_windows``, constructs an undirected NVG per
    window via ``build_nvg``, and preserves window boundaries and series
    metadata.

    Parameters
    ----------
    rolling_windows : dict
        Dictionary returned by ``windows.generate_rolling_windows``, containing:

        series_name : str
            Name of the analyzed time series (e.g., ``'FEMINICIDIO'``).
        window_size : int
            Window length used to slice the series.
        step : int
            Step size between consecutive windows.
        windows : list[dict]
            List of window dictionaries, each containing ``'window_id'``,
            ``'start_idx'``, ``'end_idx'``, ``'start_date'``, ``'end_date'``,
            and ``'df_window'`` (pd.DataFrame with columns
            ``['node_id', 'timestamp', 'frequency']``).

    Returns
    -------
    dict
        Dictionary containing rolling graph metadata and graph instances:

        series_name : str
            Name of the analyzed time series.
        window_size : int
            Window length used in the rolling partition.
        step : int
            Step size between consecutive windows.
        windows : list[dict]
            List of window graph dictionaries, each containing:

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
            graph : nx.Graph
                Natural Visibility Graph constructed for the window.
    """
    series_name = rolling_windows["series_name"]

    results = {
        "series_name": series_name,
        "window_size": rolling_windows["window_size"],
        "step": rolling_windows["step"],
        "windows": [],
    }

    for window_data in rolling_windows["windows"]:
        g_window = build_nvg(
            {
                "series_name": f"{series_name}_W{window_data['window_id']}",
                "data": window_data["df_window"],
            },
            messages=False,
        )

        results["windows"].append({
            "window_id": window_data["window_id"],
            "start_idx": window_data["start_idx"],
            "end_idx": window_data["end_idx"],
            "start_date": window_data["start_date"],
            "end_date": window_data["end_date"],
            "graph": g_window,
        })

    if results["windows"]:
        first_win = results["windows"][0]
        last_win = results["windows"][-1]

        logger.info(
            "EXITO ✅. Resumen Rolling Window: %s grafos generados | "
            "W=%s, S=%s | Rango: %s -> %s | Nodos por ventana: %s",
            len(results["windows"]),
            results["window_size"],
            results["step"],
            f"{first_win['start_date']:%Y-%m-%d}",
            f"{last_win['end_date']:%Y-%m-%d}",
            get_node_count(first_win["graph"]),
        )
    else:
        logger.warning("No se recibieron ventanas para construir grafos.")

    return results
# endregion





# region get_graph_name
def get_graph_name(g: nx.Graph, default: str = "unnamed") -> str:
    """Return the graph name stored in the graph attributes.

    Safely retrieves ``'graph_name'`` from ``g.graph``, falling back to
    ``default`` if the key is missing or its value is ``None``.

    Parameters
    ----------
    g : nx.Graph
        NetworkX graph instance to inspect.
    default : str, default="unnamed_graph"
        Fallback name returned when ``'graph_name'`` is absent or ``None``.

    Returns
    -------
    str
        Name of the graph, or ``default`` if undefined.
    """
    name = g.graph.get("graph_name")
    return str(name) if name is not None else default
# endregion





# region get_node_count
def get_node_count(g: nx.Graph) -> int:
    """Return the node count stored in the graph attributes.

    Parameters
    ----------
    g : nx.Graph
        NetworkX graph containing the ``'node_count'`` attribute.

    Returns
    -------
    int
        Total number of nodes in the graph.

    Raises
    ------
    KeyError
        If ``'node_count'`` is not present in ``g.graph``.
    """
    return g.graph.get("node_count", g.number_of_nodes())
# endregion






# region get_edge_count
def get_edge_count(g: nx.Graph) -> int:
    """Return the edge count stored in the graph attributes.

    Parameters
    ----------
    g : nx.Graph
        NetworkX graph containing the ``'edge_count'`` attribute.

    Returns
    -------
    int
        Total number of edges in the graph.

    Raises
    ------
    KeyError
        If ``'edge_count'`` is not present in ``g.graph``.
    """
    return g.graph.get("edge_count", g.number_of_edges())
# endregion






# region get_average_degree
def get_average_degree(g: nx.Graph) -> float:
    """Return the average degree stored in the graph attributes.

    Parameters
    ----------
    g : nx.Graph
        NetworkX graph containing the ``'average_degree'`` attribute.

    Returns
    -------
    float
        Average degree of the graph.

    Raises
    ------
    KeyError
        If ``'average_degree'`` is not present in ``g.graph``.
    """
    if "average_degree" in g.graph:
        return g.graph["average_degree"]
    n_nodes = g.number_of_nodes()
    return sum(deg for _, deg in g.degree()) / n_nodes if n_nodes > 0 else 0.0
# endregion






# region get_density
def get_density(g: nx.Graph) -> float:
    """Return the density stored in the graph attributes.

    Parameters
    ----------
    g : nx.Graph
        NetworkX graph containing the ``'density'`` attribute.

    Returns
    -------
    float
        Density of the graph.

    Raises
    ------
    KeyError
        If ``'density'`` is not present in ``g.graph``.
    """
    return float(g.graph["density"])
# endregion






