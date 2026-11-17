# region Libraries
import logging
import random
import re

import networkx as nx
import numpy as np
import pandas as pd

from datetime import datetime, timezone
from uuid import uuid4

from . import graph

logger = logging.getLogger("Delitos")



# endregion






# region regular_graph
def regular_graph(n: int, k: int) -> nx.Graph:
    """Conecta cada nodo con sus siguientes k nodos, hasta el final de la cadena."""

    g = nx.Graph()
    g.add_nodes_from(range(n))

    for i in range(n):
        for j in range(i + 1, min(i + k + 1, n)):
            g.add_edge(i, j)

    return g
# endregion






# region generate_theoretical_models
def generate_theoretical_models(
    g: nx.Graph,
) -> dict[str, nx.Graph]:
    """Generate a catalog of theoretical reference graphs matching size and density.

    Constructs standard random, scale-free, small-world, regular, lattice, and
    deterministic topologies parameterized by the node count and edge density
    of ``g``. Models whose structural constraints are incompatible with
    ``n_nodes`` (such as ``'Hypercube'`` when ``n_nodes`` is not a power of 2)
    are omitted.

    Parameters
    ----------
    g : nx.Graph
        Empirical visibility graph whose node count and density parameterize
        the reference models.

    Returns
    -------
    dict[str, nx.Graph]
        Mapping from model label to generated NetworkX graph. Returns an empty
        dictionary if ``g`` is not a ``nx.Graph`` or has ``<= 1`` nodes.
    """
    if not isinstance(g, nx.Graph):
        return {}

    n_nodes = graph.get_node_count(g)
    if n_nodes <= 1:
        return {}

    density_val = float(nx.density(g))
    sqrt_nodes = int(np.sqrt(n_nodes))

    models = {
        f"Erdos Renyi (p={density_val:.4f})": (
            nx.erdos_renyi_graph(n_nodes, density_val)
        ),

        "Barabasi Albert (m=2)": (
            nx.barabasi_albert_graph(n_nodes, 2)
            if n_nodes > 2
            else None
        ),
        "Barabasi Albert (m=3)": (
            nx.barabasi_albert_graph(n_nodes, 3)
            if n_nodes > 3
            else None
        ),
        "Barabasi Albert (m=4)": (
            nx.barabasi_albert_graph(n_nodes, 4)
            if n_nodes > 4
            else None
        ),

        f"Watts Strogatz (p={density_val:.4f})": (
            nx.watts_strogatz_graph(n_nodes, 4, density_val)
            if n_nodes >= 4
            else None
        ),

        "Random Regular (k=2)": (
            nx.random_regular_graph(2, n_nodes)
            if n_nodes > 2
            else None
        ),

        # Preserve both labels for compatibility.
        "Enlace Igualitario 2": regular_graph(n_nodes, 2),
        "Enlace Igualitario 3": regular_graph(n_nodes, 3),

        "Star": (
            nx.star_graph(n_nodes - 1)
            if n_nodes > 0
            else None
        ),

        "Grid 2D": (
            nx.grid_2d_graph(sqrt_nodes, sqrt_nodes)
            if sqrt_nodes > 0
            else None
        ),

        "Ladder": (
            nx.ladder_graph(n_nodes // 2)
            if n_nodes >= 2
            else None
        ),

        "Hypercube": (
            nx.hypercube_graph(int(np.log2(n_nodes)))
            if (
                n_nodes > 0
                and (n_nodes & (n_nodes - 1)) == 0
            )
            else None
        ),

        "Complete": nx.complete_graph(n_nodes),

        # Equivalent structure to Enlace Igualitario 1.
        "Path": nx.path_graph(n_nodes),

        "Cycle": nx.cycle_graph(n_nodes),

        "Circular Ladder": (
            nx.circular_ladder_graph(n_nodes // 2)
            if n_nodes >= 2
            else None
        ),
    }

    excluded = [
        name for name, model_graph in models.items()
        if model_graph is None
    ]

    if excluded:
        logger.debug(
            "Modelos excluidos para N=%s: %s",
            n_nodes,
            ", ".join(excluded),
        )

    return {
        name: model_graph
        for name, model_graph in models.items()
        if model_graph is not None
    }
# endregion





# region simulate_resilience_profile
def simulate_resilience_profile(
    g: nx.Graph,
    sorting_strategy: str = "attack",
    seed: int | None = None,
) -> np.ndarray:
    """Compute the topological fragmentation curve of a single graph under node removal.

    Sequentially removes all nodes from a copy of ``g`` following either a
    degree-targeted order (``'attack'``) or a uniform random order
    (``'random'``), recording the number of connected components after each
    node removal.

    Does not mutate the input graph.

    Parameters
    ----------
    g : nx.Graph
        Undirected graph to evaluate.
    sorting_strategy : {'attack', 'random'}, default='attack'
        Node removal order strategy:

        - ``'attack'``: removes nodes in descending order of initial degree.
        - ``'random'``: removes nodes in a uniform random permutation.
    seed : int | None, default=None
        Random seed used when ``sorting_strategy='random'`` for reproducible
        node removal sequences.

    Returns
    -------
    np.ndarray
        One-dimensional array of length ``g.number_of_nodes()`` containing the
        connected component count after each sequential node removal. Returns an
        empty array if ``g`` has no nodes.

    Raises
    ------
    ValueError
        If ``sorting_strategy`` is neither ``'attack'`` nor ``'random'``.
    """
    g_copy = g.copy()
    n_nodes = g_copy.number_of_nodes()

    if n_nodes == 0:
        return np.array([])

    if sorting_strategy == "attack":
        ordered_nodes = sorted(
            g_copy.nodes(),
            key=lambda u: g_copy.degree(u),
            reverse=True,
        )

    elif sorting_strategy == "random":
        if seed is None:
            logger.warning(
                "Remoción aleatoria sin semilla explícita: "
                "el orden de eliminación puede variar entre ejecuciones."
            )

        rng = random.Random(seed)
        ordered_nodes = rng.sample(
            list(g_copy.nodes()),
            n_nodes,
        )

    else:
        raise ValueError(
            f"Estrategia '{sorting_strategy}' no reconocida. "
            "Use 'attack' o 'random'."
        )

    components = []

    for node in ordered_nodes:
        if node in g_copy:
            g_copy.remove_node(node)

        components.append(
            nx.number_connected_components(g_copy)
        )

    return np.array(components)
# endregion





# region compute_normalized_curve_distance
def compute_normalized_curve_distance(y1: np.ndarray, y2: np.ndarray,) -> float:
    """Calcula la distancia euclidiana entre curvas normalizadas por su máximo.

    Recorta ambas curvas a la longitud de la más corta y divide cada
    segmento por su propio máximo, siempre que este sea positivo.

    Normaliza las amplitudes de las curvas; no normaliza la distancia
    por el número de puntos. El resultado no está necesariamente entre
    0 y 1 y puede depender de la longitud comparada.

    Si alguna curva está vacía, devuelve NaN.
    """

    # PENDIENTE: revisar la comparabilidad entre curvas de grafos con
    # distinto número de nodos. El recorte actual descarta el tramo
    # sobrante y compara posiciones por número de nodos eliminados,
    # no por proporción de nodos eliminados.
    min_len = min(len(y1), len(y2))

    if min_len == 0:
        return float(np.nan)

    y1_trim, y2_trim = y1[:min_len], y2[:min_len]

    max_y1 = max(y1_trim) if len(y1_trim) > 0 else 0
    max_y2 = max(y2_trim) if len(y2_trim) > 0 else 0

    y1_norm = y1_trim / max_y1 if max_y1 > 0 else y1_trim
    y2_norm = y2_trim / max_y2 if max_y2 > 0 else y2_trim

    return float(np.sqrt(np.sum((y1_norm - y2_norm) ** 2)))
# endregion





# region resilience_analysis_2
def resilience_analysis_2(
    grafo: nx.Graph,
    seed: int | None = None,    *,
    ganadores_previos: dict[str, str] | None = None,
) -> dict:
    """Analiza la fragmentación de un grafo y sus modelos teóricos.

    Conserva identidad, periodo, grafos, métricas, curvas, distancias
    ordenadas y el modelo más cercano por estrategia.

    Las notas indican "empate" si varios modelos comparten exactamente
    la distancia mínima; en caso contrario, quedan vacías.
    En empates se conserva el orden del catálogo.

    La semilla controla únicamente la remoción aleatoria.
    No genera gráficas ni modifica el grafo recibido.
    """

    logger.info("Iniciando análisis comparado de resiliencia")

    if grafo.number_of_nodes() == 0:
        logger.warning("El grafo recibido está vacío.")
        return {}

    crime_name = grafo.graph["crime_name"]

    # 1. Periodo del grafo empírico.
    fechas = [
        pd.to_datetime(datos["fecha_hecho"])
        for _, datos in grafo.nodes(data=True)
        if "fecha_hecho" in datos
    ]

    start_date = min(fechas) if fechas else None
    end_date = max(fechas) if fechas else None

    # 2. Métricas básicas, sin depender de compute_graph_metrics.
    def metricas_basicas(red: nx.Graph) -> dict:
        n_nodes = red.number_of_nodes()

        return {
            "nodos": n_nodes,
            "enlaces": red.number_of_edges(),
            "grado_promedio": (
                sum(grado for _, grado in red.degree()) / n_nodes
                if n_nodes > 0
                else 0.0
            ),
            "densidad": float(nx.density(red)),
        }

    # 3. Generar el catálogo una sola vez.
    modelos = generate_theoretical_models(grafo)

    resultados = {
        "crime_name": crime_name,
        "start_date": start_date,
        "end_date": end_date,
        "seed_remocion": seed,

        "red_empirica": {
            "grafo": grafo,
            "metricas": metricas_basicas(grafo),
            "curvas": {},
        },

        "modelos": {},

        "distancias_ordenadas": {
            "ataque": [],
            "aleatorio": [],
        },

        "mejores_modelos": {
            "ataque": None,
            "aleatorio": None,
        },

        "notas": {
            "ataque": "",
            "aleatorio": "",
        },
    }

    # Conservar todos los modelos entregados por el generador.
    for nombre_modelo, grafo_modelo in modelos.items():
        k_match = re.search(r"[kK]=(\d+)", nombre_modelo)
        p_match = re.search(r"[pP]=([\d\.]+)", nombre_modelo)

        resultados["modelos"][nombre_modelo] = {
            "grafo": grafo_modelo,
            "metricas": metricas_basicas(grafo_modelo),
            "parametros": {
                "k": int(k_match.group(1)) if k_match else None,
                "p": float(p_match.group(1)) if p_match else None,
            },
            "curvas": {},
            "distancias": {},
        }

    if not modelos:
        logger.warning(
            "No se generaron modelos teóricos. "
            "Solo se calcularán las curvas empíricas."
        )

    # 4. Calcular curvas y distancias por estrategia.
    for estrategia in ("ataque", "aleatorio"):
        curva_empirica = simulate_resilience_profile(
            grafo,
            sorting_strategy=estrategia,
            seed=seed,
        )

        resultados["red_empirica"]["curvas"][estrategia] = (
            curva_empirica
        )

        distancias = []

        for nombre_modelo, datos_modelo in resultados["modelos"].items():
            curva_modelo = simulate_resilience_profile(datos_modelo["grafo"], sorting_strategy=estrategia, seed=seed,)

            distancia = float(
                compute_normalized_curve_distance(curva_empirica, curva_modelo,)
            )

            datos_modelo["curvas"][estrategia] = curva_modelo
            datos_modelo["distancias"][estrategia] = distancia

            distancias.append({
                "modelo": nombre_modelo,
                "distancia": distancia,
            })

        # Orden estable: conservar el orden del catálogo en empates.
        distancias_ordenadas = sorted(distancias, key=lambda item: item["distancia"], )

        resultados["distancias_ordenadas"][estrategia] = (distancias_ordenadas)

        if not distancias_ordenadas:
            continue

        # 5. Resolver el empate en primer lugar por continuidad.
        distancia_minima = distancias_ordenadas[0]["distancia"]
        empatados = [
            item["modelo"]
            for item in distancias_ordenadas
            if item["distancia"] == distancia_minima
        ]

        if len(empatados) > 1:
            previo = (ganadores_previos or {}).get(estrategia)

            if previo in empatados:
                # Mover solo al ganador previo; la distancia sigue siendo la misma.
                posicion = next(
                    i for i, item in enumerate(distancias_ordenadas)
                    if item["modelo"] == previo
                )
                distancias_ordenadas.insert(0, distancias_ordenadas.pop(posicion))

        ganador = distancias_ordenadas[0]
        resultados["mejores_modelos"][estrategia] = ganador.copy()

        # 6. Registrar únicamente los empates exactos en primer lugar.
        if len(empatados) > 1:
            resultados["notas"][estrategia] = "empate"
            logger.warning(
                "Empate | Primer Lugar | d=%.4f | %s | %s",
                distancia_minima,
                ", ".join(empatados),
                ganador["modelo"],
            )

    logger.info(
        "EXITO ✅. Resiliencia calculada para '%s': "
        "red empírica y %s modelos teóricos.",
        crime_name,
        len(modelos),
    )

    return resultados
# endregion






# region rolling_resilience_analysis
def rolling_resilience_analysis(
    rolling_graphs: dict,
    seed: int | None = None,
) -> dict:
    """Analyze topological resilience across all rolling-window visibility graphs.

    Evaluates node-removal resilience for each window graph produced by
    ``graph.build_rolling_graphs``, propagating winning reference models from
    each window to the next while preserving experiment metadata and window
    boundaries.

    Does not mutate ``rolling_graphs`` or write files to disk.

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
    seed : int | None, default=None
        Random seed forwarded to ``resilience_analysis`` for reproducible
        stochastic node removals.

    Returns
    -------
    dict
        Dictionary containing experiment execution metadata and per-window
        resilience results:

        experiment_id : str
            Hexadecimal UUID identifying the experiment execution.
        executed_at : str
            ISO-8601 UTC timestamp when the analysis started.
        series_name : str
            Name of the analyzed time series.
        window_size : int
            Window length used in the rolling partition.
        step : int
            Step size between consecutive windows.
        removal_seed : int | None
            Random seed used for stochastic node removal simulations.
        windows : list[dict]
            List of window resilience dictionaries (empty if input ``'windows'``
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
            resilience_results : dict
                Resilience curves, model fits, and fragmentation metrics
                returned by ``resilience_analysis``.
    """
    experiment_id = uuid4().hex
    executed_at = datetime.now(timezone.utc).isoformat()

    results = {
        "experiment_id": experiment_id,
        "executed_at": executed_at,
        "series_name": rolling_graphs["series_name"],
        "window_size": rolling_graphs["window_size"],
        "step": rolling_graphs["step"],
        "removal_seed": seed,
        "windows": [],
    }

    logger.info(
        "Iniciando análisis de resiliencia sobre %s ventanas | "
        "W=%s, S=%s | Experimento=%s",
        len(rolling_graphs["windows"]),
        rolling_graphs["window_size"],
        rolling_graphs["step"],
        experiment_id,
    )

    if not rolling_graphs["windows"]:
        logger.warning("No hay grafos por ventana para analizar.")
        return results

    previous_winners = {}

    for window_item in rolling_graphs["windows"]:
        logger.info(
            "Analizando resiliencia de la ventana %s",
            window_item["window_id"],
        )

        analysis = resilience_analysis(
            g=window_item["graph"],
            seed=seed,
            previous_winners=previous_winners,
        )

        previous_winners = {
            strategy: winner["model"]
            for strategy, winner in analysis.get("best_models", {}).items()
            if winner is not None
        }

        results["windows"].append({
            "window_id": window_item["window_id"],
            "start_idx": window_item["start_idx"],
            "end_idx": window_item["end_idx"],
            "start_date": window_item["start_date"],
            "end_date": window_item["end_date"],
            "resilience_results": analysis,
        })

    logger.info(
        "EXITO ✅. Experimento %s: %s ventanas procesadas.",
        experiment_id,
        len(results["windows"]),
    )

    return results
# endregion







# region preparar_para_visor
def preparar_para_visor(obj):
    """Convierte objetos complejos en strings simples para que el visor interactivo no colapse."""
    if isinstance(obj, dict):
        return {k: preparar_para_visor(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [preparar_para_visor(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return f"[Array NumPy colapsado: {obj.size} registros]"
    elif isinstance(obj, nx.Graph):
        return f"[Grafo NetworkX: {obj.number_of_nodes()} nodos, {obj.number_of_edges()} enlaces]"
    elif isinstance(obj, pd.Timestamp):
        return str(obj.date())
    
    # Retorna tipos nativos intactos
    return str(obj) if not isinstance(obj, (int, float, str, bool, type(None))) else obj
# endregion






# region resilience_simulator
def resilience_simulator(g: nx.Graph) -> dict:
    """Record a targeted initial-degree attack simulation on an undirected graph.

    Orders nodes once in descending order of initial degree while preserving
    node insertion order on ties (matching ``simulate_resilience_profile``).
    Records the intact network at step 0 and tracks connected-component
    fragmentation metrics after each node removal without reconstructing
    visibility edges or mutating the input graph.

    Parameters
    ----------
    g : nx.Graph
        Simple undirected NetworkX graph to simulate.

    Returns
    -------
    dict
        Dictionary containing graph metadata, a copy of the original graph,
        the node removal sequence, and the step-by-step attack log:

        series_name : str | None
            Series or graph identifier extracted from ``g.graph['series_name']``
            or ``g.graph['graph_name']``.
        start_date : pd.Timestamp | None
            Earliest timestamp found in ``g.graph['start_date']`` or node
            ``'timestamp'`` attributes.
        end_date : pd.Timestamp | None
            Latest timestamp found in ``g.graph['end_date']`` or node
            ``'timestamp'`` attributes.
        strategy : str
            Attack strategy identifier (``'attack'``).
        criterion : str
            Node-ordering criterion (``'descending_initial_degree'``).
        initial_nodes : int
            Initial node count before any removals.
        initial_edges : int
            Initial edge count before any removals.
        graph : nx.Graph
            Unmodified copy of the input graph.
        removal_order : list
            Nodes ordered by descending initial degree.
        attack_log : list[dict]
            Step-by-step fragmentation records from step 0 through
            ``initial_nodes``.

    Raises
    ------
    TypeError
        If ``g`` is not an ``nx.Graph`` instance.
    ValueError
        If ``g`` is directed or a multigraph.
    """
    if not isinstance(g, nx.Graph):
        raise TypeError("g debe ser un grafo de NetworkX.")

    if g.is_directed() or g.is_multigraph():
        raise ValueError("Se requiere un grafo simple no dirigido.")

    logger.info("Iniciando simulación de ataque por grado inicial")

    original_graph = g.copy()
    active_graph = original_graph.copy()

    initial_nodes = original_graph.number_of_nodes()
    initial_edges = original_graph.number_of_edges()
    initial_degrees = dict(original_graph.degree())

    # Orden estable: reproduce el criterio del simulador existente.
    removal_order = sorted(
        original_graph.nodes(),
        key=lambda node: initial_degrees[node],
        reverse=True,
    )

    # Recuperar el periodo sin exigir atributos temporales al grafo.
    timestamps = pd.to_datetime([
        data["timestamp"]
        for _, data in original_graph.nodes(data=True)
        if data.get("timestamp") is not None
    ], errors="raise")
    timestamps = timestamps[~pd.isna(timestamps)]

    start_date = original_graph.graph.get("start_date")
    end_date = original_graph.graph.get("end_date")

    if start_date is None and len(timestamps):
        start_date = timestamps.min()

    if end_date is None and len(timestamps):
        end_date = timestamps.max()

    # Conservar metadatos existentes, incluido window_id si está presente.
    results = dict(original_graph.graph)
    results.update({
        "series_name": (
            original_graph.graph.get("series_name")
            or original_graph.graph.get("graph_name")
        ),
        "start_date": start_date,
        "end_date": end_date,
        "strategy": "attack",
        "criterion": "descending_initial_degree",
        "initial_nodes": initial_nodes,
        "initial_edges": initial_edges,
        "graph": original_graph,
        "removal_order": removal_order,
        "attack_log": [],
    })

    def record_state(
        step: int,
        removed_node,
        degree_before: int | None,
        previous_components: int | None,
    ) -> dict:
        """Describe the current fragmentation state without copying the graph."""
        sizes = sorted(
            (len(component)
             for component in nx.connected_components(active_graph)),
            reverse=True,
        )

        remaining_nodes = active_graph.number_of_nodes()
        n_components = len(sizes)
        largest = sizes[0] if sizes else 0

        return {
            "step": step,
            "removed_node": removed_node,
            "initial_degree": (
                initial_degrees[removed_node] if step > 0 else None
            ),
            "degree_before_removal": degree_before,
            "removed_nodes": step,
            "removed_pct": (
                100.0 * step / initial_nodes if initial_nodes else 0.0
            ),
            "remaining_nodes": remaining_nodes,
            "remaining_edges": active_graph.number_of_edges(),
            "n_components": n_components,
            "component_sizes": sizes,
            "largest_component": largest,
            "largest_original_pct": (
                100.0 * largest / initial_nodes if initial_nodes else None
            ),
            "largest_remaining_pct": (
                100.0 * largest / remaining_nodes if remaining_nodes else None
            ),
            "isolated_nodes": sum(
                degree == 0 for _, degree in active_graph.degree()
            ),
            "component_change": (
                n_components - previous_components
                if previous_components is not None
                else None
            ),
        }

    # Paso 0: grafo intacto.
    state = record_state(
        step=0,
        removed_node=None,
        degree_before=None,
        previous_components=None,
    )
    results["attack_log"].append(state)

    # Pasos 1..N: eliminar el nodo y todas sus aristas.
    for step, node in enumerate(removal_order, start=1):
        degree_before = active_graph.degree(node)
        previous_components = state["n_components"]

        active_graph.remove_node(node)

        state = record_state(
            step=step,
            removed_node=node,
            degree_before=degree_before,
            previous_components=previous_components,
        )
        results["attack_log"].append(state)

    logger.info(
        "EXITO ✅. Ataque registrado: %s eliminaciones y %s estados.",
        initial_nodes,
        len(results["attack_log"]),
    )

    return results
# endregion







# region resilience_analysis
def resilience_analysis(
    g: nx.Graph,
    seed: int | None = None,
    previous_winners: dict | None = None,
) -> dict:
    """Analyze topological fragmentation of a graph against theoretical reference models.

    Evaluates node-removal resilience curves under targeted (``'attack'``) and
    random (``'random'``) strategies for both the empirical graph and a
    catalog of theoretical reference models, ranking models by normalized curve
    distance and flagging exact first-place ties.

    Does not generate plots or mutate the input graph.

    Parameters
    ----------
    g : nx.Graph
        Undirected Natural Visibility Graph to analyze. Nodes may store a
        ``'timestamp'`` attribute used to extract the covered time span.
    seed : int | None, default=None
        Random seed forwarded to ``simulate_resilience_profile`` for
        reproducible stochastic node removals.
    previous_winners : dict | None, default=None
        Optional mapping of previous-window winning model names per strategy,
        reserved for continuity tracking across rolling windows.

    Returns
    -------
    dict
        Dictionary containing series identity, time span, empirical network and
        theoretical model metrics, fragmentation curves, sorted curve distances,
        closest model per strategy (``'best_models'``), and tie notes
        (``'notes'``). Returns an empty dictionary if ``g`` has no nodes.
    """
    logger.info("Iniciando análisis comparado de resiliencia")

    if graph.get_node_count(g) == 0:
        logger.warning("El grafo recibido está vacío.")
        return {}

    series_name = graph.get_graph_name(g)

    # 1. Empirical graph time span from standardized node timestamps.
    timestamps = [
        pd.to_datetime(attrs["timestamp"])
        for _, attrs in g.nodes(data=True)
        if "timestamp" in attrs and pd.notna(attrs["timestamp"])
    ]

    start_date = min(timestamps) if timestamps else None
    end_date = max(timestamps) if timestamps else None

    # 2. Lightweight structural summary compatible with both NVG and pure NetworkX models.
    def basic_metrics(net: nx.Graph) -> dict:
        n_nodes = net.number_of_nodes()
        return {
            "node_count": n_nodes,
            "edge_count": net.number_of_edges(),
            "average_degree": (
                sum(deg for _, deg in net.degree()) / n_nodes
                if n_nodes > 0
                else 0.0
            ),
            "density": float(nx.density(net)),
        }

    # 3. Generate theoretical reference models once.
    models = generate_theoretical_models(g)

    results = {
        "series_name": series_name,
        "start_date": start_date,
        "end_date": end_date,
        "removal_seed": seed,

        "empirical_network": {
            "graph": g,
            "metrics": basic_metrics(g),
            "curves": {},
        },

        "models": {},

        "sorted_distances": {
            "attack": [],
            "random": [],
        },

        "best_models": {
            "attack": None,
            "random": None,
        },

        "notes": {
            "attack": "",
            "random": "",
        },
    }

    # Preserve all theoretical models returned by the generator.
    for model_name, model_graph in models.items():
        k_match = re.search(r"[kK]=(\d+)", model_name)
        p_match = re.search(r"[pP]=([\d\.]+)", model_name)

        results["models"][model_name] = {
            "graph": model_graph,
            "metrics": basic_metrics(model_graph),
            "parameters": {
                "k": int(k_match.group(1)) if k_match else None,
                "p": float(p_match.group(1)) if p_match else None,
            },
            "curves": {},
            "distances": {},
        }

    if not models:
        logger.warning(
            "No se generaron modelos teóricos. "
            "Solo se calcularán las curvas empíricas."
        )

    # 4. Compute resilience profiles and distances per strategy.
    for strategy in ("attack", "random"):
        empirical_curve = simulate_resilience_profile(
            g,
            sorting_strategy=strategy,
            seed=seed,
        )

        results["empirical_network"]["curves"][strategy] = empirical_curve

        distances = []

        for model_name, model_data in results["models"].items():
            model_curve = simulate_resilience_profile(
                model_data["graph"],
                sorting_strategy=strategy,
                seed=seed,
            )

            distance = float(
                compute_normalized_curve_distance(
                    empirical_curve,
                    model_curve,
                )
            )

            model_data["curves"][strategy] = model_curve
            model_data["distances"][strategy] = distance

            distances.append({
                "model": model_name,
                "distance": distance,
            })

        # Stable sort: preserve catalog order on exact ties.
        sorted_distances = sorted(
            distances,
            key=lambda item: item["distance"],
        )

        results["sorted_distances"][strategy] = sorted_distances

        if not sorted_distances:
            continue

        # 5. Select the closest model from the sorted ranking.
        results["best_models"][strategy] = sorted_distances[0].copy()

        # 6. Flag exact first-place ties only.
        min_distance = sorted_distances[0]["distance"]

        tied_models = [
            item["model"]
            for item in sorted_distances
            if item["distance"] == min_distance
        ]

        if len(tied_models) > 1:
            results["notes"][strategy] = "tie"
            logger.warning(
                "Empate | Primer Lugar | d=%.4f | %s | %s",
                min_distance,
                ", ".join(tied_models),
                sorted_distances[0]["model"],
            )

    logger.info(
        "EXITO ✅. Resiliencia calculada para '%s': "
        "red empírica y %s modelos teóricos.",
        series_name,
        len(models),
    )

    return results
# endregion





