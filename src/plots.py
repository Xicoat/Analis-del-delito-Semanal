# region Libraries
import logging
import re

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import networkx as nx
import math

import plotly.graph_objects as go

from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.ticker import PercentFormatter

import re
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

logger = logging.getLogger("Delitos")

from . import reports
# endregion







# region plot_histogram
def plot_histogram(node_metrics: dict) -> go.Figure:
    """Plot weekly incident counts and node identifiers from a node metrics report.

    Renders a white-background Plotly bar chart of weekly frequencies overlaid
    with circular node-ID markers above each bar. All rendered chart titles,
    axis labels, and trace names are formatted in Spanish.

    Parameters
    ----------
    node_metrics : dict
        Dictionary returned by ``metrics.compute_node_metrics``, containing:

        graph_name : str
            Name of the analyzed graph/series (falls back to ``'series_name'``).
        data : pd.DataFrame
            DataFrame with columns ``['node_id', 'timestamp', 'frequency']``.

    Returns
    -------
    go.Figure
        Configured Plotly figure. Returns an empty ``go.Figure()`` if
        ``node_metrics['data']`` is empty.

    Raises
    ------
    ValueError
        If required columns are missing, node IDs are null or duplicated,
        timestamps cannot be parsed, or frequencies are non-finite or negative.
    """
    logger.info("Preparando visualización de conteos semanales")

    df_node_metrics = node_metrics["data"]
    series_name = node_metrics.get(
        "graph_name",
        node_metrics.get("series_name", node_metrics.get("crime_name", "")),
    )

    if df_node_metrics.empty:
        logger.warning("El reporte de métricas por nodo está vacío.")
        return go.Figure()

    # 1. Extract and validate required columns from the standardized node report.
    date_col = "timestamp"
    count_col = "frequency"

    required_cols = ["node_id", date_col, count_col]
    missing = set(required_cols) - set(df_node_metrics.columns)

    if missing:
        raise ValueError(
            f"Faltan columnas requeridas: {sorted(missing)}"
        )

    df_plot = df_node_metrics[required_cols].copy()

    if df_plot["node_id"].isna().any():
        raise ValueError("Hay identificadores de nodo faltantes.")

    if df_plot["node_id"].duplicated().any():
        raise ValueError("Hay identificadores de nodo duplicados.")

    df_plot[date_col] = pd.to_datetime(
        df_plot[date_col],
        errors="coerce",
    )

    if df_plot[date_col].isna().any():
        raise ValueError(
            "Hay fechas faltantes o inválidas en el reporte por nodo."
        )

    df_plot[count_col] = pd.to_numeric(
        df_plot[count_col],
        errors="raise",
    )

    counts_array = df_plot[count_col].to_numpy(
        dtype=float,
        na_value=np.nan,
    )

    if (
        not np.isfinite(counts_array).all()
        or (counts_array < 0).any()
    ):
        raise ValueError(
            "Las cantidades deben ser valores finitos no negativos."
        )

    # Preserve sequential ordering by original node identifier.
    df_plot = df_plot.sort_values("node_id")

    nodes = df_plot["node_id"].tolist()
    dates = df_plot[date_col].tolist()
    counts = df_plot[count_col].tolist()

    max_val = max(counts)
    offset = max_val * 0.05 if max_val > 0 else 1

    fig = go.Figure()

    # 2. Weekly incident count bars (labels in Spanish).
    fig.add_trace(
        go.Bar(
            x=dates,
            y=counts,
            marker=dict(
                color="white",
                line=dict(color="black", width=1),
            ),
            name="Delitos",
        )
    )

    # 3. Node ID markers above each bar.
    y_markers = [val + offset for val in counts]
    text_labels = [str(n) for n in nodes]

    fig.add_trace(
        go.Scatter(
            x=dates,
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

    # 4. Titles and axes layout (in Spanish).
    formatted_series = str(series_name).title()
    min_date = min(dates)
    max_date = max(dates)

    title_str = (
        "Asignación de Nodos en la Serie de Tiempo Semanal de"
        f"<br><b>{formatted_series}</b><br>"
        f"<sub>Periodo: {min_date.strftime('%Y-%m-%d')} "
        f"a {max_date.strftime('%Y-%m-%d')}</sub>"
    )

    visible_ticks = dates[::16]

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

    logger.info("EXITO ✅.")
    return fig
# endregion






# region plot_visibility_graph_from_network
def plot_visibility_graph_from_network(g: nx.Graph) -> go.Figure:
    """Plot the weekly time series bars, node IDs, and visibility edges of an NVG.

    Renders a white-background Plotly bar chart of weekly frequencies overlaid
    with red visibility edges, vertex cusp markers, and circular node-ID
    badges above each bar. All rendered chart titles, axis labels, and trace
    names are formatted in Spanish.

    Parameters
    ----------
    g : nx.Graph
        Natural Visibility Graph whose nodes store ``'timestamp'`` and
        ``'frequency'`` attributes, and whose ``g.graph`` metadata optionally
        contains ``'series_name'``.

    Returns
    -------
    go.Figure
        Configured Plotly figure. Returns an empty ``go.Figure()`` if ``g``
        contains no nodes.
    """
    logger.info("Preparando visualización del grafo de visibilidad")

    if g.number_of_nodes() == 0:
        logger.warning("El grafo recibido no contiene nodos.")
        return go.Figure()

    series_name = (
        g.graph.get("series_name")
        or g.graph.get("crime_name")
        or g.graph.get("name", "")
    )

    # 1. Existing node coordinates.
    nodes = sorted(g.nodes())
    x_bars = [g.nodes[i]["timestamp"] for i in nodes]
    y_bars = [g.nodes[i]["frequency"] for i in nodes]

    max_val = max(y_bars)
    offset = max_val * 0.05 if max_val > 0 else 1

    # 2. Precomputed visibility edge segments.
    edge_x = []
    edge_y = []

    for u, v in g.edges():
        edge_x.extend([
            g.nodes[u]["timestamp"],
            g.nodes[v]["timestamp"],
            None,
        ])
        edge_y.extend([
            g.nodes[u]["frequency"],
            g.nodes[v]["frequency"],
            None,
        ])

    fig = go.Figure()

    # Trace 1: Weekly time series bars (labels in Spanish).
    fig.add_trace(
        go.Bar(
            x=x_bars,
            y=y_bars,
            marker=dict(
                color="white",
                line=dict(color="black", width=1),
            ),
            name="Serie Temporal",
        )
    )

    # Trace 2: Visibility network edges.
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

    # Trace 3: Cusp markers at bar tops.
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

    # Trace 4: Numbered node badges above bars.
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

    # 3. Presentation layout (in Spanish).
    title_text = "Red de Visibilidad Natural (NVG)"

    if series_name:
        title_text += f"<br><b>{str(series_name).title()}</b>"

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

    logger.info("EXITO ✅.")
    return fig
# endregion






# region plot_network
def plot_network(g: nx.Graph, metrics: dict) -> None:
    """Plot the visibility network highlighting precomputed center, diameter, and radius.

    Renders a spring-layout Matplotlib visualization of the graph, overlaying
    diameter edges in red, radius edges in green, and central nodes in yellow,
    alongside a legend summarizing key topological metrics. All rendered chart
    titles and legend labels are formatted in Spanish.

    Parameters
    ----------
    g : nx.Graph
        NetworkX graph whose nodes optionally store ``'timestamp'`` attributes.
    metrics : dict
        Dictionary returned by ``metrics.compute_graph_metrics``, containing
        ``'graph_name'``, ``'density'``, and optional keys
        ``'diameter_edges'``, ``'radius_edges'``, ``'center'``,
        ``'diameter'``, and ``'radius'``.

    Returns
    -------
    None
        Displays the figure via ``plt.show()``.
    """

    logger.info("Preparando visualización de la red")

    if g.number_of_nodes() == 0:
        logger.warning("El grafo recibido está vacío.")
        return

    graph_name = metrics["graph_name"]

    # 1. Rango temporal obtenido de los nodos.
    dates = [
        pd.to_datetime(data["timestamp"])
        for _, data in g.nodes(data=True)
        if "timestamp" in data
    ]

    date_range = (
        f"({min(dates).strftime('%Y-%m-%d')} – "
        f"{max(dates).strftime('%Y-%m-%d')})"
        if dates
        else ""
    )

    chart_title = (
        "Red de Visibilidad Semanal con Métricas Clave\n"
        f"{graph_name.title()}\n{date_range}"
    )

    # 2. Lienzo y disposición visual de los nodos.
    fig, ax = plt.subplots(figsize=(15, 8))
    pos = nx.spring_layout(g, seed=42)

    # 3. Capas base: conexiones y nodos existentes.
    nx.draw_networkx_edges(
        g,
        pos,
        edge_color="royalblue",
        width=1.0,
        alpha=0.6,
        ax=ax,
    )

    nx.draw_networkx_nodes(
        g,
        pos,
        node_color="white",
        node_size=180,
        edgecolors="black",
        linewidths=0.6,
        ax=ax,
    )

    # 4. Resaltar caminos y centro previamente calculados.
    if metrics.get("diameter_edges"):
        nx.draw_networkx_edges(
            g,
            pos,
            edgelist=metrics["diameter_edges"],
            edge_color="red",
            width=2.5,
            ax=ax,
        )

    if metrics.get("radius_edges"):
        nx.draw_networkx_edges(
            g,
            pos,
            edgelist=metrics["radius_edges"],
            edge_color="green",
            width=2.5,
            ax=ax,
        )

    if metrics.get("center"):
        nx.draw_networkx_nodes(
            g,
            pos,
            nodelist=metrics["center"],
            node_color="yellow",
            node_size=220,
            edgecolors="black",
            ax=ax,
        )

    nx.draw_networkx_labels(
        g,
        pos,
        font_size=5.5,
        font_weight="bold",
        ax=ax,
    )

    # 5. Leyenda con las métricas existentes.
    legend_elements = [
        Line2D(
            [0],
            [0],
            color="w",
            label=(
                "Densidad Empírica: "
                f"{metrics['density']:.4f}"
            ),
        )
    ]

    if metrics.get("center"):
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=f"Centro: {metrics['center']}",
                markerfacecolor="yellow",
                markersize=10,
                markeredgecolor="black",
            )
        )

    if metrics.get("diameter") is not None:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color="red",
                lw=3,
                label=f"Diámetro: {metrics['diameter']}",
            )
        )

    if metrics.get("radius") is not None:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color="green",
                lw=3,
                label=f"Radio: {metrics['radius']}",
            )
        )

    ax.legend(
        handles=legend_elements,
        loc="best",
        facecolor="white",
        framealpha=0.8,
        fontsize=10,
    )

    ax.set_title(chart_title, fontsize=16, pad=15)
    ax.set_axis_off()

    plt.show()
    logger.info("Exito.")
# endregion






# region plot_degree_distribution_sorted
def plot_degree_distribution_sorted(
    graph_metrics: dict,
    start_date=None,
    end_date=None,
) -> None:
    """Plot precomputed node degrees sorted in descending order of connectivity.

    Renders a white-background bar chart where each bar represents a node's
    degree ordered from highest to lowest connectivity, including zero-degree
    nodes. All rendered chart titles and axis labels are formatted in Spanish.

    Parameters
    ----------
    graph_metrics : dict
        Dictionary returned by ``metrics.compute_graph_metrics``, containing
        ``'node_count'``, ``'graph_name'``, and ``'degrees'``.
    start_date : str | pd.Timestamp | None, default=None
        Optional start date of the analyzed window or series.
    end_date : str | pd.Timestamp | None, default=None
        Optional end date of the analyzed window or series.

    Returns
    -------
    None
        Displays the figure via ``plt.show()``.
    """

    logger.info("Preparando gráfico de distribución de grados")

    if graph_metrics["node_count"] == 0:
        logger.warning("El grafo recibido está vacío.")
        return

    graph_name = graph_metrics["graph_name"]

    # 1. Ordenar los grados existentes, incluidos los ceros.
    sorted_degrees = sorted(
        graph_metrics["degrees"],
        reverse=True,
    )

    # 2. Rango temporal para el título.
    if start_date is not None and end_date is not None:
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)

        if pd.notna(start_ts) and pd.notna(end_ts):
            date_range = (
                f"({start_ts.strftime('%Y-%m-%d')} – "
                f"{end_ts.strftime('%Y-%m-%d')})"
            )
        else:
            date_range = "(Periodo Desconocido)"
    else:
        date_range = "(Periodo Desconocido)"

    title = (
        "Distribución de Grados en la Red de Visibilidad Semanal\n"
        f"{graph_name.title()}\n{date_range}"
    )

    # 3. Lienzo.
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=100)
    fig.patch.set_facecolor("white")
    ax.patch.set_facecolor("white")

    # 4. Barras.
    x_pos = np.arange(len(sorted_degrees))

    ax.bar(
        x_pos,
        sorted_degrees,
        color="white",
        edgecolor="black",
        linewidth=1.2,
    )

    # 5. Estilos y formato.
    ax.set_title(
        title,
        fontsize=13,
        pad=15,
        color="black",
    )
    ax.set_ylabel(
        "Grado del nodo (Conexiones)",
        fontsize=11,
        color="black",
    )
    ax.set_xlabel(
        "Grados de los nodos ordenados por conectividad",
        fontsize=11,
        color="black",
    )

    max_degree = max(sorted_degrees) if sorted_degrees else 0
    ax.set_ylim(0, max_degree + max(1, int(max_degree * 0.08)))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")

    ax.tick_params(
        axis="both",
        direction="out",
        colors="black",
        labelsize=9,
    )
    ax.yaxis.grid(
        True,
        linestyle="--",
        alpha=0.3,
        color="gray",
    )

    fig.tight_layout()
    plt.show()

    logger.info("Exito.")
# endregion





# region plot_power_law_fit
def plot_power_law_fit(
    distribution_metrics: dict,
    start_date=None,
    end_date=None,
) -> None:
    """Plot the precomputed discrete power-law PDF fit on a log-log scale.

    Renders the empirical degree PDF alongside the fitted Clauset-Shalizi-Newman
    (CSN) power-law curve, parameter uncertainty envelopes, and an annotation
    box summarizing fit parameters and log-likelihood ratio comparison. All
    rendered chart titles, labels, and annotations are formatted in Spanish.

    Parameters
    ----------
    distribution_metrics : dict
        Dictionary returned by ``distributions.compute_distribution_metrics``,
        containing ``'status'``, ``'series_name'``, ``'fit'``,
        ``'positive_degrees'``, ``'sigma'``, ``'xmin'``,
        ``'gamma_powerlaw'``, ``'tail_node_count'``, ``'R_log'``, and
        ``'p_comp'``.
    start_date : str | pd.Timestamp | None, default=None
        Optional start date of the analyzed window or series.
    end_date : str | pd.Timestamp | None, default=None
        Optional end date of the analyzed window or series.

    Returns
    -------
    None
        Displays the figure via ``plt.show()``.
    """

    logger.info("Iniciando visualización del ajuste de Ley de Potencia")

    status = distribution_metrics["status"]

    if status != "CALCULADO":
        logger.warning(
            "No se puede generar la gráfica. Estado: %s | Motivo: %s",
            status,
            distribution_metrics.get("error"),
        )
        return

    # 1. Recuperar el nombre y los resultados previamente calculados.
    series_name = distribution_metrics["series_name"]
    fit = distribution_metrics["fit"]
    data = distribution_metrics["positive_degrees"]

    sigma_val = distribution_metrics["sigma"]
    xmin_val = distribution_metrics["xmin"]
    gamma_val = distribution_metrics["gamma_powerlaw"]
    tail_nodes = distribution_metrics["tail_node_count"]

    R = distribution_metrics["R_log"]
    p_value_PL = distribution_metrics["p_comp"]

    # Conservar la clasificación visual del plot original.
    res_log = "Power Law" if R > 0 else "Alternativa"

    # 2. Rango temporal para el título.
    if start_date is not None and end_date is not None:
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)

        if pd.notna(start_ts) and pd.notna(end_ts):
            date_range = (
                f"({start_ts.strftime('%Y-%m-%d')} – "
                f"{end_ts.strftime('%Y-%m-%d')})"
            )
        else:
            date_range = "(Periodo Desconocido)"
    else:
        date_range = "(Periodo Desconocido)"

    title = (
        "Distribución de Grado P(k) y Ajuste de Ley de Potencia (CSN)\n"
        f"Red de Visibilidad Semanal - {series_name.title()}\n"
        f"{date_range}"
    )

    # 3. Construcción del gráfico.
    fig, ax = plt.subplots(figsize=(7, 5), dpi=100)

    # Conservar las curvas auxiliares del plot legacy.
    # Este cálculo es para dibujar; no realiza nuevos ajustes.
    x_space = np.linspace(xmin_val, data.max(), 100)

    for s in np.linspace(-2, 2, 50):
        gamma_var = gamma_val + (s * sigma_val)

        y_var = (
            (gamma_var - 1)
            * (xmin_val ** (gamma_var - 1))
            * (x_space ** -gamma_var)
        )

        ax.plot(
            x_space,
            y_var,
            color="royalblue",
            alpha=0.15,
            linewidth=0.6,
            zorder=1,
        )

    # Representar el objeto de ajuste existente.
    fit.plot_pdf(
        ax=ax,
        color="black",
        marker="o",
        linestyle="None",
        markersize=5,
        label="Datos Empíricos",
    )

    fit.power_law.plot_pdf(
        ax=ax,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Ajuste Óptimo Power Law",
    )

    # 4. Conservar las anotaciones del plot legacy.
    stats_text = (
        r"$\mathbf{Métricas\ del\ Ajuste:}$" + "\n"
        f"Gamma ($\\gamma$): {gamma_val:.4f}\n"
        f"Xmin: {xmin_val:.2f}\n"
        f"Nodos (k >= Xmin): {tail_nodes} / {len(data)}\n\n"
        r"$\mathbf{Bondad\ de\ Ajuste:}$" + "\n"
        f"p-value (PL): {p_value_PL:.4f}\n\n"
        r"$\mathbf{Comparación\ (R):}$" + "\n"
        f"R: {R:.4f} ({res_log})"
    )

    ax.text(
        0.05,
        0.05,
        stats_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.8,
        ),
    )

    ax.set_title(title, fontsize=11, pad=15)
    ax.set_xlabel("Grado (k)", fontsize=10)
    ax.set_ylabel("P(k)", fontsize=10)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    plt.show()

    logger.info("Exito.")
# endregion






# region plot_model_comparison
def plot_model_comparison(
    distribution_metrics: dict,
    start_date=None,
    end_date=None,
) -> None:
    """Plot fitted tail PDFs and display their Euclidean distances to empirical data.

    Renders empirical tail probabilities alongside fitted power-law,
    log-normal, and exponential PDFs on a log-log scale, displaying each
    model's precomputed Euclidean distance in the legend. All rendered chart
    titles, axis labels, and legends are formatted in Spanish.

    Parameters
    ----------
    distribution_metrics : dict
        Dictionary returned by ``distributions.compute_distribution_metrics``,
        containing ``'status'``, ``'series_name'``, ``'k_eval'``,
        ``'tail_probabilities'``, ``'model_probabilities'``,
        ``'dist_eucl_pl'``, ``'dist_eucl_lognormal'``, and
        ``'dist_eucl_exp'``.
    start_date : str | pd.Timestamp | None, default=None
        Optional start date of the analyzed window or series.
    end_date : str | pd.Timestamp | None, default=None
        Optional end date of the analyzed window or series.

    Returns
    -------
    None
        Displays the figure via ``plt.show()``.
    """

    logger.info("Iniciando visualización de comparación multimodelo")

    status = distribution_metrics["status"]

    if status != "CALCULADO":
        logger.warning(
            "No se puede generar la gráfica. Estado: %s | Motivo: %s",
            status,
            distribution_metrics.get("error"),
        )
        return

    # 1. Recuperar el nombre y los vectores previamente calculados.
    series_name = distribution_metrics["series_name"]

    k_eval = distribution_metrics["k_eval"]
    y_emp_eval = distribution_metrics["tail_probabilities"]

    model_probabilities = distribution_metrics[
        "model_probabilities"
    ]

    y_pl = model_probabilities["power_law"]
    y_lognorm = model_probabilities["lognormal"]
    y_exp = model_probabilities["exponential"]

    # 2. Recuperar las distancias previamente calculadas.
    dist_pl = distribution_metrics["dist_eucl_pl"]
    dist_lognorm = distribution_metrics["dist_eucl_lognormal"]
    dist_exp = distribution_metrics["dist_eucl_exp"]

    # 3. Rango temporal para el título.
    if start_date is not None and end_date is not None:
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)

        if pd.notna(start_ts) and pd.notna(end_ts):
            date_range = (
                f"({start_ts.strftime('%Y-%m-%d')} – "
                f"{end_ts.strftime('%Y-%m-%d')})"
            )
        else:
            date_range = "(Periodo Desconocido)"
    else:
        date_range = "(Periodo Desconocido)"

    title = (
        "Comparación de Modelos Teóricos y Distancia Euclidiana\n"
        f"Red de Visibilidad Semanal - {series_name.title()}\n"
        f"{date_range}"
    )

    # 4. Renderizado en escala log-log.
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    ax.plot(
        k_eval,
        y_emp_eval,
        "ko",
        label=r"Datos Empíricos ($k \geq x_{min}$)",
        markersize=5,
    )
    ax.plot(
        k_eval,
        y_pl,
        "r--",
        linewidth=2,
        label=f"Power Law (D = {dist_pl:.5f})",
    )
    ax.plot(
        k_eval,
        y_lognorm,
        "b-.",
        linewidth=2,
        label=f"Log-Normal (D = {dist_lognorm:.5f})",
    )
    ax.plot(
        k_eval,
        y_exp,
        "g:",
        linewidth=2,
        label=f"Exponencial (D = {dist_exp:.5f})",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Grado (k)", fontsize=10)
    ax.set_ylabel("P(k)", fontsize=10)
    ax.set_title(title, fontsize=11, pad=12)
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)

    fig.tight_layout()
    plt.show()

    logger.info("EXITO.")
# endregion







# region plot_csn_analysis
def plot_csn_analysis(
    distribution_metrics: dict,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> None:
    """Plot precomputed Clauset-Shalizi-Newman (CSN) power-law analysis results.

    Renders the empirical degree PDF and fitted power-law curve on a log-log
    scale alongside a summary box reporting exponent estimates, tail coverage,
    Kolmogorov-Smirnov distance, and log-likelihood ratio comparison verdict.
    All rendered chart titles, labels, and annotations are formatted in Spanish.

    Parameters
    ----------
    distribution_metrics : dict
        Dictionary returned by ``distributions.compute_distribution_metrics``,
        containing ``'status'``, ``'series_name'``, ``'fit'``,
        ``'gamma_powerlaw'``, ``'sigma'``, ``'xmin'``, ``'ks_distance'``,
        ``'n_positive_degrees'``, ``'tail_node_count'``,
        ``'tail_positive_pct'``, ``'R_log'``, ``'p_comp'``, and
        ``'verdict'``.
    start_date : str | pd.Timestamp | None, default=None
        Optional start date of the analyzed window or series.
    end_date : str | pd.Timestamp | None, default=None
        Optional end date of the analyzed window or series.

    Returns
    -------
    None
        Displays the figure via ``plt.show()``.
    """
    logger.info("Iniciando visualización del análisis CSN")

    status = distribution_metrics["status"]

    if status != "CALCULADO":
        logger.warning(
            "No se puede generar la gráfica. Estado: %s | Motivo: %s",
            status,
            distribution_metrics.get("error"),
        )
        return

    # 1. Recuperar el nombre y los resultados previamente calculados.
    series_name = distribution_metrics["series_name"]
    fit = distribution_metrics["fit"]

    gamma_val = distribution_metrics["gamma_powerlaw"]
    sigma_val = distribution_metrics["sigma"]
    xmin_val = distribution_metrics["xmin"]
    ks_distance = distribution_metrics["ks_distance"]

    total_nodes = distribution_metrics["n_positive_degrees"]
    tail_nodes = distribution_metrics["tail_node_count"]
    tail_pct = distribution_metrics["tail_positive_pct"]

    R_log = distribution_metrics["R_log"]
    p_comp = distribution_metrics["p_comp"]
    verdict = distribution_metrics["verdict"]

    # 2. Rango temporal para el título.
    if start_date is not None and end_date is not None:
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)

        if pd.notna(start_ts) and pd.notna(end_ts):
            date_range = (
                f"({start_ts.strftime('%Y-%m-%d')} – "
                f"{end_ts.strftime('%Y-%m-%d')})"
            )
        else:
            date_range = "(Periodo Desconocido)"
    else:
        date_range = "(Periodo Desconocido)"

    chart_title = (
        "Distribución de Grado P(k) y Ajuste de Ley de Potencia (CSN)\n"
        f"Red de Visibilidad Semanal - {series_name.title()}\n"
        f"{date_range}"
    )

    # 3. Representar el objeto de ajuste existente.
    fig, ax = plt.subplots(figsize=(7, 5), dpi=100)

    fit.plot_pdf(
        ax=ax,
        color="black",
        marker="o",
        linestyle="None",
        markersize=5,
        label="Datos Empíricos",
    )

    fit.power_law.plot_pdf(
        ax=ax,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Ajuste Óptimo Power Law",
    )

    # 4. Conservar las anotaciones del plot legacy.
    stats_text = (
        r"$\mathbf{Métricas\ del\ Ajuste:}$" + "\n"
        f"Exponente Gamma ($\\gamma$): "
        f"{gamma_val:.4f} $\\pm$ {sigma_val:.4f}\n"
        f"Corte Inferior ($X_{{min}}$): {xmin_val:.2f}\n"
        f"Nodos ($k \\geq X_{{min}}$): "
        f"{tail_nodes} / {total_nodes} "
        f"({tail_pct:.1f}%)\n\n"
        r"$\mathbf{Validación\ Absoluta:}$" + "\n"
        f"Distancia KS (D): {ks_distance:.4f}\n\n"
        r"$\mathbf{Comparación\ Relativa:}$" + "\n"
        f"R (Log-Likelihood): {R_log:.4f}\n"
        f"p-value (Comp): {p_comp:.4f}\n"
        f"Ganador: {verdict}"
    )

    ax.text(
        0.05,
        0.05,
        stats_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.9,
        ),
    )

    ax.set_title(chart_title, fontsize=11, pad=15)
    ax.set_xlabel("Grado (k)", fontsize=10)
    ax.set_ylabel("P(k)", fontsize=10)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    plt.show()

    logger.info("EXITO.")
# endregion






# region plot_closest_resilience_models DEPRECATED
def plot_closest_resilience_models_DEPRECATED(
    df_resilience_test_report: pd.DataFrame,
) -> None:
    """Grafica las curvas guardadas de la red y sus modelos más cercanos."""

    logger.info("Preparando contraste de modelos de resiliencia")

    if df_resilience_test_report.empty:
        logger.warning("El reporte de resiliencia está vacío.")
        return

    columnas_requeridas = {
        "experiment_id",
        "window_id",
        "crime_name",
        "start_date",
        "end_date",
        "estrategia",
        "modelo",
        "distancia",
        "curva_empirica",
        "curva_modelo",
    }

    faltantes = columnas_requeridas - set(
        df_resilience_test_report.columns
    )

    if faltantes:
        raise ValueError(
            f"Faltan columnas requeridas: {sorted(faltantes)}"
        )

    df_reporte = df_resilience_test_report.copy()

    # 1. Comprobar que corresponda a un experimento y una ventana.
    metadatos = {}

    for campo in (
        "experiment_id",
        "window_id",
        "crime_name",
        "start_date",
        "end_date",
    ):
        valores = df_reporte[campo].drop_duplicates()

        if len(valores) != 1 or pd.isna(valores.iloc[0]):
            raise ValueError(
                f"'{campo}' debe tener un único valor válido. "
                "Selecciona un solo experimento y una sola ventana."
            )

        metadatos[campo] = valores.iloc[0]

    crime_name = metadatos["crime_name"]

    fecha_inicio = pd.Timestamp(metadatos["start_date"])
    fecha_fin = pd.Timestamp(metadatos["end_date"])

    rango_fechas = (
        f"({fecha_inicio.strftime('%Y-%m-%d')} – "
        f"{fecha_fin.strftime('%Y-%m-%d')})"
    )

    # El reporte utiliza 'falla'; la simulación utiliza 'aleatorio'.
    estrategias = [
        ("Ataque Dirigido (Grado Alto)", "ataque"),
        ("Falla Aleatoria", "falla"),
    ]

    seleccionados = []

    # 2. Seleccionar los modelos mediante las distancias guardadas.
    for nombre_estrategia, estrategia_key in estrategias:
        df_panel = df_reporte.loc[
            df_reporte["estrategia"].eq(estrategia_key)
        ].copy()

        if df_panel.empty:
            logger.warning(
                "No hay resultados para %s.",
                nombre_estrategia,
            )
            return

        if df_panel["modelo"].isna().any():
            raise ValueError("Hay nombres de modelos faltantes.")

        if df_panel.duplicated("modelo").any():
            raise ValueError(
                f"Hay modelos duplicados para '{estrategia_key}'."
            )

        df_panel["distancia"] = pd.to_numeric(
            df_panel["distancia"],
            errors="raise",
        )

        if not np.isfinite(
            df_panel["distancia"].to_numpy(dtype=float)
        ).all():
            raise ValueError(
                f"Hay distancias no finitas para '{estrategia_key}'."
            )

        # En empates, conservar el orden de las filas recibidas.
        mejor = (
            df_panel.sort_values("distancia", kind="stable")
            .iloc[0]
        )

        curva_base = np.asarray(
            mejor["curva_empirica"],
            dtype=float,
        )
        mejor_curva = np.asarray(
            mejor["curva_modelo"],
            dtype=float,
        )

        for nombre_curva, curva in (
            ("curva_empirica", curva_base),
            ("curva_modelo", mejor_curva),
        ):
            if (
                curva.ndim != 1
                or curva.size == 0
                or not np.isfinite(curva).all()
            ):
                raise ValueError(
                    f"'{nombre_curva}' no contiene una curva válida "
                    f"para '{estrategia_key}'."
                )

        seleccionados.append({
            "estrategia": nombre_estrategia,
            "modelo": str(mejor["modelo"]),
            "distancia": float(mejor["distancia"]),
            "curva_empirica": curva_base,
            "curva_modelo": mejor_curva,
        })

    # 3. Construir la gráfica.
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 6),
        constrained_layout=True,
    )

    for ax, seleccionado in zip(axes, seleccionados):
        mejor_nombre = seleccionado["modelo"]
        mejor_d = seleccionado["distancia"]

        ax.plot(
            seleccionado["curva_empirica"],
            label=f"Red Empírica: {crime_name.title()}",
            linewidth=2.0,
            color="royalblue",
            zorder=2,
        )

        ax.plot(
            seleccionado["curva_modelo"],
            label=f"Más cercano: {mejor_nombre}\n(D={mejor_d:.4f})",
            linestyle="-",
            linewidth=3.5,
            color="crimson",
            alpha=1.0,
            zorder=10,
        )

        ax.set_title(
            seleccionado["estrategia"],
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Nodos eliminados (Semanas)", fontsize=10)
        ax.set_ylabel("Componentes conexos", fontsize=10)
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(
        "Contraste de Desmantelamiento vs. Modelo Sintético Más Cercano\n"
        f"Red de Visibilidad Semanal - {crime_name.title()}\n"
        f"{rango_fechas}",
        fontsize=13,
        fontweight="bold",
    )

    plt.show()

    logger.info("EXITO.")
# endregion






# region plot_rolling_resilience_evolution
# [VISUALIZACIÓN] Evolución longitudinal de la distancia estructural D
# INPUT: df_resilience_test_report (pd.DataFrame), crime_name (str)
# OUTPUT: None

def plot_rolling_resilience_evolution(
    df_resilience_test_report: pd.DataFrame,
    crime_name: str,
) -> None:
    """Grafica la distancia mínima y el modelo más cercano por ventana."""

    logger.info("Iniciando visualización temporal de distancias de resiliencia")

    if df_resilience_test_report.empty:
        logger.warning("El reporte de resiliencia está vacío.")
        return

    columnas_requeridas = {
        "experiment_id",
        "crime_name",
        "window_size",
        "step_size",
        "window_id",
        "end_date",
        "estrategia",
        "modelo",
        "distancia",
    }

    faltantes = columnas_requeridas - set(
        df_resilience_test_report.columns
    )

    if faltantes:
        raise ValueError(
            f"Faltan columnas requeridas: {sorted(faltantes)}"
        )

    df_plot = df_resilience_test_report.copy()

    # 1. Validar un único experimento y sus parámetros.
    metadatos = {}

    for campo in (
        "experiment_id",
        "crime_name",
        "window_size",
        "step_size",
    ):
        valores = df_plot[campo].drop_duplicates()

        if len(valores) != 1 or pd.isna(valores.iloc[0]):
            raise ValueError(
                f"'{campo}' debe tener un único valor válido. "
                "Selecciona un solo experimento."
            )

        metadatos[campo] = valores.iloc[0]

    if metadatos["crime_name"] != crime_name:
        raise ValueError(
            "El delito del reporte no coincide con crime_name."
        )

    df_plot = df_plot.loc[
        df_plot["estrategia"].isin(["ataque", "falla"])
    ].copy()

    if df_plot.empty:
        logger.warning("No hay registros de ataque o falla.")
        return

    if df_plot[["window_id", "end_date", "modelo"]].isna().any().any():
        raise ValueError("Hay ventanas, fechas o modelos faltantes.")

    df_plot["end_date"] = pd.to_datetime(
        df_plot["end_date"],
        errors="raise",
    )

    if df_plot["end_date"].isna().any():
        raise ValueError("Hay fechas de cierre no válidas.")

    df_plot["distancia"] = pd.to_numeric(
        df_plot["distancia"],
        errors="raise",
    )

    if not np.isfinite(
        df_plot["distancia"].to_numpy(dtype=float)
    ).all():
        raise ValueError("Hay distancias no finitas.")

    df_plot["modelo"] = (
        df_plot["modelo"].astype(str).str.strip()
    )

    if df_plot["modelo"].eq("").any():
        raise ValueError("Hay nombres de modelos vacíos.")

    claves = ["window_id", "estrategia", "modelo"]

    if df_plot.duplicated(claves).any():
        raise ValueError(
            "Hay comparaciones duplicadas por ventana, estrategia y modelo."
        )

    # Cada ventana debe tener una sola fecha, común a ambas estrategias.
    if (
        df_plot.groupby("window_id")["end_date"]
        .nunique()
        .gt(1)
        .any()
    ):
        raise ValueError(
            "Una ventana contiene fechas de cierre diferentes."
        )

    ventanas = (
        df_plot[["window_id", "end_date"]]
        .drop_duplicates("window_id")
        .sort_values(["end_date", "window_id"])
        .set_index("window_id")
    )

    # 2. Seleccionar un ganador por ventana y estrategia.
    # En empates se conserva el orden de las filas recibidas.
    df_ganadores = (
        df_plot.sort_values("distancia", kind="stable")
        .drop_duplicates(
            subset=["window_id", "estrategia"],
            keep="first",
        )
        .copy()
    )

    # Conservar m y k; simplificar únicamente p de ER y WS.
    df_ganadores["modelo_visual"] = (
        df_ganadores["modelo"].str.replace(
            r"^(Erdos Renyi|Watts Strogatz)\s*\(p=[^)]*\)$",
            r"\1",
            regex=True,
        )
    )

    # 3. Construcción de los paneles.
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(15, 8),
        sharex=True,
        constrained_layout=True,
    )

    paneles = [
        {
            "ax_idx": 0,
            "estrategia": "ataque",
            "titulo": "Ataque Dirigido (Grado Alto)",
            "color": "crimson",
        },
        {
            "ax_idx": 1,
            "estrategia": "falla",
            "titulo": "Falla Aleatoria",
            "color": "royalblue",
        },
    ]

    fechas_eje = ventanas["end_date"]

    for panel in paneles:
        ax = axes[panel["ax_idx"]]

        # Alinear con todas las ventanas.
        # Una estrategia ausente deja un hueco en la línea.
        df_panel = (
            df_ganadores.loc[
                df_ganadores["estrategia"].eq(panel["estrategia"])
            ]
            .set_index("window_id")
            .reindex(ventanas.index)
        )

        ax.plot(
            fechas_eje,
            df_panel["distancia"],
            color=panel["color"],
            marker="o",
            markersize=4,
            linewidth=1.8,
            label=f"Distancia D mínima ({panel['titulo']})",
            zorder=3,
        )

        for window_id, row in df_panel.iterrows():
            if pd.isna(row["distancia"]):
                continue

            ax.annotate(
                row["modelo_visual"],
                (
                    ventanas.loc[window_id, "end_date"],
                    row["distancia"],
                ),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=7,
                rotation=40,
                alpha=0.85,
            )

        ax.set_title(
            panel["titulo"],
            fontsize=11,
            fontweight="bold",
        )
        ax.set_ylabel("Distancia Euclidiana $D$", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="upper right", fontsize=8)

        lim_inf, lim_sup = ax.get_ylim()
        ax.set_ylim(
            lim_inf,
            lim_sup + (lim_sup - lim_inf) * 0.18,
        )

    axes[1].set_xlabel(
        "Fecha de Cierre de Ventana (Semanas)",
        fontsize=10,
    )

    fig.suptitle(
        f"Evolución Longitudinal de Resiliencia Estructural "
        f"({len(ventanas)} Ventanas)\n"
        f"Delito: {crime_name.title()} | CDMX | "
        f"W={metadatos['window_size']} | "
        f"S={metadatos['step_size']}",
        fontsize=13,
        fontweight="bold",
    )

    plt.show()
    logger.info("EXITO.")
# endregion






# region plot_resilience_transitions_123
# [VISUALIZACIÓN] Transiciones de los tres modelos más próximos
# INPUT: df_resilience_test_report (pd.DataFrame), crime_name (str)
# OUTPUT: None

def plot_resilience_transitions_123(
    df_resilience_test_report: pd.DataFrame,
    crime_name: str,
) -> None:
    """Grafica tres modelos por ventana; une solamente el primer lugar."""

    logger.info("Iniciando visualización de transiciones de modelos")

    if df_resilience_test_report.empty:
        logger.warning("El reporte de resiliencia está vacío.")
        return

    columnas = ["window_id", "estrategia", "modelo", "distancia"]
    campos_metadatos = [
        "experiment_id",
        "crime_name",
        "window_size",
        "step_size",
    ]

    faltantes = (
        set(columnas + campos_metadatos)
        - set(df_resilience_test_report.columns)
    )

    if faltantes:
        raise ValueError(
            f"Faltan columnas requeridas: {sorted(faltantes)}"
        )

    # 1. Validar un único experimento y recuperar su configuración.
    metadatos = {}

    for campo in campos_metadatos:
        valores = df_resilience_test_report[campo].drop_duplicates()

        if len(valores) != 1 or pd.isna(valores.iloc[0]):
            raise ValueError(
                f"'{campo}' debe tener un único valor válido. "
                "Selecciona un solo experimento."
            )

        metadatos[campo] = valores.iloc[0]

    if metadatos["crime_name"] != crime_name:
        raise ValueError(
            "El delito del reporte no coincide con crime_name."
        )

    paneles = [
        {
            "ax_idx": 0,
            "estrategia": "ataque",
            "titulo": "Ataque Dirigido (Grado Alto)",
            "color": "crimson",
        },
        {
            "ax_idx": 1,
            "estrategia": "falla",
            "titulo": "Falla Aleatoria",
            "color": "royalblue",
        },
    ]

    estilos_adicionales = {
        2: {
            "color": "royalblue",
            "marker": "s",
            "label": "Segundo lugar",
        },
        3: {
            "color": "gray",
            "marker": "^",
            "label": "Tercer lugar",
        },
    }

    # 2. Preparar las comparaciones guardadas.
    df_plot = df_resilience_test_report[columnas].copy()
    df_plot = df_plot.loc[
        df_plot["estrategia"].isin(["ataque", "falla"])
    ].copy()

    if df_plot.empty:
        logger.warning("No hay registros de ataque o falla para graficar.")
        return

    if df_plot[["window_id", "modelo"]].isna().any().any():
        raise ValueError(
            "Hay identificadores de ventana o modelos faltantes."
        )

    df_plot["modelo"] = (
        df_plot["modelo"].astype(str).str.strip()
    )

    if df_plot["modelo"].eq("").any():
        raise ValueError("Hay nombres de modelos vacíos.")

    claves = ["window_id", "estrategia", "modelo"]

    if df_plot.duplicated(claves).any():
        raise ValueError(
            "Hay comparaciones duplicadas por ventana, estrategia y modelo. "
            "Selecciona un experimento o réplica antes de graficar."
        )

    df_plot["distancia"] = pd.to_numeric(
        df_plot["distancia"],
        errors="raise",
    )

    # Conservar todas las ventanas antes de omitir distancias no finitas.
    eje_x = np.sort(df_plot["window_id"].unique())

    distancias_validas = np.isfinite(
        df_plot["distancia"].to_numpy(dtype=float)
    )

    if not distancias_validas.all():
        logger.warning(
            "Se omiten %s comparaciones con distancia no finita.",
            int((~distancias_validas).sum()),
        )

    df_plot = df_plot.loc[distancias_validas].copy()

    if df_plot.empty:
        logger.warning("No hay distancias válidas para graficar.")
        return

    # 3. Conservar el orden y el desempate de la función original.
    df_plot = df_plot.sort_values(
        ["window_id", "estrategia", "distancia", "modelo"]
    )

    df_plot["lugar"] = (
        df_plot.groupby(["window_id", "estrategia"]).cumcount() + 1
    )

    df_top3 = df_plot.loc[df_plot["lugar"].le(3)].copy()

    # Conservar m y k. Agrupar únicamente p de ER y WS.
    df_top3["categoria"] = df_top3["modelo"].str.replace(
        r"^(Erdos Renyi|Watts Strogatz)\s*\(p=[^)]*\)$",
        r"\1",
        regex=True,
    )

    if df_top3["categoria"].eq("").any():
        raise ValueError("Hay categorías de modelos vacías.")

    df_ganadores = df_top3.loc[df_top3["lugar"].eq(1)].copy()

    todos_modelos = sorted(
        df_top3["categoria"].unique().tolist()
    )

    modelo_a_posicion = {
        modelo: posicion
        for posicion, modelo in enumerate(todos_modelos)
    }

    posiciones_y = [
        modelo_a_posicion[modelo]
        for modelo in todos_modelos
    ]

    etiquetas_x = [
        f"Window {int(window_id)}"
        for window_id in eje_x
    ]

    # 4. Construir los paneles.
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(16, 8.5),
        sharex=True,
        constrained_layout=True,
    )

    for panel in paneles:
        ax = axes[panel["ax_idx"]]

        # Alinear el primer lugar con todas las ventanas.
        # Los valores ausentes cortan la línea.
        df_primero = (
            df_ganadores.loc[
                df_ganadores["estrategia"].eq(panel["estrategia"])
            ]
            .set_index("window_id")
            .reindex(eje_x)
        )

        eje_y = df_primero["categoria"].map(modelo_a_posicion)

        ax.plot(
            eje_x,
            eje_y,
            color=panel["color"],
            linestyle="--",
            linewidth=1.2,
            alpha=0.45,
            zorder=2,
        )

        puntos_validos = eje_y.notna().to_numpy()

        # Primer lugar: color según la distancia guardada.
        if puntos_validos.any():
            scatter = ax.scatter(
                eje_x[puntos_validos],
                eje_y.to_numpy()[puntos_validos],
                c=df_primero["distancia"].to_numpy()[puntos_validos],
                cmap="viridis_r",
                s=90,
                edgecolor="black",
                linewidth=0.8,
                zorder=5,
            )

            cbar = fig.colorbar(
                scatter,
                ax=ax,
                aspect=20,
                pad=0.01,
            )
            cbar.set_label(
                "Distancia $D$ — Primer lugar",
                fontsize=8,
            )
        else:
            logger.warning(
                "No hay ganadores válidos para %s.",
                panel["estrategia"],
            )

        # Segundo y tercer lugar: colores fijos, sin líneas.
        for lugar, estilo in estilos_adicionales.items():
            df_lugar = df_top3.loc[
                df_top3["estrategia"].eq(panel["estrategia"])
                & df_top3["lugar"].eq(lugar)
            ]

            if df_lugar.empty:
                continue

            ax.scatter(
                df_lugar["window_id"],
                df_lugar["categoria"].map(modelo_a_posicion),
                color=estilo["color"],
                marker=estilo["marker"],
                label=estilo["label"],
                s=60,
                edgecolor="black",
                linewidth=0.6,
                zorder=3,
            )

        ax.set_title(
            panel["titulo"],
            fontsize=11,
            fontweight="bold",
        )
        ax.set_ylabel("Modelo Teórico", fontsize=9)

        ax.set_yticks(posiciones_y)
        ax.set_yticklabels(todos_modelos, fontsize=9)
        ax.set_ylim(-0.5, len(todos_modelos) - 0.5)
        ax.set_xlim(eje_x.min() - 0.5, eje_x.max() + 0.5)

        ax.grid(True, linestyle="--", alpha=0.35)

        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="best", fontsize=8)

    axes[1].set_xticks(eje_x)
    axes[1].set_xticklabels(
        etiquetas_x,
        rotation=90,
        fontsize=7.5,
    )
    axes[1].set_xlabel(
        "Secuencia de Ventanas Móviles ($W_i$)",
        fontsize=10,
    )

    fig.suptitle(
        f"Cambio del Modelo Teórico más Próximo a la Red "
        f"({len(eje_x)} Ventanas)\n"
        f"Delito: {crime_name.title()} | "
        f"Ventana: {metadatos['window_size']} semanas | "
        f"Paso: {metadatos['step_size']} semanas",
        fontsize=13,
        fontweight="bold",
    )

    plt.show()
    logger.info("EXITO.")
# endregion






# region plot_resilience_grid
def plot_resilience_grid(
    resilience_results: dict,
    strategy: str = "both",
) -> None:
    """Plot stored resilience curves for attack, random failure, or both strategies.

    Renders a multi-panel grid comparing the topological fragmentation curve of
    the empirical visibility network against each theoretical reference model
    (excluding ``'Path'``) for the selected node-removal strategy. All rendered
    chart titles and axis labels are formatted in Spanish.

    Parameters
    ----------
    resilience_results : dict
        Dictionary returned by ``resilience.resilience_analysis``, containing
        ``'series_name'``, ``'start_date'``, ``'end_date'``,
        ``'empirical_network'``, and ``'models'``.
    strategy : {'attack', 'random', 'both'}, default='both'
        Node-removal strategy to plot.

    Returns
    -------
    None
        Displays the figure(s) via ``plt.show()``.

    Raises
    ------
    ValueError
        If ``strategy`` is not ``'attack'``, ``'random'``, or ``'both'``.
    """
    logger.info("Preparando cuadrícula de curvas de resiliencia")

    try:
        if strategy not in ("attack", "random", "both"):
            raise ValueError(
                "strategy debe ser 'attack', 'random' o 'both'."
            )
    except ValueError as err:
        logger.error("FAIL ❌. %s", err)
        return

    if not resilience_results:
        logger.warning("El resultado de resiliencia está vacío.")
        return


    # 1. Identidad y periodo del análisis.
    series_name = resilience_results["series_name"]
    start_date = resilience_results["start_date"]
    end_date = resilience_results["end_date"]

    if start_date is not None and end_date is not None:
        date_range = (
            f"({pd.Timestamp(start_date).strftime('%Y-%m-%d')} – "
            f"{pd.Timestamp(end_date).strftime('%Y-%m-%d')})"
        )
    else:
        date_range = "(Periodo Desconocido)"

    # 2. Reunir los resultados existentes para su representación.
    catalog = {
        "Red Empírica Semanal": resilience_results["empirical_network"],
    }

    catalog.update({
        name: data
        for name, data in resilience_results["models"].items()
        if name != "Path"
    })

    titles = {
        "attack": "Ataque Dirigido (Nodos de Mayor Grado)",
        "random": "Falla Aleatoria (Nodos al Azar)",
    }

    strategies = (
        ("attack", "random")
        if strategy == "both"
        else (strategy,)
    )

    # 3. Una figura por estrategia seleccionada.
    total_models = len(catalog)
    rows = math.ceil(total_models / 5)

    for strategy_key in strategies:
        fig, axes = plt.subplots(
            rows,
            5,
            figsize=(18, 3.5 * rows),
            constrained_layout=True,
            squeeze=False,
        )
        axes = axes.flatten()

        for i, (network_name, network_data) in enumerate(catalog.items()):
            curve = network_data["curves"][strategy_key]

            line_color = (
                "crimson"
                if network_name == "Red Empírica Semanal"
                else "royalblue"
            )

            axes[i].plot(
                curve,
                color=line_color,
                linewidth=1.8,
            )
            axes[i].set_title(network_name, fontsize=9)
            axes[i].set_xlabel("Nodos removidos", fontsize=8)
            axes[i].set_ylabel("Componentes conexos", fontsize=8)
            axes[i].grid(True, linestyle="--", alpha=0.3)

        # Ocultar paneles sin datos.
        for j in range(total_models, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(
            f"{titles[strategy_key]}\n"
            f"Red de Visibilidad Semanal - {series_name.title()}\n"
            f"{date_range}",
            fontsize=13,
            fontweight="bold",
        )

        plt.show()

    logger.info("EXITO ✅.")
# endregion







# region plot_closest_resilience_models
def plot_closest_resilience_models(
    resilience_results: dict,
) -> None:
    """Compare stored resilience curves of the empirical network and closest model.

    Plots side-by-side node-removal fragmentation curves for targeted attack
    (``'attack'``) and random failure (``'random'``), contrasting the empirical
    visibility network against the top-ranked synthetic reference model per
    strategy. All rendered chart titles, axis labels, and legends are
    formatted in Spanish.

    Parameters
    ----------
    resilience_results : dict
        Dictionary returned by ``resilience.resilience_analysis``, containing
        ``'series_name'``, ``'start_date'``, ``'end_date'``,
        ``'best_models'``, ``'empirical_network'``, and ``'models'``.

    Returns
    -------
    None
        Displays the figure via ``plt.show()``.
    """
    logger.info("Preparando contraste de modelos más cercanos")

    if not resilience_results:
        logger.warning("El resultado de resiliencia está vacío.")
        return

    # 1. Identidad y periodo del análisis.
    series_name = resilience_results["series_name"]
    start_date = resilience_results["start_date"]
    end_date = resilience_results["end_date"]

    if start_date is not None and end_date is not None:
        date_range = (
            f"({pd.Timestamp(start_date).strftime('%Y-%m-%d')} – "
            f"{pd.Timestamp(end_date).strftime('%Y-%m-%d')})"
        )
    else:
        date_range = "(Periodo Desconocido)"

    strategies = [
        ("Ataque Dirigido (Grado Alto)", "attack"),
        ("Falla Aleatoria", "random"),
    ]

    # 2. Comprobar que exista un ganador para ambas estrategias.
    for strategy_label, strategy_key in strategies:
        if resilience_results["best_models"][strategy_key] is None:
            logger.warning(
                "No hay modelo ganador para %s.",
                strategy_label,
            )
            return

    # 3. Representar los resultados existentes.
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 6),
        constrained_layout=True,
    )

    for ax, (strategy_label, strategy_key) in zip(
        axes,
        strategies,
    ):
        winner = resilience_results["best_models"][strategy_key]

        best_name = winner["model"]
        best_distance = winner["distance"]

        base_curve = resilience_results["empirical_network"]["curves"][
            strategy_key
        ]
        best_curve = resilience_results["models"][best_name][
            "curves"
        ][strategy_key]

        ax.plot(
            base_curve,
            label=f"Red Empírica: {series_name.title()}",
            linewidth=2.5,
            color="royalblue",
            zorder=2,
        )

        ax.plot(
            best_curve,
            label=f"Más cercano: {best_name}\n(D={best_distance:.4f})",
            linestyle="--",
            color="crimson",
            linewidth=2.0,
            zorder=3,
        )

        ax.set_title(
            strategy_label,
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Nodos eliminados (Semanas)", fontsize=10)
        ax.set_ylabel("Componentes conexos", fontsize=10)
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(
        "Contraste de Desmantelamiento vs. Modelo Sintético Más Cercano\n"
        f"Red de Visibilidad Semanal - {series_name.title()}\n"
        f"{date_range}",
        fontsize=13,
        fontweight="bold",
    )

    plt.show()

    logger.info("EXITO.")
# endregion






# region plot_resilience_transitions
def plot_resilience_transitions(
    rolling_resilience_results: dict,
    strategy: str = "attack",
    top_n: int = 3,
    *,
    ax: Axes | None = None,
    model_to_position: dict[str, int] | None = None,
    norm: Normalize | None = None,
    show_colorbar: bool = True,
    show_legend: bool = True,
) -> plt.Figure | None:
    """Plot top-ranked theoretical resilience models across rolling windows.

    Connects first-place models across consecutive windows with a dashed line
    and colors their markers by normalized curve distance, while plotting
    lower-ranked models as standalone markers. All rendered chart titles,
    axis labels, and legends are formatted in Spanish.

    Does not recompute curves or distances, nor display or save the figure.

    Parameters
    ----------
    rolling_resilience_results : dict
        Dictionary returned by ``resilience.rolling_resilience_analysis``.
    strategy : {'attack', 'random'}, default='attack'
        Node-removal strategy to visualize.
    top_n : int, default=3
        Maximum number of top-ranked models to plot per window.
    ax : Axes | None, default=None
        Optional Matplotlib axes to draw onto. Creates a standalone figure when
        ``None``.
    model_to_position : dict[str, int] | None, default=None
        Optional mapping from model category name to Y-axis tick position for
        shared multi-panel grids.
    norm : Normalize | None, default=None
        Optional color normalization shared across multiple panels.
    show_colorbar : bool, default=True
        Whether to draw a colorbar next to the axes.
    show_legend : bool, default=True
        Whether to draw a legend on the axes.

    Returns
    -------
    plt.Figure | None
        Matplotlib figure containing the plot, or ``None`` if no valid window
        results exist.

    Raises
    ------
    ValueError
        If ``strategy`` is invalid, ``top_n < 1``, duplicate ``'window_id'``
        values exist, or ``model_to_position`` is missing observed models.
    """
    logger.info("Preparando transiciones de modelos de resiliencia")

    if strategy not in ("attack", "random"):
        raise ValueError("strategy debe ser 'attack' o 'random'.")

    if (
        isinstance(top_n, bool)
        or not isinstance(top_n, int)
        or top_n < 1
    ):
        raise ValueError("top_n debe ser un entero positivo.")

    if not rolling_resilience_results:
        logger.warning("El resultado de resiliencia está vacío.")
        return None

    windows = rolling_resilience_results["windows"]

    if not windows:
        logger.warning("No hay ventanas para graficar.")
        return None

    # 1. Identity and configuration.
    series_name = rolling_resilience_results["series_name"]
    window_size = rolling_resilience_results["window_size"]
    step = rolling_resilience_results["step"]

    titles = {
        "attack": "Ataque Dirigido (Grado Alto)",
        "random": "Falla Aleatoria",
    }

    line_colors = {
        "attack": "crimson",
        "random": "royalblue",
    }

    sorted_windows = sorted(
        windows,
        key=lambda win: win["window_id"],
    )

    x_axis = np.array([
        win["window_id"]
        for win in sorted_windows
    ])

    if len(np.unique(x_axis)) != len(x_axis):
        raise ValueError("Hay identificadores de ventana duplicados.")

    # 2. Extract precomputed top-ranked models per window.
    records = []

    for window in sorted_windows:
        analysis = window["resilience_results"]

        if not analysis:
            continue

        ranking = analysis["sorted_distances"][strategy]

        for rank, item in enumerate(
            ranking[:top_n],
            start=1,
        ):
            distance = float(item["distance"])

            if not np.isfinite(distance):
                logger.warning(
                    "Se omite un punto con distancia no finita: "
                    "ventana=%s, estrategia=%s, lugar=%s.",
                    window["window_id"],
                    strategy,
                    rank,
                )
                continue

            records.append({
                "window_id": window["window_id"],
                "rank": rank,
                "model": item["model"],
                "distance": distance,
            })

    if not records:
        logger.warning(
            "No hay resultados válidos para la estrategia '%s'.",
            strategy,
        )
        return None

    df_plot = pd.DataFrame(records)

    # Preserve m and k parameters; strip only p from ER and WS labels.
    df_plot["category"] = df_plot["model"].str.replace(
        r"^(Erdos Renyi|Watts Strogatz)\s*\(p=[^)]*\)$",
        r"\1",
        regex=True,
    )

    # 3. Use local Y-axis positions or shared grid positions.
    if model_to_position is None:
        models = sorted(df_plot["category"].unique().tolist())

        model_to_position = {
            model: pos
            for pos, model in enumerate(models)
        }
    else:
        missing = (
            set(df_plot["category"])
            - set(model_to_position)
        )

        if missing:
            raise ValueError(
                "Faltan modelos en las posiciones compartidas: "
                f"{sorted(missing)}"
            )

    ordered_models = sorted(
        model_to_position,
        key=model_to_position.get,
    )

    y_positions = [
        model_to_position[model]
        for model in ordered_models
    ]

    # 4. Create figure or draw on provided axes.
    external_panel = ax is not None

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(16, 6),
            constrained_layout=True,
        )
    else:
        fig = ax.figure

    # Missing window results leave gaps in the connecting line.
    df_first = (
        df_plot.loc[df_plot["rank"].eq(1)]
        .set_index("window_id")
        .reindex(x_axis)
    )

    y_axis = df_first["category"].map(model_to_position)

    ax.plot(
        x_axis,
        y_axis,
        color=line_colors[strategy],
        linestyle="--",
        linewidth=1.2,
        alpha=0.45,
        zorder=2,
    )

    valid_points = y_axis.notna().to_numpy()

    if valid_points.any():
        scatter = ax.scatter(
            x_axis[valid_points],
            y_axis.to_numpy()[valid_points],
            c=df_first["distance"].to_numpy()[valid_points],
            cmap="viridis_r",
            norm=norm,
            s=90,
            edgecolor="black",
            linewidth=0.8,
            label="Primer lugar",
            zorder=5,
        )

        if show_colorbar:
            cbar = fig.colorbar(
                scatter,
                ax=ax,
                aspect=20,
                pad=0.01,
            )
            cbar.set_label(
                "Distancia $D$ — Primer lugar",
                fontsize=8,
            )

    # 5. Additional ranks: markers only (no connecting lines).
    rank_styles = {
        2: ("royalblue", "s", "Segundo lugar"),
        3: ("gray", "^", "Tercer lugar"),
    }

    palette = plt.get_cmap("tab20")
    markers = ("D", "v", "P", "X", "<", ">", "h", "*")

    for rank in sorted(df_plot["rank"].unique()):
        if rank == 1:
            continue

        df_rank = df_plot.loc[df_plot["rank"].eq(rank)]

        if rank in rank_styles:
            color, marker, label = rank_styles[rank]
        else:
            idx = int(rank) - 4
            color = palette(idx % palette.N)
            marker = markers[idx % len(markers)]
            label = f"Lugar {rank}"

        ax.scatter(
            df_rank["window_id"],
            df_rank["category"].map(model_to_position),
            color=color,
            marker=marker,
            label=label,
            s=60,
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )

    # 6. Labels and visual presentation (in Spanish).
    ax.set_yticks(y_positions)
    ax.set_yticklabels(ordered_models, fontsize=9)
    ax.set_ylim(min(y_positions) - 0.5, max(y_positions) + 0.5)
    ax.set_ylabel("Modelo Teórico", fontsize=9)

    ax.set_xticks(x_axis)
    ax.set_xticklabels(
        [f"Ventana {int(window_id)}" for window_id in x_axis],
        rotation=90,
        fontsize=7.5,
    )
    ax.set_xlim(x_axis.min() - 0.5, x_axis.max() + 0.5)
    ax.set_xlabel(
        "Secuencia de Ventanas Móviles ($W_i$)",
        fontsize=10,
    )

    if external_panel:
        # Overall title will be set by the grid function.
        title = (
            f"W={window_size} | S={step}\n"
            f"{len(x_axis)} ventanas"
        )
    else:
        title = (
            f"Transiciones de Modelos más Próximos — {titles[strategy]}\n"
            f"Serie: {series_name.title()} | "
            f"Ventana: {window_size} semanas | Paso: {step} semanas\n"
            f"{len(x_axis)} ventanas | Hasta {top_n} modelos por ventana"
        )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)

    if show_legend:
        ax.legend(loc="best", fontsize=8)

    logger.info("EXITO ✅.")
    return fig
# endregion






# region plot_resilience_transitions_grid
def plot_resilience_transitions_grid(
    experiments: list[dict],
    strategy: str = "attack",
    top_n: int = 3,
    figsize: tuple[float, float] = (32, 18),
) -> plt.Figure:
    """Compare closest-model resilience transitions across multiple experiments.

    Arranges step sizes in rows and window sizes in columns, sharing Y-axis
    model positions, the first-place distance color scale, and the legend
    across all panels. All rendered chart titles, axis labels, and legends
    are formatted in Spanish.

    Accepts persisted experiment dictionaries loaded from disk. Does not
    recompute resilience analyses, display, or save the figure.

    Parameters
    ----------
    experiments : list[dict]
        List of experiment dictionaries loaded via ``load_rolling_experiments``,
        each containing ``'rolling_resilience_results'``.
    strategy : {'attack', 'random'}, default='attack'
        Node-removal strategy to visualize (``'attack'`` for targeted degree
        removal or ``'random'`` for uniform random failures).
    top_n : int, default=3
        Maximum number of top-ranked models to display per window.
    figsize : tuple[float, float], default=(32, 18)
        Width and height of the resulting Matplotlib figure in inches.

    Returns
    -------
    plt.Figure
        Matplotlib figure containing the multi-panel grid of resilience
        transitions.

    Raises
    ------
    ValueError
        If ``strategy`` is invalid, ``top_n < 1``, ``experiments`` is empty,
        duplicate ``(window_size, step)`` pairs exist, experiments belong to
        different series, or no valid distances are available.
    """
    if strategy not in ("attack", "random"):
        raise ValueError("strategy debe ser 'attack' o 'random'.")

    if (
        isinstance(top_n, bool)
        or not isinstance(top_n, int)
        or top_n < 1
    ):
        raise ValueError("top_n debe ser un entero positivo.")

    if not experiments:
        raise ValueError("No hay experimentos para graficar.")

    logger.info("Preparando cuadrícula de transiciones de resiliencia")

    # 1. Organize experiments by (window_size, step).
    series_names = set()
    results_by_combo = {}

    for experiment in experiments:
        res_results = experiment["rolling_resilience_results"]

        series_names.add(res_results["series_name"])
        combo = (
            res_results["window_size"],
            res_results["step"],
        )

        if combo in results_by_combo:
            raise ValueError(
                f"Hay experimentos duplicados para "
                f"W={combo[0]}, S={combo[1]}."
            )

        results_by_combo[combo] = res_results

    if len(series_names) != 1:
        raise ValueError("Los experimentos deben corresponder a la misma serie.")

    series_name = next(iter(series_names))

    window_sizes = sorted({
        window_size
        for window_size, _ in results_by_combo
    })

    steps = sorted({
        step
        for _, step in results_by_combo
    })

    # 2. Gather categories and first-place distances to share axes and color scale.
    categories = set()
    first_place_distances = []

    for res_results in results_by_combo.values():
        for window in res_results["windows"]:
            analysis = window["resilience_results"]

            if not analysis:
                continue

            ranking = analysis["sorted_distances"][strategy]

            for rank, item in enumerate(
                ranking[:top_n],
                start=1,
            ):
                distance = float(item["distance"])

                if not np.isfinite(distance):
                    continue

                category = re.sub(
                    r"^(Erdos Renyi|Watts Strogatz)\s*\(p=[^)]*\)$",
                    r"\1",
                    item["model"],
                )

                categories.add(category)

                if rank == 1:
                    first_place_distances.append(distance)

    if not categories:
        raise ValueError(
            f"No hay resultados válidos para la estrategia '{strategy}'."
        )

    if not first_place_distances:
        raise ValueError("No hay distancias válidas del primer lugar.")

    model_to_position = {
        model: pos
        for pos, model in enumerate(sorted(categories))
    }

    min_distance = min(first_place_distances)
    max_distance = max(first_place_distances)

    # Avoid a zero-width color normalization range.
    if min_distance == max_distance:
        max_distance = min_distance + 1.0

    norm = Normalize(
        vmin=min_distance,
        vmax=max_distance,
    )

    # 3. Create subplot grid.
    fig, axes = plt.subplots(
        nrows=len(steps),
        ncols=len(window_sizes),
        figsize=figsize,
        constrained_layout=True,
        squeeze=False,
    )

    panels_with_data = []
    legend_elements = {}

    for row, step in enumerate(steps):
        for col, window_size in enumerate(window_sizes):
            ax = axes[row, col]

            res_results = results_by_combo.get(
                (window_size, step)
            )

            if res_results is None:
                ax.set_visible(False)
                logger.warning(
                    "No hay experimento para W=%s, S=%s.",
                    window_size,
                    step,
                )
                continue

            panel_fig = plot_resilience_transitions(
                rolling_resilience_results=res_results,
                strategy=strategy,
                top_n=top_n,
                ax=ax,
                model_to_position=model_to_position,
                norm=norm,
                show_colorbar=False,
                show_legend=False,
            )

            if panel_fig is None:
                ax.set_title(f"W={window_size} | S={step}")
                ax.text(
                    0.5,
                    0.5,
                    "Sin resultados válidos",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_axis_off()
                continue

            panels_with_data.append(ax)

            # Collect a single legend handle per rank label.
            handles, labels = ax.get_legend_handles_labels()

            for handle, label in zip(handles, labels):
                if label not in legend_elements:
                    legend_elements[label] = handle

    # 4. Shared colorbar for first-place distances.
    if panels_with_data:
        mappable = ScalarMappable(
            norm=norm,
            cmap="viridis_r",
        )

        cbar = fig.colorbar(
            mappable,
            ax=panels_with_data,
            fraction=0.025,
            pad=0.02,
        )

        cbar.set_label(
            "Distancia $D$ — Primer lugar",
            fontsize=11,
        )

    # 5. Shared legend and figure title (in Spanish).
    if legend_elements:
        fig.legend(
            handles=list(legend_elements.values()),
            labels=list(legend_elements),
            loc="lower center",
            bbox_to_anchor=(0.5, -0.025),
            ncol=min(top_n, 6),
            fontsize=11,
        )

    strategy_label = (
        "Ataque Dirigido"
        if strategy == "attack"
        else "Falla Aleatoria"
    )

    fig.suptitle(
        f"Transiciones de Modelos más Próximos — {strategy_label}\n"
        f"{series_name.title()} | Hasta {top_n} modelos por ventana\n"
        "Filas: paso | Columnas: tamaño de ventana",
        fontsize=18,
        fontweight="bold",
    )

    logger.info(
        "EXITO ✅. Cuadrícula creada: %s filas y %s columnas.",
        len(steps),
        len(window_sizes),
    )

    return fig
# endregion






# region plot_resilience_winner_frequency
def plot_resilience_winner_frequency(
    rolling_resilience_results: dict,
    strategy: str = "attack",
    rank: int = 1,
    *,
    ax: Axes | None = None,
    ordered_models: list[str] | None = None,
) -> plt.Figure:
    """Plot the percentage of rolling windows where each model occupies a given rank.

    Reads precomputed sorted curve distances for the selected node-removal
    strategy. ``rank`` represents a 1-based position in that ranking, and
    ties preserve the deterministic ordering established during analysis.

    Each valid window contributes a single model, and the denominator includes
    only windows with a finite distance at the requested rank. Groups ``p``
    parameter variants of ``Erdos Renyi`` and ``Watts Strogatz`` while
    preserving all other model parameters. All rendered chart titles, axis
    labels, and bar annotations are formatted in Spanish.

    Does not run simulations, display, or save the figure.

    Parameters
    ----------
    rolling_resilience_results : dict
        Dictionary returned by ``resilience.rolling_resilience_analysis``.
    strategy : {'attack', 'random'}, default='attack'
        Node-removal strategy to visualize (``'attack'`` for targeted degree
        removal or ``'random'`` for uniform random failures).
    rank : int, default=1
        1-based ranking position to evaluate (e.g., ``1`` for closest model).
    ax : Axes | None, default=None
        Optional Matplotlib axes to draw onto. Creates a standalone figure when
        ``None``.
    ordered_models : list[str] | None, default=None
        Optional list specifying the Y-axis model order (used for shared
        multi-panel grids). When ``None``, models are sorted by descending
        frequency and then alphabetically.

    Returns
    -------
    plt.Figure
        Matplotlib figure containing the horizontal bar chart of model
        frequencies.

    Raises
    ------
    ValueError
        If ``strategy`` is invalid, ``rank < 1``, ``rolling_resilience_results``
        or its ``'windows'`` list is empty, no valid windows exist at ``rank``,
        or ``ordered_models`` contains duplicates or misses observed models.
    """
    if strategy not in ("attack", "random"):
        raise ValueError("strategy debe ser 'attack' o 'random'.")

    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError("rank debe ser un entero positivo.")

    if not rolling_resilience_results:
        raise ValueError("El resultado de resiliencia está vacío.")

    windows = rolling_resilience_results["windows"]

    if not windows:
        raise ValueError("No hay ventanas para graficar.")

    logger.info(
        "Preparando frecuencias del lugar %s para la estrategia '%s'.",
        rank,
        strategy,
    )

    # 1. Group only p parameter variants for display.
    def normalize_model_label(name: str) -> str:
        if name.startswith("Erdos Renyi ("):
            return "Erdos Renyi"
        if name.startswith("Watts Strogatz ("):
            return "Watts Strogatz"
        return name

    models = set()
    counts = {}
    total_windows = 0
    skipped_windows = 0

    # 2. Read requested rank position without re-sorting.
    for window in windows:
        analysis = window["resilience_results"]

        if not analysis:
            skipped_windows += 1
            continue

        distances = analysis["sorted_distances"][strategy]

        # Include models that never occupied the requested rank.
        models.update(
            normalize_model_label(item["model"])
            for item in distances
        )

        if len(distances) < rank:
            skipped_windows += 1
            continue

        selected = distances[rank - 1]

        if not np.isfinite(selected["distance"]):
            skipped_windows += 1
            continue

        model = normalize_model_label(selected["model"])
        counts[model] = counts.get(model, 0) + 1
        total_windows += 1

    if skipped_windows:
        logger.warning(
            "Se omiten %s ventanas sin una comparación válida "
            "en el lugar %s para '%s'.",
            skipped_windows,
            rank,
            strategy,
        )

    if total_windows == 0:
        raise ValueError(
            f"No hay ventanas con una comparación válida "
            f"en el lugar {rank} para '{strategy}'."
        )

    # 3. Use local frequency ordering or shared grid ordering.
    if ordered_models is None:
        ordered_models = sorted(
            models,
            key=lambda name: (-counts.get(name, 0), name),
        )
    else:
        if len(set(ordered_models)) != len(ordered_models):
            raise ValueError("La lista de modelos contiene duplicados.")

        missing = models - set(ordered_models)

        if missing:
            raise ValueError(
                "Faltan modelos en el orden compartido: "
                f"{sorted(missing)}"
            )

    frequencies = np.array([
        counts.get(name, 0)
        for name in ordered_models
    ])

    percentages = 100 * frequencies / total_windows

    # 4. Create standalone figure or draw on target panel.
    external_panel = ax is not None

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(12, max(5, len(ordered_models) * 0.45)),
            constrained_layout=True,
        )
    else:
        fig = ax.figure

    positions = np.arange(len(ordered_models))
    color = "crimson" if strategy == "attack" else "royalblue"

    ax.barh(
        positions,
        percentages,
        color=color,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_yticks(positions)
    ax.set_yticklabels(ordered_models, fontsize=9)

    # Fix top-to-bottom ordering without flipping when reusing axes.
    ax.set_ylim(len(ordered_models) - 0.5, -0.5)

    for position, (frequency, percentage) in enumerate(
        zip(frequencies, percentages)
    ):
        label = (
            f"{percentage:.1f} %"
            if external_panel
            else f"{percentage:.1f} % ({int(frequency)} ventanas)"
        )

        # Place labels inside the bar when close to 100%.
        inside = percentage >= 85

        ax.text(
            percentage - 1 if inside else percentage + 1,
            position,
            label,
            ha="right" if inside else "left",
            va="center",
            color="white" if inside else "black",
            fontsize=9,
        )

    ax.set_xlim(0, 100)
    ax.set_xlabel(f"Porcentaje de ventanas en el lugar {rank}")
    ax.set_ylabel("Modelo Teórico")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    # 5. Series identity and analysis configuration (titles in Spanish).
    series_name = rolling_resilience_results["series_name"]
    window_size = rolling_resilience_results["window_size"]
    step = rolling_resilience_results["step"]

    strategy_label = (
        "Ataque Dirigido"
        if strategy == "attack"
        else "Falla Aleatoria"
    )

    if external_panel:
        title = (
            f"W={window_size} | S={step}\n"
            f"{total_windows} ventanas válidas"
        )
    else:
        title = (
            f"Frecuencia de Modelos en el Lugar {rank} — "
            f"{strategy_label}\n"
            f"{series_name.title()}\n"
            f"Ventana: {window_size} semanas | Paso: {step} semanas | "
            f"{total_windows} ventanas válidas"
        )

    ax.set_title(title, fontsize=11)

    logger.info(
        "EXITO ✅. Frecuencias calculadas sobre %s ventanas válidas.",
        total_windows,
    )

    return fig
# endregion






# region plot_resilience_winner_frequency_grid
def plot_resilience_winner_frequency_grid(
    experiments: list[dict],
    strategy: str = "attack",
    rank: int = 1,
    figsize: tuple[float, float] = (32, 18),
) -> plt.Figure:
    """Compare model rank frequencies across multiple rolling experiments.

    Arranges step sizes in rows and window sizes in columns, sharing model
    categories, their Y-axis ordering, and the percentage scale across all
    panels. Each panel computes percentages over its own valid windows for the
    selected rank and removal strategy. All rendered chart titles and labels
    are formatted in Spanish.

    Accepts persisted experiment dictionaries loaded from disk. Does not
    recompute resilience analyses, display, or save the figure.

    Parameters
    ----------
    experiments : list[dict]
        List of experiment dictionaries loaded via ``load_rolling_experiments``,
        each containing ``'rolling_resilience_results'``.
    strategy : {'attack', 'random'}, default='attack'
        Node-removal strategy to visualize (``'attack'`` for targeted degree
        removal or ``'random'`` for uniform random failures).
    rank : int, default=1
        1-based ranking position to evaluate (e.g., ``1`` for closest model).
    figsize : tuple[float, float], default=(32, 18)
        Width and height of the resulting Matplotlib figure in inches.

    Returns
    -------
    plt.Figure
        Matplotlib figure containing the multi-panel grid of model frequencies.

    Raises
    ------
    ValueError
        If ``strategy`` is invalid, ``rank < 1``, ``experiments`` is empty,
        duplicate ``(window_size, step)`` pairs exist, experiments belong to
        different series, or no valid windows exist for the requested rank.
    """
    if strategy not in ("attack", "random"):
        raise ValueError("strategy debe ser 'attack' o 'random'.")

    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or rank < 1
    ):
        raise ValueError("rank debe ser un entero positivo.")

    if not experiments:
        raise ValueError("No hay experimentos para graficar.")

    logger.info("Preparando cuadrícula de frecuencias de resiliencia")

    # 1. Organize experiments by (window_size, step).
    series_names = set()
    results_by_combo = {}

    for experiment in experiments:
        res_results = experiment["rolling_resilience_results"]

        series_names.add(res_results["series_name"])

        combo = (
            res_results["window_size"],
            res_results["step"],
        )

        if combo in results_by_combo:
            raise ValueError(
                f"Hay experimentos duplicados para "
                f"W={combo[0]}, S={combo[1]}."
            )

        results_by_combo[combo] = res_results

    if len(series_names) != 1:
        raise ValueError(
            "Los experimentos deben corresponder a la misma serie."
        )

    series_name = next(iter(series_names))

    window_sizes = sorted({
        window_size
        for window_size, _ in results_by_combo
    })

    steps = sorted({
        step
        for _, step in results_by_combo
    })

    # 2. Gather all theoretical models to share their alphabetical ordering.
    def normalize_model_label(name: str) -> str:
        if name.startswith("Erdos Renyi ("):
            return "Erdos Renyi"
        if name.startswith("Watts Strogatz ("):
            return "Watts Strogatz"
        return name

    models = set()
    valid_combos = set()

    for combo, res_results in results_by_combo.items():
        for window in res_results["windows"]:
            analysis = window["resilience_results"]

            if not analysis:
                continue

            ranking = analysis["sorted_distances"][strategy]

            # Include models that never occupied the selected rank.
            models.update(
                normalize_model_label(item["model"])
                for item in ranking
            )

            if len(ranking) >= rank:
                distance = ranking[rank - 1]["distance"]

                if np.isfinite(distance):
                    valid_combos.add(combo)

    if not valid_combos:
        raise ValueError(
            f"No hay ventanas válidas en el lugar {rank} "
            f"para la estrategia '{strategy}'."
        )

    # Shared alphabetical ordering so every model keeps its Y-axis position.
    ordered_models = sorted(models)

    # 3. Create subplot grid.
    fig, axes = plt.subplots(
        nrows=len(steps),
        ncols=len(window_sizes),
        figsize=figsize,
        constrained_layout=True,
        squeeze=False,
    )

    for row, step in enumerate(steps):
        for col, window_size in enumerate(window_sizes):
            ax = axes[row, col]
            combo = (window_size, step)

            res_results = results_by_combo.get(combo)

            if res_results is None:
                ax.set_visible(False)
                logger.warning(
                    "No hay experimento para W=%s, S=%s.",
                    window_size,
                    step,
                )
                continue

            if combo not in valid_combos:
                ax.set_title(f"W={window_size} | S={step}")
                ax.text(
                    0.5,
                    0.5,
                    "Sin ventanas válidas\n"
                    f"para el lugar {rank}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_axis_off()

                logger.warning(
                    "Sin ventanas válidas para W=%s, S=%s, lugar=%s.",
                    window_size,
                    step,
                    rank,
                )
                continue

            # Reuse single-panel frequency plotter on the target axes.
            plot_resilience_winner_frequency(
                rolling_resilience_results=res_results,
                strategy=strategy,
                rank=rank,
                ax=ax,
                ordered_models=ordered_models,
            )

    # 4. Overall figure title (in Spanish).
    strategy_label = (
        "Ataque Dirigido"
        if strategy == "attack"
        else "Falla Aleatoria"
    )

    fig.suptitle(
        f"Frecuencia de Modelos en el Lugar {rank} — "
        f"{strategy_label}\n"
        f"{series_name.title()}\n"
        "Filas: paso | Columnas: tamaño de ventana",
        fontsize=18,
        fontweight="bold",
    )

    logger.info(
        "EXITO ✅. Cuadrícula creada: %s filas y %s columnas.",
        len(steps),
        len(window_sizes),
    )

    return fig
# endregion







# region plot_histogram_resilience
def plot_histogram_resilience(
    resilience_graph: dict,
    strategy: str,
    rank: int,
    models: list[str],
) -> go.Figure:
    """Highlight rolling-window resilience winners on the full-series histogram.

    Filters linked resilience rankings by removal strategy, rank position, and
    selected theoretical models, placing a distinct marker on the weekly bar
    corresponding to each window's central observation. ``Erdos Renyi`` and
    ``Watts Strogatz`` match across all ``p`` parameter variants while
    preserving the exact parameterized model name in hover tooltips. All
    rendered chart titles, legends, and tooltips are formatted in Spanish.

    Does not recompute metrics, rankings, or distances. Returns the figure
    without displaying it.

    Parameters
    ----------
    resilience_graph : dict
        Dictionary returned by ``merge_resilience_graph``, containing
        ``'series_name'``, ``'window_size'``, ``'step'``, ``'nodes'``, and
        ``'resilience'``.
    strategy : {'attack', 'random'}
        Node-removal strategy to filter (``'attack'`` for targeted degree
        removal or ``'random'`` for uniform random failures).
    rank : int
        Ranking position to display (integer between ``1`` and ``5``).
    models : list[str]
        List of base theoretical model names to highlight.

    Returns
    -------
    go.Figure
        Plotly figure combining the weekly series bar chart and resilience
        model markers.

    Raises
    ------
    ValueError
        If ``strategy`` is invalid, ``rank`` is not an integer in ``1..5``,
        or ``models`` contains unrecognized model names.
    """
    if strategy not in ("attack", "random"):
        raise ValueError("strategy debe ser 'attack' o 'random'.")

    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or rank not in (1, 2, 3, 4, 5)
    ):
        raise ValueError("rank debe ser un entero entre 1 y 5.")

    # 1. Stable visual styles: each model keeps a consistent color and symbol.
    model_styles = {
        "Watts Strogatz": ("#d62728", "circle"),
        "Erdos Renyi": ("#1f77b4", "square"),
        "Barabasi Albert (m=2)": ("#2ca02c", "diamond"),
        "Barabasi Albert (m=3)": ("#9467bd", "triangle-up"),
        "Barabasi Albert (m=4)": ("#ff7f0e", "triangle-down"),
        "Random Regular (k=2)": ("#8c564b", "cross"),
        "Enlace Igualitario 1": ("#e377c2", "x"),
        "Enlace Igualitario 2": ("#17becf", "star"),
        "Enlace Igualitario 3": ("#aec7e8", "star-diamond"),
        "Star": ("#bcbd22", "hexagon"),
        "Grid 2D": ("#7f7f7f", "pentagon"),
        "Ladder": ("#393b79", "triangle-left"),
        "Hypercube": ("#637939", "triangle-right"),
        "Complete": ("#8c6d31", "hexagram"),
        "Path": ("#843c39", "hourglass"),
        "Cycle": ("#7b4173", "bowtie"),
        "Circular Ladder": ("#008080", "asterisk"),
    }

    unknown = set(models) - set(model_styles)

    if unknown:
        raise ValueError(
            f"Modelos no reconocidos: {sorted(unknown)}"
        )

    # Deduplicate selected models while preserving user order.
    models = list(dict.fromkeys(models))

    df_nodes = resilience_graph["nodes"]
    df_resilience = resilience_graph["resilience"]
    series_name = resilience_graph["series_name"]

    # 2. Build base weekly histogram using the standardized node metrics schema.
    fig = plot_histogram(
        node_metrics={
            "graph_name": series_name,
            "data": df_nodes,
        }
    )

    if df_nodes.empty:
        return fig

    # Show only the highlighted models in the legend.
    fig.update_traces(showlegend=False)

    count_col = "frequency"
    counts = pd.to_numeric(
        df_nodes[count_col],
        errors="raise",
    )

    max_val = float(counts.max())
    offset = max_val * 0.05 if max_val > 0 else 1

    # Raise node ID labels slightly so they do not overlap with model markers.
    sorted_counts = (
        df_nodes.sort_values("node_id")[count_col]
        .to_numpy(dtype=float)
    )

    fig.data[1].y = sorted_counts + 2 * offset

    # 3. Filter rankings by strategy, rank, and selected models.
    selection = df_resilience.loc[
        df_resilience["strategy"].eq(strategy)
        & df_resilience["rank"].eq(rank)
    ].copy()

    def normalize_selector_label(name: str) -> str:
        if name.startswith("Erdos Renyi ("):
            return "Erdos Renyi"
        if name.startswith("Watts Strogatz ("):
            return "Watts Strogatz"
        return name

    selection["model_selector"] = selection["model"].map(
        normalize_selector_label
    )

    selection = selection.loc[
        selection["model_selector"].isin(models)
    ].copy()

    # Retrieve bar heights from the full series via the linked node_id.
    selection = selection.merge(
        df_nodes[["node_id", count_col]],
        on="node_id",
        how="left",
        validate="many_to_one",
    )

    selection["center_date"] = pd.to_datetime(
        selection["center_date"],
        errors="raise",
    )

    # 4. Add one scatter trace per selected model (tooltips in Spanish).
    for model in models:
        points = selection.loc[
            selection["model_selector"].eq(model)
        ].copy()

        if points.empty:
            logger.warning(
                "No hay puntos para '%s', estrategia='%s', posición=%s.",
                model,
                strategy,
                rank,
            )
            continue

        points = points.sort_values("center_date")

        # Format boundary dates solely for hover display.
        points["start_text"] = pd.to_datetime(
            points["start_date"]
        ).dt.strftime("%Y-%m-%d")

        points["end_text"] = pd.to_datetime(
            points["end_date"]
        ).dt.strftime("%Y-%m-%d")

        customdata = points[[
            "model",
            "window_id",
            "start_text",
            "end_text",
            "rank",
            "distance",
            "node_id",
        ]].to_numpy(dtype=object)

        color, symbol = model_styles[model]

        fig.add_trace(
            go.Scatter(
                x=points["center_date"],
                y=points[count_col],
                mode="markers",
                name=model,
                showlegend=True,
                marker=dict(
                    color=color,
                    symbol=symbol,
                    size=13,
                    line=dict(color="black", width=1),
                ),
                customdata=customdata,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Ventana: %{customdata[1]}<br>"
                    "Inicio: %{customdata[2]}<br>"
                    "Fin: %{customdata[3]}<br>"
                    "Fecha central: %{x|%Y-%m-%d}<br>"
                    "Posición: %{customdata[4]}<br>"
                    "Distancia: %{customdata[5]:.6f}<br>"
                    "Nodo: %{customdata[6]}<br>"
                    "Delitos: %{y}"
                    "<extra></extra>"
                ),
            )
        )

    # 5. Append experiment configuration to the figure title (in Spanish).
    strategy_label = (
        "Ataque dirigido"
        if strategy == "attack"
        else "Falla aleatoria"
    )

    title = (
        fig.layout.title.text
        + "<br>"
        + f"<sub>{strategy_label} | Posición {rank} | "
        + f"W={resilience_graph['window_size']} | "
        + f"S={resilience_graph['step']}</sub>"
    )

    fig.update_layout(
        title=dict(text=title),
        margin=dict(l=70, r=50, t=150, b=100),
        showlegend=True,
        legend=dict(
            title=dict(text="Modelos señalados"),
        ),
    )

    logger.info(
        "EXITO ✅. Histograma preparado con %s marcas de resiliencia.",
        len(selection),
    )

    return fig
# endregion






# region plot_metrics_distance_heatmap
# DEPRECATED
def plot_metrics_distance_heatmap(
    correlation_results: dict,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Grafica las correlaciones previamente calculadas.

    Las filas representan modelos y las columnas representan métricas.
    Las celdas grises indican correlaciones no calculables.
    Devuelve la figura sin mostrarla ni guardarla.
    """

    correlaciones = correlation_results["correlaciones"]

    if correlaciones.empty:
        raise ValueError("No hay correlaciones para representar.")

    if figsize is None:
        figsize = (
            max(12, len(correlaciones.columns) * 1.3),
            max(6, len(correlaciones.index) * 0.45 + 3),
        )

    fig, ax = plt.subplots(
        figsize=figsize, constrained_layout=True
    )

    valores = correlaciones.to_numpy(dtype=float)
    valores = np.ma.masked_invalid(valores)

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#dddddd")

    imagen = ax.imshow(
        valores,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_xticks(np.arange(len(correlaciones.columns)))
    ax.set_xticklabels(
        correlaciones.columns, rotation=45, ha="right", fontsize=9
    )

    ax.set_yticks(np.arange(len(correlaciones.index)))
    ax.set_yticklabels(correlaciones.index, fontsize=9)

    # Anotar el coeficiente o indicar que no es calculable.
    for fila in range(len(correlaciones.index)):
        for columna in range(len(correlaciones.columns)):
            valor = correlaciones.iat[fila, columna]

            if pd.isna(valor):
                texto, color = "—", "black"
            else:
                texto = f"{valor:.2f}"
                color = "white" if abs(valor) >= 0.6 else "black"

            ax.text(
                columna, fila, texto,
                ha="center", va="center", color=color, fontsize=8,
            )

    cbar = fig.colorbar(imagen, ax=ax, pad=0.02)
    cbar.set_label("Correlación de Spearman con la distancia")

    estrategia = {
        "ataque": "Ataque dirigido",
        "aleatorio": "Falla aleatoria",
    }[correlation_results["strategy"]]

    ax.set_title(
        "Relación entre métricas y distancia de fragmentación\n"
        f"{correlation_results['crime_name'].title()}\n"
        f"{estrategia} | W={correlation_results['window_size']} | "
        f"S={correlation_results['step']} | "
        f"{correlation_results['n_windows']} ventanas",
        fontsize=12,
        pad=18,
    )

    constantes = correlation_results["constantes"]
    nota = (
        "Gris: variable constante, datos insuficientes "
        "o valores ausentes. Mínimo: 3 pares válidos."
    )

    if constantes:
        detalle = ", ".join(
            f"{nombre}={valor:g}"
            for nombre, valor in constantes.items()
        )
        nota += f"\nConstantes: {detalle}."

    ax.set_xlabel(nota, fontsize=9, labelpad=18)
    ax.set_ylabel("Modelo teórico")

    return fig
# endregion





# region plot_degree_distribution
def plot_degree_distribution(
    distribution_metrics: dict,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> plt.Figure | None:
    """Plot the empirical degree distribution using bars and a guide line.

    Reads precomputed degree probabilities and formats the Y-axis as node
    percentages. Line segments connecting discrete points serve purely as a
    visual guide. Does not restrict data to the tail, recompute distributions,
    display, or save the figure. All rendered chart titles and axis labels
    are formatted in Spanish.

    Parameters
    ----------
    distribution_metrics : dict
        Dictionary returned by ``distributions.compute_distribution_metrics``,
        containing ``'k_all'``, ``'probabilities_all'``, and ``'series_name'``
        (or ``'graph_name'``).
    start_date : str | pd.Timestamp | None, default=None
        Optional start date of the analyzed window or series.
    end_date : str | pd.Timestamp | None, default=None
        Optional end date of the analyzed window or series.

    Returns
    -------
    plt.Figure | None
        Closed Matplotlib figure ready for display or export, or ``None`` if
        ``'k_all'`` is empty.

    Raises
    ------
    ValueError
        If degrees or probabilities are not 1D arrays of equal length, contain
        non-finite or negative values, or probabilities do not sum to 1.
    """
    logger.info("Preparando distribución empírica de grados")

    k = np.asarray(distribution_metrics["k_all"])
    probabilities = np.asarray(
        distribution_metrics["probabilities_all"], dtype=float
    )

    if k.size == 0:
        logger.warning("No hay grados para graficar.")
        return None

    if k.ndim != 1 or probabilities.ndim != 1:
        raise ValueError("Los grados y probabilidades deben ser vectores.")

    if k.size != probabilities.size:
        raise ValueError(
            "Los grados y probabilidades deben tener igual longitud."
        )

    if (
        not np.isfinite(k).all()
        or not np.isfinite(probabilities).all()
        or (k < 0).any()
        or (probabilities < 0).any()
        or not np.isclose(probabilities.sum(), 1.0)
    ):
        raise ValueError(
            "La distribución contiene grados o probabilidades inválidos."
        )

    series_name = (
        distribution_metrics.get("series_name")
        or distribution_metrics.get("graph_name")
        or distribution_metrics.get("crime_name", "")
    )

    # Sort degrees alongside their corresponding probabilities.
    order = np.argsort(k)
    k = k[order]
    probabilities = probabilities[order]

    date_range = "(Periodo Desconocido)"

    if start_date is not None and end_date is not None:
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)

        if pd.notna(start_ts) and pd.notna(end_ts):
            date_range = (
                f"({start_ts:%Y-%m-%d} – {end_ts:%Y-%m-%d})"
            )

    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Bars: proportion of nodes with each observed degree.
    ax.bar(
        k, probabilities,
        width=0.8,
        color="lightsteelblue",
        edgecolor="royalblue",
        linewidth=0.8,
        alpha=0.7,
        zorder=2,
    )

    # Line and markers overlaid on bars.
    ax.plot(
        k, probabilities,
        marker="o", markersize=6,
        color="royalblue", linewidth=1.2,
        markeredgecolor="black", markeredgewidth=0.7,
        zorder=3,
    )

    ax.set_title(
        "Distribución Empírica de Grados\n"
        f"{str(series_name).title()}\n{date_range}",
        fontsize=13, pad=15,
    )
    ax.set_xlabel("Grado $k$ (número de conexiones)", fontsize=11)
    ax.set_ylabel("Porcentaje de nodos", fontsize=11)

    # Underlying values remain in [0, 1]; axis ticks render as percentages.
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_ylim(0, min(1.05, float(probabilities.max()) * 1.15))

    if len(k) <= 25:
        ax.set_xticks(k)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", direction="out", labelsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    logger.info("EXITO ✅. Distribución empírica de grados graficada.")
    plt.close(fig)
    return fig
# endregion






# region plot_window_analysis
def plot_window_analysis(
    rolling_experiment: dict,
    window_id: int,
    figsize: tuple[float, float] = (28, 32),
    *,
    global_visibility_graph: nx.Graph,
    global_pos: dict | None = None,
    show_global_edges: bool = True,
) -> plt.Figure:
    """Build an eight-panel diagnostic figure combining global and local window context.

    Row 1: full time series and global visibility network with the target window
    highlighted.
    Row 2: local window time series with visibility chords and local topological
    structure (diameter, radius, and center).
    Row 3: sorted node degrees and empirical degree distribution.
    Row 4: best-matching theoretical reference models for both removal
    strategies and directed-attack resilience curves across the model catalog.

    Matches local and global nodes by exact timestamp without assuming identical
    node IDs, and verifies node frequency consistency. Does not recompute
    metrics, curves, distances, or rankings, and does not mutate input graphs.
    All rendered chart titles, axis labels, and legends are formatted in
    Spanish.

    Parameters
    ----------
    rolling_experiment : dict
        Consolidated rolling-window experiment dictionary returned by
        ``run_rolling_experiment``.
    window_id : int
        Zero-based sequential identifier of the target window to visualize.
    figsize : tuple[float, float], default=(28, 32)
        Overall figure dimensions ``(width, height)`` in inches.
    global_visibility_graph : nx.Graph
        Full-series Natural Visibility Graph built by ``graph.build_nvg``,
        containing ``'graph_name'`` in ``g.graph`` and ``'timestamp'`` /
        ``'frequency'`` attributes on each node.
    global_pos : dict | None, default=None
        Precomputed node layout mapping ``{node_id: (x, y)}`` for
        ``global_visibility_graph`` to keep positions consistent across
        windows. Computed via ``nx.spring_layout(..., seed=42)`` if ``None``.
    show_global_edges : bool, default=True
        Whether to draw background and internal edges on the global network
        panel.

    Returns
    -------
    plt.Figure
        Closed Matplotlib figure instance ready for saving or explicit display.

    Raises
    ------
    ValueError
        If the local or global graph is empty, if ``window_id`` lacks
        resilience results, if series names do not match, if node timestamps or
        frequencies are invalid/mismatched, or if ``global_pos`` misses nodes.
    """
    # 1. Recuperar los resultados de la ventana.
    window_graph = reports.get_graph_by_window(
        rolling_experiment, window_id
    )
    window_metrics = reports.get_graph_metrics_by_window(
        rolling_experiment, window_id
    )
    window_distribution = reports.get_distribution_metrics_by_window(
        rolling_experiment, window_id
    )
    window_resilience = reports.get_resilience_results_by_window(
        rolling_experiment, window_id
    )

    graph = window_graph["graph"]
    metrics = window_metrics["graph_metrics"]
    distribution = window_distribution["distribution_metrics"]
    resilience = window_resilience["resilience_results"]

    if graph.number_of_nodes() == 0:
        raise ValueError(f"El grafo de la ventana {window_id} está vacío.")

    if not resilience:
        raise ValueError(
            f"La ventana {window_id} no contiene análisis de resiliencia."
        )

    series_name = window_metrics["series_name"]
    start_date = pd.Timestamp(window_metrics["start_date"])
    end_date = pd.Timestamp(window_metrics["end_date"])

    # 2. Validar la correspondencia con el grafo global.
    global_graph = global_visibility_graph

    if global_graph.number_of_nodes() == 0:
        raise ValueError("El grafo global está vacío.")

    if global_graph.graph.get("graph_name") != series_name:
        raise ValueError("El grafo global corresponde a otra serie.")

    def index_by_date(network: nx.Graph) -> dict:
        by_date = {}

        for node, data in network.nodes(data=True):
            timestamp = pd.Timestamp(data["timestamp"])
            frequency = float(data["frequency"])

            if pd.isna(timestamp) or not np.isfinite(frequency):
                raise ValueError("Hay fechas o cantidades inválidas.")

            if timestamp in by_date:
                raise ValueError(
                    f"Hay nodos con fecha duplicada: {timestamp}."
                )

            by_date[timestamp] = (node, frequency)

        return by_date

    global_by_date = index_by_date(global_graph)
    local_by_date = index_by_date(graph)

    missing = set(local_by_date) - set(global_by_date)

    if missing:
        raise ValueError(
            "Hay fechas de la ventana ausentes en el grafo global: "
            f"{sorted(missing)}"
        )

    for timestamp, (_, frequency) in local_by_date.items():
        if not np.isclose(
            frequency,
            global_by_date[timestamp][1],
            rtol=0,
            atol=1e-9,
        ):
            raise ValueError(
                f"La cantidad local y global no coincide en {timestamp}."
            )

    selected_nodes = {
        global_by_date[timestamp][0]
        for timestamp in local_by_date
    }

    global_dates = sorted(global_by_date)
    global_nodes = [
        global_by_date[timestamp][0]
        for timestamp in global_dates
    ]
    global_frequencies = [
        global_by_date[timestamp][1]
        for timestamp in global_dates
    ]

    if global_pos is None:
        global_pos = nx.spring_layout(global_graph, seed=42)
    elif not set(global_graph).issubset(global_pos):
        raise ValueError("Faltan nodos en las posiciones globales.")

    # 3. Cuadrícula exterior: ocho secciones de igual tamaño.
    fig = plt.figure(figsize=figsize, facecolor="white")
    outer_grid = fig.add_gridspec(
        4, 2,
        left=0.06,
        right=0.98,
        bottom=0.05,
        top=0.92,
        hspace=0.35,
        wspace=0.22,
    )

    ax_global_series = fig.add_subplot(outer_grid[0, 0])
    ax_global_network = fig.add_subplot(outer_grid[0, 1])
    ax_series = fig.add_subplot(outer_grid[1, 0])
    ax_network = fig.add_subplot(outer_grid[1, 1])
    ax_degrees = fig.add_subplot(outer_grid[2, 0])
    ax_distribution = fig.add_subplot(outer_grid[2, 1])

    def configure_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
        ax.set_title(
            title, fontsize=12, fontweight="bold", pad=12
        )
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def draw_curve(ax, curve, **style) -> None:
        # Cada observación corresponde a una eliminación ya realizada.
        values = np.asarray(curve, dtype=float)
        ax.plot(
            np.arange(1, len(values) + 1),
            values,
            **style,
        )

    # 4. Histograma global con la ventana resaltada.
    numeric_dates = mdates.date2num(
        pd.DatetimeIndex(global_dates).to_pydatetime()
    )
    global_colors = [
        "crimson" if node in selected_nodes else "lightgray"
        for node in global_nodes
    ]

    ax_global_series.bar(
        numeric_dates,
        global_frequencies,
        width=5.5,
        color=global_colors,
        edgecolor="gray",
        linewidth=0.25,
        zorder=2,
    )

    highlight_start = min(local_by_date)
    highlight_end = max(local_by_date)

    ax_global_series.axvspan(
        mdates.date2num(highlight_start) - 3.5,
        mdates.date2num(highlight_end) + 3.5,
        color="crimson",
        alpha=0.10,
        zorder=1,
    )

    configure_axes(
        ax_global_series,
        "Serie completa: ubicación de la ventana",
        "Inicio de semana",
        "Cantidad",
    )

    global_locator = mdates.AutoDateLocator(
        minticks=4, maxticks=8
    )
    ax_global_series.xaxis.set_major_locator(global_locator)
    ax_global_series.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(global_locator)
    )
    ax_global_series.set_ylim(bottom=0)
    ax_global_series.margins(x=0.01, y=0.08)

    # 5. Red global con los nodos de la ventana resaltados.
    if show_global_edges:
        internal_edges = []
        remaining_edges = []

        for u, v in global_graph.edges():
            if u in selected_nodes and v in selected_nodes:
                internal_edges.append((u, v))
            else:
                remaining_edges.append((u, v))

        # Conexiones del resto de la red y conexiones hacia el exterior.
        nx.draw_networkx_edges(
            global_graph,
            global_pos,
            edgelist=remaining_edges,
            edge_color="royalblue",
            width=0.6,
            alpha=0.65,
            ax=ax_global_network,
        )

        # Conexiones internas de la ventana, sobre las aristas azules.
        nx.draw_networkx_edges(
            global_graph,
            global_pos,
            edgelist=internal_edges,
            edge_color="crimson",
            width=1.2,
            alpha=0.95,
            ax=ax_global_network,
        )

    outside_window = [
        node for node in global_nodes
        if node not in selected_nodes
    ]
    inside_window = [
        node for node in global_nodes
        if node in selected_nodes
    ]

    if outside_window:
        nx.draw_networkx_nodes(
            global_graph,
            global_pos,
            nodelist=outside_window,
            node_color="white",
            node_size=16,
            edgecolors="black",
            # alpha=0.5,
            linewidths=0.6,
            ax=ax_global_network,
        )

    nx.draw_networkx_nodes(
        global_graph,
        global_pos,
        nodelist=inside_window,
        node_color="crimson",
        node_size=45,
        edgecolors="black",
        linewidths=0.6,
        ax=ax_global_network,
    )

    ax_global_network.set_title(
        "Red global: ubicación de la ventana",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax_global_network.set_axis_off()

    context_legend = [
        Line2D(
            [], [],
            marker="o",
            linestyle="none",
            color="crimson",
            label=f"Ventana {window_id}: {len(selected_nodes)} nodos",
        ),
        Line2D(
            [], [],
            marker="o",
            linestyle="none",
            color="lightgray",
            label="Resto de la serie",
        ),
    ]

    ax_global_series.legend(
        handles=context_legend, fontsize=8
    )
    ax_global_network.legend(
        handles=context_legend, fontsize=8
    )

    # 6. Serie de la ventana y conexiones de visibilidad.
    nodes = sorted(graph.nodes())
    dates = pd.to_datetime([
        graph.nodes[node]["timestamp"]
        for node in nodes
    ])
    frequencies = np.array([
        graph.nodes[node]["frequency"]
        for node in nodes
    ], dtype=float)

    if pd.isna(dates).any():
        raise ValueError(
            "Hay fechas inválidas en los nodos de la ventana."
        )

    x_positions = mdates.date2num(dates.to_pydatetime())

    coordinates = {
        node: (x, y)
        for node, x, y in zip(nodes, x_positions, frequencies)
    }
    segments = [
        [coordinates[u], coordinates[v]]
        for u, v in graph.edges()
    ]

    ax_series.bar(
        x_positions,
        frequencies,
        width=5.5,
        color="white",
        edgecolor="black",
        linewidth=0.6,
        zorder=1,
    )
    ax_series.add_collection(
        LineCollection(
            segments,
            colors="royalblue",
            linewidths=0.6,
            alpha=0.65,
            zorder=2,
        )
    )
    ax_series.scatter(
        x_positions,
        frequencies,
        s=12,
        color="black",
        zorder=3,
    )

    for node, x, y in zip(nodes, x_positions, frequencies):
        ax_series.annotate(
            str(node),
            (x, y),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=5,
        )

    configure_axes(
        ax_series,
        "Serie y conexiones de visibilidad",
        "Inicio de semana",
        "Cantidad",
    )

    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    ax_series.xaxis.set_major_locator(locator)
    ax_series.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(locator)
    )
    ax_series.set_ylim(bottom=0)
    ax_series.margins(x=0.02, y=0.12)

    # 7. Estructura topológica local y caminos precalculados.
    pos = nx.spring_layout(graph, seed=42)

    nx.draw_networkx_edges(
        graph,
        pos,
        edge_color="royalblue",
        width=0.9,
        alpha=0.8,
        ax=ax_network,
    )

    for key, color in (
        ("diameter_edges", "red"),
        ("radius_edges", "green"),
    ):
        if metrics.get(key):
            nx.draw_networkx_edges(
                graph,
                pos,
                edgelist=metrics[key],
                edge_color=color,
                width=2.2,
                ax=ax_network,
            )

    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color="white",
        node_size=150,
        edgecolors="black",
        linewidths=0.6,
        ax=ax_network,
    )

    if metrics.get("center"):
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=metrics["center"],
            node_color="yellow",
            node_size=180,
            edgecolors="black",
            ax=ax_network,
        )

    nx.draw_networkx_labels(
        graph,
        pos,
        font_size=6,
        font_weight="bold",
        ax=ax_network,
    )

    legend_items = [
        Line2D(
            [], [],
            color="royalblue",
            label=f"Densidad: {metrics['density']:.4f}",
        )
    ]

    for key, color, label in (
        ("diameter", "red", "Diámetro"),
        ("radius", "green", "Radio"),
    ):
        if metrics.get(key) is not None:
            legend_items.append(
                Line2D(
                    [], [],
                    color=color,
                    linewidth=2,
                    label=f"{label}: {metrics[key]}",
                )
            )

    if metrics.get("center"):
        legend_items.append(
            Line2D(
                [], [],
                linestyle="none",
                marker="o",
                markerfacecolor="yellow",
                markeredgecolor="black",
                label=f"Centro: {metrics['center']}",
            )
        )

    ax_network.legend(handles=legend_items, loc="best", fontsize=8)
    ax_network.set_title(
        "Estructura topológica",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax_network.set_axis_off()

    # 8. Grados ordenados con identificadores originales.
    pairs = sorted(
        graph.degree(),
        key=lambda item: item[1],
        reverse=True,
    )
    labels = [str(node) for node, degree in pairs]
    degrees = [degree for node, degree in pairs]
    positions = np.arange(len(pairs))

    ax_degrees.bar(
        positions,
        degrees,
        color="white",
        edgecolor="black",
        linewidth=0.7,
    )
    ax_degrees.set_xticks(positions)
    ax_degrees.set_xticklabels(
        labels, rotation=90, fontsize=5
    )

    configure_axes(
        ax_degrees,
        "Grados ordenados por conectividad",
        "Identificador del nodo, ordenado por grado",
        "Grado",
    )
    ax_degrees.tick_params(axis="x", labelsize=5)
    ax_degrees.set_ylim(bottom=0)

    # 9. Distribución empírica completa de grados.
    k = np.asarray(distribution["k_all"])
    probabilities = np.asarray(
        distribution["probabilities_all"], dtype=float
    )

    configure_axes(
        ax_distribution,
        "Distribución empírica de grados",
        "Grado $k$ (número de conexiones)",
        "Porcentaje de nodos",
    )

    if k.size:
        order = np.argsort(k)
        k = k[order]
        probabilities = probabilities[order]

        ax_distribution.bar(
            k,
            probabilities,
            width=0.8,
            color="lightsteelblue",
            edgecolor="royalblue",
            linewidth=0.8,
            alpha=0.7,
            zorder=2,
        )

        ax_distribution.plot(
            k,
            probabilities,
            marker="o",
            markersize=6,
            color="royalblue",
            linewidth=1.2,
            markeredgecolor="black",
            markeredgewidth=0.7,
            zorder=3,
        )

        # Ajustar la altura al máximo, dejando 15 % de margen.
        ax_distribution.set_ylim(
            0,
            min(1.05, float(probabilities.max()) * 1.15),
        )

        if len(k) <= 25:
            ax_distribution.set_xticks(k)
        else:
            ax_distribution.xaxis.set_major_locator(
                MaxNLocator(integer=True)
            )

    else:
        ax_distribution.text(
            0.5,
            0.5,
            "Sin datos de distribución",
            ha="center",
            va="center",
            transform=ax_distribution.transAxes,
        )

    # Las probabilidades conservan sus valores; mostrar porcentajes.
    ax_distribution.yaxis.set_major_formatter(
        PercentFormatter(xmax=1.0)
    )

    # 10. Comparación con los ganadores de ambas estrategias.
    winners_block = outer_grid[3, 0].subgridspec(
        2, 1,
        height_ratios=[0.22, 0.78],
        hspace=0.35,
    )
    header = fig.add_subplot(winners_block[0])
    header.set_axis_off()
    header.text(
        0.5, 0.5,
        "Comparación con los modelos ganadores",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )

    winners_panels = winners_block[1].subgridspec(
        1, 2, wspace=0.35
    )

    for col, (strategy, title) in enumerate((
        ("attack", "Ataque dirigido"),
        ("random", "Falla aleatoria"),
    )):
        ax = fig.add_subplot(winners_panels[0, col])
        winner = resilience["best_models"][strategy]

        if winner is None:
            ax.text(
                0.5, 0.5,
                "Sin ganador",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        else:
            model_name = winner["model"]

            empirical_curve = resilience["empirical_network"]["curves"][
                strategy
            ]
            model_curve = resilience["models"][model_name]["curves"][
                strategy
            ]

            draw_curve(
                ax,
                empirical_curve,
                color="royalblue",
                linewidth=2,
                label="Red empírica",
                zorder=2,
            )
            draw_curve(
                ax,
                model_curve,
                color="crimson",
                linestyle="--",
                linewidth=1.8,
                label=f"{model_name}\nD={winner['distance']:.4f}",
                zorder=3,
            )
            ax.legend(fontsize=7, loc="best")

        configure_axes(
            ax,
            title,
            "Nodos eliminados",
            "Componentes conexos",
        )
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # 11. Catálogo completo: únicamente ataque dirigido.
    catalog = [
        ("Red empírica", resilience["empirical_network"], True)
    ]
    catalog.extend(
        (name, data, False)
        for name, data in resilience["models"].items()
        if name != "Path"
    )

    catalog_block = outer_grid[3, 1].subgridspec(
        2, 1,
        height_ratios=[0.22, 0.78],
        hspace=0.35,
    )
    header = fig.add_subplot(catalog_block[0])
    header.set_axis_off()
    header.text(
        0.5, 0.5,
        "Todos los modelos: ataque dirigido",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )

    columns = 5
    rows = math.ceil(len(catalog) / columns)

    catalog_panels = catalog_block[1].subgridspec(
        rows, columns,
        hspace=0.85,
        wspace=0.5,
    )

    for idx, (name, data, is_empirical) in enumerate(catalog):
        row, col = divmod(idx, columns)
        ax = fig.add_subplot(catalog_panels[row, col])

        draw_curve(
            ax,
            data["curves"]["attack"],
            color="crimson" if is_empirical else "royalblue",
            linewidth=1,
        )

        ax.set_title(name, fontsize=6, wrap=True)
        ax.tick_params(labelsize=5)
        ax.grid(linestyle="--", alpha=0.25)
        ax.xaxis.set_major_locator(
            MaxNLocator(nbins=3, integer=True)
        )
        ax.yaxis.set_major_locator(
            MaxNLocator(nbins=3, integer=True)
        )

        if row == rows - 1:
            ax.set_xlabel("Nodos eliminados", fontsize=6)

        if col == 0:
            ax.set_ylabel("Componentes", fontsize=6)

    # 12. Identidad y configuración del análisis.
    fig.suptitle(
        f"Análisis de la ventana {window_id} — {series_name}\n"
        f"{start_date:%Y-%m-%d} – {end_date:%Y-%m-%d} | "
        f"W={window_metrics['window_size']} | "
        f"S={window_metrics['step']} | "
        f"N={metrics['node_count']} | M={metrics['edge_count']}",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    plt.close(fig)
    return fig
# endregion






# region plot_resilience_simulation_step
def plot_resilience_simulation_step(
    simulation: dict,
    step: int = 0,
    pos: dict | None = None,
) -> plt.Figure:
    """Render a four-panel visual snapshot of a targeted attack simulation step.

    Preserves the initial network layout and reconstructs the active graph by
    removing nodes according to ``simulation['removal_order'][:step]``.
    Displays the time series with surviving visibility chords, active
    connected components, and cumulative fragmentation curves up to ``step``.
    Does not recompute the attack or mutate the stored graph. All rendered
    chart titles, axis labels, and legends are formatted in Spanish.

    Parameters
    ----------
    simulation : dict
        Simulation record dictionary returned by
        ``resilience.resilience_simulator``, containing ``'attack_log'``,
        ``'graph'``, ``'removal_order'``, ``'initial_nodes'``, and optional
        ``'series_name'``.
    step : int, default=0
        Zero-based simulation step index to visualize (from ``0`` to
        ``len(simulation['attack_log']) - 1``).
    pos : dict | None, default=None
        Precomputed node layout mapping ``{node_id: (x, y)}`` for the intact
        graph. Computed via ``nx.spring_layout(..., seed=42)`` if ``None``.

    Returns
    -------
    plt.Figure
        Closed Matplotlib figure instance ready for explicit display or saving.

    Raises
    ------
    ValueError
        If ``step`` is not an integer within valid bounds, if the simulation
        graph is empty, or if any node lacks valid ``'timestamp'`` or
        ``'frequency'`` attributes.
    """
    records = simulation["attack_log"]

    if isinstance(step, bool) or not isinstance(step, int):
        raise ValueError("step debe ser un entero.")

    if step < 0 or step >= len(records):
        raise ValueError(f"step debe estar entre 0 y {len(records) - 1}.")

    original = simulation["graph"]
    removal_order = simulation["removal_order"]

    if original.number_of_nodes() == 0:
        raise ValueError("El grafo de la simulación está vacío.")

    # Las posiciones topológicas siempre se calculan sobre la red intacta.
    if pos is None:
        pos = nx.spring_layout(original, seed=42)

    removed = removal_order[:step]
    active = original.copy()
    active.remove_nodes_from(removed)

    last_removed = removal_order[step - 1] if step > 0 else None
    next_target = removal_order[step] if step < len(removal_order) else None

    # Orden temporal fijo para la vista de visibilidad.
    dates_by_node = {}
    frequencies = {}

    for node, data in original.nodes(data=True):
        if "timestamp" not in data or "frequency" not in data:
            raise ValueError(
                "La vista temporal requiere 'timestamp' y 'frequency' "
                "en todos los nodos."
            )

        timestamp = pd.Timestamp(data["timestamp"])

        if pd.isna(timestamp):
            raise ValueError(f"El nodo {node} tiene una fecha inválida.")

        dates_by_node[node] = timestamp
        frequencies[node] = float(data["frequency"])

    nodes = sorted(original.nodes(), key=dates_by_node.get)
    index_by_node = {node: i for i, node in enumerate(nodes)}
    heights = [frequencies[node] for node in nodes]

    # Colorear cada componente por su primer nodo en el orden temporal.
    components = sorted(
        nx.connected_components(active),
        key=lambda group: (-len(group), min(index_by_node[n] for n in group)),
    )

    palette = plt.get_cmap("tab20")
    colors = {}

    for group in components:
        reference = min(index_by_node[node] for node in group)
        color = palette(reference % palette.N)

        for node in group:
            colors[node] = color

    fig, axes = plt.subplots(
        2, 2,
        figsize=(17, 11),
        constrained_layout=True,
    )
    ax_series, ax_network = axes[0]
    ax_components, ax_largest = axes[1]

    # 1. Serie original de referencia y aristas supervivientes.
    ax_series.bar(
        range(len(nodes)), heights,
        color="#DFD9D9E1", edgecolor="#2563EB",
        linewidth=0.7, zorder=1,
    )

    segments = [
        [
            (index_by_node[u], frequencies[u]),
            (index_by_node[v], frequencies[v]),
        ]
        for u, v in active.edges()
    ]

    if segments:
        ax_series.add_collection(
            LineCollection(
                segments,
                colors=[colors[u] for u, v in active.edges()],
                linewidths=0.9,
                alpha=0.8,
                zorder=2,
            )
        )

    if removed:
        ax_series.scatter(
            [index_by_node[n] for n in removed],
            [frequencies[n] for n in removed],
            color="lightgray", marker="x", s=35,
            label="Eliminados", zorder=3,
        )

    survivors = list(active.nodes())

    if survivors:
        ax_series.scatter(
            [index_by_node[n] for n in survivors],
            [frequencies[n] for n in survivors],
            c=[colors[n] for n in survivors],
            edgecolors="black", linewidths=0.5,
            s=35, zorder=4,
        )

    for node, color, marker, label in (
        (last_removed, "red", "x", "Último eliminado"),
        (next_target, "darkorange", "o", "Siguiente objetivo"),
    ):
        if node is not None:
            ax_series.scatter(
                [index_by_node[node]], [frequencies[node]],
                color=color, marker=marker, s=100,
                label=f"{label}: {node}", zorder=5,
            )

    ticks = np.unique(
        np.linspace(0, len(nodes) - 1, min(7, len(nodes))).astype(int)
    )
    ax_series.set_xticks(ticks)
    ax_series.set_xticklabels(
        [dates_by_node[nodes[i]].strftime("%Y-%m-%d") for i in ticks],
        rotation=30, ha="right", fontsize=8,
    )
    ax_series.set_xlim(-1, len(nodes))
    ax_series.set_ylim(
        0, max(1.0, max(heights) * 1.15)
    )
    ax_series.set_title("Serie y conexiones de visibilidad")
    ax_series.set_xlabel("Semana")
    ax_series.set_ylabel("Cantidad")
    ax_series.legend(fontsize=8, loc="best")

    # 2. Estructura topológica con posiciones y límites fijos.
    if removed:
        nx.draw_networkx_nodes(
            original, pos, nodelist=removed,
            node_color="lightgray", node_size=60,
            alpha=0.4, ax=ax_network,
        )

    if survivors:
        nx.draw_networkx_edges(
            active, pos,
            edge_color=[colors[u] for u, v in active.edges()],
            width=1.1, alpha=0.8, ax=ax_network,
        )
        nx.draw_networkx_nodes(
            active, pos, nodelist=survivors,
            node_color=[colors[n] for n in survivors],
            node_size=150, edgecolors="black",
            linewidths=0.5, ax=ax_network,
        )
        nx.draw_networkx_labels(
            active, pos, font_size=6, ax=ax_network,
        )

    if next_target is not None:
        nx.draw_networkx_nodes(
            original, pos, nodelist=[next_target],
            node_color="darkorange", node_size=230,
            edgecolors="black", ax=ax_network,
        )

    if last_removed is not None:
        ax_network.scatter(
            [pos[last_removed][0]], [pos[last_removed][1]],
            color="red", marker="x", s=120, zorder=5,
        )

    coordinates = np.asarray([pos[n] for n in original.nodes()])

    for dimension, set_limit in (
        (0, ax_network.set_xlim),
        (1, ax_network.set_ylim),
    ):
        minimum = coordinates[:, dimension].min()
        maximum = coordinates[:, dimension].max()
        margin = max((maximum - minimum) * 0.12, 0.1)
        set_limit(minimum - margin, maximum + margin)

    ax_network.set_title("Componentes activas y nodos eliminados")
    ax_network.set_axis_off()

    # 3. Curvas del registro, reveladas hasta el paso actual.
    history = records[:step + 1]
    steps = [state["step"] for state in history]
    counts = [state["n_components"] for state in history]
    largest_sizes = [state["largest_component"] for state in history]

    ax_components.plot(
        steps, counts, color="crimson", linewidth=1.8
    )
    ax_components.scatter(
        [step], [counts[-1]], color="black", s=45, zorder=3
    )
    ax_components.set_title("Número de componentes")
    ax_components.set_ylabel("Componentes conexos")

    ax_largest.plot(
        steps, largest_sizes, color="royalblue", linewidth=1.8
    )
    ax_largest.scatter(
        [step], [largest_sizes[-1]], color="black", s=45, zorder=3
    )
    ax_largest.set_title("Tamaño de la componente mayor")
    ax_largest.set_ylabel("Nodos")

    # Escalas constantes durante toda la reproducción.
    max_components = max(r["n_components"] for r in records)
    initial_nodes = simulation["initial_nodes"]

    ax_components.set_ylim(0, max(1, max_components) * 1.15)
    ax_largest.set_ylim(0, max(1, initial_nodes) * 1.05)

    for ax in (ax_components, ax_largest):
        ax.axvline(step, color="gray", linestyle="--", alpha=0.6)
        ax.set_xlim(0, max(1, len(removal_order)))
        ax.set_xlabel("Nodos eliminados")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(linestyle="--", alpha=0.3)

    series_name = simulation.get("series_name") or "Grafo empírico"
    fig.suptitle(
        f"{series_name} — Ataque por grado inicial\n"
        f"Paso {step}/{initial_nodes}",
        fontsize=15, fontweight="bold",
    )

    plt.close(fig)
    return fig
# endregion











