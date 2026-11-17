# region Libraries
from pathlib import Path
import re

import ipywidgets as ipw
from IPython.display import display

import traceback


import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from . import plots, reports


import networkx as nx
import pandas as pd







# endregion


# region select_report_folder
def select_report_folder(report_path: str | Path) -> ipw.Dropdown:
    """Muestra las carpetas de reportes para seleccionar un delito.

    Devuelve el widget. Su propiedad .value contiene la ruta
    seleccionada como un objeto Path.
    """

    report_path = Path(report_path)

    if not report_path.is_dir():
        raise FileNotFoundError(
            f"No existe el directorio de reportes: {report_path}"
        )

    carpetas = sorted(
        (carpeta for carpeta in report_path.iterdir() if carpeta.is_dir()),
        key=lambda carpeta: carpeta.name.casefold(),
    )

    if not carpetas:
        raise ValueError("No hay carpetas de delitos en el directorio.")

    selector = ipw.Dropdown(
        options=[(carpeta.name, carpeta) for carpeta in carpetas],
        description="Carpeta:",
        layout=ipw.Layout(width="700px"),
        style={"description_width": "initial"},
    )

    display(selector)
    return selector
# endregion






# region select_crime
def select_crime(crime_names: list[str]) -> ipw.Dropdown:
    """Muestra un selector de delitos.

    Devuelve el widget. Su propiedad .value contiene el nombre
    del delito seleccionado.
    """

    if not crime_names:
        raise ValueError("La lista de delitos está vacía.")

    selector = ipw.Dropdown(
        options=crime_names,
        description="Delito:",
        layout=ipw.Layout(width="800px"),
        style={"description_width": "initial"},
    )

    display(selector)
    return selector
# endregion






# region select_series_node
def select_series_nodes(path: str | Path) -> ipw.Dropdown:
    """Selecciona un archivo Pickle con una serie preparada.

    Acepta una carpeta o un archivo .pkl. Ordena los archivos
    del más recientemente modificado al más antiguo.

    Devuelve el widget; .value contiene la ruta seleccionada.
    """

    ruta = Path(path)

    if not ruta.exists():
        raise FileNotFoundError(f"No existe la ruta: {ruta}")

    archivo_inicial = ruta if ruta.is_file() else None
    carpeta = ruta.parent if archivo_inicial is not None else ruta

    archivos = sorted(
        (archivo for archivo in carpeta.glob("*.pkl") if archivo.is_file()),
        key=lambda archivo: archivo.stat().st_mtime,
        reverse=True,
    )

    if not archivos:
        raise FileNotFoundError(f"No hay archivos .pkl en: {carpeta}")

    if archivo_inicial is not None and archivo_inicial not in archivos:
        raise ValueError("El archivo indicado debe tener extensión .pkl.")

    selector = ipw.Dropdown(
        options=[(archivo.name, archivo) for archivo in archivos],
        value=archivo_inicial if archivo_inicial is not None else archivos[0],
        description="Serie:",
        layout=ipw.Layout(width="100%"),
        style={"description_width": "initial"},
    )

    display(selector)
    return selector
# endregion





# region select_report
def select_report(report_path: str | Path) -> ipw.Dropdown:
    """Muestra los reportes Pickle, del último modificado al más antiguo."""

    carpeta = Path(report_path)

    if not carpeta.is_dir():
        raise FileNotFoundError(f"No existe el directorio: {carpeta}")

    archivos = sorted(
        (ruta for ruta in carpeta.glob("*.pkl") if ruta.is_file()),
        key=lambda ruta: ruta.stat().st_mtime,
        reverse=True,
    )

    if not archivos:
        raise FileNotFoundError(f"No hay reportes .pkl en: {carpeta}")

    selector = ipw.Dropdown(
        options=[(archivo.name, archivo) for archivo in archivos],
        description="Reporte:",
        layout=ipw.Layout(width="100%"),
        style={"description_width": "initial"},
    )

    display(selector)
    return selector
# endregion






# region select_global_graph_file
def select_global_graph_file(
    report_path: str | Path,
    crime_name: str,
) -> ipw.Dropdown:
    """Muestra los archivos de grafos completos de un delito.

    Devuelve el widget. Su propiedad .value contiene la ruta
    del archivo seleccionado.
    """

    nombre_delito = re.sub(
        r'[<>:"/\\|?*\x00-\x1f]',
        "_",
        crime_name,
    ).strip().rstrip(". ")

    if not nombre_delito:
        raise ValueError("El nombre del delito no permite localizar una carpeta.")

    carpeta_global = Path(report_path) / nombre_delito / "global"

    if not carpeta_global.is_dir():
        raise FileNotFoundError(
            f"No existe la carpeta de grafos completos: {carpeta_global}"
        )

    archivos = sorted(carpeta_global.glob("*.pkl"))

    if not archivos:
        raise FileNotFoundError(
            f"No hay archivos .pkl en: {carpeta_global}"
        )

    selector = ipw.Dropdown(
        options=[(archivo.name, archivo) for archivo in archivos],
        description="Grafo global:",
        layout=ipw.Layout(width="100%"),
        style={"description_width": "initial"},
    )

    display(selector)
    return selector
# endregion






# region select_histogram_resilience_options
def select_histogram_resilience_options() -> dict:
    """Display interactive widgets to select resilience histogram overlay options.

    Renders ``ipywidgets`` controls for choosing the node-removal strategy,
    ranking position, and theoretical models to highlight on the weekly series
    histogram. All UI labels and instructions are displayed in Spanish.

    Returns
    -------
    dict
        Dictionary mapping parameter names (``'strategy'``, ``'rank'``,
        ``'models'``) to their corresponding ``ipywidgets`` control objects,
        matching the arguments of ``plot_histogram_resilience``.
    """
    models = [
        "Watts Strogatz",
        "Erdos Renyi",
        "Barabasi Albert (m=2)",
        "Barabasi Albert (m=3)",
        "Barabasi Albert (m=4)",
        "Random Regular (k=2)",
        "Enlace Igualitario 1",
        "Enlace Igualitario 2",
        "Enlace Igualitario 3",
        "Star",
        "Grid 2D",
        "Ladder",
        "Hypercube",
        "Complete",
        "Path",
        "Cycle",
        "Circular Ladder",
    ]

    strategy_widget = ipw.Dropdown(
        options=[
            ("Ataque dirigido", "attack"),
            ("Falla aleatoria", "random"),
        ],
        description="Estrategia:",
        style={"description_width": "initial"},
    )

    rank_widget = ipw.Dropdown(
        options=[1, 2, 3, 4, 5],
        value=1,
        description="Posición:",
        style={"description_width": "initial"},
    )

    models_widget = ipw.SelectMultiple(
        options=models,
        value=(),
        description="Modelos:",
        rows=len(models),
        layout=ipw.Layout(width="600px"),
        style={"description_width": "initial"},
    )

    display(
        ipw.VBox([
            strategy_widget,
            rank_widget,
            ipw.HTML(
                "Selecciona uno o varios modelos "
                "usando Ctrl + clic o Shift + clic."
            ),
            models_widget,
        ])
    )

    return {
        "strategy": strategy_widget,
        "rank": rank_widget,
        "models": models_widget,
    }
# endregion







# region select_window_analysis
def select_window_analysis(rolling_experiment: dict) -> ipw.VBox:
    """Muestra y actualiza las cuatro gráficas de la ventana seleccionada.

    Recupera el grafo, sus métricas y la distribución de grados.
    Utiliza los resultados guardados sin recalcularlos.
    """

    ventanas = rolling_experiment["rolling_metrics"]["windows"]
    window_ids = sorted(ventana["window_id"] for ventana in ventanas)

    if not window_ids:
        raise ValueError("El experimento no contiene ventanas.")

    if len(set(window_ids)) != len(window_ids):
        raise ValueError("Hay identificadores de ventana duplicados.")

    selector = ipw.Dropdown(
        options=[(f"Ventana {wid}", wid) for wid in window_ids],
        description="Ventana:",
        style={"description_width": "initial"},
        layout=ipw.Layout(width="300px"),
    )

    salida = ipw.Output()

    def mostrar_figura(fig):
        """Muestra las figuras devueltas y evita su salida automática."""
        if fig is None:
            return

        if isinstance(fig, Figure):
            plt.close(fig)
            display(fig)
        else:
            fig.show()

    def actualizar(window_id):
        with salida:
            salida.clear_output(wait=True)

            try:
                window_graph = reports.get_graph_by_window(
                    rolling_experiment, window_id=window_id
                )
                window_metrics = reports.get_graph_metrics_by_window(
                    rolling_experiment, window_id=window_id
                )
                window_distribution = (
                    reports.get_distribution_metrics_by_window(
                        rolling_experiment, window_id=window_id
                    )
                )

                window_resilience = reports.get_resilience_results_by_window(
                    rolling_experiment, window_id
                )   

                grafo = window_graph["grafo"]
                graph_metrics = window_metrics["graph_metrics"]
                distribution_metrics = (
                    window_distribution["distribution_metrics"]
                )
                start_date = window_metrics["start_date"]
                end_date = window_metrics["end_date"]

                print(
                    f"{window_metrics['crime_name']} | "
                    f"Ventana {window_id} | "
                    f"W={window_metrics['window_size']} | "
                    f"S={window_metrics['step']}"
                )
                print(f"Periodo: {start_date} → {end_date}")

                # 1. Serie temporal con las conexiones de visibilidad.
                mostrar_figura(
                    plots.plot_visibility_graph_from_network(grafo)
                )

                # 2. Grados individuales ordenados por conectividad.
                mostrar_figura(
                    plots.plot_degree_distribution_sorted(
                        metricas_grafo=graph_metrics,
                        start_date=start_date,
                        end_date=end_date,
                    )
                )

                # 3. Distribución empírica de grados P(k).
                mostrar_figura(
                    plots.plot_degree_distribution(
                        distribution_metrics=distribution_metrics,
                        start_date=start_date,
                        end_date=end_date,
                    )
                )

                # 4. Estructura topológica del grafo.
                mostrar_figura(
                    plots.plot_network(grafo=grafo, metricas=graph_metrics)
                )


                mostrar_figura(
                    plots.plot_closest_resilience_models(
                        window_resilience["resilience_results"]
                    )
                )

                mostrar_figura(
                    plots.plot_resilience_grid(
                        window_resilience["resilience_results"],
                        strategy = "ambas"
                        )
                )

            except Exception:
                traceback.print_exc()

    def al_cambiar(change):
        actualizar(change["new"])

    selector.observe(al_cambiar, names="value")

    panel = ipw.VBox([selector, salida])
    display(panel)
    actualizar(selector.value)

    return panel
# endregion






# region show_resilience_simulation
def show_resilience_simulation(
    simulation: dict,
    interval: int = 1000,
) -> ipw.VBox:
    """Display interactive playback controls and synchronized views for an attack simulation.

    Renders previous/next/reset buttons, an ``ipywidgets.Play`` controller, a
    synchronized step slider, and an output pane showing step-by-step
    fragmentation metrics alongside ``plots.plot_resilience_simulation_step``.
    Does not rerun the simulator or mutate its results. All rendered widget
    labels, button descriptions, and status messages are formatted in Spanish.

    Parameters
    ----------
    simulation : dict
        Simulation record dictionary returned by
        ``resilience.resilience_simulator``, containing ``'graph'`` and
        ``'attack_log'``.
    interval : int, default=1000
        Playback interval between steps in milliseconds. Must be a positive
        integer.

    Returns
    -------
    ipw.VBox
        Displayed container widget holding the playback controls, step slider,
        and synchronized output area.

    Raises
    ------
    ValueError
        If ``interval`` is not a positive integer, or if ``simulation`` has an
        empty graph or empty ``'attack_log'``.
    """
    if isinstance(interval, bool) or not isinstance(interval, int):
        raise ValueError("interval debe ser un entero positivo.")

    if interval <= 0:
        raise ValueError("interval debe ser un entero positivo.")

    graph = simulation["graph"]
    records = simulation["attack_log"]

    if graph.number_of_nodes() == 0 or not records:
        raise ValueError("La simulación no contiene una red para visualizar.")

    last_step = len(records) - 1

    # Calcular una sola disposición para toda la reproducción.
    pos = nx.spring_layout(graph, seed=42)

    previous_btn = ipw.Button(description="← Anterior")
    next_btn = ipw.Button(description="Siguiente →")
    reset_btn = ipw.Button(description="Reiniciar")

    # Play incluye sus propios controles de reproducción y pausa.
    player = ipw.Play(
        value=0,
        min=0,
        max=last_step,
        step=1,
        interval=interval,
        description="Reproducir",
    )

    step_slider = ipw.IntSlider(
        value=0,
        min=0,
        max=last_step,
        step=1,
        description="Paso:",
        continuous_update=False,
        layout=ipw.Layout(width="650px"),
    )

    link = ipw.link(
        (player, "value"),
        (step_slider, "value"),
    )

    output = ipw.Output(
        layout=ipw.Layout(
            height="1250px",
            overflow="auto",
        )
    )

    def format_percentage(value: float | None) -> str:
        return "No aplica" if value is None else f"{value:.1f} %"

    def update(step: int) -> None:
        state = records[step]
        previous_btn.disabled = step == 0
        next_btn.disabled = step == last_step

        with output:
            output.clear_output(wait=True)

            try:
                print(
                    f"Paso: {step}/{last_step} | "
                    f"Eliminados: {state['removed_pct']:.1f} %"
                )

                if step > 0:
                    print(
                        f"Último nodo: {state['removed_node']} | "
                        f"Grado inicial: {state['initial_degree']} | "
                        "Grado antes de eliminarlo: "
                        f"{state['degree_before_removal']}"
                    )

                print(
                    f"Nodos restantes: {state['remaining_nodes']} | "
                    f"Aristas restantes: {state['remaining_edges']} | "
                    f"Aislados: {state['isolated_nodes']}"
                )
                print(
                    f"Componentes: {state['n_components']} | "
                    f"Tamaños: {state['component_sizes']}"
                )

                original_pct = format_percentage(
                    state["largest_original_pct"]
                )
                remaining_pct = format_percentage(
                    state["largest_remaining_pct"]
                )

                print(
                    f"Componente mayor: {state['largest_component']} | "
                    f"Del total original: {original_pct} | "
                    f"De los restantes: {remaining_pct}"
                )

                if state["component_change"] is not None:
                    print(
                        "Cambio de componentes en este paso: "
                        f"{state['component_change']:+d}"
                    )

                fig = plots.plot_resilience_simulation_step(
                    simulation=simulation,
                    step=step,
                    pos=pos,
                )
                display(fig)

            except Exception:
                traceback.print_exc()

    def on_change(change) -> None:
        update(change["new"])

    def step_back(_) -> None:
        step_slider.value = max(0, step_slider.value - 1)

    def step_forward(_) -> None:
        step_slider.value = min(
            last_step, step_slider.value + 1
        )

    def reset_to_start(_) -> None:
        step_slider.value = 0

    previous_btn.on_click(step_back)
    next_btn.on_click(step_forward)
    reset_btn.on_click(reset_to_start)
    step_slider.observe(on_change, names="value")

    controls = ipw.HBox(
        [previous_btn, next_btn, reset_btn, player],
        layout=ipw.Layout(flex_flow="row wrap"),
    )

    panel = ipw.VBox([controls, step_slider, output])

    # Conservar el enlace mientras exista el panel.
    panel._resilience_link = link

    display(panel)
    update(0)

    return panel
# endregion






# region select_window
def select_window(rolling_experiment: dict) -> ipw.Dropdown:
    """Create and display an interactive dropdown widget to select a rolling window.

    Builds an ``ipywidgets.Dropdown`` populated with the available windows in
    ``rolling_experiment['rolling_metrics']['windows']``, formatting each
    option with its ``window_id`` and covered date range while storing the
    integer ``window_id`` as the widget value. All rendered widget labels and
    descriptions are formatted in Spanish.

    Parameters
    ----------
    rolling_experiment : dict
        Consolidated rolling-window experiment dictionary returned by
        ``run_rolling_experiment``, containing
        ``'rolling_metrics'['windows']`` with ``'window_id'``,
        ``'start_date'``, and ``'end_date'`` entries.

    Returns
    -------
    ipw.Dropdown
        Interactive dropdown widget whose ``.value`` attribute holds the
        selected ``window_id`` (int), or ``None`` if no windows exist.
    """
    windows = rolling_experiment["rolling_metrics"]["windows"]

    options = [
        (
            f"Ventana {w['window_id']} "
            f"({pd.Timestamp(w['start_date']).strftime('%Y-%m-%d')} – "
            f"{pd.Timestamp(w['end_date']).strftime('%Y-%m-%d')})",
            w["window_id"],
        )
        for w in windows
    ]

    selector = ipw.Dropdown(
        options=options,
        value=options[0][1] if options else None,
        description="Ventana:",
        style={"description_width": "initial"},
        layout=ipw.Layout(width="420px"),
    )

    display(selector)
    return selector
# endregion





