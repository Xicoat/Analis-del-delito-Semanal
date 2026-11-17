# %% RELOAD
%load_ext autoreload
%autoreload 2



import logging

logging.getLogger("Delitos").setLevel(logging.WARNING)
logging.getLogger("Delitos").setLevel(logging.INFO)



# %% INIT
# region INIT
import sys
from pathlib import Path
import pandas as pd
import importlib
from IPython.display import JSON
import matplotlib.pyplot as plt
import logging
import pickle

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

REPORT_PATH = PROJECT_ROOT / "data" / "reports"

from src import (
    benchmarks,
    config,
    data_setup,
    distributions,
    experiments,
    export,
    graph,
    metrics,
    plots,
    reports,
    resilience,
    synthetic,
    windows,
    widgets,
)

config.configure_logging()
#logging.getLogger("Delitos").setLevel(logging.WARNING)
#logging.getLogger("Delitos").setLevel(logging.INFO)
# endregion






# %% Extracción
# region Extracción
TARGET_CRIME = "HOMICIDIO POR ARMA DE FUEGO"

crime_raw = data_setup.extract_csv_data(path=config.DATA_PATH, target_crime=TARGET_CRIME)
crime_dates = data_setup.extract_dates(crime_data=crime_raw, date_column="fecha_hecho", crime_column="delito")
crime_filtered = data_setup.filter_by_date_range(crime_data=crime_dates, date_column="fecha_hecho", start=config.START_DATE, end=config.END_DATE)
crime_weekly = data_setup.events_by_week(crime_data=crime_filtered, date_column="fecha_hecho", origin_date=config.START_DATE)
crime_nodes = data_setup.assign_network_nodes(crime_data=crime_weekly)


# endregion






# %% Grafo
# region Grafo y metricas
global_visibility_graph = graph.build_visibility_graph(crime_data=crime_nodes)
node_metrics = metrics.compute_node_metrics(grafo=global_visibility_graph)
# endregion





# %% Plot Histograma
# region Histograma
fig = plots.plot_histogram(node_metrics=node_metrics)
fig.show()
# endregion 




# %% Plot Grafo
fig = plots.plot_visibility_graph_from_network(graph=global_visibility_graph)
fig.show()
# %%





# %% Plot Red
graph_metrics = metrics.compute_graph_metrics(grafo=general_visibility_graph)
plots.plot_network(grafo=general_visibility_graph, metricas=graph_metrics)
# %%





# %% Plot 
plots.plot_degree_distribution(metricas_grafo=graph_metrics)
# %%






# %% Distributions
# region Distributions
distribution_metrics = distributions.compute_distribution_metrics(metricas_grafo=graph_metrics)
plots.plot_power_law_fit(metricas_distribucion=distribution_metrics)
plots.plot_model_comparison(metricas_distribucion=distribution_metrics)
df_distancias = pd.DataFrame({
    "Modelo": ["Ley de Potencia", "Log-Normal", "Exponencial"],
    "Distancia Euclidiana": [
        distribution_metrics["dist_eucl_pl"],
        distribution_metrics["dist_eucl_lognormal"],
        distribution_metrics["dist_eucl_exp"],
    ],
}).sort_values("Distancia Euclidiana").reset_index(drop=True)

display(df_distancias)
plots.plot_csn_analysis(metricas_distribucion=distribution_metrics)
# endregion






# %% Resilience
# region Resilience
resilience_results = resilience.resilience_analysis(grafo=global_visibility_graph, seed=42)
# endregion






# %% Diccionario de resiliencia.
# region Diccionario de resiliencia.
datos_limpios = resilience.preparar_para_visor(resilience_results)
JSON(datos_limpios)
# endregion





# %% plot_resilience_grid
# region 
plots.plot_resilience_grid(resilience_results=resilience_results, strategy = "ambas")
plots.plot_closest_resilience_models(resilience_results=resilience_results)
# endregion





# %% rolling_graphs
# region rolling_graphs
rolling_windows = windows.generate_rolling_windows(crime_data=crime_nodes, window_size=52, step=4)
rolling_graphs = graph.build_rolling_graphs(rolling_windows=rolling_windows)
# endregion





# %% Métricas por ventana
# region Métricas por ventana
rolling_metrics = metrics.compute_rolling_metrics(rolling_graphs=rolling_graphs)
rolling_metrics["windows"][0]["graph_metrics"]
rolling_metrics["windows"][0]["distribution_metrics"]
reports.print_rolling_metrics(rolling_metrics=rolling_metrics)
# endregion





# %% Resiliencia por ventanas
# region rolling_resilience_analysis
rolling_resilience_results = resilience.rolling_resilience_analysis(rolling_graphs=rolling_graphs, seed=42)
reports.print_rolling_resilience_results(rolling_resilience_results=rolling_resilience_results)
# endregion






# %%
# region plot_resilience_transitions
# Ataque: tres primeros lugares.
fig = plots.plot_resilience_transitions(rolling_resilience_results=rolling_resilience_results, strategy="ataque", top_n=3)
plt.show()
# Falla aleatoria: cinco primeros lugares.
fig = plots.plot_resilience_transitions(rolling_resilience_results=rolling_resilience_results, strategy="aleatorio", top_n=5)
plt.show()
# Únicamente el ganador bajo ataque.
fig = plots.plot_resilience_transitions(rolling_resilience_results=rolling_resilience_results, strategy="ataque", top_n=1)
plt.show()
# endregion





# %%  resilience_winner
# region resilience_winner
fig = plots.plot_resilience_winner_frequency(rolling_resilience_results=rolling_resilience_results, strategy="ataque", lugar=1)
plt.show()
fig = plots.plot_resilience_winner_frequency(rolling_resilience_results=rolling_resilience_results, strategy="aleatorio", lugar=2)
plt.show()
# endregion




# %%  Save_To_File
# region  resilience_winner
# %% Ejecutar experimento
experiment_results = experiments.run_rolling_experiment(crime_data=crime_nodes, window_size=52, step=1, seed=42)

# %% Guardar experimento
ruta_reporte = export.save_rolling_experiment(experiment_results=experiment_results, report_path=config.REPORT_PATH)
# endregion




# %%
# region 1 WIDGET DATASET SELECTION
dataset_selection = widgets.select_crime(crime_names=config.DATASETS_NAMES)
# endregion

# %% 
# region 2 SETUP PATHS 
dataset_name = dataset_selection.value
dataset_path = config.DATA_SOURCES[dataset_name]
reports_folder_path = REPORT_PATH / dataset_name
series_nodes_path = REPORT_PATH / dataset_name / "series_nodes"

print(dataset_name)
print(dataset_path)
print(reports_folder_path)
print(series_nodes_path)
# endregion

# %% 
# region 4 PROCESS CSV 
series_nodes = experiments.prepare_series_nodes(dataset_name, dataset_path, 
                config.START_DATE, config.END_DATE)
# endregion

# %%
# region 5 SAVE SERIES NODES
export.save_series_nodes(series_nodes, report_path = REPORT_PATH)
# endregion

# %% 
# region 6 WIDGET SELECT SERIES NODES
print(series_nodes_path)
series_nodes_selection = widgets.select_series_nodes(series_nodes_path)
# endregion 


# %%
# region 7 LOAD SERIES_NODES FROM FILE
series_nodes_path = series_nodes_selection.value
print(series_nodes_path)
series_nodes =  data_setup.load_series_nodes(series_nodes_path)
# endregion

# %%
# region 8 BUILD GENERAL VISIBILITY GRAPH
general_visibility_graph = graph.build_visibility_graph(series_nodes, messages=True)
# endregion

# %%
# region 9 RUN ROLLING EXPERIMENTS
combinaciones = config.combinaciones
experiments.run_rolling_experiments(crime_data=series_nodes, combinaciones=combinaciones, report_path=config.REPORT_PATH, seed=42)
# endregion

# %% 
# region 10 WIDGET SELECT SINGLE REPORT
report_folder = REPORT_PATH / dataset_name
print(report_folder)
selector_reporte = widgets.select_report(report_path=report_folder)
# endregion

# %%
# region 11  LOAD SINGLE ROLLING REPORT
rolling_report_name = selector_reporte.value
print(rolling_report_name)
rolling_experiment = data_setup.load_rolling_experiment(rolling_report_name)
# endregion

# %% 
# region 12 GRAPH + RESILIENCE
rolling_resilience_rankings = reports.extract_rolling_resilience_rankings(experimento=rolling_experiment)
resilience_graph = reports.merge_resilience_graph(node_metrics=node_metrics, rolling_resilience_rankings=rolling_resilience_rankings)
# endregion


# %% 
# region 13 WIDGET LOAD MULTIPLE ROLLING REPORTS
experiments = data_setup.load_rolling_experiments(carpeta_delito_path=report_folder)
# endregion

# %%
# region 14 COMPUTE NODE METRICS
node_metrics = metrics.compute_node_metrics(grafo=general_visibility_graph)



# region PLOTS
# %% PLOT RESILIENCE TRANSITIONS GRID
fig = plots.plot_resilience_transitions_grid(experiments, strategy="ataque", top_n=3)
fig.savefig(report_folder / "transiciones_ataque_top3.png", dpi=300, bbox_inches="tight")
fig.savefig(report_folder / "transiciones_ataque_top3.pdf", bbox_inches="tight")

# %% PLOT RESILIENCE FREQUENCY GRID
strategy = "ataque"
lugar = 1
fig = plots.plot_resilience_winner_frequency_grid(experimentos=experiments, strategy=strategy, lugar=lugar)
nombre = f"frecuencias_{strategy}_lugar{lugar}_grid"
fig.savefig(reports_folder_path / f"{nombre}.png", dpi=300, bbox_inches="tight")
fig.savefig(reports_folder_path / f"{nombre}.pdf", bbox_inches="tight")
plt.close(fig)


# %% PLOT SINGLE ROLLING EXPERIMENT PREVIOUSLY SELECTED
fig = plots.plot_resilience_transitions(rolling_experiment["rolling_resilience_results"], strategy="ataque", top_n=3)
plt.show()

fig = plots.plot_resilience_winner_frequency(rolling_experiment["rolling_resilience_results"], strategy="ataque", lugar=1)
plt.show()

# %% PLOT HISTOGRAM
fig = plots.plot_histogram(node_metrics=node_metrics)
fig.show()




# endregion


# region PLOT HISTOGRAM + RESILIENCE
# %% WIDGET Mostrar controles
selector_histograma = widgets.select_histogram_resilience_options()

# %% Recuperar selección
strategy = selector_histograma["strategy"].value
posicion = selector_histograma["posicion"].value
modelos = list(selector_histograma["modelos"].value)
print(strategy, posicion, modelos)

# %% Plot
fig = plots.plot_histogram_resilience(resilience_graph=resilience_graph, strategy=strategy, posicion=posicion, modelos=modelos)
fig.show()
# endregion 



# region EXPERIMENTO 7
# %% GUARDAR HISTOGRAMA DE RESILIENCIA
crime_name = resilience_graph["crime_name"]

carpeta_delito_path = REPORT_PATH / crime_name
carpeta_delito_path.mkdir(parents=True, exist_ok=True)

nombre = (
    f"{crime_name}_histograma_resiliencia"
    f"_{strategy}_posicion{posicion}"
    f"_W{resilience_graph['window_size']}_S{resilience_graph['step']}"
)

fig.write_html(carpeta_delito_path / f"{nombre}.html")
fig.write_image(carpeta_delito_path / f"{nombre}.png", width=1600, height=900, scale=2)
fig.write_image(carpeta_delito_path / f"{nombre}.pdf", width=1600, height=900)

# endregion EXPERIMENTO 7













# %%
import sys
import subprocess

print("Python del kernel:", sys.executable)
subprocess.run([sys.executable, "-m", "pip", "show", "kaleido", "plotly"])


# %% Diccionario ventanas.
# region Diccionario de ventanas.
JSON(rolling_resilience_results)
# endregion






# %% TEST NO EJECUTAR
importlib.reload(experiments)
# %%
