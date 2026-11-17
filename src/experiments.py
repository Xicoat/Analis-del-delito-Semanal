# experiments.py

# region Libraries
import logging
import pandas as pd

from . import graph, metrics, resilience, windows
from . import config, data_setup, graph, export

from pathlib import Path

logger = logging.getLogger("Delitos")
# endregion





# region run_rolling_experiment
def run_rolling_experiment(
    series_nodes: dict,
    window_size: int,
    step: int = 1,
    seed: int | None = None,
) -> dict:
    """Execute rolling-window graph metrics and resilience analyses for a series.

    Generates rolling windows from ``series_nodes``, constructs their Natural
    Visibility Graphs once, and computes both structural node/graph metrics and
    topological resilience metrics across all windows. Window records across
    branches are linked via ``'window_id'``.

    Does not generate plots or write files to disk.

    Parameters
    ----------
    series_nodes : dict
        Dictionary containing the discretized weekly time series with node
        indices:

        series_name : str
            Name of the analyzed time series (e.g., ``'FEMINICIDIO'``).
        data : pd.DataFrame
            DataFrame with columns ``['node_id', 'timestamp', 'frequency']``.
    window_size : int
        Number of consecutive weekly nodes included in each rolling window.
    step : int, default=1
        Number of nodes to advance between consecutive windows.
    seed : int | None, default=None
        Random seed forwarded to ``resilience.rolling_resilience_analysis`` for
        reproducible stochastic node removals.

    Returns
    -------
    dict
        Dictionary containing the experiment metadata and both analysis
        branches:

        experiment_id : str
            Unique identifier of the resilience/experiment run.
        executed_at : str | pd.Timestamp
            Execution timestamp of the experiment.
        series_name : str
            Name of the analyzed time series.
        window_size : int
            Window length used in the experiment.
        step : int
            Step size used between consecutive windows.
        removal_seed : int | None
            Random seed used for stochastic node removal simulations.
        rolling_metrics : dict
            Structural metrics computed per window by
            ``metrics.compute_rolling_metrics``.
        rolling_resilience_results : dict
            Resilience curves and fragmentation metrics computed per window by
            ``resilience.rolling_resilience_analysis``.

    Raises
    ------
    ValueError
        If ``windows.generate_rolling_windows`` produces an empty list of
        windows.
    """
    logger.info(
        "Iniciando experimento de ventanas: W=%s, S=%s.",
        window_size,
        step,
    )

    # 1. Generate rolling windows from the discretized weekly series.
    rolling_windows = windows.generate_rolling_windows(
        series_nodes=series_nodes,
        window_size=window_size,
        step=step,
    )

    if not rolling_windows["windows"]:
        raise ValueError("No se generaron ventanas para el experimento.")

    # 2. Build visibility graphs once per window.
    rolling_graphs = graph.build_rolling_graphs(
        rolling_windows=rolling_windows,
    )

    # 3. Compute both analytical branches on the same graphs.
    rolling_metrics = metrics.compute_rolling_metrics(
        rolling_graphs=rolling_graphs,
    )

    rolling_resilience_results = resilience.rolling_resilience_analysis(
        rolling_graphs=rolling_graphs,
        seed=seed,
    )

    # 4. Consolidate results under a single experiment identity.
    results = {
        "experiment_id": rolling_resilience_results["experiment_id"],
        "executed_at": rolling_resilience_results["executed_at"],
        "series_name": rolling_graphs["series_name"],
        "window_size": rolling_graphs["window_size"],
        "step": rolling_graphs["step"],
        "removal_seed": seed,
        "rolling_metrics": rolling_metrics,
        "rolling_resilience_results": rolling_resilience_results,
    }

    logger.info(
        "SUCCESS ✅. Experimento %s: '%s', %s ventanas analizadas.",
        results["experiment_id"],
        results["series_name"],
        len(rolling_graphs["windows"]),
    )

    return results
# endregion






# region run_rolling_experiments
def run_rolling_experiments(
    series_nodes: dict,
    windows_experiment_sizes: list[tuple[int, int]],
    report_path: str | Path,
    seed: int | None = None,
) -> None:
    """Execute and save rolling-window experiments for each (window_size, step) pair.

    Does not generate summary reports or accumulate batch results in memory.
    If an individual combination fails, the exception is logged and execution
    continues with the remaining combinations.

    Temporarily suppresses ``INFO`` logs from internal module functions while
    preserving ``WARNING`` and ``ERROR`` messages.

    Parameters
    ----------
    series_nodes : dict
        Dictionary containing the discretized weekly time series with node
        indices:

        series_name : str
            Name of the analyzed time series (e.g., ``'FEMINICIDIO'``).
        data : pd.DataFrame
            DataFrame with columns ``['node_id', 'timestamp', 'frequency']``.
    windows_experiment_sizes : list[tuple[int, int]]
        Sequence of ``(window_size, step)`` integer pairs defining the rolling
        window dimensions to evaluate across experiments.
    report_path : str | Path
        Destination directory where serialized experiment outputs will be saved.
    seed : int | None, default=None
        Random seed forwarded to ``run_rolling_experiment`` for reproducible
        stochastic simulations (e.g., random node failures).

    Returns
    -------
    None
        Results are persisted directly to disk via ``export.save_rolling_experiment``.
    """
    batch_logger = logging.getLogger("Experimentos")
    batch_logger.setLevel(logging.INFO)

    previous_level = logger.level
    succeeded = 0
    failed = 0

    try:
        logger.setLevel(logging.WARNING)

        for index, (window_size, step) in enumerate(windows_experiment_sizes, start=1):
            batch_logger.info(
                "Experimento %s/%s | W=%s, S=%s.",
                index,
                len(windows_experiment_sizes),
                window_size,
                step,
            )

            try:
                results = run_rolling_experiment(
                    series_nodes=series_nodes,
                    window_size=window_size,
                    step=step,
                    seed=seed,
                )

                try:
                    output_path = export.save_rolling_experiment(
                        experiment_results=results,
                        report_path=report_path,
                        replace_existing = True,
                    )
                finally:
                    # Release reference before the next experiment.
                    del results

                succeeded += 1
                batch_logger.info(
                    "SUCCESS ✅ | W=%s, S=%s | Archivo: %s",
                    window_size,
                    step,
                    output_path.resolve(),
                )

            except Exception:
                failed += 1
                batch_logger.exception(
                    "FAILURE ❌ | W=%s, S=%s.",
                    window_size,
                    step,
                )

        if failed > 0:
            batch_logger.error(
                "FAILURE ❌. Lote finalizado con errores: %s exitosos y %s fallidos.",
                succeeded,
                failed,
            )
        else:
            batch_logger.info(
                "SUCCESS ✅. Lote finalizado: %s exitosos y %s fallidos.",
                succeeded,
                failed,
            )

    finally:
        # Restore the logger level prior to the batch run.
        logger.setLevel(previous_level)
# endregion













