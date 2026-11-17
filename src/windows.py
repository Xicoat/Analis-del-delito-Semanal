# region Libraries
import logging
from typing import Iterator

import pandas as pd

logger = logging.getLogger("Delitos")
# endregion






# region generate_rolling_windows
def generate_rolling_windows(
    series_nodes: dict,
    window_size: int,
    step: int = 1,
) -> dict:
    """Generate complete rolling windows and preserve series metadata.

    Slices the input time series into overlapping or disjoint windows of length
    ``window_size`` advancing by ``step`` rows. Each window stores boundary
    positions, start/end timestamps, and a DataFrame with local ``'node_id'``
    values reindexed sequentially from ``0`` to ``window_size - 1``.

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
        Positive integer specifying the number of consecutive weekly nodes in
        each window.
    step : int, default=1
        Positive integer specifying the number of nodes to advance between
        consecutive windows.

    Returns
    -------
    dict
        Dictionary containing windowing metadata and window slices:

        series_name : str
            Name of the analyzed time series.
        window_size : int
            Window length used to slice the series.
        step : int
            Step size between consecutive windows.
        windows : list[dict]
            List of window dictionaries (empty if ``len(data) < window_size``),
            each containing:

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
            df_window : pd.DataFrame
                Window slice with columns
                ``['node_id', 'timestamp', 'frequency']`` and local
                ``'node_id'`` reindexed from ``0`` to ``window_size - 1``.

    Raises
    ------
    ValueError
        If ``window_size`` or ``step`` is not a positive integer.
    """
    for param_name, param_value in (("window_size", window_size), ("step", step)):
        if isinstance(param_value, bool) or not isinstance(param_value, int) or param_value <= 0:
            raise ValueError(f"{param_name} debe ser un entero positivo.")

    logger.info(
        "Generando ventanas móviles (W=%s, S=%s)",
        window_size,
        step,
    )

    df = series_nodes["data"]
    n = len(df)

    results = {
        "series_name": series_nodes["series_name"],
        "window_size": window_size,
        "step": step,
        "windows": [],
    }

    if n < window_size:
        logger.warning(
            "Longitud de datos (%s) insuficiente para ventana (%s).",
            n,
            window_size,
        )
        return results

    for window_id, start_idx in enumerate(
        range(0, n - window_size + 1, step)
    ):
        end_idx = start_idx + window_size
        sub_df = df.iloc[start_idx:end_idx].copy().reset_index(drop=True)

        start_date = sub_df["timestamp"].iloc[0]
        end_date = sub_df["timestamp"].iloc[-1]

        # Local sequential node identifiers for building this window's NVG.
        sub_df["node_id"] = range(len(sub_df))

        results["windows"].append({
            "window_id": window_id,
            "start_idx": start_idx,
            "end_idx": end_idx - 1,
            "start_date": start_date,
            "end_date": end_date,
            "df_window": sub_df,
        })

    logger.info(
        "EXITO ✅. %s ventanas generadas.",
        len(results["windows"]),
    )

    return results
# endregion



