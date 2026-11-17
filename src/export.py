# export.py

# region Libraries
import logging
import pickle
import re
import os
import stat
import time
import shutil
from pathlib import Path
from uuid import uuid4
import shutil

import networkx as nx


logger = logging.getLogger("Delitos")
# endregion




# region _handle_remove_readonly
def _handle_remove_readonly(func, path: str, exc_info: BaseException) -> None:
    """Force delete read-only or temporarily locked files on Windows.

    Parameters
    ----------
    func : callable
        Filesystem removal function that raised the exception (e.g.,
        ``os.unlink`` or ``os.rmdir``).
    path : str
        Target file or directory path that failed to be removed.
    exc_info : BaseException
        Exception instance raised during removal (Python 3.12+ ``onexc`` API).
    """
    os.chmod(path, stat.S_IWRITE)
    func(path)
# endregion





# region save_rolling_experiment
def save_rolling_experiment(
    experiment_results: dict,
    report_path: str | Path,
    replace_existing: bool = True,
) -> Path:
    """Serialize a rolling-window experiment to disk and optionally replace older runs.

    Writes the new ``.pkl`` file first, verifies that it exists and is non-empty,
    and only then removes previous ``.pkl`` files matching the same
    ``(series_name, window_size, step)`` combination.

    Parameters
    ----------
    experiment_results : dict
        Dictionary returned by ``experiments.run_rolling_experiment``, containing
        ``'series_name'``, ``'window_size'``, ``'step'``, and ``'experiment_id'``.
    report_path : str | Path
        Base output directory where the series subdirectory will be created.
    replace_existing : bool, default=True
        If ``True``, deletes older ``.pkl`` files for the same ``(window_size, step)``
        only after the new file has been written and verified on disk.

    Returns
    -------
    Path
        Path to the newly saved ``.pkl`` file.

    Raises
    ------
    RuntimeError
        If the newly written file is missing or empty on disk.
    """
    series_name = experiment_results["series_name"]
    window_size = experiment_results["window_size"]
    step = experiment_results["step"]
    experiment_id = experiment_results["experiment_id"]

    series_dir = Path(report_path) / series_name
    series_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{series_name}_W{window_size}_S{step}_"
    output_path = series_dir / f"{prefix}{experiment_id}.pkl"

    # 1. Write the new file first (never touch old files before this succeeds).
    with output_path.open("wb") as f_out:
        pickle.dump(experiment_results, f_out, protocol=pickle.HIGHEST_PROTOCOL)

    # 2. Verify that the new file exists and actually contains data.
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(
            f"El archivo guardado está vacío o no existe: {output_path}"
        )

    # 3. Safely remove previous files for the same (W, S) combination.
    if replace_existing:
        for old_file in series_dir.glob(f"{prefix}*.pkl"):
            if old_file.resolve() != output_path.resolve():
                old_file.unlink()
                logger.info("Archivo anterior reemplazado: %s", old_file.name)

    logger.info("EXITO ✅. Experimento guardado en: %s", output_path)
    return output_path
# endregion






# region save_global_visibility_graph
def save_global_visibility_graph(
    grafo: nx.Graph,
    report_path: str | Path,
) -> Path:
    """Guarda el grafo completo del delito en un archivo Pickle.

    Conserva los nodos, sus atributos, las aristas y los metadatos
    del grafo. Guarda en la subcarpeta 'global' del delito.

    Devuelve la ruta del archivo creado.
    """

    crime_name = grafo.graph["crime_name"]

    nombre_delito = re.sub(
        r'[<>:"/\\|?*\x00-\x1f]',
        "_",
        crime_name,
    ).strip().rstrip(". ")

    if not nombre_delito:
        raise ValueError("El nombre del delito no permite crear una carpeta.")

    carpeta_global = Path(report_path) / nombre_delito / "global"
    carpeta_global.mkdir(parents=True, exist_ok=True)

    identificador = uuid4().hex
    ruta = carpeta_global / f"{nombre_delito}_GLOBAL_{identificador}.pkl"

    with ruta.open("xb") as archivo:
        pickle.dump(
            grafo,
            archivo,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    logger.info("EXITO. Grafo completo guardado en: %s", ruta.resolve())
    return ruta
# endregion





# region save_series_nodes
def save_series_nodes(series_nodes: dict, report_path: str | Path) -> Path:
    """Serialize a prepared node-assigned time series to a Pickle file.

    Stores the dictionary inside a ``'series_nodes'`` subdirectory under the
    sanitized series folder to keep base series separate from experiment outputs.

    Parameters
    ----------
    series_nodes : dict
        Dictionary containing the following items:

        series_name : str
            Name of the analyzed time series.
        data : pd.DataFrame
            DataFrame with columns ``['node_id', 'timestamp', 'frequency']``.
    report_path : str | Path
        Base output directory where the series folder will be created.

    Returns
    -------
    Path
        Path to the newly created ``.pkl`` file.

    Raises
    ------
    ValueError
        If ``series_name`` cannot be sanitized into a valid directory name.
    FileExistsError
        If a file with the generated UUID already exists at the target path.
    """

    series_name = series_nodes["series_name"]

    folder_name = re.sub(
        r'[<>:"/\\|?*\x00-\x1f]', "_", series_name
    ).strip().rstrip(". ")

    if not folder_name:
        raise ValueError("El nombre del delito no permite crear una carpeta.")

    folder_path = Path(report_path) / folder_name / "series_nodes"
    folder_path.mkdir(parents=True, exist_ok=True)

    uid = uuid4().hex
    file_path = folder_path / f"{folder_name}_SERIES_NODES_{uid}.pkl"

    with file_path.open("xb") as f:
        pickle.dump(series_nodes, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("EXITO. Serie guardada en: %s", file_path.resolve())
    return file_path
# endregion






# region reset_reports_folder
def reset_reports_folder(reports_folder_path: str | Path) -> None:
    """Remove all contents inside the target reports directory.

    Preserves the root directory itself while deleting all nested files,
    symlinks, and subdirectories. Includes retry logic to tolerate transient
    locks from Windows Explorer or cloud sync clients (e.g., OneDrive).

    Parameters
    ----------
    reports_folder_path : str | Path
        Path to the existing directory whose contents will be purged.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If ``reports_folder_path`` does not exist, is not a directory, or is a
        symbolic link.
    PermissionError
        If a subdirectory remains locked by an external process after retries.
    """
    carpeta = Path(reports_folder_path)

    if not carpeta.is_dir() or carpeta.is_symlink():
        raise ValueError(f"La ruta no es un directorio válido: {carpeta}")

    for item in carpeta.iterdir():
        if item.is_symlink() or item.is_file():
            item.unlink()
        elif item.is_dir():
            for intento in range(3):
                try:
                    shutil.rmtree(item, onexc=_handle_remove_readonly)
                    break
                except PermissionError:
                    if intento == 2:
                        raise
                    time.sleep(0.5)

    logger.info("RESET. Contenido eliminado de: %s", carpeta.resolve())
# endregion






