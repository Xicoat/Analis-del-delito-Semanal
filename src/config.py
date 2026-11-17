# region Setup
import logging
import sys
from pathlib import Path

import pandas as pd
# endsetup

# region Constants
# Fija el directorio raiz
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Path Data sources
DATA_PATH = PROJECT_ROOT / "data" / "delitos" / "csv"
SYNTHETIC_PATH = PROJECT_ROOT / "data" / "delitos" / "sinteticos"
REPORT_PATH = PROJECT_ROOT / "data" / "reports"

# Valores predeterminados
START_DATE = pd.Timestamp("2016-01-01")
END_DATE = pd.Timestamp("2024-06-30")
# Columna del csv que contiene la fecha del delito
DATE_COL = "fecha_hecho"

# Configuración default para ventanas moviles
WINDOW_SIZE = 52
STEP_SIZE = 12
# endregion 

windows_experiment_sizes = [(12, 1),(12, 4),(12, 12),(26, 1),(26, 4),(26, 12),(52, 1),
    (52, 4),(52, 12),(104, 1),(104, 4),(104, 12),]

# region logging
def configure_logging() -> None:
    """Configura la salida de logs al iniciar una sesión."""
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s - [%(funcName)s] - %(levelname)s - %(message)s",
        force=True,
    )
# endregion


DATA_SOURCES = {
    "HOMICIDIO POR ARMA DE FUEGO": DATA_PATH,
    "FEMINICIDIO": DATA_PATH,
    "ROBO DE VEHICULO DE SERVICIO PARTICULAR SIN VIOLENCIA": DATA_PATH,
    "ROBO A CASA HABITACION CON VIOLENCIA": DATA_PATH,
    "ROBO A CASA HABITACION SIN VIOLENCIA": DATA_PATH,
    "ROBO A TRANSEUNTE SALIENDO DEL CAJERO CON VIOLENCIA": DATA_PATH,
    "ROBO A TRANSEUNTE SALIENDO DEL BANCO CON VIOLENCIA": DATA_PATH,
    "DELITO SINWAVE": SYNTHETIC_PATH,
    "DELITO SQUARE": SYNTHETIC_PATH,
    "DELITO TRIANGULAR": SYNTHETIC_PATH,
    "DELITO SPIKE": SYNTHETIC_PATH,
    "DELITO SPIKE X3": SYNTHETIC_PATH,
    "DELITO ESCALERA": SYNTHETIC_PATH,
    "DELITO ESCALERA NOISE": SYNTHETIC_PATH,
    "DELITO 5 CICLOS LIMPIO": SYNTHETIC_PATH,
    "DELITO CICLO 5 RUIDO 4": SYNTHETIC_PATH,
}


# region DATASET NAMES
DATASETS_NAMES = list(DATA_SOURCES.keys())
# endregion