import logging

from src.config import DATA_PATH, configure_logging
from src.data_setup import extract_crime_data


logger = logging.getLogger("Delitos")


def main() -> None:
    configure_logging()

    # Parámetros de esta ejecución.
    target_crime = "ROBO A CASA HABITACION SIN VIOLENCIA"

    # Extracción de datos.
    df_raw = extract_crime_data(
        path=DATA_PATH,
        target_crime=target_crime,
    )

    logger.info("Registros extraídos: %s", len(df_raw))

    # Aquí continúa el procesamiento.


if __name__ == "__main__":
    main()