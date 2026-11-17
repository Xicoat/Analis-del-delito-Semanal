# region Libraries
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("Delitos")
# endregion


# region generate_synthetic_csv
def generate_synthetic_csv(
    n_weeks: int,
    output_path: str | Path,
    periodo: int = 12,
    start_date: str | pd.Timestamp = "2016-01-01",
) -> dict[str, Path]:
    """Genera cinco CSV sintéticos con una fila por evento.

    Produce señales sinusoidal, cuadrada, triangular y dos series
    con picos periódicos. Cada observación de la señal representa
    el número de eventos de un bloque de siete días.

    Reglas internas:
        - Periodo: Configurable en semanas (por defecto 12).
        - Ondas: entre 10 y 90 eventos.
        - Spike: base 10 e incremento de 80 cada ciclo del periodo.
        - Spike X3: base 10 e incremento de 240 en las mismas semanas.

    Los picos aparecen al final de cada ciclo del periodo indicado.
    No agrega ruido. Redondea los conteos a enteros.

    Distribuye los eventos dentro de cada semana y conserva la
    estructura de columnas de los CSV de delitos.

    Devuelve un diccionario con los nombres de las series y sus rutas.
    Los archivos generados incluyen el sufijo '_P{periodo}' en su nombre.
    No sobrescribe archivos existentes.
    """

    if (
        isinstance(n_weeks, bool)
        or not isinstance(n_weeks, int)
        or n_weeks < 1
    ):
        raise ValueError("n_weeks debe ser un entero positivo.")

    if (
        isinstance(periodo, bool)
        or not isinstance(periodo, int)
        or periodo < 2
    ):
        raise ValueError("periodo debe ser un entero de al menos 2 semanas.")

    inicio = pd.Timestamp(start_date)

    if pd.isna(inicio):
        raise ValueError("start_date debe ser una fecha válida.")

    inicio = inicio.normalize()
    carpeta = Path(output_path)

    # 1. Definir las señales semanales dependientes del periodo.
    semanas = np.arange(n_weeks)
    fase = (semanas % periodo) / periodo

    sinusoidal = np.rint(
        50 + 40 * np.sin(2 * np.pi * fase)
    ).astype(int)

    cuadrada = np.where(fase < 0.5, 90, 10)

    triangular = np.rint(
        10 + 80 * (1 - np.abs(2 * fase - 1))
    ).astype(int)

    hay_pico = (semanas + 1) % periodo == 0
    spike = np.where(hay_pico, 90, 10)
    spike_x3 = np.where(hay_pico, 250, 10)

    senales = {
        "DELITO SINWAVE": sinusoidal,
        "DELITO SQUARE": cuadrada,
        "DELITO TRIANGULAR": triangular,
        "DELITO SPIKE": spike,
        "DELITO SPIKE X3": spike_x3,
    }

    # Se inyecta el sufijo del periodo en el nombre del archivo
    rutas = {
        nombre: carpeta / f"{nombre.replace(' ', '_')}_P{periodo}.csv"
        for nombre in senales
    }

    existentes = [ruta.name for ruta in rutas.values() if ruta.exists()]

    if existentes:
        raise FileExistsError(
            f"Ya existen estos archivos: {existentes}"
        )

    carpeta.mkdir(parents=True, exist_ok=True)

    meses = [
        "Enero", "Febrero", "Marzo", "Abril",
        "Mayo", "Junio", "Julio", "Agosto",
        "Septiembre", "Octubre", "Noviembre", "Diciembre",
    ]

    inicios_semanales = pd.date_range(
        start=inicio,
        periods=n_weeks,
        freq="7D",
    )

    segundos_semana = 7 * 24 * 60 * 60

    # 2. Convertir los conteos en registros individuales.
    for nombre, conteos in senales.items():
        fechas = []

        for fecha_semana, cantidad in zip(inicios_semanales, conteos):
            # Distribución uniforme dentro del bloque, sin invadir el siguiente.
            desplazamientos = (
                np.arange(int(cantidad), dtype=np.int64)
                * segundos_semana
                // int(cantidad)
            )

            fechas.extend(
                fecha_semana
                + pd.to_timedelta(desplazamientos, unit="s")
            )

        fechas = pd.DatetimeIndex(fechas)

        fechas_texto = fechas.strftime("%Y-%m-%d")
        horas_texto = fechas.strftime("%H:%M:%S")
        nombres_mes = [meses[mes - 1] for mes in fechas.month]

        df = pd.DataFrame({
            "anio_inicio": fechas.year,
            "mes_inicio": nombres_mes,
            "fecha_inicio": fechas_texto,
            "hora_inicio": horas_texto,
            "anio_hecho": fechas.year,
            "mes_hecho": nombres_mes,
            "fecha_hecho": fechas_texto,
            "hora_hecho": horas_texto,
            "delito": nombre,
            "categoria_delito": "DATOS SINTETICOS",
            "competencia": "FICTICIA",
            "fiscalia": "FICTICIA",
            "agencia": "FICTICIA",
            "unidad_investigacion": "FICTICIA",
            "colonia_hecho": "FICTICIA",
            "colonia_catalogo": "FICTICIA",
            "alcaldia_hecho": "FICTICIA",
            "alcaldia_catalogo": "FICTICIA",
            "municipio_hecho": "FICTICIO",
            "latitud": np.nan,
            "longitud": np.nan,
        })

        # 3. Verificar los conteos usando la columna que leerá el proceso.
        conteos_recuperados = (
            pd.DataFrame({
                "fecha_hecho": pd.to_datetime(df["fecha_hecho"]),
            })
            .resample("168h", on="fecha_hecho", origin=inicio)
            .size()
            .to_numpy()
        )

        if not np.array_equal(conteos_recuperados, conteos):
            raise ValueError(
                f"Los registros generados no reproducen la señal '{nombre}'."
            )

        # 4. Guardar el CSV sin una columna de índice adicional.
        df.to_csv(
            rutas[nombre],
            index=False,
            encoding="utf-8",
            mode="x",
        )

        logger.info(
            "EXITO. %s: %s semanas, %s eventos. Archivo: %s",
            nombre,
            n_weeks,
            len(df),
            rutas[nombre].resolve(),
        )

    return rutas
# endregion