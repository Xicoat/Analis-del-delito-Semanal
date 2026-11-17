# region Libraries
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# endregion






# region generar_dataset_escalera_clean
def generar_dataset_escalera_clean(
    nombre="escalera_5_25_50_25_5",
    carpeta_salida="datasets_sinteticos",
):
    inicio = pd.Timestamp("2016-01-01")
    fin = pd.Timestamp("2024-06-30")

    columnas = [
        "anio_inicio", "mes_inicio", "fecha_inicio", "hora_inicio",
        "anio_hecho", "mes_hecho", "fecha_hecho", "hora_hecho",
        "delito", "categoria_delito", "competencia", "fiscalia",
        "agencia", "unidad_investigacion", "colonia_hecho",
        "colonia_catalogo", "alcaldia_hecho", "alcaldia_catalogo",
        "municipio_hecho", "latitud", "longitud",
    ]

    # Bloques consecutivos de 7 días, anclados al 1 de enero de 2016.
    semanas = pd.date_range(inicio, fin, freq="7D")
    secciones = np.array_split(np.arange(len(semanas)), 5)

    frecuencias = np.zeros(len(semanas), dtype=int)

    for indices, frecuencia in zip(secciones, [5, 25, 50, 25, 5]):
        frecuencias[indices] = frecuencia

    # Cada fecha representa un registro: +1 al conteo de su semana.
    # Se distribuyen los registros entre los días sin usar aleatoriedad.
    fechas = []

    for semana, frecuencia in zip(semanas, frecuencias):
        dias_disponibles = min(7, (fin - semana).days + 1)

        for j in range(int(frecuencia)):
            fechas.append(
                semana + pd.Timedelta(days=j % dias_disponibles)
            )

    fechas = pd.Series(fechas, dtype="datetime64[ns]")

    df = pd.DataFrame("", index=range(len(fechas)), columns=columnas)

    # Para este dataset sintético, inicio y hecho tienen la misma fecha.
    for sufijo in ["inicio", "hecho"]:
        df[f"anio_{sufijo}"] = fechas.dt.year
        df[f"mes_{sufijo}"] = fechas.dt.month
        df[f"fecha_{sufijo}"] = fechas.dt.strftime("%Y-%m-%d")
        df[f"hora_{sufijo}"] = "12:00:00"

    df["delito"] = nombre

    # Reconstruir la serie desde los registros, incluyendo semanas con cero.
    indice_semana = (fechas - inicio).dt.days // 7

    conteos = (
        indice_semana.value_counts()
        .reindex(range(len(semanas)), fill_value=0)
        .sort_index()
        .to_numpy()
    )

    # Verificar que el CSV representa exactamente la escalera solicitada.
    assert np.array_equal(conteos, frecuencias)

    serie_semanal = pd.DataFrame({
        "semana": np.arange(1, len(semanas) + 1),
        "fecha_inicio_semana": semanas,
        "frecuencia": conteos,
    })

    carpeta = Path(carpeta_salida)
    carpeta.mkdir(parents=True, exist_ok=True)

    ruta = carpeta / f"{nombre}.csv"
    df.to_csv(ruta, index=False, encoding="utf-8-sig")

    print(f"CSV generado: {ruta.resolve()}")
    print(f"Registros: {len(df):,}")
    print(f"Bloques semanales: {len(serie_semanal)}")

    return df, serie_semanal
# endregion






# region generar_dataset_escalera_noise
def generar_dataset_escalera_noise(
    nombre="escalera_5_25_50_25_5",
    carpeta_salida="datasets_sinteticos",
    ruido_max=0,
    seed=42,
):
    inicio = pd.Timestamp("2016-01-01")
    fin = pd.Timestamp("2024-06-30")

    columnas = [
        "anio_inicio", "mes_inicio", "fecha_inicio", "hora_inicio",
        "anio_hecho", "mes_hecho", "fecha_hecho", "hora_hecho",
        "delito", "categoria_delito", "competencia", "fiscalia",
        "agencia", "unidad_investigacion", "colonia_hecho",
        "colonia_catalogo", "alcaldia_hecho", "alcaldia_catalogo",
        "municipio_hecho", "latitud", "longitud",
    ]

    # Bloques consecutivos de 7 días, anclados al 1 de enero de 2016.
    semanas = pd.date_range(inicio, fin, freq="7D")
    secciones = np.array_split(np.arange(len(semanas)), 5)

    frecuencias = np.zeros(len(semanas), dtype=int)

    for indices, frecuencia in zip(secciones, [5, 25, 50, 25, 5]):
        frecuencias[indices] = frecuencia

    # Ruido aditivo uniforme entero, independiente para cada semana.
    if not isinstance(ruido_max, (int, np.integer)) or ruido_max < 0:
        raise ValueError(
            "ruido_max debe ser un entero mayor o igual que cero."
        )

    if ruido_max > 0:
        rng = np.random.default_rng(seed)

        ruido = rng.integers(
            low=-ruido_max,
            high=ruido_max + 1,
            size=len(frecuencias),
        )

        # Los conteos no pueden ser negativos.
        frecuencias = np.maximum(frecuencias + ruido, 0)

    # Cada fecha representa un registro: +1 al conteo de su semana.
    # La distribución entre días sigue siendo determinista.
    fechas = []

    for semana, frecuencia in zip(semanas, frecuencias):
        dias_disponibles = min(7, (fin - semana).days + 1)

        for j in range(int(frecuencia)):
            fechas.append(
                semana + pd.Timedelta(days=j % dias_disponibles)
            )

    fechas = pd.Series(fechas, dtype="datetime64[ns]")

    df = pd.DataFrame(
        "",
        index=range(len(fechas)),
        columns=columnas,
    )

    # Para este dataset sintético, inicio y hecho tienen la misma fecha.
    for sufijo in ["inicio", "hecho"]:
        df[f"anio_{sufijo}"] = fechas.dt.year
        df[f"mes_{sufijo}"] = fechas.dt.month
        df[f"fecha_{sufijo}"] = fechas.dt.strftime("%Y-%m-%d")
        df[f"hora_{sufijo}"] = "12:00:00"

    df["delito"] = nombre

    # Reconstruir la serie desde los registros, incluyendo semanas con cero.
    indice_semana = (fechas - inicio).dt.days // 7

    conteos = (
        indice_semana.value_counts()
        .reindex(range(len(semanas)), fill_value=0)
        .sort_index()
        .to_numpy()
    )

    # Comprobar que los registros reproducen las frecuencias generadas.
    assert np.array_equal(conteos, frecuencias)

    serie_semanal = pd.DataFrame({
        "semana": np.arange(1, len(semanas) + 1),
        "fecha_inicio_semana": semanas,
        "frecuencia": conteos,
    })

    carpeta = Path(carpeta_salida)
    carpeta.mkdir(parents=True, exist_ok=True)

    ruta = carpeta / f"{nombre}.csv"
    df.to_csv(ruta, index=False, encoding="utf-8-sig")

    print(f"CSV generado: {ruta.resolve()}")
    print(f"Registros: {len(df):,}")
    print(f"Bloques semanales: {len(serie_semanal)}")

    return df, serie_semanal
# endregion





# region generar_dataset_periodico_clean
def generar_dataset_periodico_clean(
    nombre="periodico_1_25_1_50_estable_50",
    carpeta_salida="datasets_sinteticos",
    ciclos=None,
    periodo=None,
):
    inicio = pd.Timestamp("2016-01-01")
    fin = pd.Timestamp("2024-06-30")

    columnas = [
        "anio_inicio", "mes_inicio", "fecha_inicio", "hora_inicio",
        "anio_hecho", "mes_hecho", "fecha_hecho", "hora_hecho",
        "delito", "categoria_delito", "competencia", "fiscalia",
        "agencia", "unidad_investigacion", "colonia_hecho",
        "colonia_catalogo", "alcaldia_hecho", "alcaldia_catalogo",
        "municipio_hecho", "latitud", "longitud",
    ]

    # Bloques consecutivos de 7 días, anclados al 1 de enero de 2016.
    semanas = pd.date_range(inicio, fin, freq="7D")
    secciones = np.array_split(np.arange(len(semanas)), 3)

    # Las dos secciones periódicas deben tener la misma longitud.
    assert len(secciones[0]) == len(secciones[1])

    # Indicar ciclos o periodo, pero no ambos.
    if ciclos is not None and periodo is not None:
        raise ValueError(
            "Indica solo 'ciclos' o 'periodo'; el otro se calcula."
        )

    # Si no se especifica ninguno, usar cinco ciclos por sección.
    if ciclos is None and periodo is None:
        ciclos = 5

    semanas_por_seccion = len(secciones[0])

    if ciclos is not None:
        if not np.isfinite(ciclos) or ciclos <= 0:
            raise ValueError("'ciclos' debe ser un número positivo y finito.")

        periodo = semanas_por_seccion / ciclos

    else:
        if not np.isfinite(periodo) or periodo < 2:
            raise ValueError(
                "'periodo' debe ser finito y de al menos 2 semanas."
            )

        ciclos = semanas_por_seccion / periodo

    # Evitar solicitar oscilaciones demasiado rápidas para datos semanales.
    if periodo < 2:
        raise ValueError(
            "Demasiados ciclos: el periodo debe ser de al menos 2 semanas."
        )

    frecuencias = np.ones(len(semanas), dtype=int)

    # Onda periódica entre 1 y el máximo de cada sección.
    for indices, maximo in zip(secciones[:2], [25, 50]):
        t = np.arange(len(indices))
        onda = (1 - np.cos(2 * np.pi * t / periodo)) / 2

        frecuencias[indices] = np.rint(
            1 + (maximo - 1) * onda
        ).astype(int)

    # Tercera sección: frecuencia constante.
    frecuencias[secciones[2]] = 50

    assert np.all(frecuencias >= 1)

    # Cada fecha representa un registro: +1 al conteo de su semana.
    # Distribución determinista entre los días disponibles.
    fechas = []

    for semana, frecuencia in zip(semanas, frecuencias):
        dias_disponibles = min(7, (fin - semana).days + 1)

        for j in range(int(frecuencia)):
            fechas.append(
                semana + pd.Timedelta(days=j % dias_disponibles)
            )

    fechas = (
        pd.Series(fechas, dtype="datetime64[ns]")
        .sort_values(ignore_index=True)
    )

    df = pd.DataFrame("", index=range(len(fechas)), columns=columnas)

    # Para este dataset sintético, inicio y hecho tienen la misma fecha.
    for sufijo in ["inicio", "hecho"]:
        df[f"anio_{sufijo}"] = fechas.dt.year
        df[f"mes_{sufijo}"] = fechas.dt.month
        df[f"fecha_{sufijo}"] = fechas.dt.strftime("%Y-%m-%d")
        df[f"hora_{sufijo}"] = "12:00:00"

    df["delito"] = nombre

    # Reconstruir la serie desde los registros.
    indice_semana = (fechas - inicio).dt.days // 7

    conteos = (
        indice_semana.value_counts()
        .reindex(range(len(semanas)), fill_value=0)
        .sort_index()
        .to_numpy()
    )

    # Verificar las frecuencias, la cobertura semanal y las fechas.
    assert np.array_equal(conteos, frecuencias)
    assert np.all(conteos >= 1)
    assert fechas.between(inicio, fin).all()

    serie_semanal = pd.DataFrame({
        "semana": np.arange(1, len(semanas) + 1),
        "fecha_inicio_semana": semanas,
        "frecuencia": conteos,
    })

    carpeta = Path(carpeta_salida)
    carpeta.mkdir(parents=True, exist_ok=True)

    ruta = carpeta / f"{nombre}.csv"
    df.to_csv(ruta, index=False, encoding="utf-8-sig")

    print(f"CSV generado: {ruta.resolve()}")
    print(f"Registros: {len(df):,}")
    print(f"Bloques semanales: {len(serie_semanal)}")
    print(f"Periodo: {periodo:g} semanas")
    print(f"Ciclos por sección periódica: {ciclos}")

    return df, serie_semanal
# endregion






# region generar_dataset_periodico_noise
def generar_dataset_periodico_noise(
    nombre="periodico_1_25_1_50_estable_50_ruido",
    carpeta_salida="datasets_sinteticos",
    ciclos=None,
    periodo=None,
    ruido_max=0,
    seed=42,
):
    inicio = pd.Timestamp("2016-01-01")
    fin = pd.Timestamp("2024-06-30")

    columnas = [
        "anio_inicio", "mes_inicio", "fecha_inicio", "hora_inicio",
        "anio_hecho", "mes_hecho", "fecha_hecho", "hora_hecho",
        "delito", "categoria_delito", "competencia", "fiscalia",
        "agencia", "unidad_investigacion", "colonia_hecho",
        "colonia_catalogo", "alcaldia_hecho", "alcaldia_catalogo",
        "municipio_hecho", "latitud", "longitud",
    ]

    # Bloques consecutivos de 7 días, anclados al inicio.
    semanas = pd.date_range(inicio, fin, freq="7D")
    secciones = np.array_split(np.arange(len(semanas)), 3)

    assert len(secciones[0]) == len(secciones[1])

    # Configuración de ciclos y periodo.
    if ciclos is not None and periodo is not None:
        raise ValueError(
            "Indica solo 'ciclos' o 'periodo'; el otro se calcula."
        )

    if ciclos is None and periodo is None:
        ciclos = 5

    semanas_por_seccion = len(secciones[0])

    if ciclos is not None:
        if not np.isfinite(ciclos) or ciclos <= 0:
            raise ValueError(
                "'ciclos' debe ser un número positivo y finito."
            )

        periodo = semanas_por_seccion / ciclos

    else:
        if not np.isfinite(periodo) or periodo < 2:
            raise ValueError(
                "'periodo' debe ser finito y de al menos 2 semanas."
            )

        ciclos = semanas_por_seccion / periodo

    if periodo < 2:
        raise ValueError(
            "Demasiados ciclos: el periodo debe ser de al menos 2 semanas."
        )

    if (
        isinstance(ruido_max, (bool, np.bool_))
        or not isinstance(ruido_max, (int, np.integer))
        or ruido_max < 0
    ):
        raise ValueError(
            "'ruido_max' debe ser un entero mayor o igual que cero."
        )

    # Frecuencias base de las dos secciones periódicas.
    frecuencias_base = np.ones(len(semanas), dtype=int)

    for indices, maximo in zip(secciones[:2], [25, 50]):
        t = np.arange(len(indices))
        onda = (1 - np.cos(2 * np.pi * t / periodo)) / 2

        frecuencias_base[indices] = np.rint(
            1 + (maximo - 1) * onda
        ).astype(int)

    # Tercera sección: nivel base constante en 50.
    frecuencias_base[secciones[2]] = 50

    # Ruido aditivo uniforme entero, independiente por semana.
    ruido = np.zeros(len(semanas), dtype=int)

    if ruido_max > 0:
        rng = np.random.default_rng(seed)

        ruido = rng.integers(
            low=-ruido_max,
            high=ruido_max + 1,
            size=len(semanas),
        )

    # Garantizar al menos un registro por semana.
    frecuencias = np.maximum(frecuencias_base + ruido, 1)

    # Cada fecha representa un registro.
    # Distribución determinista entre los días disponibles.
    fechas = []

    for semana, frecuencia in zip(semanas, frecuencias):
        dias_disponibles = min(7, (fin - semana).days + 1)

        for j in range(int(frecuencia)):
            fechas.append(
                semana + pd.Timedelta(days=j % dias_disponibles)
            )

    fechas = (
        pd.Series(fechas, dtype="datetime64[ns]")
        .sort_values(ignore_index=True)
    )

    df = pd.DataFrame(
        "",
        index=range(len(fechas)),
        columns=columnas,
    )

    # Inicio y hecho tienen la misma fecha.
    for sufijo in ["inicio", "hecho"]:
        df[f"anio_{sufijo}"] = fechas.dt.year
        df[f"mes_{sufijo}"] = fechas.dt.month
        df[f"fecha_{sufijo}"] = fechas.dt.strftime("%Y-%m-%d")
        df[f"hora_{sufijo}"] = "12:00:00"

    df["delito"] = nombre

    # Reconstruir la serie desde los registros.
    indice_semana = (fechas - inicio).dt.days // 7

    conteos = (
        indice_semana.value_counts()
        .reindex(range(len(semanas)), fill_value=0)
        .sort_index()
        .to_numpy()
    )

    assert np.array_equal(conteos, frecuencias)
    assert np.all(conteos >= 1)
    assert fechas.between(inicio, fin).all()

    serie_semanal = pd.DataFrame({
        "semana": np.arange(1, len(semanas) + 1),
        "fecha_inicio_semana": semanas,
        "frecuencia": conteos,
    })

    carpeta = Path(carpeta_salida)
    carpeta.mkdir(parents=True, exist_ok=True)

    ruta = carpeta / f"{nombre}.csv"
    df.to_csv(ruta, index=False, encoding="utf-8-sig")

    print(f"CSV generado: {ruta.resolve()}")
    print(f"Registros: {len(df):,}")
    print(f"Bloques semanales: {len(serie_semanal)}")
    print(f"Periodo: {periodo:g} semanas")
    print(f"Ciclos por sección periódica: {ciclos:g}")
    print(f"Ruido solicitado: ±{ruido_max}; mínimo semanal: 1")

    return df, serie_semanal
# endregion











