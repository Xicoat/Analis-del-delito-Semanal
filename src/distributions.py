# region Libraries
import logging
import warnings

import numpy as np
import powerlaw
from scipy.optimize import OptimizeWarning


logger = logging.getLogger("Delitos")
# endregion






# region compute_distribution_metrics
def compute_distribution_metrics(
    graph_metrics: dict,
    min_positive_degrees: int = 1,
) -> dict:
    """Compute empirical degree distributions and power-law/lognormal/exponential fits.

    Evaluates the full empirical degree distribution (including zero-degree
    nodes) and fits discrete power-law, lognormal, and exponential tail models
    on positive degrees via ``powerlaw.Fit``, computing Euclidean distances and
    Vuong's log-likelihood ratio comparison.

    Parameters
    ----------
    graph_metrics : dict
        Dictionary returned by ``compute_graph_metrics``, containing at least
        ``'graph_name'`` (str), ``'node_count'`` (int), and ``'degrees'``
        (Sequence[int]).
    min_positive_degrees : int, default=1
        Minimum number of positive-degree nodes required to attempt parametric
        distribution fitting.

    Returns
    -------
    dict
        Dictionary containing series identity, empirical degree distributions,
        tail samples, fitted model objects, Euclidean distances, likelihood
        comparison statistics (``'R_log'``, ``'p_comp'``, ``'verdict'``), and
        execution status metadata (``'status'``, ``'stage'``, ``'error'``).

    Raises
    ------
    ValueError
        If ``'degrees'`` is missing, non-1D, contains negative/non-integer
        values, mismatches ``'node_count'``, or if ``min_positive_degrees < 1``.
    """
    logger.info("Calculando distribuciones y métricas de sus ajustes")

    if "degrees" not in graph_metrics:
        raise ValueError(
            "El resultado de compute_graph_metrics debe incluir 'degrees'."
        )

    if (
        isinstance(min_positive_degrees, bool)
        or not isinstance(min_positive_degrees, int)
        or min_positive_degrees < 1
    ):
        raise ValueError(
            "min_positive_degrees debe ser un entero positivo."
        )

    degrees = np.asarray(graph_metrics["degrees"], dtype=float)

    if degrees.ndim != 1:
        raise ValueError(
            "'degrees' debe ser una secuencia unidimensional."
        )

    if (
        not np.isfinite(degrees).all()
        or (degrees < 0).any()
        or (degrees != np.floor(degrees)).any()
    ):
        raise ValueError(
            "Los grados deben ser enteros no negativos y finitos."
        )

    degrees = degrees.astype(np.int64)

    if (
        "node_count" in graph_metrics
        and graph_metrics["node_count"] != len(degrees)
    ):
        raise ValueError(
            "El número de grados no coincide con el campo 'node_count'."
        )

    # Full empirical degree distribution, including zero-degree nodes.
    k_all, frequencies_all = np.unique(
        degrees,
        return_counts=True,
    )

    probabilities_all = (
        frequencies_all / len(degrees)
        if len(degrees) > 0
        else np.array([], dtype=float)
    )

    degree_entropy = (
        -float(
            np.sum(
                probabilities_all
                * np.log(probabilities_all + 1e-12)
            )
        )
        if len(degrees) > 0
        else np.nan
    )

    # Parametric fits use strictly positive degrees only.
    positive_degrees = degrees[degrees > 0]

    results = {
        # Series identity.
        "series_name": graph_metrics["graph_name"],

        # Full empirical data and distribution.
        "node_count": len(degrees),
        "degrees": degrees.copy(),
        "k_all": k_all,
        "frequencies_all": frequencies_all,
        "probabilities_all": probabilities_all,
        "degree_entropy": degree_entropy,

        # Sample used for parametric fitting.
        "positive_degrees": positive_degrees.copy(),
        "n_positive_degrees": len(positive_degrees),
        "min_positive_degrees": min_positive_degrees,

        # Power-law fit outputs.
        "gamma_powerlaw": np.nan,
        "sigma": np.nan,
        "xmin": np.nan,
        "ks_distance": np.nan,

        # Empirical tail distribution.
        "tail_degrees": np.array([], dtype=np.int64),
        "tail_node_count": 0,
        "tail_positive_pct": np.nan,
        "k_eval": np.array([], dtype=np.int64),
        "tail_frequencies": np.array([], dtype=np.int64),
        "tail_probabilities": np.array([], dtype=float),

        # Fitted model objects and probability vectors.
        "fit": None,
        "fitted_models": {},
        "model_probabilities": {},

        # Euclidean distances on observed tail degrees.
        "dist_eucl_pl": np.nan,
        "dist_eucl_lognormal": np.nan,
        "dist_eucl_exp": np.nan,

        # Relative comparison: power law vs. lognormal.
        "R_log": np.nan,
        "p_comp": np.nan,
        "verdict": None,

        # Execution status tracking.
        "status": "NO_CALCULADO",
        "stage": None,
        "error": None,
    }

    if len(degrees) == 0:
        results["status"] = "GRAFO_VACIO"
        logger.warning("No hay grados para analizar.")
        return results

    if len(positive_degrees) == 0:
        results["status"] = "SIN_GRADOS_POSITIVOS"
        logger.warning("No hay grados positivos para ajustar.")
        return results

    if len(positive_degrees) < min_positive_degrees:
        results["status"] = "MUESTRA_INSUFICIENTE"
        logger.warning(
            "Hay %s grados positivos; se requieren al menos %s.",
            len(positive_degrees),
            min_positive_degrees,
        )
        return results

    try:
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore",
                category=OptimizeWarning,
            )

            # Single shared Fit object for downstream comparisons.
            results["stage"] = "AJUSTE_LEY_POTENCIA"

            fit = powerlaw.Fit(
                positive_degrees,
                discrete=True,
                verbose=False,
            )

            results["fit"] = fit
            power_law_model = fit.power_law

            gamma = float(power_law_model.alpha)
            sigma = float(power_law_model.sigma)
            xmin = float(power_law_model.xmin)
            ks_distance = float(power_law_model.D)

            results.update({
                "gamma_powerlaw": gamma,
                "sigma": sigma,
                "xmin": xmin,
                "ks_distance": ks_distance,
            })

            if not np.isfinite([gamma, sigma, xmin, ks_distance]).all():
                raise ValueError(
                    "El ajuste produjo parámetros o distancia KS no finitos."
                )

            # Tail sample and empirical probabilities.
            results["stage"] = "DISTRIBUCION_COLA"

            tail_degrees = positive_degrees[positive_degrees >= xmin]

            if len(tail_degrees) == 0:
                results["status"] = "COLA_VACIA"
                logger.warning("No hay grados con k >= xmin.")
                return results

            k_eval, tail_frequencies = np.unique(
                tail_degrees,
                return_counts=True,
            )

            tail_probabilities = tail_frequencies / len(tail_degrees)

            results.update({
                "tail_degrees": tail_degrees,
                "tail_node_count": len(tail_degrees),
                "tail_positive_pct": (
                    100 * len(tail_degrees) / len(positive_degrees)
                ),
                "k_eval": k_eval,
                "tail_frequencies": tail_frequencies,
                "tail_probabilities": tail_probabilities,
            })

            # Model fits and Euclidean distances over the observed tail support.
            results["stage"] = "MODELOS_Y_DISTANCIAS"

            models = {
                "power_law": power_law_model,
                "lognormal": fit.lognormal,
                "exponential": fit.exponential,
            }

            results["fitted_models"] = models

            distance_fields = {
                "power_law": "dist_eucl_pl",
                "lognormal": "dist_eucl_lognormal",
                "exponential": "dist_eucl_exp",
            }

            for model_name, model in models.items():
                probabilities = np.asarray(
                    model.pdf(k_eval),
                    dtype=float,
                )

                if (
                    probabilities.shape != tail_probabilities.shape
                    or not np.isfinite(probabilities).all()
                ):
                    raise ValueError(
                        f"Probabilidades inválidas para '{model_name}'."
                    )

                results["model_probabilities"][model_name] = probabilities

                results[distance_fields[model_name]] = float(
                    np.linalg.norm(tail_probabilities - probabilities)
                )

            # Log-likelihood ratio comparison.
            results["stage"] = "COMPARACION_VEROSIMILITUDES"

            R_log, p_comp = fit.distribution_compare(
                "power_law",
                "lognormal",
                normalized_ratio=True,
            )

            results["R_log"] = float(R_log)
            results["p_comp"] = float(p_comp)

            if not np.isfinite([R_log, p_comp]).all():
                raise ValueError(
                    "La comparación produjo R o valor p no finitos."
                )

            if p_comp < 0.05:
                verdict = "Power Law" if R_log > 0 else "Lognormal"
            else:
                verdict = "Inconclusivo (Ambas)"

            results["verdict"] = verdict

        results["status"] = "CALCULADO"
        results["stage"] = "COMPLETADO"

        logger.info("EXITO ✅. Distribuciones y comparaciones calculadas.")

    except Exception as err:
        # Preserve partial computations and record the failing stage.
        results["status"] = "ERROR_CALCULO"
        results["error"] = f"{type(err).__name__}: {err}"

        logger.exception(
            "Fallo al calcular distribuciones en la etapa %s.",
            results["stage"],
        )

    return results
# endregion






# region 
# endregion


