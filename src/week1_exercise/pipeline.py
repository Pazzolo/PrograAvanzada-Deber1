from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

FEATURES = ("gpu_utilization", "cpu_utilization", "memory_gb")
WEIGHTS = np.array([0.50, 0.30, 0.20], dtype=float)


class ServerMeasurements:
    """Representa una matriz numérica y los nombres de sus características."""

    def __init__(self, values: np.ndarray, feature_names: tuple[str, ...]):
        # Se normaliza la entrada a un arreglo de NumPy para poder inspeccionar
        # su número de dimensiones y su tipo de dato.
        values = np.asarray(values)

        # La validación de dimensiones va primero: si el arreglo no es bidimensional,
        # values.shape[1] no existe y el error sería un IndexError en lugar de ValueError.
        if values.ndim != 2:
            raise ValueError(
                f"values debe ser una matriz bidimensional; se recibió ndim={values.ndim}."
            )

        # Los datos deben ser numéricos. Se revisa el dtype en lugar de intentar una
        # conversión, para rechazar matrices de texto como [["high", "50", "20"]].
        if not np.issubdtype(values.dtype, np.number):
            raise ValueError(
                f"values debe contener datos numéricos; se recibió dtype={values.dtype}."
            )

        # Los nombres se guardan siempre como tupla (estructura inmutable).
        feature_names = tuple(feature_names)

        # Cada columna de la matriz necesita exactamente un nombre de característica.
        if values.shape[1] != len(feature_names):
            raise ValueError(
                f"El número de columnas ({values.shape[1]}) no coincide con la "
                f"cantidad de nombres de características ({len(feature_names)})."
            )

        # Se almacena en punto flotante para garantizar un tipo homogéneo en todo el pipeline.
        self.values = values.astype(float)
        self.feature_names = feature_names

    def __repr__(self) -> str:
        # Representación breve solicitada en el enunciado: shape y nombres de características.
        return (
            f"ServerMeasurements(shape={self.values.shape}, "
            f"features={self.feature_names})"
        )


def build_measurements(df: pd.DataFrame) -> ServerMeasurements:
    """Construye ServerMeasurements usando las columnas definidas en FEATURES."""
    # Se verifica que el DataFrame contenga las tres características esperadas.
    faltantes = [columna for columna in FEATURES if columna not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en el DataFrame: {faltantes}.")

    # Selección de las características y conversión a una matriz NumPy de flotantes.
    values = df.loc[:, list(FEATURES)].to_numpy(dtype=float)

    return ServerMeasurements(values, FEATURES)


def compute_load_score(batch: ServerMeasurements) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retorna zscores, load_score y medias por característica."""
    # TODO 2: normalización por columnas + broadcasting con WEIGHTS.
    raise NotImplementedError


def enrich_dataframe(df: pd.DataFrame, load_score: np.ndarray) -> pd.DataFrame:
    """Añade load_score y requires_review sin modificar el DataFrame original."""
    # TODO 3
    raise NotImplementedError


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Construye el resumen por servidor solicitado en el enunciado."""
    # TODO 4
    raise NotImplementedError


def save_and_validate_parquet(df: pd.DataFrame, output_path: str | Path) -> pd.DataFrame:
    """Guarda, lee y valida el round trip del DataFrame analizado."""
    # TODO 5
    raise NotImplementedError
