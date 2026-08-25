from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

FEATURES = ("gpu_utilization", "cpu_utilization", "memory_gb")
WEIGHTS = np.array([0.50, 0.30, 0.20], dtype=float)


class ServerMeasurements:
    """Representa una matriz numérica y los nombres de sus características."""

    def __init__(self, values: np.ndarray, feature_names: tuple[str, ...]):
        # TODO 1: almacenar los datos y validar las invariantes solicitadas.
        raise NotImplementedError

    def __repr__(self) -> str:
        # TODO 1: mostrar de forma breve shape y feature_names.
        raise NotImplementedError


def build_measurements(df: pd.DataFrame) -> ServerMeasurements:
    """Construye ServerMeasurements usando las columnas definidas en FEATURES."""
    # TODO 1
    raise NotImplementedError


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
