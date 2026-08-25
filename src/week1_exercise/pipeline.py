from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

FEATURES = ("gpu_utilization", "cpu_utilization", "memory_gb")
WEIGHTS = np.array([0.50, 0.30, 0.20], dtype=float)


class ServerMeasurements:
    """Representa una matriz numérica y los nombres de sus características."""

    def __init__(self, values: np.ndarray, feature_names: tuple[str, ...]):
        values = np.asarray(values)

        # Va primero: en un arreglo 1D, values.shape[1] lanzaría IndexError.
        if values.ndim != 2:
            raise ValueError(
                f"values debe ser una matriz bidimensional; se recibió ndim={values.ndim}."
            )

        if not np.issubdtype(values.dtype, np.number):
            raise ValueError(
                f"values debe contener datos numéricos; se recibió dtype={values.dtype}."
            )

        feature_names = tuple(feature_names)

        if values.shape[1] != len(feature_names):
            raise ValueError(
                f"El número de columnas ({values.shape[1]}) no coincide con la "
                f"cantidad de nombres de características ({len(feature_names)})."
            )

        self.values = values.astype(float)
        self.feature_names = feature_names

    def __repr__(self) -> str:
        return (
            f"ServerMeasurements(shape={self.values.shape}, "
            f"features={self.feature_names})"
        )


def build_measurements(df: pd.DataFrame) -> ServerMeasurements:
    """Construye ServerMeasurements usando las columnas definidas en FEATURES."""
    faltantes = [columna for columna in FEATURES if columna not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en el DataFrame: {faltantes}.")

    values = df.loc[:, list(FEATURES)].to_numpy(dtype=float)

    return ServerMeasurements(values, FEATURES)


def compute_load_score(batch: ServerMeasurements) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retorna zscores, load_score y medias por característica."""
    values = batch.values

    means = values.mean(axis=0)
    stds = values.std(axis=0)

    # Evita la división por cero cuando una columna es constante.
    stds_seguras = np.where(stds == 0, 1.0, stds)

    zscores = (values - means) / stds_seguras
    load_score = zscores @ WEIGHTS

    return zscores, load_score, means


def enrich_dataframe(df: pd.DataFrame, load_score: np.ndarray) -> pd.DataFrame:
    """Añade load_score y requires_review sin modificar el DataFrame original."""
    if len(load_score) != len(df):
        raise ValueError(
            f"load_score tiene {len(load_score)} valores pero el DataFrame "
            f"tiene {len(df)} filas."
        )

    # Copia independiente para no modificar el DataFrame original.
    analysis = df.copy()

    analysis["load_score"] = load_score

    requiere_revision = (analysis["load_score"] > 1.5) | (analysis["temperature_c"] > 80)
    analysis["requires_review"] = requiere_revision.astype(bool)

    return analysis


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Construye el resumen por servidor solicitado en el enunciado."""
    summary = (
        df.groupby("server")
        .agg(
            observations=("timestamp", "size"),
            mean_power_w=("power_w", "mean"),
            max_temperature_c=("temperature_c", "max"),
            mean_load=("load_score", "mean"),
            review_count=("requires_review", "sum"),
        )
        .reset_index()
    )

    # Orden exacto de columnas exigido por el enunciado.
    return summary.loc[
        :,
        [
            "server",
            "observations",
            "mean_power_w",
            "max_temperature_c",
            "mean_load",
            "review_count",
        ],
    ]


def save_and_validate_parquet(df: pd.DataFrame, output_path: str | Path) -> pd.DataFrame:
    """Guarda, lee y valida el round trip del DataFrame analizado."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # preserve_index=False evita guardar el índice como una columna extra.
    tabla_original = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(tabla_original, output_path)

    tabla_leida = pq.read_table(output_path)

    filas_ok = tabla_leida.num_rows == tabla_original.num_rows
    columnas_ok = tabla_leida.column_names == tabla_original.column_names
    esquema_ok = tabla_leida.schema == tabla_original.schema
    valores_ok = tabla_leida.equals(tabla_original)

    print(f"Round trip verificado: {output_path}")
    print(f"Filas: {filas_ok}")
    print(f"Columnas: {columnas_ok}")
    print(f"Esquema: {esquema_ok}")
    print(f"Valores: {valores_ok}")

    if not (filas_ok and columnas_ok and esquema_ok and valores_ok):
        raise ValueError(
            "El round trip de Parquet no conservó el contrato de los datos."
        )

    return tabla_leida.to_pandas()
