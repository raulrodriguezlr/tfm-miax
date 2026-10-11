"""Cacheo local de klines en Parquet: un fichero por mes en `{base_dir}/{symbol}/{interval}/`."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from miax.ingest.klines import KLINE_COLUMNS, KLINE_DTYPES
from miax.utils.io import read_parquet, resolve_repo_path, write_parquet
from miax.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_BASE_DIR = "data/klines"
_MONTH_FORMAT = "%Y-%m"
_MONTH_STEM = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9_.-]+$")


def _validate_segment(name: str) -> str:
    """Valida que `name` sea un unico tramo de ruta (sin separadores ni `..`) y lo devuelve."""
    if name in ("", ".", "..") or not _SAFE_SEGMENT.fullmatch(name):
        raise ValueError(
            f"Nombre de ruta no valido: {name!r} (solo letras, digitos, '_', '-' y '.'; "
            "sin separadores, ':' ni '..')"
        )
    return name


def _month_bounds_ms(month: str) -> tuple[int, int]:
    """Devuelve el primer ms del mes `YYYY-MM` (UTC) y el primer ms del mes siguiente."""
    start = pd.Timestamp(f"{month}-01", tz="UTC")
    end = start + pd.DateOffset(months=1)
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)


def save_klines(
    df: pd.DataFrame,
    symbol: str,
    interval: str,
    *,
    base_dir: str | Path = DEFAULT_BASE_DIR,
) -> list[Path]:
    """Guarda las klines en un Parquet por mes (UTC); solo se persisten las columnas canonicas."""
    symbol = _validate_segment(symbol)
    interval = _validate_segment(interval)
    missing = [c for c in KLINE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas de klines: {missing}")
    if df.empty:
        logger.info("Sin velas que guardar de %s %s", symbol, interval)
        return []

    # Las columnas extra se descartan al guardar para que el round-trip sea fiel.
    ordered = df[KLINE_COLUMNS].sort_values("open_time", kind="stable").reset_index(drop=True)
    # open_time esta en ms UTC: el mes se deriva en UTC para que no dependa de la zona local.
    months = pd.to_datetime(ordered["open_time"], unit="ms", utc=True).dt.strftime(_MONTH_FORMAT)
    directory = resolve_repo_path(base_dir) / symbol / interval

    paths: list[Path] = []
    for month, chunk in ordered.groupby(months, sort=True):
        path = write_parquet(chunk.reset_index(drop=True), directory / f"{month}.parquet")
        logger.info("Guardado %s %s %s: %d velas", symbol, interval, month, len(chunk))
        paths.append(path)
    return paths


def load_klines(
    symbol: str,
    interval: str,
    *,
    base_dir: str | Path = DEFAULT_BASE_DIR,
    start_ms: int | None = None,
    end_ms: int | None = None,
) -> pd.DataFrame:
    """Lee las klines cacheadas (solo las de [start_ms, end_ms] si se indica), por `open_time`."""
    directory = (
        resolve_repo_path(base_dir) / _validate_segment(symbol) / _validate_segment(interval)
    )
    frames: list[pd.DataFrame] = []
    for path in sorted(directory.glob("*.parquet")):
        if not _MONTH_STEM.fullmatch(path.stem):
            logger.warning("Se ignora el fichero ajeno al cacheo: %s", path.name)
            continue
        month_start, month_end = _month_bounds_ms(path.stem)
        # Se descartan los meses que no solapan con el rango pedido sin abrirlos.
        if end_ms is not None and month_start > end_ms:
            continue
        if start_ms is not None and month_end <= start_ms:
            continue
        frames.append(read_parquet(path))

    if not frames:
        return pd.DataFrame({c: pd.Series(dtype=t) for c, t in KLINE_DTYPES.items()})

    result = pd.concat(frames, ignore_index=True).sort_values("open_time", kind="stable")
    if start_ms is not None:
        result = result[result["open_time"] >= start_ms]
    if end_ms is not None:
        result = result[result["open_time"] <= end_ms]
    return result.reset_index(drop=True)[KLINE_COLUMNS]
