"""Reanudacion de la descarga de klines: solo se pide lo que falta tras lo ya cacheado."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from miax.ingest.http import HttpClient
from miax.ingest.klines import DEFAULT_BASE_URL, MAX_LIMIT, download_klines
from miax.ingest.store import (
    _MONTH_FORMAT,
    _MONTH_STEM,
    DEFAULT_BASE_DIR,
    _month_bounds_ms,
    _validate_segment,
    load_klines,
    save_klines,
)
from miax.utils.io import read_parquet, resolve_repo_path
from miax.utils.logging import get_logger

logger = get_logger(__name__)


def cached_last_open_time(
    symbol: str,
    interval: str,
    *,
    base_dir: str | Path = DEFAULT_BASE_DIR,
) -> int | None:
    """Devuelve el `open_time` mas reciente cacheado de `symbol`/`interval`, o None si no hay."""
    directory = (
        resolve_repo_path(base_dir) / _validate_segment(symbol) / _validate_segment(interval)
    )
    months = sorted(p for p in directory.glob("*.parquet") if _MONTH_STEM.fullmatch(p.stem))
    # Se recorre del mes mas reciente al mas antiguo hasta dar con uno que tenga velas.
    for path in reversed(months):
        frame = read_parquet(path)
        if not frame.empty:
            return int(frame["open_time"].max())
    return None


def update_symbol(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    *,
    client: HttpClient | None = None,
    base_dir: str | Path = DEFAULT_BASE_DIR,
    base_url: str = DEFAULT_BASE_URL,
    limit: int = MAX_LIMIT,
) -> int:
    """Descarga solo las velas posteriores a la ultima cacheada y devuelve cuantas son nuevas."""
    last_cached = cached_last_open_time(symbol, interval, base_dir=base_dir)
    # Reanudacion solo hacia delante: nunca se pide un instante ya guardado.
    resume = start_ms if last_cached is None else max(last_cached + 1, start_ms)
    if resume > end_ms:
        logger.info("Nada que descargar de %s %s: el rango ya esta cacheado", symbol, interval)
        return 0

    logger.info("Reanudando %s %s desde open_time=%d", symbol, interval, resume)
    new = download_klines(
        symbol, interval, resume, end_ms, client=client, base_url=base_url, limit=limit
    )
    if new.empty:
        logger.info("Nada que descargar de %s %s: Binance no devolvio velas", symbol, interval)
        return 0

    # save_klines sobrescribe el mes entero: se fusiona con lo cacheado del mes de `resume`
    # en adelante para no truncar el mes parcial. Los meses anteriores no se tocan.
    resume_month = pd.to_datetime(resume, unit="ms", utc=True).strftime(_MONTH_FORMAT)
    month_start, _ = _month_bounds_ms(resume_month)
    # Todo lo cacheado es <= last_cached < resume <= end_ms: el tope end_ms no recorta datos.
    cached = load_klines(symbol, interval, base_dir=base_dir, start_ms=month_start, end_ms=end_ms)
    merged = (
        pd.concat([cached, new], ignore_index=True)
        .drop_duplicates("open_time", keep="last")
        .sort_values("open_time", kind="stable")
        .reset_index(drop=True)
    )
    save_klines(merged, symbol, interval, base_dir=base_dir)
    logger.info("Descargadas %d velas nuevas de %s %s", len(new), symbol, interval)
    return len(new)
