"""Reanudacion de la descarga de klines: solo se pide lo que falta tras lo ya cacheado."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from miax.ingest.http import HttpClient
from miax.ingest.klines import (
    DEFAULT_BASE_URL,
    MAX_LIMIT,
    download_klines,
    interval_ms,
    slot_bounds,
)
from miax.ingest.store import (
    _MONTH_FORMAT,
    _MONTH_STEM,
    DEFAULT_BASE_DIR,
    _month_bounds_ms,
    _validate_segment,
    cache_bounds,
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
    _merge_into_cache(new, symbol, interval, base_dir=base_dir, start_ms=month_start, end_ms=end_ms)
    logger.info("Descargadas %d velas nuevas de %s %s", len(new), symbol, interval)
    return len(new)


def _merge_into_cache(
    new: pd.DataFrame,
    symbol: str,
    interval: str,
    *,
    base_dir: str | Path,
    start_ms: int,
    end_ms: int,
) -> None:
    """Fusiona `new` con lo cacheado en [start_ms, end_ms] y guarda los meses afectados enteros."""
    cached = load_klines(symbol, interval, base_dir=base_dir, start_ms=start_ms, end_ms=end_ms)
    merged = (
        pd.concat([cached, new], ignore_index=True)
        .drop_duplicates("open_time", keep="last")
        .sort_values("open_time", kind="stable")
        .reset_index(drop=True)
    )
    save_klines(merged, symbol, interval, base_dir=base_dir)


def _month_spans_ms(first_ms: int, last_ms: int) -> list[tuple[int, int]]:
    """Da `(inicio, fin_exclusivo)` en ms de cada mes UTC que toca [first_ms, last_ms], en orden."""
    month = pd.to_datetime(first_ms, unit="ms", utc=True).strftime(_MONTH_FORMAT)
    last_month = pd.to_datetime(last_ms, unit="ms", utc=True).strftime(_MONTH_FORMAT)
    spans: list[tuple[int, int]] = []
    while month <= last_month:
        span = _month_bounds_ms(month)
        spans.append(span)
        month = pd.to_datetime(span[1], unit="ms", utc=True).strftime(_MONTH_FORMAT)
    return spans


def _fill_forwards(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    *,
    client: HttpClient | None,
    base_dir: str | Path,
    base_url: str,
    limit: int,
) -> int:
    """Baja [start_ms, end_ms] mes a mes, de menor a mayor; devuelve las velas nuevas."""
    total = 0
    # Mes a mes y guardando cada uno: poca memoria y, si se corta, se reanuda donde se quedo.
    for month_start, month_end in _month_spans_ms(start_ms, end_ms):
        block_start = max(start_ms, month_start)
        block_end = min(end_ms, month_end - 1)
        if not _has_candles(interval, block_start, block_end):
            continue
        # update_symbol reanuda desde la ultima vela cacheada: cada mes continua el anterior.
        total += update_symbol(
            symbol,
            interval,
            block_start,
            block_end,
            client=client,
            base_dir=base_dir,
            base_url=base_url,
            limit=limit,
        )
    return total


def _fill_backwards(
    symbol: str,
    interval: str,
    start_ms: int,
    first_cached: int,
    *,
    client: HttpClient | None,
    base_dir: str | Path,
    base_url: str,
    limit: int,
) -> int:
    """Baja [start_ms, first_cached - 1] mes a mes de mayor a menor; devuelve las velas nuevas."""
    logger.info("Rellenando %s %s hacia atras desde open_time=%d", symbol, interval, start_ms)
    total = 0
    # De mayor a menor: si se corta, la cache sigue continua (empieza mas tarde, sin huecos).
    for month_start, month_end in reversed(_month_spans_ms(start_ms, first_cached - 1)):
        block_start = max(start_ms, month_start)
        block_end = min(first_cached - 1, month_end - 1)
        if not _has_candles(interval, block_start, block_end):
            continue
        new = download_klines(
            symbol, interval, block_start, block_end, client=client, base_url=base_url, limit=limit
        )
        if new.empty:
            # El par aun no existia: los meses anteriores tambien estaran vacios.
            logger.info(
                "%s %s: sin velas desde %d hacia atras, se deja de retroceder",
                symbol,
                interval,
                block_end,
            )
            break
        # save_klines reescribe el mes entero: se fusiona con lo cacheado en ese mes (solo el
        # mas reciente puede tener datos, porque la cache empieza en `first_cached`).
        _merge_into_cache(
            new, symbol, interval, base_dir=base_dir, start_ms=month_start, end_ms=month_end - 1
        )
        total += len(new)
    logger.info("Descargadas %d velas nuevas de %s %s hacia atras", total, symbol, interval)
    return total


def _has_candles(interval: str, first_ms: int, last_ms: int) -> bool:
    """Dice si hay alguna vela de `interval` que abra en [first_ms, last_ms] (ambos inclusive)."""
    if first_ms > last_ms:
        return False
    if interval_ms(interval) is None:
        return True  # sin duracion fija no se puede saber: se deja decidir a Binance
    return slot_bounds(first_ms, last_ms + 1, interval)[2] > 0


def fill_range(
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
    """Deja cacheado [start_ms, end_ms] bajando solo lo que falta antes y despues de lo cacheado.

    Va por meses UTC y guarda cada uno antes de pedir el siguiente (memoria y reanudacion tras un
    corte); devuelve las velas nuevas y no rellena huecos interiores (los ve `validate`).
    """
    if start_ms > end_ms:
        logger.info("Nada que descargar de %s %s: rango vacio", symbol, interval)
        return 0
    bounds = cache_bounds(symbol, interval, base_dir=base_dir)
    if bounds is None:
        # Sin cache: es una descarga normal del rango entero, mes a mes.
        new = _fill_forwards(
            symbol,
            interval,
            start_ms,
            end_ms,
            client=client,
            base_dir=base_dir,
            base_url=base_url,
            limit=limit,
        )
        if new == 0:
            logger.warning("%s %s: sin velas en Binance para su rango", symbol, interval)
        return new

    first, last = bounds
    total = 0
    if _has_candles(interval, start_ms, first - 1):
        # Hacia atras hasta la primera vela cacheada: la cache sigue siendo continua. Solo si
        # cabe alguna vela: un fin como `period_end - 1 ms` no abre vela y no se pide en balde.
        before = _fill_backwards(
            symbol,
            interval,
            start_ms,
            first,
            client=client,
            base_dir=base_dir,
            base_url=base_url,
            limit=limit,
        )
        if before == 0:
            logger.warning("%s %s: sin velas en Binance antes de lo cacheado", symbol, interval)
        total += before
    if _has_candles(interval, last + 1, end_ms):
        # Hacia delante desde `last + 1` (no desde `start_ms`): sin hueco con lo cacheado.
        after = _fill_forwards(
            symbol,
            interval,
            last + 1,
            end_ms,
            client=client,
            base_dir=base_dir,
            base_url=base_url,
            limit=limit,
        )
        if after == 0:
            logger.warning("%s %s: sin velas en Binance despues de lo cacheado", symbol, interval)
        total += after
    return total
