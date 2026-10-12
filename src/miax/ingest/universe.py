"""Filtro de universo por liquidez: volumen diario en USDT medido en una ventana historica."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from miax.ingest.http import HttpClient
from miax.ingest.klines import DEFAULT_BASE_URL, download_klines
from miax.ingest.ratelimit import RateLimiter
from miax.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_MIN_QUOTE_VOLUME = 10_000_000.0  # USDT por dia (mediana)
VOLUME_INTERVAL = "1d"
VOLUME_COLUMN = "median_quote_volume"


def median_daily_quote_volume(
    symbols: Sequence[str],
    start_ms: int,
    end_ms: int,
    *,
    client: HttpClient | None = None,
    base_url: str = DEFAULT_BASE_URL,
) -> pd.Series:
    """Mediana del volumen diario en USDT de cada simbolo en la ventana; NaN si no hay velas."""
    if start_ms > end_ms:
        raise ValueError(f"start_ms ({start_ms}) no puede ser mayor que end_ms ({end_ms})")
    own_client = client is None
    http = HttpClient(rate_limiter=RateLimiter()) if client is None else client
    medians: dict[str, float] = {}
    try:
        # La ventana debe ser anterior al periodo de estudio: asi la seleccion no mira al futuro.
        for symbol in dict.fromkeys(symbols):  # sin duplicados, conserva el orden
            candles = download_klines(
                symbol, VOLUME_INTERVAL, start_ms, end_ms, client=http, base_url=base_url
            )
            # Solo velas cerradas dentro de la ventana: evita colar volumen posterior a end_ms.
            candles = candles[candles["close_time"] <= end_ms]
            if candles.empty:
                medians[symbol] = float("nan")
            else:
                medians[symbol] = float(candles["quote_asset_volume"].median())
    finally:
        if own_client:
            http.close()
    return pd.Series(medians, name=VOLUME_COLUMN, dtype="float64")


def filter_by_volume(
    pairs: pd.DataFrame,
    volumes: pd.Series,
    min_quote_volume: float = DEFAULT_MIN_QUOTE_VOLUME,
) -> pd.DataFrame:
    """Conserva los pares con mediana de volumen >= umbral, ordenados de mayor a menor volumen."""
    if min_quote_volume < 0:
        raise ValueError(f"min_quote_volume no puede ser negativo, recibido {min_quote_volume}")
    if "symbol" not in pairs.columns:
        raise ValueError(f"pairs necesita la columna 'symbol'; columnas: {list(pairs.columns)}")
    joined = pairs.copy()
    joined[VOLUME_COLUMN] = joined["symbol"].map(volumes).astype("float64")
    no_candles = joined[VOLUME_COLUMN].isna()
    below = ~no_candles & (joined[VOLUME_COLUMN] < min_quote_volume)
    kept = joined[~no_candles & ~below]
    kept = kept.sort_values(VOLUME_COLUMN, ascending=False, kind="stable").reset_index(drop=True)
    logger.info(
        "Universo por volumen (umbral %.0f): %d entran, %d descartados por umbral, "
        "%d descartados por no tener velas",
        min_quote_volume,
        len(kept),
        int(below.sum()),
        int(no_candles.sum()),
    )
    return kept
