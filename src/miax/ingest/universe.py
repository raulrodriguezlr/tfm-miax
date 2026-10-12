"""Filtros de universo por liquidez (volumen diario en USDT) y por antiguedad de listado.

La antiguedad se mide con la primera vela diaria de cada par.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from miax.ingest.http import HttpClient
from miax.ingest.klines import DEFAULT_BASE_URL, KLINES_PATH, download_klines
from miax.ingest.ratelimit import RateLimiter
from miax.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_MIN_QUOTE_VOLUME = 10_000_000.0  # USDT por dia (mediana)
VOLUME_INTERVAL = "1d"
VOLUME_COLUMN = "median_quote_volume"

DAY_MS = 86_400_000
DEFAULT_MIN_LISTING_DAYS = 90  # dias de cotizacion exigidos antes de la referencia
LISTING_INTERVAL = "1d"
LISTING_KLINES_WEIGHT = 2  # peso de /klines en Binance spot
LISTING_COLUMN = "first_candle_ms"


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


def first_candle_time(
    symbols: Sequence[str],
    reference_ms: int,
    *,
    client: HttpClient | None = None,
    base_url: str = DEFAULT_BASE_URL,
) -> pd.Series:
    """Apertura de la primera vela diaria de cada simbolo no posterior a reference_ms (NA si no)."""
    if reference_ms < 0:
        raise ValueError(f"reference_ms no puede ser negativo, recibido {reference_ms}")
    own_client = client is None
    http = HttpClient(rate_limiter=RateLimiter()) if client is None else client
    url = f"{base_url.rstrip('/')}{KLINES_PATH}"
    first_candles: dict[str, int | pd.NAType] = {}
    try:
        for symbol in dict.fromkeys(symbols):  # sin duplicados, conserva el orden
            # endTime=reference_ms: la consulta nunca mira nada posterior a la referencia.
            params = {
                "symbol": symbol,
                "interval": LISTING_INTERVAL,
                "startTime": 0,
                "endTime": reference_ms,
                "limit": 1,
            }
            block = http.get(url, params=params, weight=LISTING_KLINES_WEIGHT).json()
            first_candles[symbol] = pd.NA if not block else int(block[0][0])
    finally:
        if own_client:
            http.close()
    return pd.Series(first_candles, name=LISTING_COLUMN, dtype="Int64")


def filter_by_listing_age(
    pairs: pd.DataFrame,
    first_candles: pd.Series,
    reference_ms: int,
    min_listing_days: int = DEFAULT_MIN_LISTING_DAYS,
) -> pd.DataFrame:
    """Conserva los pares que cotizan desde al menos min_listing_days dias antes de reference_ms."""
    if min_listing_days < 0:
        raise ValueError(f"min_listing_days no puede ser negativo, recibido {min_listing_days}")
    if "symbol" not in pairs.columns:
        raise ValueError(f"pairs necesita la columna 'symbol'; columnas: {list(pairs.columns)}")
    if not first_candles.index.is_unique:
        raise ValueError("first_candles tiene simbolos repetidos en el indice")
    joined = pairs.copy()
    joined[LISTING_COLUMN] = joined["symbol"].map(first_candles).astype("Int64")
    # El corte solo depende de reference_ms y de la primera vela: no se usa ningun dato posterior.
    cutoff = reference_ms - min_listing_days * DAY_MS
    no_candles = joined[LISTING_COLUMN].isna()
    too_new = ~no_candles & (joined[LISTING_COLUMN] > cutoff)
    kept = joined[~no_candles & ~too_new].reset_index(drop=True)
    logger.info(
        "Universo por antiguedad (minimo %d dias): %d entran, %d descartados por recientes, "
        "%d descartados por no tener velas",
        min_listing_days,
        len(kept),
        int(too_new.sum()),
        int(no_candles.sum()),
    )
    return kept
