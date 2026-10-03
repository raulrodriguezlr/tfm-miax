"""Descarga paginada de klines de Binance: un rango largo se pide en bloques de hasta 1000 velas."""

from __future__ import annotations

from typing import Any

import pandas as pd

from miax.ingest.http import HttpClient
from miax.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_BASE_URL = "https://api.binance.com"
KLINES_PATH = "/api/v3/klines"
MAX_LIMIT = 1000  # maximo de velas por llamada que admite Binance

# Columnas crudas de Binance, en el orden de la respuesta, con su dtype.
KLINE_DTYPES: dict[str, str] = {
    "open_time": "int64",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
    "close_time": "int64",
    "quote_asset_volume": "float64",
    "num_trades": "int64",
    "taker_buy_base": "float64",
    "taker_buy_quote": "float64",
    "ignore": "float64",
}
KLINE_COLUMNS = list(KLINE_DTYPES)


def _to_frame(rows: list[list[Any]]) -> pd.DataFrame:
    """Estructura las filas crudas de Binance en un DataFrame con nombres y dtypes."""
    frame = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    return frame.astype(KLINE_DTYPES)


def download_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    *,
    client: HttpClient | None = None,
    base_url: str = DEFAULT_BASE_URL,
    limit: int = MAX_LIMIT,
) -> pd.DataFrame:
    """Descarga las klines de `symbol` entre `start_ms` y `end_ms` (inclusive), paginando."""
    if not 1 <= limit <= MAX_LIMIT:
        raise ValueError(f"limit debe estar entre 1 y {MAX_LIMIT}, recibido {limit}")

    own_client = client is None
    http = HttpClient() if client is None else client
    url = f"{base_url.rstrip('/')}{KLINES_PATH}"
    rows: list[list[Any]] = []
    cursor = start_ms
    try:
        while cursor <= end_ms:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": limit,
            }
            block = http.get(url, params=params).json()
            if not block:
                break
            last_open_time = int(block[-1][0])
            # El cursor solo avanza hacia delante, a la vela siguiente a la ultima recibida:
            # nunca se pide un instante ya descargado ni se mira mas alla de end_ms.
            next_cursor = last_open_time + 1
            if next_cursor <= cursor:
                # Se comprueba antes de anadir: un bloque sin avance son velas ya vistas.
                logger.warning("El cursor no avanza (%d); se corta la paginacion", cursor)
                break
            rows.extend(block)
            cursor = next_cursor
    finally:
        if own_client:
            http.close()

    logger.info("Descargadas %d velas de %s %s", len(rows), symbol, interval)
    return _to_frame(rows)
