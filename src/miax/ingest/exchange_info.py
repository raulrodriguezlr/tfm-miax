"""Catalogo de pares de Binance: consulta `/exchangeInfo` y lo estructura en un DataFrame."""

from __future__ import annotations

from typing import Any

import pandas as pd

from miax.ingest.http import HttpClient
from miax.ingest.klines import DEFAULT_BASE_URL
from miax.utils.logging import get_logger

logger = get_logger(__name__)

EXCHANGE_INFO_PATH = "/api/v3/exchangeInfo"
EXCHANGE_INFO_WEIGHT = 20  # peso de la llamada segun Binance, para el limitador

# Columnas del catalogo, con su dtype.
PAIR_DTYPES: dict[str, str] = {
    "symbol": "str",
    "status": "str",
    "base_asset": "str",
    "quote_asset": "str",
    "is_spot_trading_allowed": "bool",
    "is_margin_trading_allowed": "bool",
}
PAIR_COLUMNS = list(PAIR_DTYPES)


def fetch_exchange_info(
    *,
    client: HttpClient | None = None,
    base_url: str = DEFAULT_BASE_URL,
) -> dict[str, Any]:
    """Pide `/exchangeInfo` a Binance y devuelve el JSON crudo (unos 17 MB, peso 20)."""
    own_client = client is None
    http = HttpClient() if client is None else client
    url = f"{base_url.rstrip('/')}{EXCHANGE_INFO_PATH}"
    try:
        payload: dict[str, Any] = http.get(url, weight=EXCHANGE_INFO_WEIGHT).json()
    finally:
        if own_client:
            http.close()
    return payload


def pairs_from_exchange_info(payload: dict[str, Any]) -> pd.DataFrame:
    """Convierte los simbolos de `/exchangeInfo` en un DataFrame; ValueError si faltan datos."""
    if "symbols" not in payload:
        raise ValueError(
            "La respuesta de exchangeInfo no trae la clave 'symbols'; "
            f"claves recibidas: {sorted(payload)}"
        )
    rows = []
    for position, item in enumerate(payload["symbols"]):
        try:
            row = {
                "symbol": item["symbol"],
                "status": item["status"],
                "base_asset": item["baseAsset"],
                "quote_asset": item["quoteAsset"],
                "is_spot_trading_allowed": item.get("isSpotTradingAllowed", False),
                "is_margin_trading_allowed": item.get("isMarginTradingAllowed", False),
            }
        except KeyError as exc:
            raise ValueError(
                f"Falta el campo {exc} en el simbolo {item.get('symbol', '?')!r} "
                f"(posicion {position} de la lista)"
            ) from exc
        rows.append(row)
    frame = pd.DataFrame(rows, columns=PAIR_COLUMNS).astype(PAIR_DTYPES)
    return frame.sort_values("symbol").reset_index(drop=True)


def list_pairs(
    *,
    quote_asset: str | None = None,
    status: str | None = None,
    client: HttpClient | None = None,
    base_url: str = DEFAULT_BASE_URL,
) -> pd.DataFrame:
    """Lista el catalogo completo de pares, con filtros opcionales por activo cotizado y estado."""
    payload = fetch_exchange_info(client=client, base_url=base_url)
    pairs = pairs_from_exchange_info(payload)
    logger.info(
        "Catalogo de %d pares; por estado: %s",
        len(pairs),
        pairs["status"].value_counts().to_dict(),
    )
    if quote_asset is not None:
        pairs = pairs[pairs["quote_asset"] == quote_asset]
    if status is not None:
        pairs = pairs[pairs["status"] == status]
    return pairs.reset_index(drop=True)
