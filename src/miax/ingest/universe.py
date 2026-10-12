"""Filtros de universo por liquidez (volumen diario en USDT) y por antiguedad de listado.

La antiguedad se mide con la primera vela diaria de cada par. Tambien reparte el universo
entre categorias de activos (L1, L2, DeFi, memecoins...) con un tope por categoria.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd
import yaml

from miax.ingest.http import HttpClient
from miax.ingest.klines import DEFAULT_BASE_URL, KLINES_PATH, download_klines
from miax.ingest.ratelimit import RateLimiter
from miax.utils.config import CONFIGS_DIR, ConfigError
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

CATEGORY_COLUMN = "category"
DEFAULT_CATEGORIES_FILE = "categories.yaml"
DEFAULT_MAX_PER_CATEGORY = 10
DEFAULT_MAX_PAIRS = 50
DEFAULT_MIN_CATEGORIES = 4
DEFAULT_EXCLUDED_CATEGORIES = ("stablecoin", "pegged")  # precio plano o replica de otro activo
MAX_LOGGED_UNCATEGORIZED = 20  # activos sin categoria que se listan en el log


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


def load_categories(path: str | Path = DEFAULT_CATEGORIES_FILE) -> dict[str, str]:
    """Lee el YAML de categorias y devuelve el mapa activo base -> categoria."""
    candidate = Path(path)
    resolved = candidate if candidate.is_absolute() else CONFIGS_DIR / candidate
    if not resolved.is_file():
        raise ConfigError(
            f"No se encuentra el fichero de categorias '{path}' (probado: {resolved})."
        )
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"El YAML '{resolved}' debe ser un mapeo categoria -> lista de activos.")

    asset_to_category: dict[str, str] = {}
    for category, assets in raw.items():
        if not isinstance(category, str):
            raise ConfigError(f"La categoria {category!r} de '{resolved}' debe ser texto.")
        if not isinstance(assets, list):
            raise ConfigError(
                f"La categoria '{category}' de '{resolved}' debe ser una lista de activos, "
                f"no {type(assets).__name__}."
            )
        for asset in assets:
            if not isinstance(asset, str) or not asset.strip():
                raise ConfigError(
                    f"La categoria '{category}' de '{resolved}' tiene un activo invalido: "
                    f"{asset!r}."
                )
            normalized = asset.strip().upper()
            previous = asset_to_category.setdefault(normalized, category)
            if previous != category:
                raise ConfigError(
                    f"El activo '{normalized}' esta en dos categorias: '{previous}' y '{category}'."
                )
    return asset_to_category


def diversify_universe(
    pairs: pd.DataFrame,
    categories: Mapping[str, str],
    *,
    max_per_category: int = DEFAULT_MAX_PER_CATEGORY,
    max_pairs: int = DEFAULT_MAX_PAIRS,
    min_categories: int = DEFAULT_MIN_CATEGORIES,
    excluded_categories: Sequence[str] = DEFAULT_EXCLUDED_CATEGORIES,
) -> pd.DataFrame:
    """Reparte el universo entre categorias con tope por categoria y total, en orden de entrada."""
    if max_per_category < 1:
        raise ValueError(f"max_per_category debe ser >= 1, recibido {max_per_category}")
    if max_pairs < 1:
        raise ValueError(f"max_pairs debe ser >= 1, recibido {max_pairs}")
    if min_categories < 1:
        raise ValueError(f"min_categories debe ser >= 1, recibido {min_categories}")
    if "base_asset" not in pairs.columns:
        raise ValueError(f"pairs necesita la columna 'base_asset'; columnas: {list(pairs.columns)}")

    # La categoria es una clasificacion estatica por activo: no usa precios, no hay fuga temporal.
    labelled = pairs.copy()
    normalized_categories = {str(k).strip().upper(): v for k, v in categories.items()}
    labelled[CATEGORY_COLUMN] = (
        labelled["base_asset"].astype(str).str.upper().map(normalized_categories)
    )
    no_category = labelled[CATEGORY_COLUMN].isna()
    excluded = ~no_category & labelled[CATEGORY_COLUMN].isin(list(excluded_categories))
    candidates = labelled[~no_category & ~excluded]

    # pairs llega ordenado por prioridad (volumen): head() conserva ese orden en cada grupo.
    capped = candidates.groupby(CATEGORY_COLUMN, sort=False).head(max_per_category)
    kept = capped.head(max_pairs).reset_index(drop=True)

    counts = kept[CATEGORY_COLUMN].value_counts().to_dict()
    uncategorized = list(dict.fromkeys(labelled.loc[no_category, "base_asset"]))
    shown = uncategorized[:MAX_LOGGED_UNCATEGORIZED]
    if len(uncategorized) > MAX_LOGGED_UNCATEGORIZED:
        shown_text = f"{shown} y {len(uncategorized) - MAX_LOGGED_UNCATEGORIZED} mas"
    else:
        shown_text = str(shown)
    logger.info(
        "Universo diversificado: %d pares entran en %d categorias %s; %d descartados por "
        "categoria excluida, %d sin categoria %s, %d por tope de categoria (%d), %d por tope "
        "total (%d)",
        len(kept),
        len(counts),
        counts,
        int(excluded.sum()),
        int(no_category.sum()),
        shown_text,
        len(candidates) - len(capped),
        max_per_category,
        len(capped) - len(kept),
        max_pairs,
    )
    if len(counts) < min_categories:
        raise ValueError(
            f"El universo tiene {len(counts)} categorias ({sorted(counts)}), "
            f"menos del minimo exigido ({min_categories})"
        )
    return kept
