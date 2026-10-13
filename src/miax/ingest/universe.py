"""Filtros de universo por liquidez (volumen diario en USDT) y por antiguedad de listado.

La antiguedad se mide con la primera vela diaria de cada par. Tambien reparte el universo
entre categorias de activos (L1, L2, DeFi, memecoins...) con un tope por categoria, y congela
el resultado por periodos en un fichero versionado (`configs/universe.yaml`).
"""

from __future__ import annotations

import dataclasses
import math
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import yaml

from miax.ingest.exchange_info import list_pairs
from miax.ingest.http import HttpClient
from miax.ingest.klines import DAY_MS, DEFAULT_BASE_URL, KLINES_PATH, KLINES_WEIGHT, download_klines
from miax.ingest.ratelimit import RateLimiter
from miax.utils.config import CONFIGS_DIR, ConfigError, load_config
from miax.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_MIN_QUOTE_VOLUME = 10_000_000.0  # USDT por dia (mediana)
VOLUME_INTERVAL = "1d"
VOLUME_COLUMN = "median_quote_volume"

DEFAULT_MIN_LISTING_DAYS = 90  # dias de cotizacion exigidos antes de la referencia
LISTING_INTERVAL = "1d"
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
            block = http.get(url, params=params, weight=KLINES_WEIGHT).json()
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


DEFAULT_UNIVERSE_PARAMS_FILE = "universe_params.yaml"
DEFAULT_UNIVERSE_FILE = "universe.yaml"
UNIVERSE_HEADER = (
    "# Generado por `python -m miax.ingest universe` (MIAX-028). No editar a mano: regenerar.\n"
)
OUTPUT_COLUMNS = ("symbol", "category", "median_quote_volume", "first_candle")
DATE_DTYPE = "datetime64[us, UTC]"
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_MS = pd.Timedelta(milliseconds=1)
_EPOCH = pd.Timestamp(0, tz="UTC")


@dataclasses.dataclass(frozen=True)
class UniverseParams:
    """Parametros del universo por periodos, tal como se leen de universe_params.yaml."""

    quote_asset: str
    first_period_start: str
    last_period_end: str
    period_months: int
    volume_window_months: int
    min_listing_days: int
    min_quote_volume: float
    max_per_category: int
    max_pairs: int
    min_categories: int
    categories_file: str


class UniversePeriod(NamedTuple):
    """Periodo de estudio y su ventana de volumen previa, en milisegundos UTC."""

    window_start_ms: int  # ventana [window_start_ms, window_end_ms], ambos inclusivos
    window_end_ms: int  # siempre period_start_ms - 1
    period_start_ms: int  # periodo [period_start_ms, period_end_ms)
    period_end_ms: int  # exclusivo


def _to_ms(timestamp: pd.Timestamp) -> int:
    """Milisegundos desde la epoca de un Timestamp UTC."""
    return int((timestamp - _EPOCH) // _MS)


def _ms_to_date(ms: int) -> str:
    """Fecha UTC `YYYY-MM-DD` de un instante en milisegundos."""
    return pd.Timestamp(ms, unit="ms", tz="UTC").strftime("%Y-%m-%d")


def _parse_iso_date(name: str, value: str) -> pd.Timestamp:
    """Lee una fecha `YYYY-MM-DD` como medianoche UTC; ConfigError si no es ISO."""
    if not _ISO_DATE.fullmatch(value):
        raise ConfigError(f"El campo '{name}' debe tener formato YYYY-MM-DD, recibido {value!r}.")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ConfigError(f"El campo '{name}' no es una fecha valida: {value!r}.") from error
    return pd.Timestamp(parsed, tz="UTC")


def load_universe_params(path: str | Path = DEFAULT_UNIVERSE_PARAMS_FILE) -> UniverseParams:
    """Lee y valida los parametros del universo; ConfigError si alguno es invalido."""
    params = load_config(path, UniverseParams)
    start = _parse_iso_date("first_period_start", params.first_period_start)
    end = _parse_iso_date("last_period_end", params.last_period_end)
    if start >= end:
        raise ConfigError(
            f"first_period_start ({params.first_period_start}) debe ser anterior a "
            f"last_period_end ({params.last_period_end})."
        )
    minimums = {
        "period_months": (params.period_months, 1),
        "volume_window_months": (params.volume_window_months, 1),
        "min_listing_days": (params.min_listing_days, 0),
        "min_quote_volume": (params.min_quote_volume, 0),
        "max_per_category": (params.max_per_category, 1),
        "max_pairs": (params.max_pairs, 1),
        "min_categories": (params.min_categories, 1),
    }
    if not math.isfinite(params.min_quote_volume):
        raise ConfigError(
            f"El campo 'min_quote_volume' debe ser finito, recibido {params.min_quote_volume}."
        )
    for name, (value, minimum) in minimums.items():
        if value < minimum:
            raise ConfigError(f"El campo '{name}' debe ser >= {minimum}, recibido {value}.")
    return params


def universe_periods(params: UniverseParams) -> list[UniversePeriod]:
    """Divide el rango en periodos de period_months, cada uno con su ventana de volumen previa."""
    if params.period_months < 1:
        raise ValueError(f"period_months debe ser >= 1, recibido {params.period_months}")
    if params.volume_window_months < 1:
        raise ValueError(
            f"volume_window_months debe ser >= 1, recibido {params.volume_window_months}"
        )
    first = _parse_iso_date("first_period_start", params.first_period_start)
    last_end = _parse_iso_date("last_period_end", params.last_period_end)
    periods: list[UniversePeriod] = []
    step = 0
    while True:
        # Se suma desde el origen (no en cadena) para que el dia del mes no derive.
        start = first + pd.DateOffset(months=step * params.period_months)
        if start >= last_end:
            break
        # El fin tambien se suma desde el origen: coincide con el inicio del siguiente, sin huecos.
        end = min(first + pd.DateOffset(months=(step + 1) * params.period_months), last_end)
        window_start = start - pd.DateOffset(months=params.volume_window_months)
        period_start_ms = _to_ms(start)
        # Sin fuga: la ventana acaba 1 ms antes del periodo, asi que su seleccion no ve nada de el.
        periods.append(
            UniversePeriod(
                window_start_ms=_to_ms(window_start),
                window_end_ms=period_start_ms - 1,
                period_start_ms=period_start_ms,
                period_end_ms=_to_ms(end),
            )
        )
        step += 1
    return periods


def prefilter_by_category(
    pairs: pd.DataFrame,
    categories: Mapping[str, str],
    excluded_categories: Sequence[str] = DEFAULT_EXCLUDED_CATEGORIES,
) -> pd.DataFrame:
    """Conserva los pares con categoria util (conocida y no excluida), en su orden de entrada."""
    if "base_asset" not in pairs.columns:
        raise ValueError(f"pairs necesita la columna 'base_asset'; columnas: {list(pairs.columns)}")
    # Misma normalizacion que diversify_universe; la categoria es estatica, sin fuga temporal.
    normalized = {str(k).strip().upper(): v for k, v in categories.items()}
    labels = pairs["base_asset"].astype(str).str.upper().map(normalized)
    usable = labels.notna() & ~labels.isin(list(excluded_categories))
    return pairs[usable].reset_index(drop=True)


def period_candidates(
    first_candles: pd.Series, period: UniversePeriod, params: UniverseParams
) -> list[str]:
    """Simbolos con primera vela anterior al corte de antiguedad del periodo, en el orden dado."""
    # Mismo corte que filter_by_listing_age: depende solo del inicio de la ventana del periodo.
    cutoff = period.window_start_ms - params.min_listing_days * DAY_MS
    old_enough = first_candles[first_candles.notna() & (first_candles <= cutoff)]
    return [str(symbol) for symbol in old_enough.index]


def select_period_universe(
    pairs: pd.DataFrame,
    first_candles: pd.Series,
    volumes: pd.Series,
    categories: Mapping[str, str],
    period: UniversePeriod,
    params: UniverseParams,
) -> pd.DataFrame:
    """Elige el universo de un periodo con antiguedad y volumen medidos antes de que empiece."""
    # first_candles puede venir consultado con una referencia posterior (p. ej. hoy): no hay fuga
    # porque solo se compara con el corte del periodo, y la primera vela es un hecho conocido en ese
    # corte. La referencia de antiguedad es el inicio de la ventana, no el del periodo.
    aged = filter_by_listing_age(
        pairs, first_candles, period.window_start_ms, params.min_listing_days
    )
    liquid = filter_by_volume(aged, volumes, params.min_quote_volume)

    normalized = {str(k).strip().upper(): v for k, v in categories.items()}
    labels = liquid["base_asset"].astype(str).str.upper().map(normalized)
    usable = labels.notna() & ~labels.isin(list(DEFAULT_EXCLUDED_CATEGORIES))
    label = _ms_to_date(period.period_start_ms)
    if not usable.any():
        logger.warning(
            "Periodo %s: ningun par sobrevive a los filtros; el universo queda vacio", label
        )
        return liquid.iloc[0:0].assign(**{CATEGORY_COLUMN: pd.Series(dtype=object)})

    selected = diversify_universe(
        liquid,
        categories,
        max_per_category=params.max_per_category,
        max_pairs=params.max_pairs,
        min_categories=1,
    )
    present = sorted(selected[CATEGORY_COLUMN].unique())
    if len(present) < params.min_categories:
        logger.warning(
            "Periodo %s: solo %d categorias %s, menos del minimo (%d)",
            label,
            len(present),
            present,
            params.min_categories,
        )
    return selected


def build_universe(
    params: UniverseParams,
    categories: Mapping[str, str],
    *,
    client: HttpClient,
    now_ms: int,
    base_url: str = DEFAULT_BASE_URL,
) -> list[tuple[UniversePeriod, pd.DataFrame]]:
    """Descarga catalogo, primeras velas y volumenes y elige el universo de cada periodo."""
    periods = universe_periods(params)
    logger.info("%d periodos a construir", len(periods))

    # D-13: el catalogo incluye los pares en BREAK, contra el sesgo de supervivencia.
    pairs = list_pairs(quote_asset=params.quote_asset, client=client, base_url=base_url)

    # Prefiltro barato: sin categoria util no puede entrar nunca, y ahorra peticiones.
    candidates = prefilter_by_category(pairs, categories)
    logger.info(
        "%d pares contra %s; %d candidatos con categoria util, %d fuera",
        len(pairs),
        params.quote_asset,
        len(candidates),
        len(pairs) - len(candidates),
    )

    # La primera vela es un hecho fijo: se consulta una sola vez y cada periodo la compara
    # con su propio corte, sin que ningun dato posterior al corte influya en la decision.
    first_candles = first_candle_time(
        candidates["symbol"], now_ms, client=client, base_url=base_url
    )

    universes: list[tuple[UniversePeriod, pd.DataFrame]] = []
    for period in periods:
        # La ventana de volumen acaba 1 ms antes del periodo: no se mira nada del periodo.
        volumes = median_daily_quote_volume(
            period_candidates(first_candles, period, params),
            period.window_start_ms,
            period.window_end_ms,
            client=client,
            base_url=base_url,
        )
        selected = select_period_universe(
            candidates, first_candles, volumes, categories, period, params
        )
        counts = selected[CATEGORY_COLUMN].value_counts().to_dict()
        logger.info(
            "%s: %d pares, categorias %s",
            _ms_to_date(period.period_start_ms),
            len(selected),
            counts,
        )
        universes.append((period, selected))
    return universes


def _resolve_universe_path(path: str | Path) -> Path:
    """Las rutas relativas se resuelven contra configs/."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else CONFIGS_DIR / candidate


def write_universe(
    universes: Sequence[tuple[UniversePeriod, pd.DataFrame]],
    params: UniverseParams,
    path: str | Path = DEFAULT_UNIVERSE_FILE,
    *,
    generated_at: str | None = None,
) -> Path:
    """Congela el universo de cada periodo en un YAML versionado, con escritura atomica."""
    periods: list[dict[str, object]] = []
    for period, selected in universes:
        pairs = [
            {
                "symbol": str(row["symbol"]),
                "category": str(row[CATEGORY_COLUMN]),
                "median_quote_volume": round(float(row[VOLUME_COLUMN]), 2),
                "first_candle": _ms_to_date(int(row[LISTING_COLUMN])),
            }
            for _, row in selected.iterrows()
        ]
        counts = Counter(str(pair["category"]) for pair in pairs)
        periods.append(
            {
                "period_start": _ms_to_date(period.period_start_ms),
                "period_end": _ms_to_date(period.period_end_ms),
                "volume_window": [
                    _ms_to_date(period.window_start_ms),
                    _ms_to_date(period.window_end_ms),
                ],
                "n_pairs": len(pairs),
                "categories": dict(counts),
                "pairs": pairs,
            }
        )
    document = {
        "generated_at": generated_at or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "params": dataclasses.asdict(params),
        "periods": periods,
    }
    body = yaml.safe_dump(document, sort_keys=False, allow_unicode=True)

    resolved = _resolve_universe_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(resolved.name + ".tmp")
    try:
        temporary.write_text(UNIVERSE_HEADER + body, encoding="utf-8")
        os.replace(temporary, resolved)
    finally:
        temporary.unlink(missing_ok=True)  # solo queda si la escritura o el reemplazo fallaron
    logger.info("Universo de %d periodos escrito en %s", len(periods), resolved)
    return resolved


def load_universe(path: str | Path = DEFAULT_UNIVERSE_FILE) -> pd.DataFrame:
    """Lee el universo congelado: una fila por par y periodo, en el orden del fichero.

    Un periodo con `pairs: []` no aporta filas; las fechas son siempre `datetime64[us, UTC]`.
    """
    resolved = _resolve_universe_path(path)
    if not resolved.is_file():
        raise ConfigError(f"No se encuentra el fichero de universo '{path}' (probado: {resolved}).")
    try:
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ConfigError(f"El YAML '{resolved}' no se puede leer: {error}") from error
    if not isinstance(raw, dict) or not isinstance(raw.get("periods"), list):
        raise ConfigError(f"El YAML '{resolved}' debe ser un mapeo con una lista 'periods'.")

    rows: list[dict[str, object]] = []
    try:
        for period in raw["periods"]:
            for pair in period["pairs"]:
                rows.append(
                    {
                        "period_start": str(period["period_start"]),
                        "period_end": str(period["period_end"]),
                        "symbol": str(pair["symbol"]),
                        "category": str(pair["category"]),
                        "median_quote_volume": float(pair["median_quote_volume"]),
                        "first_candle": str(pair["first_candle"]),
                    }
                )
    except (KeyError, TypeError, ValueError) as error:
        raise ConfigError(f"El YAML '{resolved}' esta mal formado: {error!r}.") from error

    columns = ["period_start", "period_end", *OUTPUT_COLUMNS]
    frame = pd.DataFrame(rows, columns=columns)
    for column in ("period_start", "period_end", "first_candle"):
        try:
            parsed = pd.to_datetime(frame[column], format="%Y-%m-%d", utc=True)
            frame[column] = parsed.astype(DATE_DTYPE)  # mismo dtype aunque no haya filas
        except ValueError as error:
            raise ConfigError(
                f"El YAML '{resolved}' tiene fechas invalidas en '{column}'."
            ) from error
    frame["median_quote_volume"] = frame["median_quote_volume"].astype("float64")
    return frame
