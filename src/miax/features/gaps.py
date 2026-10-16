"""Clasificacion por causa de cada slot de la rejilla comun, con o sin vela con operaciones."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import IntEnum
from pathlib import Path

import numpy as np
import pandas as pd

from miax.features.alignment import align_klines, grid_ms
from miax.ingest.klines import DAY_MS
from miax.ingest.store import DEFAULT_BASE_DIR, cache_bounds
from miax.utils.logging import get_logger

logger = get_logger(__name__)

# La clasificacion es EX POST: separar HALTED de DELISTED y fijar la vida de la serie exige saber
# si la serie vuelve, y eso es futuro. Sirve para limpiar y para la memoria, NUNCA como feature
# del modelo; lo unico que se conoce en `t` es si hay una observacion buena (`VALID`), MIAX-035.

_INT64_MAX = int(np.iinfo("int64").max)
_INT64_MIN = int(np.iinfo("int64").min)
RUN_COLUMNS = ["symbol", "cause", "start", "end", "n_slots"]


class GapCause(IntEnum):
    """Causa de cada slot de la rejilla; se guarda como `int8`.

    La clasificacion es ex post (mira al futuro de la serie): no se usa como feature del modelo.
    """

    VALID = 0
    NO_TRADES = 1
    NOT_LISTED = 2
    HALTED = 3
    OUTAGE = 4
    DELISTED = 5
    NO_DATA = 6


def classify_slots(
    num_trades: pd.DataFrame,
    *,
    first_ms: Mapping[str, int],
    last_ms: Mapping[str, int],
    listed_ms: Mapping[str, int] | None = None,
    data_end_ms: int | Mapping[str, int] | None = None,
    outage_min_share: float = 0.5,
    outage_min_symbols: int = 2,
) -> pd.DataFrame:
    """Etiqueta con un `GapCause` (`int8`) cada slot de `num_trades` (NaN = sin vela).

    `first_ms`/`last_ms` son la vida de cada serie en toda la cache y `data_end_ms` es la ultima
    `open_time` descargada (inclusiva), un entero para todos o un mapeo por simbolo (ver
    `download_ends_from_universe`); el simbolo que no este en el mapeo, o `None`, usa el maximo de
    `last_ms` de los pasados, asi que `DELISTED` depende de ellos y conviene darlo explicito.
    `OUTAGE` exige que falten al menos `outage_min_symbols` simbolos y `outage_min_share` de los
    vivos, y depende de los pasados.
    """
    if not 0 < outage_min_share <= 1:
        raise ValueError(f"outage_min_share debe estar en (0, 1]: {outage_min_share}")
    if outage_min_symbols < 1:
        raise ValueError(f"outage_min_symbols debe ser >= 1: {outage_min_symbols}")
    names = [str(c) for c in num_trades.columns]
    if len(set(names)) != len(names):
        raise ValueError("num_trades tiene simbolos repetidos en las columnas")
    listed = listed_ms or {}
    n_symbols = len(names)

    first = np.full(n_symbols, _INT64_MAX, dtype="int64")
    last = np.full(n_symbols, _INT64_MIN, dtype="int64")
    listed_eff = np.full(n_symbols, _INT64_MIN, dtype="int64")
    has_bounds = np.zeros(n_symbols, dtype=bool)
    for j, symbol in enumerate(names):
        if (symbol in first_ms) != (symbol in last_ms):
            raise ValueError(f"{symbol}: hay que dar first_ms y last_ms a la vez")
        if symbol in first_ms:
            first[j], last[j], has_bounds[j] = first_ms[symbol], last_ms[symbol], True
            if first[j] > last[j]:
                raise ValueError(f"{symbol}: first_ms es posterior a last_ms")
        if symbol in listed:
            listed_eff[j] = listed[symbol]
            # first_candle trae precision de dia: si la primera vela cacheada cae ese mismo dia,
            # el listado exacto es esa vela; si no, el listado es el inicio del dia dado.
            if has_bounds[j] and first[j] // DAY_MS == listed[symbol] // DAY_MS:
                listed_eff[j] = first[j]
    default_end = max((last_ms[s] for s in names if s in last_ms), default=0)
    end = np.full(n_symbols, default_end, dtype="int64")
    if isinstance(data_end_ms, Mapping):
        for j, symbol in enumerate(names):
            if symbol in data_end_ms:
                end[j] = data_end_ms[symbol]
    elif data_end_ms is not None:
        end[:] = data_end_ms

    slots = grid_ms(num_trades.index)[:, None]
    values = num_trades.to_numpy(dtype="float64")
    has = ~np.isnan(values)
    if (outside := has & ((slots < first) | (slots > last))).any():
        bad = [names[j] for j in np.flatnonzero(outside.any(axis=0))]
        raise ValueError(f"Hay velas fuera de los limites de cache (first_ms/last_ms) de: {bad}")

    causes = np.full(values.shape, GapCause.NO_DATA, dtype="int8")
    causes[has] = GapCause.NO_TRADES
    causes[has & (values > 0)] = GapCause.VALID

    missing = ~has
    alive = (slots >= first) & (slots <= last)
    # Antes de la primera vela de la cache solo se afirma NOT_LISTED si el listado es conocido.
    causes[missing & (slots < first) & (slots < listed_eff)] = GapCause.NOT_LISTED
    interior = missing & alive
    causes[interior] = GapCause.HALTED
    # `slots <= end`: lo descargado de cada simbolo llega hasta su `end` inclusive; mas alla, nada.
    causes[missing & has_bounds & (slots > last) & (slots <= end)] = GapCause.DELISTED

    # Caida de Binance: en ese slot faltan al menos `outage_min_symbols` simbolos y una proporcion
    # alta de los vivos. Contemporanea (solo mira ese slot), pero depende de que simbolos se pasen.
    n_alive = alive.sum(axis=1)
    n_missing = interior.sum(axis=1)
    share = n_missing / np.maximum(n_alive, 1)
    outage_rows = (n_missing >= outage_min_symbols) & (share >= outage_min_share)
    causes[interior & outage_rows[:, None]] = GapCause.OUTAGE
    return pd.DataFrame(causes, index=num_trades.index, columns=num_trades.columns)


def gap_runs(causes: pd.DataFrame) -> pd.DataFrame:
    """Lista los tramos consecutivos con la misma causa (sin `VALID`); `end` es el ultimo slot."""
    values = causes.to_numpy(dtype="int8")
    frames: list[pd.DataFrame] = []
    for j, symbol in enumerate(causes.columns):
        column = values[:, j]
        if not len(column):
            continue
        # Cada cambio de causa entre slots contiguos abre un tramo nuevo; solo mira lo ya etiquetado
        cuts = np.flatnonzero(np.diff(column)) + 1
        starts = np.concatenate(([0], cuts))
        ends = np.concatenate((cuts, [len(column)])) - 1
        keep = column[starts] != GapCause.VALID
        starts, ends = starts[keep], ends[keep]
        frames.append(
            pd.DataFrame(
                {
                    "symbol": [symbol] * len(starts),
                    "cause": [GapCause(int(c)).name for c in column[starts]],
                    "start": causes.index[starts],
                    "end": causes.index[ends],
                    "n_slots": (ends - starts + 1).astype("int64"),
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=RUN_COLUMNS).astype({"n_slots": "int64"})
    return pd.concat(frames, ignore_index=True)[RUN_COLUMNS]


def cause_counts(causes: pd.DataFrame) -> pd.DataFrame:
    """Cuenta los slots de cada simbolo (filas) por causa (columnas, todas las de `GapCause`)."""
    values = causes.to_numpy(dtype="int8")
    counts = {cause.name: (values == cause).sum(axis=0) for cause in GapCause}
    return pd.DataFrame(counts, index=pd.Index(causes.columns, name="symbol"), dtype="int64")


def listings_from_universe(universe: pd.DataFrame) -> dict[str, int]:
    """Da el inicio (ms UTC) de la primera vela 1d de cada simbolo del universo, el minimo."""
    missing = [c for c in ("symbol", "first_candle") if c not in universe.columns]
    if missing:
        raise ValueError(f"Faltan columnas del universo: {missing}")
    earliest = universe.groupby("symbol")["first_candle"].min()
    return {str(symbol): int(ts.timestamp() * 1000) for symbol, ts in earliest.items()}


def download_ends_from_universe(universe: pd.DataFrame) -> dict[str, int]:
    """Da la ultima `open_time` (ms UTC) pedida de cada simbolo: su ultimo `period_end` menos 1."""
    missing = [c for c in ("symbol", "period_end") if c not in universe.columns]
    if missing:
        raise ValueError(f"Faltan columnas del universo: {missing}")
    # La cache de cada par se descargo de su primer a su ultimo periodo del universo: mas alla de
    # eso no hay datos, y eso no es un deslistado. `period_end` es exclusivo, de ahi el -1.
    latest = universe.groupby("symbol")["period_end"].max()
    return {str(symbol): int(ts.timestamp() * 1000) - 1 for symbol, ts in latest.items()}


def classify_klines(
    symbols: Sequence[str],
    interval: str,
    start_ms: int,
    end_ms: int,
    *,
    base_dir: str | Path = DEFAULT_BASE_DIR,
    listed_ms: Mapping[str, int] | None = None,
    data_end_ms: int | Mapping[str, int] | None = None,
    outage_min_share: float = 0.5,
    outage_min_symbols: int = 2,
) -> pd.DataFrame:
    """Clasifica los slots de `[start_ms, end_ms)` de la cache con `classify_slots`.

    `data_end_ms` es la ultima `open_time` inclusiva, un entero o un mapeo por simbolo (no el
    `end_ms` del rango); por defecto `DELISTED` depende de los simbolos pasados, conviene darlo.
    """
    names = list(dict.fromkeys(symbols))
    num_trades = align_klines(
        names, interval, start_ms, end_ms, fields=("num_trades",), base_dir=base_dir
    )["num_trades"]
    first_ms: dict[str, int] = {}
    last_ms: dict[str, int] = {}
    for symbol in names:
        # La vida de la serie se mide en TODA la cache, no solo en el rango pedido: si no, un
        # hueco en el borde del rango se confundiria con un listado o un deslistado.
        bounds = cache_bounds(symbol, interval, base_dir=base_dir)
        if bounds is None:
            logger.warning(
                "Sin cache de %s %s: sus slots seran NO_DATA o NOT_LISTED", symbol, interval
            )
            continue
        first_ms[symbol], last_ms[symbol] = bounds
    return classify_slots(
        num_trades,
        first_ms=first_ms,
        last_ms=last_ms,
        listed_ms=listed_ms,
        data_end_ms=data_end_ms,
        outage_min_share=outage_min_share,
        outage_min_symbols=outage_min_symbols,
    )
