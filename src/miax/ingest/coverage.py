"""Informe de cobertura temporal de la cache de klines: velas presentes, huecos y operaciones."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from miax.ingest.integrity import find_gaps
from miax.ingest.klines import candle_offset_ms, interval_ms
from miax.ingest.store import DEFAULT_BASE_DIR, load_klines
from miax.utils.logging import get_logger

logger = get_logger(__name__)

COVERAGE_COLUMNS = [
    "symbol",
    "first_open",
    "last_open",
    "expected",
    "present",
    "coverage_pct",
    "traded_pct",
    "n_gaps",
    "max_gap_candles",
    "missing_before",
    "missing_after",
]
_REQUIRED_COLUMNS = ["open_time", "num_trades"]
_DATETIME_COLUMNS = ["first_open", "last_open"]
_MS_PER_NS = 1_000_000


def _timestamp(open_time_ms: int | None) -> pd.Timestamp:
    """Convierte ms UTC a Timestamp con zona UTC; NaT si es None."""
    if open_time_ms is None:
        return pd.NaT
    return pd.Timestamp(open_time_ms, unit="ms", tz="UTC")


def _ceil_div(numerator: int, denominator: int) -> int:
    """Division entera redondeada hacia arriba (denominador positivo)."""
    return -(-numerator // denominator)


def _pct(count: int, expected: int) -> float:
    """Porcentaje de `count` sobre `expected`; NaN si no se esperaba ninguna vela."""
    return 100.0 * count / expected if expected > 0 else float("nan")


def _finish(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Fija las columnas y el dtype datetime UTC de las fechas, tambien si no hay filas."""
    frame = frame[columns].copy()
    for column in _DATETIME_COLUMNS:
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame.reset_index(drop=True)


def coverage_from_frame(
    candles: pd.DataFrame,
    interval: str,
    start_ms: int | None = None,
    end_ms: int | None = None,
) -> dict[str, object]:
    """Mide la cobertura de unas velas; con rango `[start_ms, end_ms)` cuenta tambien los bordes."""
    missing = [c for c in _REQUIRED_COLUMNS if c not in candles.columns]
    if missing:
        raise ValueError(f"Faltan columnas para medir la cobertura: {missing}")
    step = interval_ms(interval)
    if step is None:
        raise ValueError(f"El intervalo {interval!r} no tiene duracion fija: no hay cobertura")
    if (start_ms is None) != (end_ms is None):
        raise ValueError("start_ms y end_ms se dan juntos o ninguno")
    if start_ms is not None and end_ms is not None and start_ms >= end_ms:
        raise ValueError(f"Rango vacio o invertido: [{start_ms}, {end_ms})")

    # Cada vela se cuenta una vez, aunque la cache la repita.
    unique = candles.drop_duplicates(subset="open_time").sort_values("open_time", kind="stable")
    offset = candle_offset_ms(interval)

    if start_ms is None or end_ms is None:
        # Solo cuentan las velas sobre la rejilla del intervalo, igual que con rango.
        inside = unique[(unique["open_time"] - offset) % step == 0]
        expected = 0
        first_slot = last_slot = 0
        if not inside.empty:
            first_slot = int(inside["open_time"].iloc[0])
            last_slot = int(inside["open_time"].iloc[-1])
            expected = (last_slot - first_slot) // step + 1
    else:
        times = unique["open_time"]
        # Solo cuentan las velas dentro del rango y sobre la rejilla del intervalo.
        inside = unique[(times >= start_ms) & (times < end_ms) & ((times - offset) % step == 0)]
        first_k = _ceil_div(start_ms - offset, step)
        last_k = _ceil_div(end_ms - offset, step) - 1
        expected = max(0, last_k - first_k + 1)
        first_slot = offset + first_k * step
        last_slot = offset + last_k * step

    present = len(inside)
    traded = int((inside["num_trades"] > 0).sum())
    if present:
        first_open = int(inside["open_time"].iloc[0])
        last_open = int(inside["open_time"].iloc[-1])
        missing_before = (first_open - first_slot) // step
        missing_after = (last_slot - last_open) // step
    else:
        first_open = last_open = None
        # Sin velas, todo el rango esperado queda "antes" y nada "despues".
        missing_before = expected if start_ms is not None else 0
        missing_after = 0

    # Los huecos se miden entre velas consecutivas del propio conjunto: solo miran hacia atras.
    gaps = find_gaps(inside["open_time"], interval)
    return {
        "first_open": _timestamp(first_open),
        "last_open": _timestamp(last_open),
        "expected": expected,
        "present": present,
        "coverage_pct": _pct(present, expected),
        "traded_pct": _pct(traded, expected),
        "n_gaps": len(gaps),
        "max_gap_candles": int(gaps["missing_candles"].max()) if not gaps.empty else 0,
        "missing_before": int(missing_before),
        "missing_after": int(missing_after),
    }


def _symbol_coverage(
    symbol: str,
    interval: str,
    base_dir: str | Path,
    start_ms: int | None,
    end_ms: int | None,
) -> dict[str, object]:
    """Lee la cache de `symbol` (si hay) y devuelve su fila de cobertura, con `symbol`."""
    # `load_klines` es inclusivo en `end_ms`; el rango de cobertura es semiabierto.
    candles = load_klines(
        symbol,
        interval,
        base_dir=base_dir,
        start_ms=start_ms,
        end_ms=None if end_ms is None else end_ms - 1,
    )
    return {"symbol": symbol, **coverage_from_frame(candles, interval, start_ms, end_ms)}


def coverage_report(
    symbols: Sequence[str],
    interval: str,
    *,
    base_dir: str | Path = DEFAULT_BASE_DIR,
    start_ms: int | None = None,
    end_ms: int | None = None,
) -> pd.DataFrame:
    """Devuelve una fila de cobertura por simbolo, con `present = 0` si no hay cache."""
    rows: list[dict[str, object]] = []
    for symbol in dict.fromkeys(symbols):
        row = _symbol_coverage(symbol, interval, base_dir, start_ms, end_ms)
        logger.info(
            "Cobertura %s %s: %d de %d velas (%.2f %%), %.2f %% con operaciones",
            symbol,
            interval,
            row["present"],
            row["expected"],
            row["coverage_pct"],
            row["traded_pct"],
        )
        rows.append(row)
    return _finish(pd.DataFrame(rows, columns=COVERAGE_COLUMNS), COVERAGE_COLUMNS)


def universe_coverage(
    universe: pd.DataFrame,
    interval: str,
    *,
    base_dir: str | Path = DEFAULT_BASE_DIR,
    now_ms: int | None = None,
) -> pd.DataFrame:
    """Mide la cobertura de cada par en su periodo del universo; informe descriptivo, no selector.

    No debe usarse para elegir los pares del mismo periodo que mide: seria mirar al futuro.
    """
    columns = ["period_start", "period_end", "category", *COVERAGE_COLUMNS]
    rows: list[dict[str, object]] = []
    for item in universe.itertuples(index=False):
        start_ms = int(item.period_start.value // _MS_PER_NS)
        end_ms = int(item.period_end.value // _MS_PER_NS)
        if now_ms is not None:
            # Un periodo en curso se mide solo hasta "ahora"; el informe describe, no decide.
            end_ms = min(end_ms, now_ms)
        if end_ms <= start_ms:
            continue  # Periodo futuro: aun no hay nada que medir.
        row = _symbol_coverage(item.symbol, interval, base_dir, start_ms, end_ms)
        rows.append(
            {
                "period_start": item.period_start,
                "period_end": item.period_end,
                "category": item.category,
                **row,
            }
        )
    frame = pd.DataFrame(rows, columns=columns)
    for column in ("period_start", "period_end"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return _finish(frame, columns)
