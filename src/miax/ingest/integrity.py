"""Validador de integridad de la cache de klines: detecta ficheros corruptos y huecos."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from miax.ingest.klines import KLINE_DTYPES, candle_offset_ms, interval_ms
from miax.ingest.store import _MONTH_STEM, DEFAULT_BASE_DIR, _month_bounds_ms, _validate_segment
from miax.utils.io import read_parquet, resolve_repo_path
from miax.utils.logging import get_logger

logger = get_logger(__name__)

ISSUE_COLUMNS = ["file", "check", "severity", "count", "detail"]
GAP_COLUMNS = ["gap_start_ms", "gap_end_ms", "missing_candles"]
ERROR = "error"
WARNING = "warning"
_NO_FILE = ""  # `file` en la funcion pura: lo rellena `validate_klines`


def _issue(file: str, check: str, severity: str, count: int, detail: str) -> dict[str, object]:
    """Construye una fila de incidencia con las claves de `ISSUE_COLUMNS`."""
    return {"file": file, "check": check, "severity": severity, "count": count, "detail": detail}


def _to_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Convierte filas de incidencia en un DataFrame con las columnas canonicas."""
    return pd.DataFrame(rows, columns=ISSUE_COLUMNS)


def check_klines_frame(
    df: pd.DataFrame, interval: str, *, month: str | None = None
) -> pd.DataFrame:
    """Revisa un fichero de klines y devuelve sus incidencias, con `file` vacio.

    `month` (`YYYY-MM`) activa `wrong_month`; `count` son las filas (columnas en `schema`).
    """
    missing = [c for c in KLINE_DTYPES if c not in df.columns]
    wrong_dtype = [c for c, t in KLINE_DTYPES.items() if c in df.columns and str(df[c].dtype) != t]
    if missing or wrong_dtype:
        # Sin esquema valido las demas comprobaciones no tienen sentido (ni son seguras).
        detail = f"faltan {missing}; dtype distinto en {wrong_dtype}"
        return _to_frame(
            [_issue(_NO_FILE, "schema", ERROR, len(missing) + len(wrong_dtype), detail)]
        )

    rows: list[dict[str, object]] = []
    open_time = df["open_time"]

    dup = int(open_time.duplicated().sum())
    if dup:
        rows.append(_issue(_NO_FILE, "duplicates", ERROR, dup, f"{dup} open_time repetidos"))

    # diff entre filas contiguas del propio fichero: solo mira hacia atras, sin fuga.
    unordered = int((open_time.diff() < 0).sum())
    if unordered:
        rows.append(_issue(_NO_FILE, "unordered", ERROR, unordered, "open_time decrece"))

    step = interval_ms(interval)
    if step is not None:
        offset = candle_offset_ms(interval)
        misaligned = int(((open_time - offset) % step != 0).sum())
        if misaligned:
            detail = f"open_time fuera de la rejilla de {interval}"
            rows.append(_issue(_NO_FILE, "misaligned", ERROR, misaligned, detail))
        bad_close = int((df["close_time"] != open_time + step - 1).sum())
        if bad_close:
            detail = f"close_time distinto de open_time + {step} - 1"
            rows.append(_issue(_NO_FILE, "close_time", ERROR, bad_close, detail))

    if month is not None:
        start, end = _month_bounds_ms(month)
        wrong_month = int(((open_time < start) | (open_time >= end)).sum())
        if wrong_month:
            detail = f"velas fuera del mes {month}"
            rows.append(_issue(_NO_FILE, "wrong_month", ERROR, wrong_month, detail))

    prices = df[["open", "high", "low", "close"]]
    bad_ohlc = (
        (df["low"] > prices[["open", "close"]].min(axis=1))
        | (df["high"] < prices[["open", "close"]].max(axis=1))
        | (prices < 0).any(axis=1)
        | (df["volume"] < 0)
    )
    if bad_ohlc.any():
        detail = "low/high incoherentes con open/close, o precio o volumen negativos"
        rows.append(_issue(_NO_FILE, "ohlc", ERROR, int(bad_ohlc.sum()), detail))

    return _to_frame(rows)


def find_gaps(open_times: pd.Series, interval: str) -> pd.DataFrame:
    """Devuelve los huecos entre velas consecutivas: primer y ultimo `open_time` que faltan."""
    step = interval_ms(interval)
    if step is None:
        return pd.DataFrame(columns=GAP_COLUMNS, dtype="int64")
    # Se ordena y se quitan repetidos; cada hueco se mide solo con la vela anterior.
    ordered = open_times.drop_duplicates().sort_values().reset_index(drop=True)
    previous = ordered.shift(1)
    jump = ordered - previous
    # `>= 2 * step`: un salto de un solo paso no deja velas sin cubrir.
    mask = jump >= 2 * step
    return pd.DataFrame(
        {
            "gap_start_ms": (previous[mask] + step).astype("int64"),
            "gap_end_ms": (ordered[mask] - step).astype("int64"),
            "missing_candles": (jump[mask] // step - 1).astype("int64"),
        }
    ).reset_index(drop=True)


def _month_files(symbol: str, interval: str, base_dir: str | Path) -> tuple[Path, list[Path]]:
    """Devuelve la carpeta de `symbol`/`interval` y sus ficheros de mes ordenados."""
    directory = (
        resolve_repo_path(base_dir) / _validate_segment(symbol) / _validate_segment(interval)
    )
    paths = sorted(p for p in directory.glob("*.parquet") if _MONTH_STEM.fullmatch(p.stem))
    return directory, paths


def count_month_files(
    symbol: str, interval: str, *, base_dir: str | Path = DEFAULT_BASE_DIR
) -> int:
    """Cuenta los ficheros de mes cacheados de `symbol`/`interval` (0 si no hay cache)."""
    return len(_month_files(symbol, interval, base_dir)[1])


def validate_klines(
    symbol: str, interval: str, *, base_dir: str | Path = DEFAULT_BASE_DIR
) -> pd.DataFrame:
    """Revisa la cache de `symbol`/`interval` fichero a fichero y devuelve las incidencias.

    Las incidencias globales (`duplicates` entre ficheros y `gaps`) llevan en `file` la
    etiqueta `SIMBOLO/intervalo`.
    """
    # Solo lee: nunca corrige ni borra nada de la cache; eso lo decide un humano.
    directory, paths = _month_files(symbol, interval, base_dir)
    if not paths:
        logger.warning("Sin ficheros que validar de %s %s en %s", symbol, interval, directory)
        return _to_frame([])

    rows: list[dict[str, object]] = []
    valid_times: list[pd.Series] = []
    within_dups = 0
    for path in paths:
        try:
            frame = read_parquet(path)
        except Exception as exc:  # noqa: BLE001 - cualquier fallo de lectura es corrupcion
            rows.append(_issue(path.name, "unreadable", ERROR, 1, f"{type(exc).__name__}: {exc}"))
            continue
        issues = check_klines_frame(frame, interval, month=path.stem)
        issues["file"] = path.name
        rows.extend(issues.to_dict("records"))
        if (issues["check"] == "schema").any():
            continue
        within_dups += int(frame["open_time"].duplicated().sum())
        valid_times.append(frame["open_time"])

    label = f"{symbol}/{interval}"
    if valid_times:
        all_times = pd.concat(valid_times, ignore_index=True)
        cross = int(all_times.duplicated().sum()) - within_dups
        if cross > 0:
            detail = f"{cross} open_time repetidos entre ficheros"
            rows.append(_issue(label, "duplicates", ERROR, cross, detail))
        gaps = find_gaps(all_times, interval)
        if not gaps.empty:
            missing = int(gaps["missing_candles"].sum())
            largest = int(gaps["missing_candles"].max())
            detail = f"{len(gaps)} huecos, el mayor de {largest} velas"
            rows.append(_issue(label, "gaps", WARNING, missing, detail))

    result = _to_frame(rows)
    errors = int((result["severity"] == ERROR).sum())
    logger.info(
        "Validado %s %s: %d ficheros, %d errores, %d avisos",
        symbol,
        interval,
        len(paths),
        errors,
        len(result) - errors,
    )
    return result
