"""Cache del panel maestro en Parquet: un directorio por periodo y un fichero por feature."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from miax.features.alignment import build_time_grid
from miax.features.contract import validate_panel
from miax.features.panel import FEATURE_NAMES, FEATURES_VERSION, Panel, build_panel
from miax.ingest.store import DEFAULT_BASE_DIR, month_files
from miax.ingest.universe import to_ms
from miax.utils.io import read_parquet, resolve_repo_path, write_parquet
from miax.utils.logging import get_logger

logger = get_logger(__name__)

PANEL_DIR = "data/processed/panel"
PANEL_FORMAT_VERSION = 1
META_FILE = "meta.json"
VALID_FILE = "valid"
_DAY_MS = 86_400_000
_MINUTE_MS = 60_000


def _stamp(ms: int) -> str:
    """Fecha `YYYY-MM-DD` de un instante UTC, con la hora si no cae en medianoche."""
    moment = pd.Timestamp(ms, unit="ms", tz="UTC")
    if ms % _DAY_MS == 0:
        return moment.strftime("%Y-%m-%d")
    # Con segundos y ms si no es minuto exacto: dos limites distintos no comparten carpeta.
    if ms % _MINUTE_MS == 0:
        return moment.strftime("%Y-%m-%dT%H%M")
    return moment.strftime("%Y-%m-%dT%H%M%S") + f"{ms % 1000:03d}"


def panel_path(
    interval: str, start_ms: int, end_ms: int, *, cache_dir: str | Path = PANEL_DIR
) -> Path:
    """Directorio de la cache de un periodo: `{cache_dir}/{interval}/{inicio}_{fin}`."""
    return resolve_repo_path(cache_dir) / interval / f"{_stamp(start_ms)}_{_stamp(end_ms)}"


def cache_fingerprint(
    symbols: Sequence[str], interval: str, *, base_dir: str | Path = DEFAULT_BASE_DIR
) -> dict[str, list[list[Any]]]:
    """Nombre y tamano de cada fichero de mes de la cache de velas de cada simbolo."""
    # Nombre y tamano, no `mtime`: descomprimir la cache en otra maquina no la invalida.
    result: dict[str, list[list[Any]]] = {}
    for symbol in dict.fromkeys(symbols):
        _, paths = month_files(symbol, interval, base_dir)
        result[symbol] = [[path.name, path.stat().st_size] for path in paths]
    return result


def _native(value: object) -> int | float | bool | str:
    """Convierte escalares de numpy a int/float nativos para `json.dumps`."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} no es serializable en meta.json")


def _write_meta(directory: Path, meta_text: str) -> None:
    """Escribe `meta.json` (ya serializado) de forma atomica."""
    target = directory / META_FILE
    tmp = target.with_name(target.name + ".tmp")
    try:
        tmp.write_text(meta_text, encoding="utf-8")
        # os.replace es atomico: sin meta.json completo, el directorio no es una cache valida.
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _read_meta(directory: Path) -> dict[str, Any]:
    """Lee `meta.json`; `ValueError` si falta, no se entiende o es de otro formato o calculo."""
    target = directory / META_FILE
    if not target.is_file():
        raise ValueError(f"{directory} no es una cache valida: falta {META_FILE}")
    try:
        meta = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{target} no se puede leer: {exc}") from exc
    if not isinstance(meta, dict) or meta.get("format_version") != PANEL_FORMAT_VERSION:
        raise ValueError(f"{target}: version de formato distinta de {PANEL_FORMAT_VERSION}")
    if meta.get("features_version") != FEATURES_VERSION:
        raise ValueError(
            f"{target}: version del calculo de features distinta de {FEATURES_VERSION}"
        )
    return meta


def save_panel(
    panel: Panel,
    start_ms: int,
    end_ms: int,
    *,
    params: dict[str, Any],
    fingerprint: dict[str, list[list[Any]]],
    cache_dir: str | Path = PANEL_DIR,
) -> Path:
    """Guarda el panel de `[start_ms, end_ms)` en su directorio y escribe `meta.json` el ultimo."""
    grid = build_time_grid(start_ms, end_ms, panel.interval)
    if not panel.index.equals(grid):
        raise ValueError("el indice del panel no coincide con la rejilla de [start_ms, end_ms)")
    if set(fingerprint) != set(panel.symbols):
        raise ValueError(
            f"la huella no coincide con los simbolos del panel: {sorted(fingerprint)} "
            f"frente a {sorted(panel.symbols)}"
        )
    # El meta se serializa ANTES de escribir nada: un fallo aqui no deja Parquets sin meta.json.
    meta_text = json.dumps(
        {
            "format_version": PANEL_FORMAT_VERSION,
            "features_version": FEATURES_VERSION,
            "interval": panel.interval,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "symbols": list(panel.symbols),
            "features": list(panel.features),
            "params": dict(params),
            "created_at": datetime.now(UTC).isoformat(),
            "fingerprint": fingerprint,
        },
        indent=2,
        default=_native,
    )
    directory = panel_path(panel.interval, start_ms, end_ms, cache_dir=cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    # Sin meta.json la carpeta no es cache: se borra antes de tocar nada, por si hay una anterior.
    (directory / META_FILE).unlink(missing_ok=True)
    for name, frame in panel.features.items():
        write_parquet(frame, directory / f"{name}.parquet")
    write_parquet(panel.valid, directory / f"{VALID_FILE}.parquet")
    _write_meta(directory, meta_text)
    logger.info("Panel guardado en %s", directory)
    return directory


def load_panel(
    path: str | Path,
    *,
    features: Sequence[str] | None = None,
    symbols: Sequence[str] | None = None,
) -> Panel:
    """Carga un panel guardado, leyendo solo las features y los simbolos pedidos (y `valid`)."""
    directory = resolve_repo_path(path)
    meta = _read_meta(directory)
    stored_features: list[str] = list(meta["features"])
    stored_symbols: list[str] = list(meta["symbols"])
    names = stored_features if features is None else list(features)
    selected = stored_symbols if symbols is None else list(symbols)
    if not names or not selected:
        raise ValueError("features y symbols no pueden ser vacios")
    for label, wanted, available in (
        ("features", names, stored_features),
        ("symbols", selected, stored_symbols),
    ):
        unknown = [item for item in wanted if item not in available]
        if unknown:
            raise ValueError(f"{label} desconocidos: {unknown}; hay {available}")
        if len(set(wanted)) != len(wanted):
            raise ValueError(f"{label} repetidos: {wanted}")
    missing = [
        f"{stem}.parquet"
        for stem in (*names, VALID_FILE)
        if not (directory / f"{stem}.parquet").is_file()
    ]
    if missing:
        raise ValueError(f"{directory} esta incompleta: faltan {missing}")

    def read(stem: str) -> pd.DataFrame:
        # Parquet lee solo esas columnas; se reordenan por si el orden de lectura difiriese.
        return read_parquet(directory / f"{stem}.parquet", columns=selected)[selected]

    try:
        frames = {name: read(name) for name in names}
        valid = read(VALID_FILE)
    except (KeyError, OSError, pa.ArrowException) as exc:
        # Para que `cached_panel` (que captura ValueError) reconstruya en vez de dar traceback.
        raise ValueError(f"{directory} esta incompleta o corrupta: {exc!r}") from exc
    grid = build_time_grid(int(meta["start_ms"]), int(meta["end_ms"]), meta["interval"])
    if not valid.index.equals(grid):
        raise ValueError(f"{directory}: el indice guardado no coincide con la rejilla del meta")
    panel = Panel(features=frames, valid=valid, interval=meta["interval"], symbols=tuple(selected))
    # `ValueError` si el significado no cumple el contrato: `cached_panel` reconstruye.
    validate_panel(panel)
    return panel


def _stale_reason(meta: dict[str, Any], expected: dict[str, Any]) -> str | None:
    """Primer campo del `meta.json` que no coincide con lo pedido, o None si la cache vale."""
    for key, value in expected.items():
        if meta.get(key) != value:
            return key
    return None


def cached_panel(
    symbols: Sequence[str],
    interval: str,
    start_ms: int,
    end_ms: int,
    *,
    base_dir: str | Path = DEFAULT_BASE_DIR,
    cache_dir: str | Path = PANEL_DIR,
    rebuild: bool = False,
    vol_window: int = 60,
    vol_min_valid_share: float = 0.8,
    activity_window: int = 10080,
    activity_min_valid_share: float = 0.5,
) -> Panel:
    """Carga el panel de la cache si vale para lo pedido; si no, lo construye y lo guarda."""
    names = tuple(dict.fromkeys(symbols))
    params = {
        "vol_window": vol_window,
        "vol_min_valid_share": vol_min_valid_share,
        "activity_window": activity_window,
        "activity_min_valid_share": activity_min_valid_share,
    }
    # La huella cubre TODA la cache de velas (no solo el rango): `classify_klines` usa la vida
    # completa de la serie, asi que una descarga fuera del rango tambien puede cambiar el panel.
    fingerprint = cache_fingerprint(names, interval, base_dir=base_dir)
    directory = panel_path(interval, start_ms, end_ms, cache_dir=cache_dir)
    if not rebuild and (directory / META_FILE).is_file():
        try:
            meta = _read_meta(directory)
            stale = _stale_reason(
                meta,
                {
                    "interval": interval,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "symbols": list(names),
                    "features": list(FEATURE_NAMES),
                    "params": params,
                    "fingerprint": fingerprint,
                },
            )
            if stale is None:
                logger.info("Panel cargado de la cache: %s", directory)
                return load_panel(directory)
            logger.info("Cache del panel obsoleta (%s distinto): se reconstruye", stale)
        except ValueError as exc:
            logger.warning("Cache del panel no utilizable (%s): se reconstruye", exc)
    panel = build_panel(names, interval, start_ms, end_ms, base_dir=base_dir, **params)
    save_panel(panel, start_ms, end_ms, params=params, fingerprint=fingerprint, cache_dir=cache_dir)
    return panel


def cached_period_panels(
    universe: pd.DataFrame,
    interval: str,
    *,
    base_dir: str | Path = DEFAULT_BASE_DIR,
    cache_dir: str | Path = PANEL_DIR,
    rebuild: bool = False,
    vol_window: int = 60,
    vol_min_valid_share: float = 0.8,
    activity_window: int = 10080,
    activity_min_valid_share: float = 0.5,
) -> Iterator[tuple[pd.Timestamp, Panel]]:
    """Como `period_panels`, pero cada panel pasa por la cache; de uno en uno y en orden."""
    missing = [c for c in ("period_start", "period_end", "symbol") if c not in universe.columns]
    if missing:
        raise ValueError(f"el universo no tiene las columnas {missing}")
    ordered = universe.sort_values("period_start", kind="stable")
    for period_start, group in ordered.groupby("period_start", sort=True):
        if group["period_end"].nunique() != 1:
            raise ValueError(f"el periodo {period_start} tiene varios period_end distintos")
        period_end = pd.Timestamp(group["period_end"].iloc[0])
        if pd.Timestamp(period_start).tzinfo is None or period_end.tzinfo is None:
            raise ValueError(
                f"el periodo {period_start} debe tener period_start y period_end en UTC"
            )
        # `period_end` es exclusivo; `to_ms` evita `Timestamp.value`, que en pandas 3 es ns.
        yield (
            pd.Timestamp(period_start),
            cached_panel(
                list(dict.fromkeys(group["symbol"])),
                interval,
                to_ms(pd.Timestamp(period_start)),
                to_ms(period_end),
                base_dir=base_dir,
                cache_dir=cache_dir,
                rebuild=rebuild,
                vol_window=vol_window,
                vol_min_valid_share=vol_min_valid_share,
                activity_window=activity_window,
                activity_min_valid_share=activity_min_valid_share,
            ),
        )
