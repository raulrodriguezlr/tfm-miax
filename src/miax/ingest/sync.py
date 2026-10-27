"""Completa la cache de klines con lo que necesitan los universos: un tramo continuo por par."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from miax.ingest.batch import DEFAULT_MAX_CONSECUTIVE_FAILURES, run_symbol_batch
from miax.ingest.http import HttpClient
from miax.ingest.klines import DAY_MS, DEFAULT_BASE_URL
from miax.ingest.resume import fill_range
from miax.ingest.store import DEFAULT_BASE_DIR
from miax.ingest.universe import to_ms
from miax.utils.logging import get_logger

logger = get_logger(__name__)

RANGE_COLUMNS = ["symbol", "start_ms", "end_ms"]


def universe_download_ranges(
    universes: Sequence[pd.DataFrame], *, warmup_days: int
) -> pd.DataFrame:
    """Da por simbolo `[start_ms, end_ms]`: del primer `period_start` menos el calentamiento.

    Termina en el ultimo `period_end` menos 1 ms (es exclusivo); va ordenado por simbolo.
    """
    if warmup_days < 0:
        raise ValueError(f"warmup_days no puede ser negativo, recibido {warmup_days}")
    frames = [u[["symbol", "period_start", "period_end"]] for u in universes if not u.empty]
    if not frames:
        return pd.DataFrame({"symbol": [], "start_ms": [], "end_ms": []}).astype(
            {"symbol": "object", "start_ms": "int64", "end_ms": "int64"}
        )
    joined = pd.concat(frames, ignore_index=True)
    bounds = joined.groupby("symbol", sort=True).agg(
        first_start=("period_start", "min"), last_end=("period_end", "max")
    )
    rows = [
        {
            "symbol": str(symbol),
            # El calentamiento solo ensancha el tramo hacia el pasado: no mira al futuro.
            "start_ms": to_ms(row.first_start) - warmup_days * DAY_MS,
            # period_end es exclusivo: la ultima vela del periodo abre antes de el.
            "end_ms": to_ms(row.last_end) - 1,
        }
        for symbol, row in bounds.iterrows()
    ]
    return pd.DataFrame(rows, columns=RANGE_COLUMNS).astype(
        {"start_ms": "int64", "end_ms": "int64"}
    )


def sync_universe(
    ranges: pd.DataFrame,
    interval: str,
    *,
    client: HttpClient,
    base_dir: str = DEFAULT_BASE_DIR,
    base_url: str = DEFAULT_BASE_URL,
    end_cap_ms: int | None = None,
    max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES,
) -> dict[str, int | None]:
    """Completa la cache de cada simbolo de `ranges`; None si falla, NOT_PROCESSED si se corta."""
    spans = {
        str(row.symbol): (int(row.start_ms), int(row.end_ms))
        for row in ranges.itertuples(index=False)
    }

    def fetch_one(symbol: str) -> int:
        start_ms, end_ms = spans[symbol]
        if end_cap_ms is not None:
            # Solo velas ya cerradas: el tope recorta el fin, nunca amplia el rango.
            end_ms = min(end_ms, end_cap_ms)
        # Un unico cliente (y su limitador) para todos: el presupuesto de peso es por IP.
        return fill_range(
            symbol,
            interval,
            start_ms,
            end_ms,
            client=client,
            base_dir=base_dir,
            base_url=base_url,
        )

    return run_symbol_batch(
        list(spans),
        fetch_one,
        interval=interval,
        max_consecutive_failures=max_consecutive_failures,
        log=logger,
    )
