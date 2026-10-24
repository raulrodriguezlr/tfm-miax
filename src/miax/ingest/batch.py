"""Tanda de descargas por simbolo: orden, cortacircuitos y clasificacion de errores."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

import requests

from miax.ingest.http import HttpError, RetriesExhaustedError
from miax.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_MAX_CONSECUTIVE_FAILURES = 3
# Marca de un simbolo no intentado porque el cortacircuitos corto la tanda (distinta de None).
NOT_PROCESSED = -1


def run_symbol_batch(
    symbols: Sequence[str],
    fetch_one: Callable[[str], int],
    *,
    interval: str,
    max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES,
    log: logging.Logger = logger,
) -> dict[str, int | None]:
    """Ejecuta `fetch_one` por simbolo en orden; None si falla, NOT_PROCESSED si se corta."""
    results: dict[str, int | None] = {}
    total = len(symbols)
    systemic_failures = 0
    for position, symbol in enumerate(symbols, start=1):
        if systemic_failures >= max_consecutive_failures:
            # Cortacircuitos: tras N fallos sistemicos seguidos (429/418 o red caida) seguir
            # golpeando la API solo alargaria un posible baneo de la IP.
            log.error(
                "Tanda cortada: %d fallos sistemicos consecutivos (reintentos agotados); "
                "quedan sin procesar %d simbolos",
                systemic_failures,
                total - position + 1,
            )
            for pending in symbols[position - 1 :]:
                results[pending] = NOT_PROCESSED
            break
        log.info("[%d/%d] %s: descargando %s", position, total, symbol, interval)
        try:
            new = fetch_one(symbol)
        except RetriesExhaustedError as exc:
            # Fallo sistemico: cuenta para el cortacircuitos (solo si son consecutivos).
            systemic_failures += 1
            log.error(
                "[%d/%d] Fallo la descarga de %s %s: %s", position, total, symbol, interval, exc
            )
            results[symbol] = None
            continue
        except (HttpError, ValueError, OSError, requests.exceptions.RequestException) as exc:
            # Fallo propio del simbolo: no aborta la tanda y resetea el contador sistemico.
            systemic_failures = 0
            log.error(
                "[%d/%d] Fallo la descarga de %s %s: %s", position, total, symbol, interval, exc
            )
            results[symbol] = None
            continue
        systemic_failures = 0
        log.info("[%d/%d] %s: %d velas nuevas", position, total, symbol, new)
        results[symbol] = new

    failed = [symbol for symbol, new in results.items() if new is None]
    skipped = [symbol for symbol, new in results.items() if new == NOT_PROCESSED]
    log.info(
        "Resumen: %d simbolos OK, %d fallidos%s%s",
        total - len(failed) - len(skipped),
        len(failed),
        f" ({', '.join(failed)})" if failed else "",
        f", {len(skipped)} sin procesar ({', '.join(skipped)})" if skipped else "",
    )
    return results
