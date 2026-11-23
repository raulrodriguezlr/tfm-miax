"""CLI de las figuras y tablas: `python -m miax.viz raw-correlation` y `factor-comparison`."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

import pandas as pd

from miax.ingest.klines import interval_ms
from miax.utils import get_logger
from miax.utils.config import ConfigError
from miax.viz.factor_comparison import build_factor_comparison_report
from miax.viz.raw_correlation import build_raw_correlation_report

logger = get_logger(__name__)

DEFAULT_INTERVAL = "1m"
DEFAULT_PERIOD = "2025-01-01"


def _build_parser() -> argparse.ArgumentParser:
    """Construye el parser de la CLI, con un subcomando por figura."""
    parser = argparse.ArgumentParser(prog="python -m miax.viz")
    subparsers = parser.add_subparsers(dest="command", required=True)
    raw = subparsers.add_parser(
        "raw-correlation",
        description="Figura y tablas de la correlacion cruda de un periodo del universo.",
    )
    raw.add_argument(
        "--period",
        default=DEFAULT_PERIOD,
        help="Inicio del periodo, YYYY-MM-DD en UTC (por defecto 2025-01-01).",
    )
    raw.add_argument(
        "--interval", default=DEFAULT_INTERVAL, help="Intervalo de las velas (por defecto 1m)."
    )
    raw.add_argument("--rebuild", action="store_true", help="Reconstruye el panel de la cache.")
    comparison = subparsers.add_parser(
        "factor-comparison",
        description="Tabla comparativa de los factores BTC, media y PCA antes de 2026.",
    )
    comparison.add_argument(
        "--train-months",
        type=int,
        default=2,
        help="Meses de cada periodo para ajustar el factor (por defecto 2).",
    )
    comparison.add_argument(
        "--min-symbols",
        type=int,
        default=30,
        help="Monedas minimas con datos para que un periodo entre en la tabla (por defecto 30).",
    )
    comparison.add_argument(
        "--interval", default=DEFAULT_INTERVAL, help="Intervalo de las velas (por defecto 1m)."
    )
    return parser


def _run_raw_correlation(args: argparse.Namespace) -> int:
    """Subcomando `raw-correlation`: 2 si los argumentos no valen, 1 si falla el trabajo."""
    try:
        period = pd.Timestamp(args.period, tz="UTC")
    except ValueError:
        logger.error("Periodo no valido: %r; se espera YYYY-MM-DD", args.period)
        return 2
    if interval_ms(args.interval) is None or args.interval != DEFAULT_INTERVAL:
        logger.error(
            "Intervalo no valido: %r; el panel solo admite %s", args.interval, DEFAULT_INTERVAL
        )
        return 2
    try:
        paths = build_raw_correlation_report(period, interval=args.interval, rebuild=args.rebuild)
    except (ValueError, RuntimeError, OSError, ConfigError) as exc:
        logger.error("Fallo el informe de correlacion cruda: %s", exc)
        return 1
    logger.info("Figura: %s", paths.png)
    return 0


def _run_factor_comparison(args: argparse.Namespace) -> int:
    """Subcomando `factor-comparison`: 2 si los argumentos no valen, 1 si falla el trabajo."""
    if args.train_months < 1 or args.min_symbols < 2:
        logger.error("--train-months debe ser >= 1 y --min-symbols >= 2")
        return 2
    if interval_ms(args.interval) is None or args.interval != DEFAULT_INTERVAL:
        logger.error(
            "Intervalo no valido: %r; el panel solo admite %s", args.interval, DEFAULT_INTERVAL
        )
        return 2
    try:
        paths = build_factor_comparison_report(
            interval=args.interval, train_months=args.train_months, min_symbols=args.min_symbols
        )
    except (ValueError, RuntimeError, OSError, ConfigError) as exc:
        logger.error("Fallo la comparativa de factores: %s", exc)
        return 1
    logger.info("Tablas: %s y %s", paths.by_period_csv, paths.summary_csv)
    return 0


# Registro de subcomandos: uno nuevo es una entrada mas (y su parser en `_build_parser`).
_COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "raw-correlation": _run_raw_correlation,
    "factor-comparison": _run_factor_comparison,
}


def main(argv: list[str] | None = None) -> int:
    """Ejecuta la CLI y devuelve el codigo de salida: 0 bien, 1 fallo del trabajo, 2 argumentos."""
    args = _build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    return _COMMANDS[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
