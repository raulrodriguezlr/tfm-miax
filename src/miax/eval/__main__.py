"""CLI de la evaluacion: `python -m miax.eval beta-study`."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

import pandas as pd

from miax.eval.beta_study import (
    FACTOR_MAKERS,
    compare_schemes,
    fold_windows,
    write_tables,
)
from miax.features.resample import bar_factor
from miax.utils import get_logger
from miax.utils.config import ConfigError

logger = get_logger(__name__)

# Los 12 trimestres de prueba de D-37 (2023T1 a 2025T4).
DEFAULT_TESTS = tuple(
    f"{year}-{month:02d}-01" for year in (2023, 2024, 2025) for month in (1, 4, 7, 10)
)
DEFAULT_WINDOWS = (30, 90)
DEFAULT_HORIZONS = ("1m", "5m", "15m")


def _build_parser() -> argparse.ArgumentParser:
    """Construye el parser de la CLI, con un subcomando por estudio."""
    parser = argparse.ArgumentParser(prog="python -m miax.eval")
    subparsers = parser.add_subparsers(dest="command", required=True)
    study = subparsers.add_parser(
        "beta-study",
        description="Betas moviles frente a betas por fold, y betas a 1 min frente a 5 y 15 min.",
    )
    study.add_argument(
        "--tests",
        nargs="+",
        default=list(DEFAULT_TESTS),
        help=(
            "Inicios de trimestre de prueba, YYYY-MM-DD en UTC (por defecto 2023T1 a 2025T4). "
            "La regla de decision exige min_wins=9 folds ganados, valor absoluto que no se "
            "reescala: con menos de 12 folds (con 4 nunca pasa) es mas conservadora."
        ),
    )
    study.add_argument(
        "--windows",
        nargs="+",
        type=int,
        default=list(DEFAULT_WINDOWS),
        help="Ventanas de la beta movil, en dias (por defecto 30 90).",
    )
    study.add_argument(
        "--horizons",
        nargs="+",
        default=list(DEFAULT_HORIZONS),
        help=(
            "Horizontes de ajuste de la beta (por defecto 1m 5m 15m); 1m equivale al esquema "
            "fold, que ya es la referencia."
        ),
    )
    study.add_argument(
        "--rebuild", action="store_true", help="Reconstruye los paneles de la cache."
    )
    return parser


def _run_beta_study(args: argparse.Namespace) -> int:
    """Subcomando `beta-study`: 2 si los argumentos no valen, 1 si falla el trabajo."""
    try:
        starts = [pd.Timestamp(text, tz="UTC") for text in args.tests]
        for start in starts:
            fold_windows(start)
    except ValueError as exc:
        logger.error(
            "Trimestre no valido (%s); se espera el inicio de un trimestre YYYY-MM-DD", exc
        )
        return 2
    if not args.tests or any(w < 1 for w in args.windows):
        logger.error("Las ventanas deben ser enteros >= 1 dia")
        return 2
    try:
        for horizon in args.horizons:
            bar_factor("1m", horizon)
    except ValueError as exc:
        logger.error("Horizonte no valido: %s", exc)
        return 2
    try:
        result = compare_schemes(
            starts,
            factors=FACTOR_MAKERS,
            windows=args.windows,
            horizons=args.horizons,
            rebuild=args.rebuild,
        )
        write_tables(result)
    except (ValueError, RuntimeError, OSError, ConfigError) as exc:
        logger.error("Fallo el estudio de betas: %s", exc)
        return 1
    return 0


# Registro de subcomandos: uno nuevo es una entrada mas (y su parser en `_build_parser`).
_COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {"beta-study": _run_beta_study}


def main(argv: list[str] | None = None) -> int:
    """Ejecuta la CLI y devuelve el codigo de salida: 0 bien, 1 fallo del trabajo, 2 argumentos."""
    try:
        args = _build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    except SystemExit as exc:  # argparse sale con 2 ante argumentos invalidos
        return int(exc.code) if isinstance(exc.code, int) else 2
    return _COMMANDS[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
