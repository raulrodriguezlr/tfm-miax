"""CLI de las features: `python -m miax.features panel`."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

import pandas as pd
import yaml

from miax.features.panel_store import PANEL_DIR, cached_period_panels
from miax.ingest.klines import interval_ms
from miax.ingest.store import DEFAULT_BASE_DIR
from miax.ingest.universe import DEFAULT_UNIVERSE_FILE, load_universe, to_ms
from miax.utils import get_logger
from miax.utils.config import ConfigError

logger = get_logger(__name__)

DEFAULT_INTERVAL = "1m"


def _build_parser() -> argparse.ArgumentParser:
    """Construye el parser de la CLI, con un subcomando por comando."""
    parser = argparse.ArgumentParser(prog="python -m miax.features")
    subparsers = parser.add_subparsers(dest="command", required=True)
    panel = subparsers.add_parser(
        "panel",
        description="Construye y cachea en Parquet el panel de cada periodo del universo.",
    )
    panel.add_argument(
        "--interval", default=DEFAULT_INTERVAL, help="Intervalo de las velas (por defecto 1m)."
    )
    panel.add_argument(
        "--universe-file",
        default=DEFAULT_UNIVERSE_FILE,
        help="YAML del universo (relativo a configs/).",
    )
    panel.add_argument(
        "--period",
        action="append",
        default=None,
        help="Inicio del periodo, YYYY-MM-DD en UTC; se puede repetir. Por defecto, todos.",
    )
    panel.add_argument(
        "--rebuild", action="store_true", help="Reconstruye aunque haya cache valida."
    )
    panel.add_argument("--base-dir", default=DEFAULT_BASE_DIR, help="Cache de velas.")
    panel.add_argument("--cache-dir", default=PANEL_DIR, help="Carpeta de la cache del panel.")
    return parser


def _select_periods(universe: pd.DataFrame, periods: list[str]) -> pd.DataFrame:
    """Filas del universo con `period_start` en `periods`; `ValueError` si alguno no existe."""
    wanted = {to_ms(pd.Timestamp(text, tz="UTC")) for text in periods}
    starts = universe["period_start"].map(lambda moment: to_ms(pd.Timestamp(moment)))
    unknown = wanted - set(starts)
    if unknown:
        known = sorted(pd.Timestamp(ms, unit="ms").strftime("%Y-%m-%d") for ms in set(starts))
        missing = sorted(pd.Timestamp(ms, unit="ms").strftime("%Y-%m-%d") for ms in unknown)
        raise ValueError(f"periodos que no estan en el universo: {missing}; hay {known}")
    return universe[starts.isin(wanted)]


def _run_panel(args: argparse.Namespace) -> int:
    """Subcomando `panel`: 2 si los argumentos no valen, 1 si falla la construccion."""
    try:
        universe = load_universe(args.universe_file)
        if args.period:
            universe = _select_periods(universe, args.period)
    except (ConfigError, yaml.YAMLError) as exc:
        logger.error("Universo no valido: %s", exc)
        return 2
    except ValueError as exc:
        logger.error("Periodo no valido: %s", exc)
        return 2
    if universe.empty:
        logger.error("El universo no tiene ningun periodo con pares")
        return 2
    if interval_ms(args.interval) is None or args.interval != DEFAULT_INTERVAL:
        logger.error(
            "Intervalo no valido: %r; build_panel solo admite %s", args.interval, DEFAULT_INTERVAL
        )
        return 2
    try:
        for period_start, panel in cached_period_panels(
            universe,
            args.interval,
            base_dir=args.base_dir,
            cache_dir=args.cache_dir,
            rebuild=args.rebuild,
        ):
            logger.info("Periodo %s: panel %s", period_start.date(), panel.shape)
    except (ValueError, RuntimeError, OSError) as exc:
        logger.error("Fallo la construccion del panel: %s", exc)
        return 1
    return 0


# Registro de subcomandos: uno nuevo es una entrada mas (y su parser en `_build_parser`).
_COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {"panel": _run_panel}


def main(argv: list[str] | None = None) -> int:
    """Ejecuta la CLI y devuelve el codigo de salida: 0 bien, 1 fallo del trabajo, 2 argumentos."""
    args = _build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    return _COMMANDS[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
