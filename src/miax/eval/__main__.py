"""CLI de la evaluacion: subcomandos `beta-study`, `idio-signal`, `loo-factor` y `market-lag`."""

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
from miax.eval.idiosyncratic_signal import study_signal, write_signal_tables
from miax.eval.loo_factor import DEFAULT_TESTS, compare_loo, write_loo_tables
from miax.eval.market_lag import compare_market_lag, write_market_lag_tables
from miax.features.resample import bar_factor
from miax.utils import get_logger
from miax.utils.config import ConfigError

logger = get_logger(__name__)

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
    signal = subparsers.add_parser(
        "idio-signal",
        description="Contraste de senal idiosincratica: queda algo que modelar tras el factor.",
    )
    signal.add_argument(
        "--tests",
        nargs="+",
        default=list(DEFAULT_TESTS),
        help="Inicios de trimestre de prueba, YYYY-MM-DD en UTC (por defecto 2023T1 a 2025T4).",
    )
    signal.add_argument(
        "--horizons",
        nargs="+",
        default=list(DEFAULT_HORIZONS),
        help="Horizontes de los retornos (por defecto 1m 5m 15m); solo 1m decide.",
    )
    signal.add_argument("--max-lag", type=int, default=5, help="Retardos 1..L en barras.")
    signal.add_argument("--n-shifts", type=int, default=99, help="Desplazamientos del nulo.")
    signal.add_argument(
        "--rebuild", action="store_true", help="Reconstruye los paneles de la cache."
    )
    loo = subparsers.add_parser(
        "loo-factor",
        description="Factor media y PCA sin la propia moneda (leave-one-out) frente a D-35 y D-36.",
    )
    loo.add_argument(
        "--tests",
        nargs="+",
        default=list(DEFAULT_TESTS),
        help="Inicios de trimestre de prueba, YYYY-MM-DD en UTC (por defecto 2023T1 a 2025T4).",
    )
    loo.add_argument("--rebuild", action="store_true", help="Reconstruye los paneles de la cache.")
    lag = subparsers.add_parser(
        "market-lag",
        description="Residuo con y sin el retraso respecto al mercado, y si sigue la senal.",
    )
    lag.add_argument(
        "--tests",
        nargs="+",
        default=list(DEFAULT_TESTS),
        help="Inicios de trimestre de prueba, YYYY-MM-DD en UTC (por defecto 2023T1 a 2025T4).",
    )
    lag.add_argument("--lags", type=int, default=5, help="Retardos 1..L en minutos.")
    lag.add_argument(
        "--skip-signal", action="store_true", help="Solo el diagnostico, sin el contraste de senal."
    )
    lag.add_argument("--rebuild", action="store_true", help="Reconstruye los paneles de la cache.")
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


def _run_idio_signal(args: argparse.Namespace) -> int:
    """Subcomando `idio-signal`: 2 si los argumentos no valen, 1 si falla el trabajo."""
    try:
        starts = [pd.Timestamp(text, tz="UTC") for text in args.tests]
        for start in starts:
            fold_windows(start)
    except ValueError as exc:
        logger.error(
            "Trimestre no valido (%s); se espera el inicio de un trimestre YYYY-MM-DD", exc
        )
        return 2
    if not args.tests or args.max_lag < 1 or args.n_shifts < 1:
        logger.error("Hacen falta trimestres, --max-lag >= 1 y --n-shifts >= 1")
        return 2
    try:
        for horizon in args.horizons:
            bar_factor("1m", horizon)
    except ValueError as exc:
        logger.error("Horizonte no valido: %s", exc)
        return 2
    try:
        result = study_signal(
            starts,
            factors=FACTOR_MAKERS,
            horizons=args.horizons,
            max_lag=args.max_lag,
            n_shifts=args.n_shifts,
            rebuild=args.rebuild,
        )
        write_signal_tables(result)
    except (ValueError, RuntimeError, OSError, ConfigError) as exc:
        logger.error("Fallo el estudio de senal idiosincratica: %s", exc)
        return 1
    verdict = result.decision.iloc[-1]["verdict"]
    logger.info("Veredicto de senal idiosincratica: %s", verdict)
    return 0


def _run_loo_factor(args: argparse.Namespace) -> int:
    """Subcomando `loo-factor`: 2 si los argumentos no valen, 1 si falla el trabajo."""
    try:
        starts = [pd.Timestamp(text, tz="UTC") for text in args.tests]
        for start in starts:
            fold_windows(start)
    except ValueError as exc:
        logger.error(
            "Trimestre no valido (%s); se espera el inicio de un trimestre YYYY-MM-DD", exc
        )
        return 2
    try:
        result = compare_loo(args.tests, rebuild=args.rebuild)
        write_loo_tables(result)
    except (ValueError, RuntimeError, OSError, ConfigError) as exc:
        logger.error("Fallo el estudio del factor leave-one-out: %s", exc)
        return 1
    logger.info("Resumen leave-one-out (mediana entre folds):\n%s", result.summary.to_string())
    return 0


def _run_market_lag(args: argparse.Namespace) -> int:
    """Subcomando `market-lag`: 2 si los argumentos no valen, 1 si falla el trabajo."""
    try:
        starts = [pd.Timestamp(text, tz="UTC") for text in args.tests]
        for start in starts:
            fold_windows(start)
    except ValueError as exc:
        logger.error(
            "Trimestre no valido (%s); se espera el inicio de un trimestre YYYY-MM-DD", exc
        )
        return 2
    if not args.tests or args.lags < 1:
        logger.error("Hacen falta trimestres y --lags >= 1")
        return 2
    try:
        result = compare_market_lag(
            args.tests,
            lags=args.lags,
            run_signal=not args.skip_signal,
            rebuild=args.rebuild,
        )
        write_market_lag_tables(result)
    except (ValueError, RuntimeError, OSError, ConfigError) as exc:
        logger.error("Fallo el estudio del retraso respecto al mercado: %s", exc)
        return 1
    summary = result.diagnostics.groupby("model", sort=False)["abs_corr_lag1"].median()
    logger.info("Mediana entre folds de abs_corr_lag1 por modelo:\n%s", summary.to_string())
    if result.signal_decision is not None:
        verdicts = result.signal_decision[
            result.signal_decision["factor"].str.startswith("verdict")
        ]
        logger.info("Decision de senal:\n%s", verdicts.to_string(index=False))
    return 0


# Registro de subcomandos: uno nuevo es una entrada mas (y su parser en `_build_parser`).
_COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "beta-study": _run_beta_study,
    "idio-signal": _run_idio_signal,
    "loo-factor": _run_loo_factor,
    "market-lag": _run_market_lag,
}


def main(argv: list[str] | None = None) -> int:
    """Ejecuta la CLI y devuelve el codigo de salida: 0 bien, 1 fallo del trabajo, 2 argumentos."""
    try:
        args = _build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    except SystemExit as exc:  # argparse sale con 2 ante argumentos invalidos
        return int(exc.code) if isinstance(exc.code, int) else 2
    return _COMMANDS[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
