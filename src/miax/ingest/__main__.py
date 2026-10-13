"""CLI de la ingesta: `python -m miax.ingest [download|validate|coverage|pairs|universe]`."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import pandas as pd
import requests
import yaml

from miax.ingest.coverage import coverage_report, universe_coverage
from miax.ingest.exchange_info import list_pairs
from miax.ingest.http import HttpClient, HttpError, RetriesExhaustedError
from miax.ingest.integrity import ERROR, count_month_files, validate_klines
from miax.ingest.klines import DAY_MS, DEFAULT_BASE_URL, candle_offset_ms, interval_ms
from miax.ingest.ratelimit import RateLimiter
from miax.ingest.resume import update_symbol
from miax.ingest.store import DEFAULT_BASE_DIR
from miax.ingest.universe import (
    DEFAULT_UNIVERSE_FILE,
    DEFAULT_UNIVERSE_PARAMS_FILE,
    build_universe,
    load_categories,
    load_universe,
    load_universe_params,
    write_universe,
)
from miax.utils import get_logger
from miax.utils.config import ConfigError
from miax.utils.io import write_parquet

logger = get_logger(__name__)

DEFAULT_INTERVAL = "1m"
DEFAULT_LOOKBACK_DAYS = 30
DEFAULT_MAX_CONSECUTIVE_FAILURES = 3
# Marca de un simbolo no intentado porque el cortacircuitos corto la tanda (distinta de None).
NOT_PROCESSED = -1
_MAX_UNCACHED_EXAMPLES = 10
_DATE_FORMAT = "%Y-%m-%d"
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_MS = timedelta(milliseconds=1)


def _utc_now() -> datetime:
    """Devuelve el instante actual en UTC (punto de parcheo para los tests)."""
    return datetime.now(UTC)


def _to_ms(moment: datetime) -> int:
    """Convierte un datetime con zona horaria a milisegundos UTC desde la epoca, sin redondeos."""
    return (moment - _EPOCH) // _MS


def _last_closed_ms(interval: str, now_ms: int) -> int | None:
    """Ultimo ms de la ultima vela cerrada de `interval`, o None si no tiene duracion fija."""
    step = interval_ms(interval)
    if step is None:
        return None
    offset = candle_offset_ms(interval)
    # Inicio de la vela en curso menos 1 ms: la vela en curso (sin cerrar) queda fuera.
    return (now_ms - offset) // step * step + offset - 1


def _day_start_ms(text: str) -> int:
    """Convierte `YYYY-MM-DD` al inicio de ese dia UTC en ms; lanza ValueError si no parsea."""
    return _to_ms(datetime.strptime(text, _DATE_FORMAT).replace(tzinfo=UTC))


def _add_download_parser(subparsers: argparse._SubParsersAction) -> None:
    """Anade el subcomando `download`."""
    parser = subparsers.add_parser(
        "download",
        description=(
            "Descarga klines de Binance y los cachea en Parquet, reanudando lo ya bajado. "
            "Otros subcomandos: validate (integridad de la cache), coverage (cobertura temporal), "
            "pairs (catalogo de pares), universe (congela el universo por periodos)."
        ),
    )
    parser.add_argument(
        "--symbol",
        required=True,
        help="Simbolo de Binance, o varios separados por comas: BTCUSDT,ETHUSDT,SOLUSDT.",
    )
    parser.add_argument(
        "--interval", default=DEFAULT_INTERVAL, help="Intervalo de las velas (por defecto 1m)."
    )
    parser.add_argument(
        "--start",
        default=None,
        help=(
            "Inicio del rango, YYYY-MM-DD en UTC (00:00:00 de ese dia). "
            f"Por defecto, hace {DEFAULT_LOOKBACK_DAYS} dias."
        ),
    )
    parser.add_argument(
        "--end",
        default=None,
        help=(
            "Fin del rango, YYYY-MM-DD en UTC, inclusivo (hasta las 23:59:59.999 de ese dia). "
            "Por defecto, ahora. Se recorta a la ultima vela cerrada: la vela en curso no se baja."
        ),
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="URL base de la API.")
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR, help="Carpeta de la cache.")


def _add_validate_parser(subparsers: argparse._SubParsersAction) -> None:
    """Anade el subcomando `validate`."""
    parser = subparsers.add_parser(
        "validate", description="Revisa la integridad de la cache de klines (solo lectura)."
    )
    parser.add_argument(
        "--symbol", required=True, help="Simbolo de Binance, o varios separados por comas."
    )
    parser.add_argument(
        "--interval", default=DEFAULT_INTERVAL, help="Intervalo de las velas (por defecto 1m)."
    )
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR, help="Carpeta de la cache.")
    parser.add_argument(
        "--output", default=None, help="Ruta de un Parquet donde guardar las incidencias."
    )


def _add_coverage_parser(subparsers: argparse._SubParsersAction) -> None:
    """Anade el subcomando `coverage`."""
    parser = subparsers.add_parser(
        "coverage",
        description=(
            "Informe de cobertura temporal de la cache de klines: fechas, % de velas presentes "
            "y % de minutos con operaciones (solo lectura)."
        ),
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--symbol", default=None, help="Simbolo de Binance, o varios separados por comas."
    )
    target.add_argument(
        "--universe",
        action="store_true",
        help="Mide cada par en cada periodo del universo congelado.",
    )
    parser.add_argument(
        "--universe-file",
        default=DEFAULT_UNIVERSE_FILE,
        help="YAML del universo (relativo a configs/); solo con --universe.",
    )
    parser.add_argument(
        "--interval", default=DEFAULT_INTERVAL, help="Intervalo de las velas (por defecto 1m)."
    )
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR, help="Carpeta de la cache.")
    parser.add_argument(
        "--start",
        default=None,
        help="Inicio del rango, YYYY-MM-DD en UTC; solo con --symbol y junto a --end.",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Fin del rango, YYYY-MM-DD en UTC, inclusivo; solo con --symbol y junto a --start.",
    )
    parser.add_argument(
        "--output", default=None, help="Ruta de un Parquet donde guardar el informe."
    )


def _add_pairs_parser(subparsers: argparse._SubParsersAction) -> None:
    """Anade el subcomando `pairs`."""
    parser = subparsers.add_parser("pairs", description="Lista el catalogo de pares de Binance.")
    parser.add_argument(
        "--quote", default=None, help="Filtra por activo cotizado (por defecto, todos)."
    )
    parser.add_argument(
        "--status", default=None, help="Filtra por estado del par (por defecto, todos)."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="URL base de la API.")
    parser.add_argument(
        "--output", default=None, help="Ruta de un Parquet donde guardar los pares."
    )


def _add_universe_parser(subparsers: argparse._SubParsersAction) -> None:
    """Anade el subcomando `universe`."""
    parser = subparsers.add_parser(
        "universe",
        description=(
            "Construye el universo por periodos (hace llamadas a Binance) y lo congela en un "
            "YAML versionado."
        ),
    )
    parser.add_argument(
        "--params",
        default=DEFAULT_UNIVERSE_PARAMS_FILE,
        help="Parametros del universo (relativo a configs/).",
    )
    parser.add_argument(
        "--out", default=DEFAULT_UNIVERSE_FILE, help="YAML de salida (relativo a configs/)."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="URL base de la API.")


def _build_parser() -> argparse.ArgumentParser:
    """Construye el parser de argumentos de la CLI, con un subcomando por comando."""
    parser = argparse.ArgumentParser(prog="python -m miax.ingest")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_download_parser(subparsers)
    _add_validate_parser(subparsers)
    _add_coverage_parser(subparsers)
    _add_pairs_parser(subparsers)
    _add_universe_parser(subparsers)
    return parser


def _parse_symbols(text: str) -> list[str]:
    """Separa simbolos por comas, en mayusculas, sin espacios ni vacios ni repetidos, en orden."""
    return list(dict.fromkeys(item.strip().upper() for item in text.split(",") if item.strip()))


def download_symbols(
    symbols: list[str],
    interval: str,
    start_ms: int,
    end_ms: int,
    *,
    client: HttpClient,
    base_dir: str = DEFAULT_BASE_DIR,
    base_url: str = DEFAULT_BASE_URL,
    max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES,
) -> dict[str, int | None]:
    """Baja los simbolos en orden con un mismo cliente; None si falla, NOT_PROCESSED si se corta."""
    results: dict[str, int | None] = {}
    total = len(symbols)
    systemic_failures = 0
    for position, symbol in enumerate(symbols, start=1):
        if systemic_failures >= max_consecutive_failures:
            # Cortacircuitos: tras N fallos sistemicos seguidos (429/418 o red caida) seguir
            # golpeando la API solo alargaria un posible baneo de la IP.
            logger.error(
                "Tanda cortada: %d fallos sistemicos consecutivos (reintentos agotados); "
                "quedan sin procesar %d simbolos",
                systemic_failures,
                total - position + 1,
            )
            for pending in symbols[position - 1 :]:
                results[pending] = NOT_PROCESSED
            break
        logger.info("[%d/%d] %s: descargando %s", position, total, symbol, interval)
        # Un unico cliente (y su limitador) para todos: el presupuesto de peso es por IP.
        try:
            new = update_symbol(
                symbol,
                interval,
                start_ms,
                end_ms,
                client=client,
                base_dir=base_dir,
                base_url=base_url,
            )
        except RetriesExhaustedError as exc:
            # Fallo sistemico: cuenta para el cortacircuitos (solo si son consecutivos).
            systemic_failures += 1
            logger.error(
                "[%d/%d] Fallo la descarga de %s %s: %s", position, total, symbol, interval, exc
            )
            results[symbol] = None
            continue
        except (HttpError, ValueError, OSError, requests.exceptions.RequestException) as exc:
            # Fallo propio del simbolo: no aborta la tanda y resetea el contador sistemico.
            systemic_failures = 0
            logger.error(
                "[%d/%d] Fallo la descarga de %s %s: %s", position, total, symbol, interval, exc
            )
            results[symbol] = None
            continue
        systemic_failures = 0
        logger.info("[%d/%d] %s: %d velas nuevas", position, total, symbol, new)
        results[symbol] = new

    failed = [symbol for symbol, new in results.items() if new is None]
    skipped = [symbol for symbol, new in results.items() if new == NOT_PROCESSED]
    logger.info(
        "Resumen: %d simbolos OK, %d fallidos%s%s",
        total - len(failed) - len(skipped),
        len(failed),
        f" ({', '.join(failed)})" if failed else "",
        f", {len(skipped)} sin procesar ({', '.join(skipped)})" if skipped else "",
    )
    return results


def _run_download(args: argparse.Namespace) -> int:
    """Subcomando `download`: baja y cachea klines; devuelve el codigo de salida."""
    symbols = _parse_symbols(args.symbol)

    now = _utc_now()
    try:
        if args.start is None:
            # El rango por defecto mira hacia atras desde ahora: no pide nada posterior a "ahora".
            start_ms = _to_ms(now - timedelta(days=DEFAULT_LOOKBACK_DAYS))
        else:
            start_ms = _day_start_ms(args.start)
        if args.end is None:
            end_ms = _to_ms(now)
        else:
            # Fin inclusivo del dia: inicio del dia siguiente menos 1 ms.
            end_ms = _day_start_ms(args.end) + DAY_MS - 1
    except ValueError as exc:
        logger.error("Fecha invalida, se espera YYYY-MM-DD: %s", exc)
        return 2

    # Solo se piden velas ya cerradas (close_time <= ahora): una vela en curso quedaria cacheada
    # con OHLCV parcial y la reanudacion (ultima + 1) nunca la refrescaria.
    last_closed_ms = _last_closed_ms(args.interval, _to_ms(now))
    if last_closed_ms is None:
        logger.warning(
            "No se recorta la vela en curso para el intervalo %s: duracion no fija", args.interval
        )
    else:
        end_ms = min(end_ms, last_closed_ms)

    if start_ms > end_ms:
        logger.error(
            "El inicio (%d) es posterior al fin (%d, ultima vela cerrada si es anterior): "
            "rango vacio",
            start_ms,
            end_ms,
        )
        return 2

    if not symbols:
        logger.error("--symbol no contiene ningun simbolo")
        return 2

    logger.info(
        "Rango %s en [%d, %d] ms UTC, %d simbolos", args.interval, start_ms, end_ms, len(symbols)
    )
    # Un solo cliente con su limitador para toda la tanda; se cierra al terminar.
    with HttpClient(rate_limiter=RateLimiter()) as client:
        results = download_symbols(
            symbols,
            args.interval,
            start_ms,
            end_ms,
            client=client,
            base_dir=args.base_dir,
            base_url=args.base_url,
        )
    ok = all(new is not None and new != NOT_PROCESSED for new in results.values())
    return 0 if ok else 1


def _run_validate(args: argparse.Namespace) -> int:
    """Subcomando `validate`: revisa la cache; devuelve 1 si hay algun error, 0 si no."""
    symbols = _parse_symbols(args.symbol)
    if not symbols:
        logger.error("--symbol no contiene ningun simbolo")
        return 2
    frames = []
    uncached = 0
    try:
        for symbol in symbols:
            if count_month_files(symbol, args.interval, base_dir=args.base_dir) == 0:
                # Sin cache no hay nada que validar: se trata como problema, no como cache sana.
                logger.error("%s %s: no hay cache para validar", symbol, args.interval)
                uncached += 1
                continue
            issues = validate_klines(symbol, args.interval, base_dir=args.base_dir)
            errors = int((issues["severity"] == ERROR).sum())
            logger.info(
                "%s %s: %d errores, %d avisos", symbol, args.interval, errors, len(issues) - errors
            )
            frames.append(issues.assign(symbol=symbol))
    except ValueError as exc:
        logger.error("Simbolo o intervalo no valido: %s", exc)
        return 2
    result = pd.concat(frames, ignore_index=True) if frames else None
    if args.output is not None and result is not None:
        try:
            write_parquet(result, args.output)
        except OSError as exc:
            logger.error("No se pudo escribir %s: %s", args.output, exc)
            return 1
        logger.info("Incidencias guardadas en %s", args.output)
    has_errors = result is not None and bool((result["severity"] == ERROR).any())
    return 1 if (has_errors or uncached) else 0


def _log_symbol_coverage(report: pd.DataFrame, interval: str) -> None:
    """Loguea una linea por simbolo con su % de velas presentes y de minutos con operaciones."""
    for row in report.itertuples(index=False):
        logger.info(
            "%s %s: %.2f %% presente, %.2f %% con operaciones, %d huecos (%s a %s)",
            row.symbol,
            interval,
            row.coverage_pct,
            row.traded_pct,
            row.n_gaps,
            row.first_open,
            row.last_open,
        )


def _log_period_coverage(report: pd.DataFrame, interval: str) -> None:
    """Loguea una linea por periodo del universo con la mediana y el minimo de cobertura."""
    for period_start, group in report.groupby("period_start", sort=True):
        weakest = (
            group.loc[group["traded_pct"].idxmin()] if group["traded_pct"].notna().any() else None
        )
        logger.info(
            "Periodo %s %s: %d pares, presente mediana %.2f %% (min %.2f %%), "
            "con operaciones mediana %.2f %% (min %.2f %%%s)",
            period_start.date(),
            interval,
            len(group),
            group["coverage_pct"].median(),
            group["coverage_pct"].min(),
            group["traded_pct"].median(),
            group["traded_pct"].min(),
            "" if weakest is None else f", {weakest['symbol']}",
        )


def _run_coverage(args: argparse.Namespace) -> int:
    """Subcomando `coverage`: informe; 2 si los argumentos no valen, 1 si faltan velas."""
    # Fin exclusivo (ultimo ms cerrado + 1): la vela en curso no cuenta como esperada.
    last_closed = _last_closed_ms(args.interval, _to_ms(_utc_now()))
    if last_closed is None:
        logger.error("El intervalo %s no tiene duracion fija: no hay cobertura", args.interval)
        return 2
    closed_end_ms = last_closed + 1
    if args.universe and (args.start is not None or args.end is not None):
        logger.error("--start y --end solo se usan con --symbol")
        return 2
    if not args.universe and args.universe_file != DEFAULT_UNIVERSE_FILE:
        logger.error("--universe-file solo se usa con --universe")
        return 2

    start_ms = end_ms = None
    if not args.universe:
        symbols = _parse_symbols(args.symbol)
        if not symbols:
            logger.error("--symbol no contiene ningun simbolo")
            return 2
        if (args.start is None) != (args.end is None):
            logger.error("--start y --end se dan juntos o ninguno")
            return 2
        if args.start is not None:
            try:
                start_ms = _day_start_ms(args.start)
                # Fin exclusivo: inicio del dia siguiente al de --end, que es inclusivo.
                end_ms = _day_start_ms(args.end) + DAY_MS
            except ValueError as exc:
                logger.error("Fecha invalida, se espera YYYY-MM-DD: %s", exc)
                return 2
            # Un fin en el futuro se recorta a la ultima vela cerrada.
            end_ms = min(end_ms, closed_end_ms)
            if start_ms >= end_ms:
                logger.error("El inicio es posterior al fin (o a la ultima vela cerrada): vacio")
                return 2

    try:
        if args.universe:
            universe = load_universe(args.universe_file)
            # "Ahora" solo recorta periodos en curso; el informe describe, no selecciona pares.
            report = universe_coverage(
                universe, args.interval, base_dir=args.base_dir, now_ms=closed_end_ms
            )
            if report.empty:
                logger.error("El informe del universo no tiene filas: periodos futuros o vacio")
                return 1
            _log_period_coverage(report, args.interval)
        else:
            report = coverage_report(
                symbols, args.interval, base_dir=args.base_dir, start_ms=start_ms, end_ms=end_ms
            )
            _log_symbol_coverage(report, args.interval)
    except (ConfigError, yaml.YAMLError) as exc:
        logger.error("Universo no valido: %s", exc)
        return 2
    except ValueError as exc:
        logger.error("Simbolo, intervalo o rango no valido: %s", exc)
        return 2

    uncached = report.loc[report["present"] == 0]
    if not uncached.empty:
        if args.universe:
            examples = [
                f"{row.period_start.date()}:{row.symbol}"
                for row in uncached.head(_MAX_UNCACHED_EXAMPLES).itertuples(index=False)
            ]
            extra = len(uncached) - len(examples)
            logger.error(
                "%d pares-periodo sin velas: %s%s",
                len(uncached),
                ", ".join(examples),
                f" y {extra} mas" if extra > 0 else "",
            )
        else:
            logger.error("Sin velas en el rango: %s", ", ".join(dict.fromkeys(uncached["symbol"])))
    if args.output is not None:
        try:
            write_parquet(report, args.output)
        except OSError as exc:
            logger.error("No se pudo escribir %s: %s", args.output, exc)
            return 1
        logger.info("Informe guardado en %s", args.output)
    return 1 if not uncached.empty else 0


def _run_pairs(args: argparse.Namespace) -> int:
    """Subcomando `pairs`: lista el catalogo de pares; devuelve 1 si falla la llamada HTTP."""
    try:
        with HttpClient(rate_limiter=RateLimiter()) as client:
            pairs = list_pairs(
                quote_asset=args.quote, status=args.status, client=client, base_url=args.base_url
            )
    except (HttpError, ValueError, requests.exceptions.RequestException) as exc:
        logger.error("Fallo la consulta del catalogo de pares: %s", exc)
        return 1
    logger.info(
        "%d pares (quote=%s, status=%s); por estado: %s",
        len(pairs),
        args.quote,
        args.status,
        pairs["status"].value_counts().to_dict(),
    )
    if args.output is not None:
        write_parquet(pairs, args.output)
        logger.info("Pares guardados en %s", args.output)
    return 0


def _run_universe(args: argparse.Namespace) -> int:
    """Subcomando `universe`: congela el universo; 2 si la config no vale, 1 si falla el trabajo."""
    try:
        params = load_universe_params(args.params)
        categories = load_categories(params.categories_file)
    except (ConfigError, yaml.YAMLError) as exc:
        logger.error("Configuracion del universo no valida: %s", exc)
        return 2
    try:
        with HttpClient(rate_limiter=RateLimiter()) as client:
            # "Ahora" solo acota la consulta de primeras velas: son un hecho fijo, sin fuga.
            universes = build_universe(
                params,
                categories,
                client=client,
                now_ms=_to_ms(_utc_now()),
                base_url=args.base_url,
            )
        write_universe(universes, params, args.out)
    except (HttpError, ValueError, OSError, requests.exceptions.RequestException) as exc:
        logger.error("Fallo la construccion del universo: %s", exc)
        return 1
    return 0


# Registro de subcomandos: uno nuevo es una entrada mas (y su parser en `_build_parser`).
_COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "download": _run_download,
    "validate": _run_validate,
    "coverage": _run_coverage,
    "pairs": _run_pairs,
    "universe": _run_universe,
}


def main(argv: list[str] | None = None) -> int:
    """Ejecuta la CLI y devuelve el codigo de salida: 0 si todo va bien, distinto de 0 si no."""
    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list or args_list[0] not in _COMMANDS:
        # Compatibilidad: sin subcomando explicito se asume `download`.
        args_list.insert(0, "download")
    args = _build_parser().parse_args(args_list)
    return _COMMANDS[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
