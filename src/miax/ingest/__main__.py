"""CLI de la ingesta: `python -m miax.ingest --symbol BTCUSDT` descarga y cachea klines."""

from __future__ import annotations

import argparse
import re
from datetime import UTC, datetime, timedelta

from miax.ingest.http import HttpClient, HttpError
from miax.ingest.klines import DEFAULT_BASE_URL
from miax.ingest.ratelimit import RateLimiter
from miax.ingest.resume import update_symbol
from miax.ingest.store import DEFAULT_BASE_DIR
from miax.utils import get_logger

logger = get_logger(__name__)

DEFAULT_INTERVAL = "1m"
DEFAULT_LOOKBACK_DAYS = 30
_DATE_FORMAT = "%Y-%m-%d"
_DAY_MS = 24 * 60 * 60 * 1000
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_MS = timedelta(milliseconds=1)
_INTERVAL_PATTERN = re.compile(r"^(\d+)([smhdw])$")
_UNIT_MS = {
    "s": 1000,
    "m": 60 * 1000,
    "h": 60 * 60 * 1000,
    "d": _DAY_MS,
    "w": 7 * _DAY_MS,
}
# La epoca (jueves) no cae en lunes: las velas semanales de Binance abren en lunes 00:00 UTC.
_WEEK_OFFSET_MS = 4 * _DAY_MS


def _utc_now() -> datetime:
    """Devuelve el instante actual en UTC (punto de parcheo para los tests)."""
    return datetime.now(UTC)


def _to_ms(moment: datetime) -> int:
    """Convierte un datetime con zona horaria a milisegundos UTC desde la epoca, sin redondeos."""
    return (moment - _EPOCH) // _MS


def _last_closed_ms(interval: str, now_ms: int) -> int | None:
    """Ultimo ms de la ultima vela cerrada de `interval`, o None si no tiene duracion fija."""
    match = _INTERVAL_PATTERN.match(interval)
    if match is None:
        return None
    unit = match.group(2)
    interval_ms = int(match.group(1)) * _UNIT_MS[unit]
    if interval_ms == 0:
        return None
    offset = _WEEK_OFFSET_MS if unit == "w" else 0
    # Inicio de la vela en curso menos 1 ms: la vela en curso (sin cerrar) queda fuera.
    return (now_ms - offset) // interval_ms * interval_ms + offset - 1


def _day_start_ms(text: str) -> int:
    """Convierte `YYYY-MM-DD` al inicio de ese dia UTC en ms; lanza ValueError si no parsea."""
    return _to_ms(datetime.strptime(text, _DATE_FORMAT).replace(tzinfo=UTC))


def _build_parser() -> argparse.ArgumentParser:
    """Construye el parser de argumentos de la CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m miax.ingest",
        description="Descarga klines de Binance y los cachea en Parquet, reanudando lo ya bajado.",
    )
    parser.add_argument("--symbol", required=True, help="Simbolo de Binance, p. ej. BTCUSDT.")
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
    return parser


def main(argv: list[str] | None = None) -> int:
    """Ejecuta la CLI y devuelve el codigo de salida: 0 si todo va bien, distinto de 0 si no."""
    args = _build_parser().parse_args(argv)

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
            end_ms = _day_start_ms(args.end) + _DAY_MS - 1
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

    try:
        with HttpClient(rate_limiter=RateLimiter()) as client:
            new = update_symbol(
                args.symbol,
                args.interval,
                start_ms,
                end_ms,
                client=client,
                base_dir=args.base_dir,
                base_url=args.base_url,
            )
    except (HttpError, ValueError, OSError) as exc:
        logger.error("Fallo la descarga de %s %s: %s", args.symbol, args.interval, exc)
        return 1

    logger.info(
        "Resumen: %s %s en [%d, %d] ms UTC, %d velas nuevas",
        args.symbol,
        args.interval,
        start_ms,
        end_ms,
        new,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
