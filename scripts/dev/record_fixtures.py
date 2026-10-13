"""Graba de Binance los fixtures crudos que usan los tests de ingesta sin red (MIAX-032).

Uso: `python scripts/dev/record_fixtures.py [--out-dir tests/fixtures/binance]`. Usa la red y
sobrescribe los ficheros; es idempotente salvo por lo que cambie Binance (p. ej. exchangeInfo).
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from miax.ingest.exchange_info import fetch_exchange_info
from miax.ingest.http import HttpClient
from miax.ingest.klines import DEFAULT_BASE_URL, KLINES_PATH, KLINES_WEIGHT, MAX_LIMIT
from miax.ingest.ratelimit import RateLimiter
from miax.utils.logging import get_logger

logger = get_logger("record_fixtures")

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO / "tests" / "fixtures" / "binance"

MINUTE_MS = 60 * 1000
DAY_MS = 24 * 60 * MINUTE_MS

# Dia completo de BTCUSDT: 1.440 velas de 1m.
BTC_SYMBOL = "BTCUSDT"
BTC_DAY = datetime(2024, 3, 1, tzinfo=UTC)
BTC_EXPECTED_ROWS = 1440

# LUNAUSDT: se baja una ventana amplia, se localiza el salto mayor y se guardan solo los bordes.
LUNA_SYMBOL = "LUNAUSDT"
LUNA_SEARCH_START = datetime(2022, 5, 10, tzinfo=UTC)
LUNA_SEARCH_END = datetime(2022, 6, 5, tzinfo=UTC)
LUNA_EDGE_ROWS = 100
LUNA_MIN_GAP_MS = 60 * MINUTE_MS  # un salto menor de 1 h no se considera el hueco de mayo

# Velas diarias: primera vela de cada simbolo mas el ultimo trimestre de 2023.
DAILY_SYMBOLS = ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "UNIUSDT", "LINKUSDT", "USDCUSDT", "EOSUSDT"]
DAILY_START = datetime(2023, 10, 1, tzinfo=UTC)
DAILY_END = datetime(2023, 12, 31, tzinfo=UTC)

# exchangeInfo recortado: simbolos fijos mas unos pocos elegidos por estado (orden alfabetico).
EXTRA_TRADING_SYMBOLS = ["SOLUSDT", "ADAUSDT", "BNBUSDT", "ETHBTC"]
EXTRA_BREAK_USDT = 2  # pares USDT en BREAK, ademas de los fijos
EXTRA_BREAK_OTHER = 1  # pares no USDT en BREAK, para probar el filtro por activo cotizado


def _ms(moment: datetime) -> int:
    """Convierte un instante UTC a milisegundos desde la epoca."""
    return int(moment.timestamp() * 1000)


def _fetch_raw_klines(
    client: HttpClient, symbol: str, interval: str, start_ms: int, end_ms: int
) -> list[list[Any]]:
    """Pide /klines paginando hacia delante y devuelve las filas tal cual las da Binance."""
    url = f"{DEFAULT_BASE_URL}{KLINES_PATH}"
    rows: list[list[Any]] = []
    cursor = start_ms
    while cursor <= end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": MAX_LIMIT,
        }
        block = client.get(url, params=params, weight=KLINES_WEIGHT).json()
        if not block:
            break
        # El cursor solo avanza a la vela siguiente a la ultima recibida: no se mira atras.
        next_cursor = int(block[-1][0]) + 1
        if next_cursor <= cursor:
            raise RuntimeError(f"El cursor no avanza en {symbol} {interval} ({cursor})")
        rows.extend(block)
        cursor = next_cursor
    return rows


def _fetch_first_candle(client: HttpClient, symbol: str, interval: str) -> list[list[Any]]:
    """Pide la primera vela disponible del simbolo (startTime=0, limit=1)."""
    params = {"symbol": symbol, "interval": interval, "startTime": 0, "limit": 1}
    return client.get(
        f"{DEFAULT_BASE_URL}{KLINES_PATH}", params=params, weight=KLINES_WEIGHT
    ).json()


def record_btc_day(client: HttpClient) -> list[list[Any]]:
    """Graba el 2024-03-01 completo de BTCUSDT en 1m y comprueba que son 1.440 velas."""
    start_ms = _ms(BTC_DAY)
    end_ms = start_ms + BTC_EXPECTED_ROWS * MINUTE_MS - 1
    rows = _fetch_raw_klines(client, BTC_SYMBOL, "1m", start_ms, end_ms)
    if len(rows) != BTC_EXPECTED_ROWS:
        raise RuntimeError(f"{BTC_SYMBOL}: {len(rows)} velas, se esperaban {BTC_EXPECTED_ROWS}")
    return rows


def record_luna_gap(client: HttpClient) -> list[list[Any]]:
    """Graba las 100 velas 1m a cada lado del mayor hueco de LUNAUSDT (mayo de 2022)."""
    rows = _fetch_raw_klines(
        client, LUNA_SYMBOL, "1m", _ms(LUNA_SEARCH_START), _ms(LUNA_SEARCH_END)
    )
    if len(rows) < 2:
        raise RuntimeError(f"{LUNA_SYMBOL}: solo {len(rows)} velas en la ventana de busqueda")
    # Solo compara velas consecutivas ya descargadas; el hueco es el salto mayor entre dos.
    jumps = [(int(rows[i + 1][0]) - int(rows[i][0]), i) for i in range(len(rows) - 1)]
    jump, idx = max(jumps)
    if jump < LUNA_MIN_GAP_MS:
        raise RuntimeError(f"{LUNA_SYMBOL}: no hay hueco (salto maximo {jump // MINUTE_MS} min)")
    before, after = rows[: idx + 1], rows[idx + 1 :]
    if len(before) < LUNA_EDGE_ROWS or len(after) < LUNA_EDGE_ROWS:
        raise RuntimeError(
            f"{LUNA_SYMBOL}: bordes cortos ({len(before)} antes, {len(after)} despues)"
        )
    logger.info(
        "Hueco de %s: de %s a %s (%d min)",
        LUNA_SYMBOL,
        datetime.fromtimestamp(int(rows[idx][0]) / 1000, tz=UTC).isoformat(),
        datetime.fromtimestamp(int(rows[idx + 1][0]) / 1000, tz=UTC).isoformat(),
        jump // MINUTE_MS,
    )
    return before[-LUNA_EDGE_ROWS:] + after[:LUNA_EDGE_ROWS]


def record_daily(client: HttpClient) -> dict[str, list[list[Any]]]:
    """Graba, por simbolo, su primera vela 1d mas las velas del 2023-10-01 al 2023-12-31."""
    start_ms = _ms(DAILY_START)
    end_ms = _ms(DAILY_END)
    result: dict[str, list[list[Any]]] = {}
    for symbol in DAILY_SYMBOLS:
        first = _fetch_first_candle(client, symbol, "1d")
        window = _fetch_raw_klines(client, symbol, "1d", start_ms, end_ms)
        if not first or not window:
            raise RuntimeError(
                f"{symbol}: sin velas 1d (primera={len(first)}, ventana={len(window)})"
            )
        # Si la primera vela ya cae en la ventana no se duplica; el orden es por open_time.
        merged = {int(row[0]): row for row in first + window}
        result[symbol] = [merged[key] for key in sorted(merged)]
        logger.info("%s 1d: %d velas (primera %s)", symbol, len(result[symbol]), first[0][0])
    return result


def _symbol_names(payload: dict[str, Any], status: str, quote_usdt: bool) -> list[str]:
    """Simbolos del catalogo con ese estado, USDT o no USDT, en orden alfabetico."""
    names = [
        item["symbol"]
        for item in payload["symbols"]
        if item["status"] == status and (item["quoteAsset"] == "USDT") == quote_usdt
    ]
    return sorted(names)


def record_exchange_info(client: HttpClient) -> dict[str, Any]:
    """Graba /exchangeInfo recortado a unos 15 simbolos, con cada entrada intacta."""
    payload = fetch_exchange_info(client=client)
    by_symbol = {item["symbol"]: item for item in payload["symbols"]}
    fixed = [*DAILY_SYMBOLS, LUNA_SYMBOL, *EXTRA_TRADING_SYMBOLS]
    missing = [name for name in fixed if name not in by_symbol]
    if missing:
        raise RuntimeError(f"exchangeInfo no trae los simbolos fijos: {missing}")

    chosen = list(dict.fromkeys(fixed))
    usdt_break = [n for n in _symbol_names(payload, "BREAK", True) if n not in chosen]
    other_break = [n for n in _symbol_names(payload, "BREAK", False) if n not in chosen]
    chosen += usdt_break[:EXTRA_BREAK_USDT] + other_break[:EXTRA_BREAK_OTHER]

    for name in ("EOSUSDT", LUNA_SYMBOL):
        logger.info("%s esta en estado %s", name, by_symbol[name]["status"])
    trimmed = dict(payload)  # conserva el resto de claves (timezone, rateLimits, ...) tal cual
    trimmed["symbols"] = [by_symbol[name] for name in sorted(chosen)]
    logger.info("exchangeInfo recortado a %d simbolos: %s", len(chosen), sorted(chosen))
    return trimmed


def _dump_rows(rows: list[list[Any]], indent: str = "") -> str:
    """Serializa una lista de filas con una fila por linea, para diffs legibles."""
    lines = [f"{indent}  {json.dumps(row, ensure_ascii=False)}" for row in rows]
    return "[\n" + ",\n".join(lines) + f"\n{indent}]"


def _dump_daily(daily: dict[str, list[list[Any]]]) -> str:
    """Serializa el dict simbolo -> filas, una fila por linea y orden de claves estable."""
    blocks = [f"  {json.dumps(symbol)}: {_dump_rows(rows, '  ')}" for symbol, rows in daily.items()]
    return "{\n" + ",\n".join(blocks) + "\n}"


def _write(path: Path, text: str) -> None:
    """Escribe el texto en UTF-8 con salto de linea final, sobrescribiendo."""
    path.write_text(text + "\n", encoding="utf-8", newline="\n")
    logger.info("Escrito %s (%d bytes)", path, path.stat().st_size)


def main(argv: list[str] | None = None) -> int:
    """Descarga todo en memoria y, si no falla nada, sobrescribe los cuatro ficheros."""
    parser = argparse.ArgumentParser(description="Graba los fixtures de Binance (usa la red).")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="carpeta de salida")
    args = parser.parse_args(argv)

    with HttpClient(rate_limiter=RateLimiter()) as client:
        btc = record_btc_day(client)
        luna = record_luna_gap(client)
        daily = record_daily(client)
        info = record_exchange_info(client)

    # Todo se descarga antes de escribir: un fallo a medias no deja fixtures mezclados.
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _write(out_dir / "klines_BTCUSDT_1m.json", _dump_rows(btc))
    _write(out_dir / "klines_LUNAUSDT_1m.json", _dump_rows(luna))
    _write(out_dir / "klines_1d.json", _dump_daily(daily))
    _write(out_dir / "exchange_info.json", json.dumps(info, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
