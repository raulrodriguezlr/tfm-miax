"""Tests de los subcomandos de la CLI de ingesta (`download`, `validate`, `pairs`), sin red."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from miax.ingest import HttpError, save_klines
from miax.ingest import __main__ as cli
from miax.ingest.klines import KLINE_COLUMNS, KLINE_DTYPES
from miax.utils.io import read_parquet

NOW = datetime(2026, 10, 9, 12, 0, 0, tzinfo=UTC)
HOUR_MS = 3_600_000
T0 = 1_767_225_600_000  # 2026-01-01T00:00:00Z
SYMBOL = "BTCUSDT"


class _Spy:
    """Sustituto de `update_symbol` o `list_pairs` que registra las llamadas."""

    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Anota los argumentos y devuelve el resultado fijo."""
        self.calls.append((args, kwargs))
        return self.result


class _ListHandler(logging.Handler):
    """Handler que guarda los mensajes formateados con su nivel."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[tuple[int, str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Guarda nivel y mensaje."""
        self.records.append((record.levelno, record.getMessage()))

    @property
    def errors(self) -> list[str]:
        """Mensajes de nivel ERROR o superior."""
        return [m for level, m in self.records if level >= logging.ERROR]


@pytest.fixture
def logs() -> Iterator[_ListHandler]:
    """Engancha un handler al logger de la CLI mientras dura el test."""
    handler = _ListHandler()
    cli.logger.addHandler(handler)
    yield handler
    cli.logger.removeHandler(handler)


def _frame(open_times: list[int]) -> pd.DataFrame:
    """Klines coherentes y con dtypes canonicos para esos `open_time` horarios."""
    n = len(open_times)
    data = {
        "open_time": open_times,
        "open": [100.0] * n,
        "high": [101.0] * n,
        "low": [99.0] * n,
        "close": [100.5] * n,
        "volume": [1.5] * n,
        "close_time": [t + HOUR_MS - 1 for t in open_times],
        "quote_asset_volume": [150.0] * n,
        "num_trades": [10] * n,
        "taker_buy_base": [0.7] * n,
        "taker_buy_quote": [70.0] * n,
        "ignore": [0.0] * n,
    }
    return pd.DataFrame(data).astype(KLINE_DTYPES)[KLINE_COLUMNS]


def _cache(root: Path, frame: pd.DataFrame) -> None:
    """Guarda `frame` como cache horaria de BTCUSDT en `root`."""
    save_klines(frame, SYMBOL, "1h", base_dir=root)


def _times(count: int) -> list[int]:
    """`count` open_time horarios consecutivos desde T0."""
    return [T0 + i * HOUR_MS for i in range(count)]


def _validate(root: Path, *extra: str, symbol: str = SYMBOL) -> int:
    """Ejecuta `validate` sobre la cache de `root`."""
    argv = ["validate", "--symbol", symbol, "--interval", "1h", "--base-dir", str(root), *extra]
    return cli.main(argv)


# --- compatibilidad con la CLI previa ---


def test_legacy_invocation_without_subcommand_goes_to_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`main(["--symbol", "X"])` sin subcomando sigue yendo a download."""
    spy = _Spy(7)
    monkeypatch.setattr(cli, "update_symbol", spy)
    monkeypatch.setattr(cli, "_utc_now", lambda: NOW)
    assert cli.main(["--symbol", "BTCUSDT", "--start", "2026-10-01"]) == 0
    assert len(spy.calls) == 1
    assert spy.calls[0][0][0] == "BTCUSDT"


def test_explicit_download_subcommand(monkeypatch: pytest.MonkeyPatch) -> None:
    """`main(["download", "--symbol", "X"])` hace lo mismo que sin subcomando."""
    spy = _Spy(7)
    monkeypatch.setattr(cli, "update_symbol", spy)
    monkeypatch.setattr(cli, "_utc_now", lambda: NOW)
    assert cli.main(["download", "--symbol", "ETHUSDT", "--start", "2026-10-01"]) == 0
    assert len(spy.calls) == 1
    assert spy.calls[0][0][0] == "ETHUSDT"


# --- validate ---


def test_validate_healthy_cache_returns_zero(tmp_path: Path) -> None:
    """Una cache sana da codigo de salida 0."""
    _cache(tmp_path, _frame(_times(5)))
    assert _validate(tmp_path) == 0


def test_validate_cache_with_error_returns_one(tmp_path: Path) -> None:
    """Una cache con un error de integridad da codigo 1."""
    frame = _frame(_times(5))
    frame.loc[2, "volume"] = -1.0
    _cache(tmp_path, frame)
    assert _validate(tmp_path) == 1


def test_validate_only_warnings_returns_zero(tmp_path: Path) -> None:
    """Solo avisos (un hueco) no hacen fallar el comando."""
    _cache(tmp_path, _frame([*_times(3), T0 + 10 * HOUR_MS]))
    assert _validate(tmp_path) == 0


def test_validate_symbol_without_cache_returns_one_and_logs_error(
    tmp_path: Path, logs: _ListHandler
) -> None:
    """Un simbolo sin cache no se da por sano: codigo 1 y error en el log."""
    assert _validate(tmp_path, symbol="ETHUSDT") == 1
    assert any("ETHUSDT" in m and "no hay cache" in m for m in logs.errors)


@pytest.mark.parametrize("symbol", ["../x", "a/b", "..", "BTC USDT"])
def test_validate_invalid_symbol_returns_two(
    tmp_path: Path, logs: _ListHandler, symbol: str
) -> None:
    """Un simbolo con caracteres no validos da 2 y un error en el log, sin traceback."""
    assert _validate(tmp_path, symbol=symbol) == 2
    assert logs.errors


def test_validate_output_writes_parquet_with_symbol_column(tmp_path: Path) -> None:
    """`--output` guarda las incidencias en Parquet con la columna `symbol`."""
    frame = _frame(_times(5))
    frame.loc[1, "high"] = 1.0
    _cache(tmp_path, frame)
    out = tmp_path / "salida" / "incidencias.parquet"
    assert _validate(tmp_path, "--output", str(out)) == 1
    result = read_parquet(out)
    assert "symbol" in result.columns
    assert set(result["symbol"]) == {SYMBOL}
    assert "ohlc" in set(result["check"])


def test_validate_multiple_symbols_aggregates(tmp_path: Path) -> None:
    """Con varios simbolos, un error en uno basta para devolver 1 y se anotan los dos."""
    _cache(tmp_path, _frame(_times(4)))
    bad = _frame(_times(4))
    bad.loc[0, "volume"] = -1.0
    save_klines(bad, "ETHUSDT", "1h", base_dir=tmp_path)
    out = tmp_path / "multi.parquet"
    code = cli.main(
        [
            "validate",
            "--symbol",
            "BTCUSDT,ETHUSDT",
            "--interval",
            "1h",
            "--base-dir",
            str(tmp_path),
            "--output",
            str(out),
        ]
    )
    assert code == 1
    assert set(read_parquet(out)["symbol"]) == {"ETHUSDT"}


# --- pairs ---


def _pairs_frame() -> pd.DataFrame:
    """Catalogo minimo de pares."""
    return pd.DataFrame(
        {
            "symbol": ["BTCUSDT", "ETHUSDT"],
            "status": ["TRADING", "BREAK"],
            "quote_asset": ["USDT", "USDT"],
        }
    )


def test_pairs_without_flags_passes_no_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sin flags, `list_pairs` se llama con quote_asset=None y status=None."""
    spy = _Spy(_pairs_frame())
    monkeypatch.setattr(cli, "list_pairs", spy)
    assert cli.main(["pairs"]) == 0
    assert len(spy.calls) == 1
    kwargs = spy.calls[0][1]
    assert kwargs["quote_asset"] is None
    assert kwargs["status"] is None


def test_pairs_filters_are_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    """`--quote` y `--status` llegan a `list_pairs`."""
    spy = _Spy(_pairs_frame())
    monkeypatch.setattr(cli, "list_pairs", spy)
    assert cli.main(["pairs", "--quote", "USDT", "--status", "TRADING"]) == 0
    kwargs = spy.calls[0][1]
    assert kwargs["quote_asset"] == "USDT"
    assert kwargs["status"] == "TRADING"


def test_pairs_output_writes_parquet(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`--output` guarda los pares en Parquet."""
    monkeypatch.setattr(cli, "list_pairs", _Spy(_pairs_frame()))
    out = tmp_path / "pares.parquet"
    assert cli.main(["pairs", "--output", str(out)]) == 0
    pd.testing.assert_frame_equal(read_parquet(out), _pairs_frame())


def test_pairs_http_error_returns_one(monkeypatch: pytest.MonkeyPatch, logs: _ListHandler) -> None:
    """Un `HttpError` al consultar el catalogo da codigo 1 y error en el log, sin excepcion."""

    def boom(*args: object, **kwargs: object) -> pd.DataFrame:
        raise HttpError("fallo simulado")

    monkeypatch.setattr(cli, "list_pairs", boom)
    assert cli.main(["pairs"]) == 1
    assert logs.errors
