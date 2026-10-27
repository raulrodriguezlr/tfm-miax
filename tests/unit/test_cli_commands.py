"""Tests de los subcomandos de la CLI de ingesta (download, validate, coverage, pairs, universe)."""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar, Self

import pandas as pd
import pytest
import yaml

from miax.ingest import HttpError, save_klines
from miax.ingest import __main__ as cli
from miax.ingest import universe as universe_module
from miax.ingest.klines import KLINE_COLUMNS, KLINE_DTYPES
from miax.ingest.universe import UniverseParams, UniversePeriod, load_universe, universe_periods
from miax.utils.config import ConfigError
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


# --- universe ---

UNIVERSE_PARAMS = UniverseParams(
    quote_asset="USDT",
    first_period_start="2020-01-01",
    last_period_end="2021-07-01",
    period_months=12,
    volume_window_months=3,
    min_listing_days=30,
    min_quote_volume=10_000_000.0,
    max_per_category=10,
    max_pairs=50,
    min_categories=1,
    categories_file="categories.yaml",
)
UNIVERSE_NOW = datetime(2026, 10, 13, 12, 0, 0, tzinfo=UTC)


class _FakeClient:
    """Cliente falso que solo sirve de gestor de contexto y recuerda si se cerro."""

    instances: ClassVar[list[_FakeClient]] = []

    def __init__(self, **kwargs: object) -> None:
        """Acepta y descarta el rate_limiter."""
        self.closed = False
        _FakeClient.instances.append(self)

    def __enter__(self) -> Self:
        """Entra en el contexto."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Marca el cliente como cerrado."""
        self.closed = True


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> type[_FakeClient]:
    """Sustituye `HttpClient` de la CLI por un cliente falso sin red."""
    _FakeClient.instances = []
    monkeypatch.setattr(cli, "HttpClient", _FakeClient)
    monkeypatch.setattr(cli, "_utc_now", lambda: UNIVERSE_NOW)
    return _FakeClient


def _universe_files(tmp_path: Path, **overrides: object) -> tuple[Path, Path, Path]:
    """Escribe params y categorias en tmp_path y devuelve (params, categorias, salida)."""
    cats = tmp_path / "cats.yaml"
    cats.write_text("l1: [A, B]\nl2: [D]\nstablecoin: [USDC]\n", encoding="utf-8")
    values = {
        **dataclasses.asdict(UNIVERSE_PARAMS),
        "categories_file": str(cats),
        **overrides,
    }
    params = tmp_path / "params.yaml"
    params.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    return params, cats, tmp_path / "out.yaml"


def _universe_argv(params: Path, out: Path, *extra: str) -> list[str]:
    """Argumentos del subcomando `universe`."""
    return ["universe", "--params", str(params), "--out", str(out), *extra]


def _selected(symbols: Sequence[str]) -> pd.DataFrame:
    """Universo de un periodo con las columnas que necesita `write_universe`."""
    return pd.DataFrame(
        {
            "symbol": list(symbols),
            "category": ["l1"] * len(symbols),
            "median_quote_volume": [5e7] * len(symbols),
            "first_candle_ms": pd.array([1_500_000_000_000] * len(symbols), dtype="Int64"),
        }
    )


def _fake_build(*args: object, **kwargs: object) -> list[tuple[UniversePeriod, pd.DataFrame]]:
    """Resultado fijo de `build_universe`: un universo con dos pares por periodo."""
    return [(period, _selected(["AUSDT", "BUSDT"])) for period in universe_periods(UNIVERSE_PARAMS)]


def test_universe_success_writes_file_and_closes_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_client: type[_FakeClient],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Con la red parcheada el subcomando escribe el YAML en tmp_path y devuelve 0."""
    params, cats, out = _universe_files(tmp_path)
    spy_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def build(*args: object, **kwargs: object) -> list[tuple[UniversePeriod, pd.DataFrame]]:
        spy_calls.append((args, kwargs))
        assert not fake_client.instances[0].closed  # el cliente sigue abierto durante la descarga
        return _fake_build()

    monkeypatch.setattr(cli, "build_universe", build)
    code = cli.main([*_universe_argv(params, out), "--base-url", "https://example.invalid"])

    assert code == 0
    assert out.is_file()
    assert [c.closed for c in fake_client.instances] == [True]
    args, kwargs = spy_calls[0]
    assert args[0] == dataclasses.replace(UNIVERSE_PARAMS, categories_file=str(cats))
    assert args[1] == {"A": "l1", "B": "l1", "D": "l2", "USDC": "stablecoin"}
    assert kwargs["client"] is fake_client.instances[0]
    assert kwargs["base_url"] == "https://example.invalid"
    assert kwargs["now_ms"] == cli._to_ms(UNIVERSE_NOW)
    loaded = load_universe(out)
    assert sorted(set(loaded["symbol"])) == ["AUSDT", "BUSDT"]
    assert len(loaded) == 4  # 2 pares x 2 periodos
    assert capsys.readouterr().out == ""


def test_universe_end_to_end_with_network_functions_patched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_client: type[_FakeClient],
) -> None:
    """Con solo la red parcheada, la CLI y build_universe escriben un universo valido."""
    params, _, out = _universe_files(tmp_path)

    def fake_pairs(**kwargs: object) -> pd.DataFrame:
        assets = ["A", "B", "D", "USDC", "ZZZ"]
        return pd.DataFrame({"symbol": [f"{a}USDT" for a in assets], "base_asset": assets})

    def fake_first(symbols: Sequence[str], reference_ms: int, **kwargs: object) -> pd.Series:
        return pd.Series({s: 1_400_000_000_000 for s in symbols}, dtype="Int64")

    def fake_stats(symbols: Sequence[str], start: int, end: int, **kwargs: object) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "median_quote_volume": pd.Series({s: 5e7 for s in symbols}, dtype="float64"),
                "last_traded_ms": pd.Series(
                    {s: end + 1 - 86_400_000 for s in symbols}, dtype="Int64"
                ),
            }
        )

    monkeypatch.setattr(universe_module, "list_pairs", fake_pairs)
    monkeypatch.setattr(universe_module, "first_candle_time", fake_first)
    monkeypatch.setattr(universe_module, "window_volume_stats", fake_stats)

    assert cli.main(_universe_argv(params, out)) == 0
    loaded = load_universe(out)
    assert sorted(set(loaded["symbol"])) == ["AUSDT", "BUSDT", "DUSDT"]  # sin USDC ni ZZZ
    assert fake_client.instances[0].closed


def test_universe_defaults_are_the_repo_config_files(
    monkeypatch: pytest.MonkeyPatch, fake_client: type[_FakeClient]
) -> None:
    """Sin --params ni --out se usan los ficheros por defecto de configs/."""
    seen: dict[str, object] = {}

    def fake_params(path: object) -> UniverseParams:
        seen["params"] = path
        return UNIVERSE_PARAMS

    def fake_write(universes: object, params: object, path: object) -> Path:
        seen["out"] = path
        return Path("x")

    monkeypatch.setattr(cli, "load_universe_params", fake_params)
    monkeypatch.setattr(cli, "load_categories", lambda path: {"A": "l1"})
    monkeypatch.setattr(cli, "build_universe", _fake_build)
    monkeypatch.setattr(cli, "write_universe", fake_write)
    assert cli.main(["universe"]) == 0
    assert seen == {"params": "universe_params.yaml", "out": "universe.yaml"}


def test_universe_invalid_params_returns_two_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_client: type[_FakeClient],
    logs: _ListHandler,
) -> None:
    """Un parametro invalido (ConfigError) da 2, un error en el log y no abre el cliente."""
    params, _, out = _universe_files(tmp_path, max_pairs=0)
    build = _Spy([])
    monkeypatch.setattr(cli, "build_universe", build)
    assert cli.main(_universe_argv(params, out)) == 2
    assert logs.errors
    assert build.calls == []
    assert fake_client.instances == []
    assert not out.exists()


def test_universe_missing_params_file_returns_two(
    tmp_path: Path, fake_client: type[_FakeClient], logs: _ListHandler
) -> None:
    """Un fichero de parametros inexistente es ConfigError: codigo 2."""
    assert cli.main(_universe_argv(tmp_path / "nope.yaml", tmp_path / "out.yaml")) == 2
    assert logs.errors


def test_universe_missing_categories_file_returns_two(
    tmp_path: Path, fake_client: type[_FakeClient], logs: _ListHandler
) -> None:
    """Un fichero de categorias inexistente es ConfigError: codigo 2."""
    params, _, out = _universe_files(tmp_path, categories_file=str(tmp_path / "nope.yaml"))
    assert cli.main(_universe_argv(params, out)) == 2
    assert logs.errors
    assert fake_client.instances == []


def test_universe_broken_params_yaml_returns_two(
    tmp_path: Path, fake_client: type[_FakeClient], logs: _ListHandler
) -> None:
    """Un YAML de parametros con sintaxis rota (YAMLError) da 2, sin traceback."""
    params = tmp_path / "broken.yaml"
    params.write_text("quote_asset: [unclosed\n", encoding="utf-8")
    assert cli.main(_universe_argv(params, tmp_path / "out.yaml")) == 2
    assert logs.errors
    assert fake_client.instances == []


def test_universe_broken_categories_yaml_returns_two(
    tmp_path: Path, fake_client: type[_FakeClient], logs: _ListHandler
) -> None:
    """Un YAML de categorias con sintaxis rota (YAMLError) da 2, sin traceback."""
    params, cats, out = _universe_files(tmp_path)
    cats.write_text("l1: [A, B\n", encoding="utf-8")
    assert cli.main(_universe_argv(params, out)) == 2
    assert logs.errors
    assert fake_client.instances == []
    assert not out.exists()


def test_universe_config_error_raised_by_loader_returns_two(
    monkeypatch: pytest.MonkeyPatch,
    fake_client: type[_FakeClient],
    logs: _ListHandler,
) -> None:
    """Cualquier ConfigError al cargar la configuracion se traduce en el codigo 2."""

    def boom(path: object) -> UniverseParams:
        raise ConfigError("simulado")

    monkeypatch.setattr(cli, "load_universe_params", boom)
    assert cli.main(["universe"]) == 2
    assert any("simulado" in m for m in logs.errors)


@pytest.mark.parametrize("error", [HttpError("fallo simulado"), ValueError("valor simulado")])
def test_universe_build_failure_returns_one_and_closes_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_client: type[_FakeClient],
    logs: _ListHandler,
    error: Exception,
) -> None:
    """Un fallo de HTTP o de valor al construir da 1, error en el log y cliente cerrado."""
    params, _, out = _universe_files(tmp_path)

    def boom(*args: object, **kwargs: object) -> None:
        raise error

    monkeypatch.setattr(cli, "build_universe", boom)
    assert cli.main(_universe_argv(params, out)) == 1
    assert logs.errors
    assert [c.closed for c in fake_client.instances] == [True]
    assert not out.exists()


def test_universe_write_oserror_returns_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_client: type[_FakeClient],
    logs: _ListHandler,
) -> None:
    """Un OSError al escribir el YAML da 1 y un error en el log, sin traceback."""
    params, _, out = _universe_files(tmp_path)

    def boom(*args: object, **kwargs: object) -> Path:
        raise OSError("disco lleno")

    monkeypatch.setattr(cli, "build_universe", _fake_build)
    monkeypatch.setattr(cli, "write_universe", boom)
    assert cli.main(_universe_argv(params, out)) == 1
    assert any("disco lleno" in m for m in logs.errors)
    assert [c.closed for c in fake_client.instances] == [True]


def test_universe_real_write_failure_returns_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_client: type[_FakeClient],
    logs: _ListHandler,
) -> None:
    """Si --out cuelga de un fichero (no de un directorio), la escritura real falla con codigo 1."""
    params, _, _ = _universe_files(tmp_path)
    blocker = tmp_path / "es_un_fichero"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setattr(cli, "build_universe", _fake_build)
    assert cli.main(_universe_argv(params, blocker / "out.yaml")) == 1
    assert logs.errors


def test_universe_help_lists_its_options(capsys: pytest.CaptureFixture[str]) -> None:
    """`universe --help` lista --params, --out y --base-url y sale con 0."""
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["universe", "--help"])
    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    for option in ("--params", "--out", "--base-url"):
        assert option in help_text


def test_universe_failure_paths_print_nothing_to_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_client: type[_FakeClient],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Ni los errores de configuracion ni los del trabajo escriben nada en stdout."""
    bad, _, out = _universe_files(tmp_path, max_pairs=0)
    assert cli.main(_universe_argv(bad, out)) == 2
    good, _, _ = _universe_files(tmp_path)

    def boom(*args: object, **kwargs: object) -> None:
        raise HttpError("fallo simulado")

    monkeypatch.setattr(cli, "build_universe", boom)
    assert cli.main(_universe_argv(good, out)) == 1
    assert capsys.readouterr().out == ""


# --- coverage ---

COVERAGE_NOW = datetime(2026, 1, 1, 10, 30, 0, tzinfo=UTC)  # la vela de las 10:00 sigue en curso


@pytest.fixture
def coverage_clock(monkeypatch: pytest.MonkeyPatch) -> datetime:
    """Fija el reloj de la CLI en `COVERAGE_NOW` para que la vela en curso sea reproducible."""
    monkeypatch.setattr(cli, "_utc_now", lambda: COVERAGE_NOW)
    return COVERAGE_NOW


def _coverage(root: Path, *extra: str, symbol: str = SYMBOL) -> int:
    """Ejecuta `coverage --symbol` sobre la cache horaria de `root`."""
    argv = ["coverage", "--symbol", symbol, "--interval", "1h", "--base-dir", str(root), *extra]
    return cli.main(argv)


def _universe_yaml(path: Path, periods: list[tuple[str, str, list[str]]]) -> Path:
    """Escribe un universo congelado minimo con (inicio, fin, simbolos) por periodo."""
    document = {
        "periods": [
            {
                "period_start": start,
                "period_end": end,
                "pairs": [
                    {
                        "symbol": symbol,
                        "category": "l1",
                        "median_quote_volume": 5e7,
                        "first_candle": "2020-01-01",
                    }
                    for symbol in symbols
                ],
            }
            for start, end, symbols in periods
        ]
    }
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def _coverage_universe(root: Path, universe_file: Path, *extra: str, interval: str = "1h") -> int:
    """Ejecuta `coverage --universe` con el YAML y la cache indicados."""
    argv = [
        "coverage",
        "--universe",
        "--universe-file",
        str(universe_file),
        "--interval",
        interval,
        "--base-dir",
        str(root),
        *extra,
    ]
    return cli.main(argv)


def _info_messages(logs: _ListHandler) -> list[str]:
    """Mensajes de nivel INFO guardados por el handler."""
    return [m for level, m in logs.records if level == logging.INFO]


def test_coverage_healthy_cache_returns_zero_and_logs_percentages(
    tmp_path: Path, coverage_clock: datetime, logs: _ListHandler
) -> None:
    """Una cache completa da 0 y el log lleva el % presente y el % con operaciones."""
    _cache(tmp_path, _frame(_times(5)))
    assert _coverage(tmp_path) == 0
    summary = [m for m in _info_messages(logs) if m.startswith(SYMBOL)]
    assert len(summary) == 1
    assert "100.00 % presente" in summary[0]
    assert "100.00 % con operaciones" in summary[0]
    assert not logs.errors


def test_coverage_symbol_without_cache_returns_one_and_logs_error(
    tmp_path: Path, coverage_clock: datetime, logs: _ListHandler
) -> None:
    """Un simbolo sin cache da codigo 1 y un error con su nombre."""
    assert _coverage(tmp_path, symbol="ETHUSDT") == 1
    assert any("ETHUSDT" in m for m in logs.errors)


def test_coverage_multiple_symbols_one_without_cache_returns_one(
    tmp_path: Path, coverage_clock: datetime, logs: _ListHandler
) -> None:
    """Con varios simbolos, uno sin cache basta para devolver 1, y solo se nombra ese."""
    _cache(tmp_path, _frame(_times(5)))
    assert _coverage(tmp_path, symbol="BTCUSDT,ETHUSDT") == 1
    assert any("ETHUSDT" in m for m in logs.errors)
    assert not any("BTCUSDT" in m for m in logs.errors)


def test_coverage_prints_nothing_to_stdout(
    tmp_path: Path, coverage_clock: datetime, capsys: pytest.CaptureFixture[str]
) -> None:
    """Ni el caso feliz ni los errores escriben en stdout."""
    _cache(tmp_path, _frame(_times(5)))
    assert _coverage(tmp_path) == 0
    assert _coverage(tmp_path, symbol="ETHUSDT") == 1
    assert _coverage(tmp_path, "--start", "2026-01-01") == 2
    assert capsys.readouterr().out == ""


def test_coverage_symbol_list_is_uppercased_and_deduplicated(
    tmp_path: Path, coverage_clock: datetime
) -> None:
    """La lista de simbolos se normaliza como en validate: mayusculas y sin repetidos."""
    _cache(tmp_path, _frame(_times(5)))
    out = tmp_path / "cov.parquet"
    assert _coverage(tmp_path, "--output", str(out), symbol="btcusdt, BTCUSDT") == 0
    assert list(read_parquet(out)["symbol"]) == ["BTCUSDT"]


def test_coverage_output_writes_readable_parquet(tmp_path: Path, coverage_clock: datetime) -> None:
    """`--output` guarda el informe con las columnas de cobertura."""
    frame = _frame(_times(5))
    frame.loc[1, "num_trades"] = 0
    frame.loc[3, "num_trades"] = 0
    _cache(tmp_path, frame)
    out = tmp_path / "salida" / "cov.parquet"
    assert _coverage(tmp_path, "--output", str(out)) == 0
    result = read_parquet(out)
    assert list(result["symbol"]) == [SYMBOL]
    assert result.loc[0, "expected"] == 5
    assert result.loc[0, "present"] == 5
    assert result.loc[0, "traded_pct"] == pytest.approx(60.0)
    assert result.loc[0, "coverage_pct"] == pytest.approx(100.0)


def test_coverage_output_is_written_even_when_returning_one(
    tmp_path: Path, coverage_clock: datetime
) -> None:
    """Con un simbolo sin cache el codigo es 1 pero el Parquet se escribe igualmente."""
    _cache(tmp_path, _frame(_times(5)))
    out = tmp_path / "cov.parquet"
    assert _coverage(tmp_path, "--output", str(out), symbol="BTCUSDT,ETHUSDT") == 1
    result = read_parquet(out)
    assert list(result["symbol"]) == ["BTCUSDT", "ETHUSDT"]
    assert list(result["present"]) == [5, 0]


def test_coverage_output_write_failure_returns_one(
    tmp_path: Path, coverage_clock: datetime, logs: _ListHandler
) -> None:
    """Si `--output` cuelga de un fichero, la escritura falla con codigo 1 y un error."""
    _cache(tmp_path, _frame(_times(5)))
    blocker = tmp_path / "es_un_fichero"
    blocker.write_text("x", encoding="utf-8")
    assert _coverage(tmp_path, "--output", str(blocker / "cov.parquet")) == 1
    assert logs.errors


def test_coverage_range_measures_borders(tmp_path: Path, coverage_clock: datetime) -> None:
    """Con `--start/--end` se miden los bordes: 5 velas de las 10 cerradas el 1-ene."""
    _cache(tmp_path, _frame(_times(5)))
    out = tmp_path / "cov.parquet"
    code = _coverage(tmp_path, "--start", "2026-01-01", "--end", "2026-01-01", "--output", str(out))
    assert code == 0
    row = read_parquet(out).iloc[0]
    # El dia entero acaba tras la ultima vela cerrada (10:00): quedan 10 ranuras, no 24.
    assert row["expected"] == 10
    assert row["present"] == 5
    assert row["missing_before"] == 0
    assert row["missing_after"] == 5


def test_coverage_future_end_is_clipped_to_last_closed_candle(
    tmp_path: Path, coverage_clock: datetime
) -> None:
    """Un `--end` en el futuro no cuenta velas futuras como ausentes."""
    _cache(tmp_path, _frame(_times(10)))
    out = tmp_path / "cov.parquet"
    code = _coverage(tmp_path, "--start", "2026-01-01", "--end", "2030-01-01", "--output", str(out))
    assert code == 0
    row = read_parquet(out).iloc[0]
    assert row["expected"] == 10
    assert row["present"] == 10
    assert row["missing_after"] == 0
    assert row["coverage_pct"] == pytest.approx(100.0)


def test_coverage_open_candle_is_neither_expected_nor_present(
    tmp_path: Path, coverage_clock: datetime
) -> None:
    """Con rango, la vela en curso (10:00) esta en cache pero no es presente ni esperada."""
    _cache(tmp_path, _frame(_times(11)))  # 00:00 a 10:00, la ultima aun sin cerrar
    out = tmp_path / "cov.parquet"
    code = _coverage(tmp_path, "--start", "2026-01-01", "--end", "2026-01-01", "--output", str(out))
    assert code == 0
    row = read_parquet(out).iloc[0]
    assert row["expected"] == 10
    assert row["present"] == 10


def test_coverage_range_in_the_future_returns_two(
    tmp_path: Path, coverage_clock: datetime, logs: _ListHandler
) -> None:
    """Si el rango queda vacio tras recortar a la ultima vela cerrada, codigo 2."""
    _cache(tmp_path, _frame(_times(5)))
    assert _coverage(tmp_path, "--start", "2026-01-02", "--end", "2026-01-03") == 2
    assert logs.errors


@pytest.mark.parametrize(
    "extra",
    [
        ("--start", "2026-01-01"),
        ("--end", "2026-01-01"),
        ("--start", "01/01/2026", "--end", "2026-01-02"),
        ("--start", "2026-01-01", "--end", "manana"),
        ("--start", "2026-01-03", "--end", "2026-01-02"),
    ],
)
def test_coverage_invalid_dates_return_two(
    tmp_path: Path, coverage_clock: datetime, logs: _ListHandler, extra: tuple[str, ...]
) -> None:
    """Un solo extremo, una fecha mal escrita o un rango invertido dan 2 con error en el log."""
    _cache(tmp_path, _frame(_times(5)))
    assert _coverage(tmp_path, *extra) == 2
    assert logs.errors


def test_coverage_monthly_interval_returns_two(
    tmp_path: Path, coverage_clock: datetime, logs: _ListHandler
) -> None:
    """`1M` no tiene duracion fija: codigo 2 y error en el log."""
    argv = ["coverage", "--symbol", SYMBOL, "--interval", "1M", "--base-dir", str(tmp_path)]
    assert cli.main(argv) == 2
    assert logs.errors


def test_coverage_monthly_interval_with_universe_returns_two(
    tmp_path: Path, coverage_clock: datetime
) -> None:
    """`1M` tambien se rechaza con `--universe`."""
    universe_file = _universe_yaml(tmp_path / "u.yaml", [("2026-01-01", "2026-01-02", ["AUSDT"])])
    assert _coverage_universe(tmp_path, universe_file, interval="1M") == 2


@pytest.mark.parametrize("symbol", [",", " , ", ""])
def test_coverage_empty_symbol_list_returns_two(
    tmp_path: Path, coverage_clock: datetime, logs: _ListHandler, symbol: str
) -> None:
    """Una lista de simbolos sin ningun simbolo da 2."""
    assert _coverage(tmp_path, symbol=symbol) == 2
    assert logs.errors


@pytest.mark.parametrize("symbol", ["../x", "a/b", "BTC USDT"])
def test_coverage_invalid_symbol_returns_two(
    tmp_path: Path, coverage_clock: datetime, logs: _ListHandler, symbol: str
) -> None:
    """Un simbolo con caracteres no validos da 2 y un error en el log, sin traceback."""
    assert _coverage(tmp_path, symbol=symbol) == 2
    assert logs.errors


def test_coverage_requires_symbol_or_universe() -> None:
    """Sin `--symbol` ni `--universe` argparse sale con 2."""
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["coverage"])
    assert exit_info.value.code == 2


def test_coverage_symbol_and_universe_are_mutually_exclusive() -> None:
    """`--symbol` y `--universe` a la vez: argparse sale con 2."""
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["coverage", "--symbol", SYMBOL, "--universe"])
    assert exit_info.value.code == 2


def test_coverage_universe_file_without_universe_returns_two(
    tmp_path: Path, coverage_clock: datetime, logs: _ListHandler
) -> None:
    """Un `--universe-file` distinto del de por defecto sin `--universe` da 2."""
    _cache(tmp_path, _frame(_times(5)))
    assert _coverage(tmp_path, "--universe-file", str(tmp_path / "otro.yaml")) == 2
    assert logs.errors


def test_coverage_default_universe_file_is_accepted_with_symbol(
    tmp_path: Path, coverage_clock: datetime
) -> None:
    """Pasar explicitamente el fichero de universo por defecto con `--symbol` no es un error."""
    _cache(tmp_path, _frame(_times(5)))
    assert _coverage(tmp_path, "--universe-file", "universe.yaml") == 0


@pytest.mark.parametrize(
    "extra",
    [
        ("--start", "2026-01-01", "--end", "2026-01-01"),
        ("--start", "2026-01-01"),
        ("--end", "2026-01-01"),
    ],
)
def test_coverage_range_with_universe_returns_two(
    tmp_path: Path, coverage_clock: datetime, logs: _ListHandler, extra: tuple[str, ...]
) -> None:
    """`--start` y/o `--end` junto a `--universe` dan 2."""
    universe_file = _universe_yaml(tmp_path / "u.yaml", [("2026-01-01", "2026-01-02", ["AUSDT"])])
    assert _coverage_universe(tmp_path, universe_file, *extra) == 2
    assert logs.errors


def test_coverage_universe_reports_each_pair_and_period(
    tmp_path: Path, coverage_clock: datetime, logs: _ListHandler
) -> None:
    """`--universe` mide cada par en su periodo, con la vela en curso fuera de lo esperado."""
    # Periodo 1 en curso (hasta 2026-01-02); periodo 2 futuro: no entra en el informe.
    universe_file = _universe_yaml(
        tmp_path / "u.yaml",
        [("2026-01-01", "2026-01-02", ["AUSDT", "BUSDT"]), ("2026-01-02", "2026-01-03", ["AUSDT"])],
    )
    save_klines(_frame(_times(11)), "AUSDT", "1h", base_dir=tmp_path)  # incluye la vela en curso
    save_klines(_frame(_times(6)), "BUSDT", "1h", base_dir=tmp_path)
    out = tmp_path / "cov.parquet"
    assert _coverage_universe(tmp_path, universe_file, "--output", str(out)) == 0
    result = read_parquet(out)
    assert list(result["symbol"]) == ["AUSDT", "BUSDT"]
    assert list(result["expected"]) == [10, 10]
    assert list(result["present"]) == [10, 6]
    assert list(result["missing_after"]) == [0, 4]
    assert {"period_start", "period_end", "category"} <= set(result.columns)
    assert any(m.startswith("Periodo 2026-01-01") for m in _info_messages(logs))
    assert not any(m.startswith("Periodo 2026-01-02") for m in _info_messages(logs))


def test_coverage_universe_default_file_is_universe_yaml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, coverage_clock: datetime
) -> None:
    """Sin `--universe-file` se lee `universe.yaml` (relativo a configs/)."""
    seen: list[object] = []

    def fake_load(path: object) -> pd.DataFrame:
        seen.append(path)
        return pd.DataFrame(columns=["period_start", "period_end", "symbol", "category"])

    monkeypatch.setattr(cli, "load_universe", fake_load)
    cli.main(["coverage", "--universe", "--interval", "1h", "--base-dir", str(tmp_path)])
    assert seen == ["universe.yaml"]


def test_coverage_universe_with_only_future_periods_returns_one(
    tmp_path: Path, coverage_clock: datetime, logs: _ListHandler
) -> None:
    """Si todos los periodos son futuros el informe queda vacio: codigo 1 y error en el log."""
    universe_file = _universe_yaml(tmp_path / "u.yaml", [("2026-01-02", "2026-01-03", ["AUSDT"])])
    assert _coverage_universe(tmp_path, universe_file) == 1
    assert logs.errors


def test_coverage_universe_without_pairs_returns_one(
    tmp_path: Path, coverage_clock: datetime, logs: _ListHandler
) -> None:
    """Un universo sin pares (periodos con `pairs: []`) tambien da un informe vacio: codigo 1."""
    universe_file = _universe_yaml(tmp_path / "u.yaml", [("2026-01-01", "2026-01-02", [])])
    assert _coverage_universe(tmp_path, universe_file) == 1
    assert logs.errors


def test_coverage_universe_pairs_without_candles_warn_with_at_most_ten_examples(
    tmp_path: Path, coverage_clock: datetime, logs: _ListHandler
) -> None:
    """Con 12 pares-periodo sin velas el aviso da 10 ejemplos `periodo:simbolo` y 'y 2 mas'."""
    symbols = [f"S{i:02d}USDT" for i in range(12)]
    universe_file = _universe_yaml(tmp_path / "u.yaml", [("2026-01-01", "2026-01-02", symbols)])
    assert _coverage_universe(tmp_path, universe_file) == 1
    messages = [m for m in logs.errors if "sin velas" in m]
    assert len(messages) == 1
    message = messages[0]
    assert "12 pares-periodo sin velas" in message
    assert message.count("2026-01-01:S") == 10
    assert "2026-01-01:S09USDT" in message
    assert "S10USDT" not in message
    assert message.endswith("y 2 mas")


def test_coverage_universe_few_pairs_without_candles_have_no_and_more_suffix(
    tmp_path: Path, coverage_clock: datetime, logs: _ListHandler
) -> None:
    """Con 3 pares sin velas se listan los tres y no aparece 'y N mas'."""
    universe_file = _universe_yaml(
        tmp_path / "u.yaml", [("2026-01-01", "2026-01-02", ["AUSDT", "BUSDT", "CUSDT"])]
    )
    assert _coverage_universe(tmp_path, universe_file) == 1
    message = next(m for m in logs.errors if "sin velas" in m)
    assert "3 pares-periodo sin velas" in message
    for symbol in ("AUSDT", "BUSDT", "CUSDT"):
        assert f"2026-01-01:{symbol}" in message
    assert " mas" not in message


def test_coverage_universe_partial_missing_returns_one_and_writes_output(
    tmp_path: Path, coverage_clock: datetime
) -> None:
    """Con un par con velas y otro sin ellas el codigo es 1 y el Parquet se escribe."""
    universe_file = _universe_yaml(
        tmp_path / "u.yaml", [("2026-01-01", "2026-01-02", ["AUSDT", "BUSDT"])]
    )
    save_klines(_frame(_times(10)), "AUSDT", "1h", base_dir=tmp_path)
    out = tmp_path / "cov.parquet"
    assert _coverage_universe(tmp_path, universe_file, "--output", str(out)) == 1
    assert list(read_parquet(out)["present"]) == [10, 0]


def test_coverage_universe_missing_file_returns_two(
    tmp_path: Path, coverage_clock: datetime, logs: _ListHandler
) -> None:
    """Un YAML de universo inexistente es ConfigError: codigo 2."""
    assert _coverage_universe(tmp_path, tmp_path / "nope.yaml") == 2
    assert logs.errors


def test_coverage_universe_broken_yaml_returns_two(
    tmp_path: Path, coverage_clock: datetime, logs: _ListHandler
) -> None:
    """Un YAML de universo con sintaxis rota da 2, sin traceback."""
    broken = tmp_path / "roto.yaml"
    broken.write_text("periods: [unclosed\n", encoding="utf-8")
    assert _coverage_universe(tmp_path, broken) == 2
    assert logs.errors


def test_coverage_universe_malformed_yaml_returns_two(
    tmp_path: Path, coverage_clock: datetime, logs: _ListHandler
) -> None:
    """Un YAML valido pero sin la lista `periods` da 2."""
    malformed = tmp_path / "mal.yaml"
    malformed.write_text("foo: bar\n", encoding="utf-8")
    assert _coverage_universe(tmp_path, malformed) == 2
    assert logs.errors


def test_coverage_universe_prints_nothing_to_stdout(
    tmp_path: Path, coverage_clock: datetime, capsys: pytest.CaptureFixture[str]
) -> None:
    """Ni el informe del universo ni sus errores escriben en stdout."""
    universe_file = _universe_yaml(tmp_path / "u.yaml", [("2026-01-01", "2026-01-02", ["AUSDT"])])
    save_klines(_frame(_times(10)), "AUSDT", "1h", base_dir=tmp_path)
    assert _coverage_universe(tmp_path, universe_file) == 0
    assert _coverage_universe(tmp_path, tmp_path / "nope.yaml") == 2
    assert capsys.readouterr().out == ""


def test_coverage_help_lists_its_options(capsys: pytest.CaptureFixture[str]) -> None:
    """`coverage --help` lista sus opciones y sale con 0."""
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["coverage", "--help"])
    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    for option in ("--symbol", "--universe", "--universe-file", "--start", "--end", "--output"):
        assert option in help_text


def test_download_help_mentions_coverage(capsys: pytest.CaptureFixture[str]) -> None:
    """La descripcion de `download` cita el subcomando `coverage`."""
    with pytest.raises(SystemExit):
        cli.main(["download", "--help"])
    assert "coverage" in capsys.readouterr().out
