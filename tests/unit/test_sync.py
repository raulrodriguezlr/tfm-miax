"""Tests de `sync` (rangos del universo, descarga por lotes y CLI) con cliente falso: sin red."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import pandas as pd
import pytest
import yaml

from miax.ingest import ClientError, HttpError, RetriesExhaustedError, cache_bounds, load_klines
from miax.ingest import __main__ as cli
from miax.ingest import batch as batch_module
from miax.ingest import resume as resume_module
from miax.ingest import sync as sync_module
from miax.ingest.batch import DEFAULT_MAX_CONSECUTIVE_FAILURES, NOT_PROCESSED, run_symbol_batch
from miax.ingest.klines import DAY_MS
from miax.ingest.store import _validate_segment
from miax.ingest.sync import sync_universe, universe_download_ranges
from miax.ingest.universe import load_universe

HOUR = 3_600_000
INTERVAL = "1h"
NOW = datetime(2026, 3, 5, 10, 30, 0, tzinfo=UTC)  # la vela de las 10:00 sigue en curso
UNIVERSE_FILES = ["universe.yaml", "universe_semiannual.yaml", "universe_annual.yaml"]


def _ms(text: str) -> int:
    """Milisegundos UTC de una fecha o fecha-hora ISO."""
    return int(pd.Timestamp(text, tz="UTC").timestamp() * 1000)


def _expected(first: int, last: int) -> list[int]:
    """Los `open_time` horarios de una cache perfecta entre `first` y `last`."""
    return list(range(first, last + 1, HOUR))


def _row(open_time: int) -> list[Any]:
    """Una vela horaria cruda de Binance determinista."""
    price = str(100.0 + open_time % 7)
    return [
        open_time,
        price,
        price,
        price,
        price,
        "1.5",
        open_time + HOUR - 1,
        "150",
        10,
        "1",
        "2",
        "0",
    ]


class _Response:
    """Respuesta simulada que solo expone `json()`."""

    def __init__(self, payload: list[list[Any]]) -> None:
        self._payload = payload

    def json(self) -> list[list[Any]]:
        """Devuelve el cuerpo simulado."""
        return self._payload


class _Binance:
    """Cliente falso por simbolo: velas horarias de `available` y errores a medida.

    `available[symbol] = (primera, ultima)` es el `open_time` de la primera y ultima vela que tiene;
    un simbolo ausente no tiene velas. `failing[symbol]` es la excepcion que lanza al pedirlo.
    """

    def __init__(
        self,
        available: dict[str, tuple[int, int]] | None = None,
        failing: dict[str, Exception] | None = None,
    ) -> None:
        self.available = available
        self.failing = failing or {}
        self.calls: list[tuple[str, int, int]] = []
        self.closed = 0

    def get(self, url: str, params: dict[str, Any] | None = None, weight: int = 1) -> _Response:
        """Registra `(symbol, startTime, endTime)` y sirve el bloque que daria Binance."""
        assert params is not None
        symbol, start, end, limit = (params[k] for k in ("symbol", "startTime", "endTime", "limit"))
        self.calls.append((symbol, start, end))
        if symbol in self.failing:
            raise self.failing[symbol]
        if self.available is None:
            first, last = 0, 10**15
        elif symbol in self.available:
            first, last = self.available[symbol]
        else:
            return _Response([])
        lowest = max(start, first)
        lowest += -lowest % HOUR
        return _Response([_row(t) for t in list(range(lowest, min(end, last) + 1, HOUR))[:limit]])

    def symbols_called(self) -> list[str]:
        """Simbolos pedidos, sin repetir y en orden de primera aparicion."""
        return list(dict.fromkeys(symbol for symbol, _, _ in self.calls))

    def close(self) -> None:
        """Cuenta los cierres."""
        self.closed += 1

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class _ListHandler(logging.Handler):
    """Handler que guarda nivel y mensaje formateado."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[tuple[int, str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Guarda nivel y mensaje."""
        self.records.append((record.levelno, record.getMessage()))

    @property
    def messages(self) -> list[str]:
        """Todos los mensajes."""
        return [m for _, m in self.records]

    @property
    def errors(self) -> list[str]:
        """Mensajes de nivel ERROR o superior."""
        return [m for level, m in self.records if level >= logging.ERROR]


@pytest.fixture
def logs() -> Iterator[_ListHandler]:
    """Recoge los logs de la CLI, de `sync`, del lote y de `resume` mientras dura el test."""
    handler = _ListHandler()
    loggers = [cli.logger, sync_module.logger, batch_module.logger, resume_module.logger]
    for logger in loggers:
        logger.addHandler(handler)
    yield handler
    for logger in loggers:
        logger.removeHandler(handler)


def _universe(rows: list[tuple[str, str, str]], dtype: str = "datetime64[us, UTC]") -> pd.DataFrame:
    """Universo de (simbolo, inicio, fin) con las columnas que usa `universe_download_ranges`."""
    frame = pd.DataFrame(rows, columns=["symbol", "period_start", "period_end"])
    for column in ("period_start", "period_end"):
        frame[column] = pd.to_datetime(frame[column], utc=True).astype(dtype)
    return frame


def _ranges(spans: dict[str, tuple[int, int]]) -> pd.DataFrame:
    """Tabla de rangos `symbol, start_ms, end_ms` como la que da `universe_download_ranges`."""
    return pd.DataFrame(
        [(s, a, b) for s, (a, b) in spans.items()], columns=["symbol", "start_ms", "end_ms"]
    )


# --- universe_download_ranges ------------------------------------------------------------


def test_ranges_merge_periods_and_universes_into_one_span_per_symbol() -> None:
    """Un simbolo en varios periodos y universos da un unico tramo: min inicio - calentamiento."""
    quarterly = _universe(
        [("AUSDT", "2023-01-01", "2023-04-01"), ("AUSDT", "2023-04-01", "2023-07-01")]
    )
    annual = _universe(
        [("AUSDT", "2022-10-01", "2023-01-01"), ("BUSDT", "2023-01-01", "2024-01-01")]
    )

    ranges = universe_download_ranges([quarterly, annual], warmup_days=7)

    assert ranges["symbol"].tolist() == ["AUSDT", "BUSDT"]
    assert ranges["start_ms"].tolist() == [
        _ms("2022-10-01") - 7 * DAY_MS,
        _ms("2023-01-01") - 7 * DAY_MS,
    ]
    # period_end es exclusivo: el tramo acaba 1 ms antes.
    assert ranges["end_ms"].tolist() == [_ms("2023-07-01") - 1, _ms("2024-01-01") - 1]
    assert ranges.dtypes["start_ms"] == "int64"
    assert ranges.dtypes["end_ms"] == "int64"


def test_ranges_bridge_a_gap_between_periods() -> None:
    """Dos periodos separados por un hueco dan un solo tramo continuo, no dos."""
    universe = _universe(
        [("AUSDT", "2023-01-01", "2023-02-01"), ("AUSDT", "2023-06-01", "2023-07-01")]
    )

    ranges = universe_download_ranges([universe], warmup_days=0)

    assert len(ranges) == 1
    assert ranges.iloc[0].tolist() == ["AUSDT", _ms("2023-01-01"), _ms("2023-07-01") - 1]


@pytest.mark.parametrize(
    "dtype", ["datetime64[us, UTC]", "datetime64[ns, UTC]", "datetime64[ms, UTC]"]
)
def test_ranges_do_not_depend_on_the_datetime_resolution(dtype: str) -> None:
    """Las fechas `us`, `ns` y `ms` dan exactamente los mismos milisegundos."""
    rows = [("AUSDT", "2023-01-01", "2023-04-01")]
    ranges = universe_download_ranges([_universe(rows, dtype)], warmup_days=3)

    assert ranges.iloc[0].tolist() == [
        "AUSDT",
        _ms("2023-01-01") - 3 * DAY_MS,
        _ms("2023-04-01") - 1,
    ]


def test_ranges_with_zero_warmup_start_at_the_first_period() -> None:
    """`warmup_days=0` no ensancha el inicio."""
    ranges = universe_download_ranges(
        [_universe([("AUSDT", "2023-01-01", "2023-02-01")])], warmup_days=0
    )

    assert ranges["start_ms"].tolist() == [_ms("2023-01-01")]


def test_ranges_reject_negative_warmup() -> None:
    """Un calentamiento negativo (acortaria el tramo) es un `ValueError`."""
    with pytest.raises(ValueError, match="warmup_days"):
        universe_download_ranges(
            [_universe([("AUSDT", "2023-01-01", "2023-02-01")])], warmup_days=-1
        )


def test_ranges_of_empty_inputs_are_an_empty_typed_table() -> None:
    """Ni lista vacia, ni universos vacios dan filas; las columnas y dtypes se mantienen."""
    empty = _universe([("AUSDT", "2023-01-01", "2023-02-01")]).iloc[:0]
    for universes in ([], [empty], [empty, empty]):
        ranges = universe_download_ranges(universes, warmup_days=7)
        assert ranges.empty
        assert list(ranges.columns) == ["symbol", "start_ms", "end_ms"]
        assert ranges.dtypes["start_ms"] == "int64"
        assert ranges.dtypes["end_ms"] == "int64"


def test_ranges_ignore_empty_universes_among_full_ones() -> None:
    """Un universo vacio entre otros con datos no estorba."""
    full = _universe([("AUSDT", "2023-01-01", "2023-02-01")])

    ranges = universe_download_ranges([full.iloc[:0], full], warmup_days=0)

    assert ranges["symbol"].tolist() == ["AUSDT"]


def test_ranges_are_sorted_by_symbol_and_do_not_mutate_inputs() -> None:
    """Salen ordenados por simbolo, uno por simbolo, y los universos de entrada no se tocan."""
    universe = _universe(
        [
            ("ZUSDT", "2023-01-01", "2023-02-01"),
            ("AUSDT", "2023-01-01", "2023-02-01"),
            ("MUSDT", "2023-01-01", "2023-02-01"),
            ("AUSDT", "2023-02-01", "2023-03-01"),
        ]
    )
    before = universe.copy()

    ranges = universe_download_ranges([universe], warmup_days=1)

    assert ranges["symbol"].tolist() == ["AUSDT", "MUSDT", "ZUSDT"]
    pd.testing.assert_frame_equal(universe, before)


def test_ranges_accept_extra_columns_like_load_universe_gives() -> None:
    """El universo real trae mas columnas (categoria, volumen...): se ignoran."""
    universe = _universe([("AUSDT", "2023-01-01", "2023-02-01")]).assign(
        category="l1", median_quote_volume=5e7
    )

    assert universe_download_ranges([universe], warmup_days=0)["symbol"].tolist() == ["AUSDT"]


@pytest.mark.parametrize("warmup_days", [0, 7, 30])
def test_ranges_of_the_real_universes_follow_the_rule(warmup_days: int) -> None:
    """Con los tres YAML reales: un tramo por simbolo, min inicio - calentamiento a max fin - 1."""
    universes = [load_universe(name) for name in UNIVERSE_FILES]
    joined = pd.concat(universes, ignore_index=True)

    ranges = universe_download_ranges(universes, warmup_days=warmup_days).set_index("symbol")

    assert ranges.index.is_unique
    assert ranges.index.tolist() == sorted(joined["symbol"].unique())
    for symbol, group in joined.groupby("symbol"):
        first = int(group["period_start"].min().timestamp() * 1000)
        last = int(group["period_end"].max().timestamp() * 1000)
        assert ranges.loc[symbol, "start_ms"] == first - warmup_days * DAY_MS
        assert ranges.loc[symbol, "end_ms"] == last - 1
        assert ranges.loc[symbol, "start_ms"] < ranges.loc[symbol, "end_ms"]


def test_every_symbol_of_the_real_universes_is_a_valid_cache_name() -> None:
    """Ningun simbolo de los universos reales (incluido el chino) lo rechaza la cache."""
    symbols = pd.concat([load_universe(name) for name in UNIVERSE_FILES])["symbol"].unique()

    assert len(symbols) > 0
    for symbol in symbols:
        assert _validate_segment(str(symbol)) == symbol


# --- run_symbol_batch --------------------------------------------------------------------


def test_batch_defaults_and_empty_input() -> None:
    """El cortacircuitos por defecto es 3, `NOT_PROCESSED` no es ni None ni un numero de velas."""
    assert DEFAULT_MAX_CONSECUTIVE_FAILURES == 3
    assert NOT_PROCESSED < 0
    assert run_symbol_batch([], lambda symbol: 1, interval=INTERVAL) == {}


def test_batch_runs_in_order_and_isolates_failures() -> None:
    """Orden de entrada, None en el fallido y los demas con su resultado."""
    seen: list[str] = []

    def fetch(symbol: str) -> int:
        seen.append(symbol)
        if symbol == "B":
            raise HttpError("x")
        return len(seen)

    results = run_symbol_batch(["A", "B", "C"], fetch, interval=INTERVAL)

    assert seen == ["A", "B", "C"]
    assert results == {"A": 1, "B": None, "C": 3}


def test_batch_uses_the_given_logger() -> None:
    """Los mensajes de la tanda salen por el logger que se le pasa."""
    handler = _ListHandler()
    logger = logging.getLogger("test-batch-logger")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False

    run_symbol_batch(["A"], lambda symbol: 2, interval=INTERVAL, log=logger)

    assert handler.messages[-1] == "Resumen: 1 simbolos OK, 0 fallidos"


# --- sync_universe -----------------------------------------------------------------------


def _spans(*symbols: str) -> dict[str, tuple[int, int]]:
    """Mismo tramo de dos dias para cada simbolo."""
    return {symbol: (_ms("2026-01-10"), _ms("2026-01-11 23:00")) for symbol in symbols}


def _sync(
    client: _Binance,
    base_dir: Path,
    spans: dict[str, tuple[int, int]],
    *,
    end_cap_ms: int | None = None,
    max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES,
) -> dict[str, int | None]:
    """Ejecuta `sync_universe` de `INTERVAL` con el falso."""
    return sync_universe(
        _ranges(spans),
        INTERVAL,
        client=client,  # type: ignore[arg-type]
        base_dir=str(base_dir),
        end_cap_ms=end_cap_ms,
        max_consecutive_failures=max_consecutive_failures,
    )


def test_sync_downloads_every_symbol_in_order(tmp_path: Path) -> None:
    """Cada simbolo queda cacheado con su tramo exacto y el resultado respeta el orden."""
    spans = {
        "BUSDT": (_ms("2026-01-10"), _ms("2026-01-11 23:00")),
        "AUSDT": (_ms("2026-02-01"), _ms("2026-02-02 23:00")),
    }
    client = _Binance()

    results = _sync(client, tmp_path, spans)

    assert results == {"BUSDT": 48, "AUSDT": 48}
    assert list(results) == ["BUSDT", "AUSDT"]
    assert client.symbols_called() == ["BUSDT", "AUSDT"]
    for symbol, (first, last) in spans.items():
        opens = load_klines(symbol, INTERVAL, base_dir=tmp_path)["open_time"].tolist()
        assert opens == _expected(first, last)


def test_sync_second_run_downloads_nothing(tmp_path: Path) -> None:
    """Repetir la sincronizacion devuelve 0 por simbolo y no llama al cliente."""
    spans = _spans("AUSDT", "BUSDT")
    _sync(_Binance(), tmp_path, spans)
    again = _Binance()

    assert _sync(again, tmp_path, spans) == {"AUSDT": 0, "BUSDT": 0}
    assert again.calls == []


def test_sync_downloads_only_the_warmup_before_the_cached_period(tmp_path: Path) -> None:
    """Con el periodo ya cacheado, el calentamiento baja justo los dias que faltan por detras."""
    universe = _universe([("AUSDT", "2026-02-10", "2026-02-12")])
    ranges = universe_download_ranges([universe], warmup_days=2)
    first_period, last_period = _ms("2026-02-10"), _ms("2026-02-11 23:00")
    seed = _Binance()
    sync_universe(
        ranges.assign(start_ms=first_period),
        INTERVAL,
        client=seed,
        base_dir=str(tmp_path),  # type: ignore[arg-type]
    )
    assert cache_bounds("AUSDT", INTERVAL, base_dir=tmp_path) == (first_period, last_period)
    client = _Binance()

    results = sync_universe(ranges, INTERVAL, client=client, base_dir=str(tmp_path))  # type: ignore[arg-type]

    assert results == {"AUSDT": 48}
    assert cache_bounds("AUSDT", INTERVAL, base_dir=tmp_path) == (
        first_period - 2 * DAY_MS,
        last_period,
    )
    assert all(end < first_period for _, _, end in client.calls)


def test_sync_stops_at_period_end_exclusive(tmp_path: Path) -> None:
    """La vela que abre justo en `period_end` no se baja (el fin es exclusivo)."""
    ranges = universe_download_ranges(
        [_universe([("AUSDT", "2026-02-01", "2026-02-03")])], warmup_days=0
    )

    sync_universe(ranges, INTERVAL, client=_Binance(), base_dir=str(tmp_path))  # type: ignore[arg-type]

    assert cache_bounds("AUSDT", INTERVAL, base_dir=tmp_path) == (
        _ms("2026-02-01"),
        _ms("2026-02-02 23:00"),
    )


def test_sync_circuit_breaker_stops_after_consecutive_systemic_failures(
    tmp_path: Path, logs: _ListHandler
) -> None:
    """Tres `RetriesExhaustedError` seguidos cortan la tanda: el resto queda NOT_PROCESSED."""
    failing: dict[str, Exception] = {s: RetriesExhaustedError("429") for s in ("A", "B", "C")}
    client = _Binance(failing=failing)

    results = _sync(client, tmp_path, _spans("A", "B", "C", "D", "E"))

    assert results == {"A": None, "B": None, "C": None, "D": NOT_PROCESSED, "E": NOT_PROCESSED}
    assert client.symbols_called() == ["A", "B", "C"]
    assert any("Tanda cortada" in m and "2 simbolos" in m for m in logs.messages)
    assert logs.messages[-1] == (
        "Resumen: 0 simbolos OK, 3 fallidos (A, B, C), 2 sin procesar (D, E)"
    )


def test_sync_circuit_breaker_threshold_is_configurable(tmp_path: Path) -> None:
    """`max_consecutive_failures=1` corta con el primer fallo sistemico."""
    client = _Binance(failing={"A": RetriesExhaustedError("418")})

    results = _sync(client, tmp_path, _spans("A", "B", "C"), max_consecutive_failures=1)

    assert results == {"A": None, "B": NOT_PROCESSED, "C": NOT_PROCESSED}
    assert client.symbols_called() == ["A"]


def test_sync_success_in_between_resets_the_systemic_counter(tmp_path: Path) -> None:
    """Dos fallos, un exito y dos fallos mas no cortan: los fallos tienen que ser consecutivos."""
    failing: dict[str, Exception] = {s: RetriesExhaustedError("429") for s in ("A", "B", "D", "E")}
    client = _Binance(failing=failing)

    results = _sync(client, tmp_path, _spans("A", "B", "C", "D", "E", "F"))

    assert results == {"A": None, "B": None, "C": 48, "D": None, "E": None, "F": 48}


def test_sync_symbol_own_failure_does_not_count_as_systemic(tmp_path: Path) -> None:
    """Un `ClientError` (400) aisla al simbolo, no corta y resetea el contador sistemico."""
    failing: dict[str, Exception] = {
        "A": RetriesExhaustedError("x"),
        "B": RetriesExhaustedError("x"),
        "C": ClientError(400, "http://x"),
        "D": RetriesExhaustedError("x"),
    }
    client = _Binance(failing=failing)

    results = _sync(client, tmp_path, _spans("A", "B", "C", "D", "E"))

    assert results == {"A": None, "B": None, "C": None, "D": None, "E": 48}


def test_sync_symbol_without_candles_is_not_a_failure(tmp_path: Path, logs: _ListHandler) -> None:
    """Un par que Binance no tiene (BCC) da 0, deja aviso y no cuenta para el cortacircuitos."""
    client = _Binance(available={"OKUSDT": (0, 10**15)})

    results = _sync(
        client, tmp_path, _spans("BCCUSDT", "BCHABCUSDT", "OKUSDT"), max_consecutive_failures=1
    )

    assert results == {"BCCUSDT": 0, "BCHABCUSDT": 0, "OKUSDT": 48}
    assert not (tmp_path / "BCCUSDT").exists()
    assert any("sin velas en Binance para su rango" in m and "BCCUSDT" in m for m in logs.messages)
    assert logs.messages[-1] == "Resumen: 3 simbolos OK, 0 fallidos"


def test_sync_end_cap_trims_the_range_end(tmp_path: Path) -> None:
    """`end_cap_ms` recorta el fin: la vela en curso y las futuras no se piden ni se guardan."""
    first, last = _ms("2026-01-10"), _ms("2026-01-20 23:00")
    cap = _ms("2026-01-12 06:59:59.999")  # la vela de las 06:00 es la ultima cerrada
    client = _Binance()

    results = _sync(client, tmp_path, {"AUSDT": (first, last)}, end_cap_ms=cap)

    assert results == {"AUSDT": len(_expected(first, _ms("2026-01-12 06:00")))}
    assert max(end for _, _, end in client.calls) <= cap
    assert cache_bounds("AUSDT", INTERVAL, base_dir=tmp_path) == (first, _ms("2026-01-12 06:00"))


def test_sync_end_cap_never_extends_a_range(tmp_path: Path) -> None:
    """Un tope posterior al fin del tramo no lo amplia."""
    first, last = _ms("2026-01-10"), _ms("2026-01-10 23:00")

    _sync(_Binance(), tmp_path, {"AUSDT": (first, last)}, end_cap_ms=_ms("2026-03-01"))

    assert cache_bounds("AUSDT", INTERVAL, base_dir=tmp_path) == (first, last)


def test_sync_end_cap_before_the_range_downloads_nothing(tmp_path: Path) -> None:
    """Un tope anterior al inicio del tramo (rango vacio) no llama y da 0."""
    client = _Binance()

    results = _sync(client, tmp_path, _spans("AUSDT"), end_cap_ms=_ms("2025-01-01"))

    assert results == {"AUSDT": 0}
    assert client.calls == []


def test_sync_empty_ranges_return_an_empty_result(tmp_path: Path) -> None:
    """Sin simbolos no hay nada que hacer."""
    assert _sync(_Binance(), tmp_path, {}) == {}


def test_sync_handles_a_non_latin_symbol(tmp_path: Path) -> None:
    """`币安人生USDT` se sincroniza como cualquier otro."""
    spans = _spans("币安人生USDT")

    assert _sync(_Binance(), tmp_path, spans) == {"币安人生USDT": 48}
    assert cache_bounds("币安人生USDT", INTERVAL, base_dir=tmp_path) is not None


# --- CLI sync ----------------------------------------------------------------------------


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
    path.write_text(yaml.safe_dump(document, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


class _Factory:
    """Sustituto de `cli.HttpClient`: entrega siempre el mismo falso y cuenta las creaciones."""

    def __init__(self, client: _Binance) -> None:
        self.client = client
        self.created = 0

    def __call__(self, *args: object, **kwargs: object) -> _Binance:
        self.created += 1
        return self.client


@pytest.fixture
def two_universes(tmp_path: Path) -> tuple[Path, Path]:
    """Dos universos: A en el 1 (dos periodos), B en el 2 y C en ambos; el 2 acaba en el futuro."""
    first = _universe_yaml(
        tmp_path / "u1.yaml",
        [
            ("2026-01-01", "2026-02-01", ["AUSDT", "CUSDT"]),
            ("2026-02-01", "2026-03-01", ["AUSDT"]),
        ],
    )
    second = _universe_yaml(
        tmp_path / "u2.yaml", [("2026-02-01", "2026-03-10", ["BUSDT", "CUSDT"])]
    )
    return first, second


def _install(monkeypatch: pytest.MonkeyPatch, client: _Binance) -> _Factory:
    """Parchea el reloj de la CLI y el cliente HTTP por el falso."""
    factory = _Factory(client)
    monkeypatch.setattr(cli, "_utc_now", lambda: NOW)
    monkeypatch.setattr(cli, "HttpClient", factory)
    return factory


def _sync_cli(cache: Path, files: str, *extra: str) -> int:
    """Ejecuta `sync` horario sobre la cache dada."""
    return cli.main(
        [
            "sync",
            "--universe-files",
            files,
            "--interval",
            INTERVAL,
            "--base-dir",
            str(cache),
            *extra,
        ]
    )


def test_cli_sync_two_universes_returns_zero_and_caches_each_symbol(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, two_universes: tuple[Path, Path]
) -> None:
    """Con dos universos: 0, un solo cliente cerrado una vez y cada par con su tramo continuo."""
    client = _Binance()
    factory = _install(monkeypatch, client)
    cache = tmp_path / "cache"

    code = _sync_cli(cache, ",".join(str(p) for p in two_universes), "--warmup-days", "1")

    assert code == 0
    assert factory.created == 1
    assert client.closed == 1
    last_closed = _ms("2026-03-05 09:00")  # la vela de las 10:00 sigue en curso
    expected = {
        "AUSDT": (_ms("2025-12-31"), _ms("2026-02-28 23:00")),
        "BUSDT": (_ms("2026-01-31"), last_closed),  # su periodo acaba el 10-mar: se recorta
        "CUSDT": (_ms("2025-12-31"), last_closed),
    }
    for symbol, (first, last) in expected.items():
        opens = load_klines(symbol, INTERVAL, base_dir=cache)["open_time"].tolist()
        assert opens == _expected(first, last)


def test_cli_sync_second_run_downloads_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, two_universes: tuple[Path, Path]
) -> None:
    """Una segunda ejecucion con la misma cache sale con 0 sin pedir ninguna vela."""
    files = ",".join(str(p) for p in two_universes)
    cache = tmp_path / "cache"
    _install(monkeypatch, _Binance())
    assert _sync_cli(cache, files) == 0
    again = _Binance()
    _install(monkeypatch, again)

    assert _sync_cli(cache, files) == 0
    assert again.calls == []


def test_cli_sync_default_warmup_is_seven_days(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, two_universes: tuple[Path, Path]
) -> None:
    """Sin `--warmup-days` el tramo empieza 7 dias antes del primer periodo."""
    _install(monkeypatch, _Binance())
    cache = tmp_path / "cache"

    assert _sync_cli(cache, str(two_universes[1])) == 0

    assert cache_bounds("BUSDT", INTERVAL, base_dir=cache)[0] == _ms("2026-02-01") - 7 * DAY_MS  # type: ignore[index]


def test_cli_sync_parser_defaults() -> None:
    """Valores por defecto del subcomando: universo trimestral, 7 dias y velas de 1m."""
    args = cli._build_parser().parse_args(["sync"])

    assert args.universe_files == "universe.yaml"
    assert args.warmup_days == 7
    assert args.interval == "1m"
    assert args.base_dir == cli.DEFAULT_BASE_DIR
    assert args.command == "sync"


def test_cli_sync_resolves_names_relative_to_configs_and_merges_the_three_universes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Los tres nombres relativos a `configs/` llegan unidos y con el tope de la ultima cerrada."""
    seen: dict[str, Any] = {}

    def fake_sync(ranges: pd.DataFrame, interval: str, **kwargs: object) -> dict[str, int | None]:
        seen.update(ranges=ranges, interval=interval, **kwargs)
        return {str(s): 0 for s in ranges["symbol"]}

    monkeypatch.setattr(cli, "sync_universe", fake_sync)
    monkeypatch.setattr(cli, "_utc_now", lambda: NOW)
    monkeypatch.setattr(cli, "HttpClient", _Factory(_Binance()))

    assert (
        cli.main(["sync", "--universe-files", ",".join(UNIVERSE_FILES), "--warmup-days", "3"]) == 0
    )

    union = {
        str(s) for s in pd.concat([load_universe(n) for n in UNIVERSE_FILES])["symbol"].unique()
    }
    ranges = seen["ranges"]
    assert set(ranges["symbol"]) == union
    assert seen["interval"] == "1m"
    # Con velas de 1m a las 10:30 la ultima cerrada abre a las 10:29 y acaba a las 10:29:59.999.
    assert seen["end_cap_ms"] == _ms("2026-03-05 10:29:59.999")


def test_cli_sync_failing_symbol_returns_one_and_keeps_the_rest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    two_universes: tuple[Path, Path],
    logs: _ListHandler,
) -> None:
    """Un simbolo que falla da codigo 1, no aborta la tanda y queda en el resumen."""
    client = _Binance(failing={"BUSDT": ClientError(400, "http://x")})
    _install(monkeypatch, client)
    cache = tmp_path / "cache"

    assert _sync_cli(cache, ",".join(str(p) for p in two_universes)) == 1

    assert client.symbols_called() == ["AUSDT", "BUSDT", "CUSDT"]
    assert cache_bounds("AUSDT", INTERVAL, base_dir=cache) is not None
    assert cache_bounds("CUSDT", INTERVAL, base_dir=cache) is not None
    assert cache_bounds("BUSDT", INTERVAL, base_dir=cache) is None
    assert logs.messages[-1] == "Resumen: 2 simbolos OK, 1 fallidos (BUSDT)"


def test_cli_sync_tripped_breaker_returns_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Si el cortacircuitos corta la tanda, el codigo es 1 y los pendientes no se piden."""
    universe = _universe_yaml(
        tmp_path / "u.yaml", [("2026-01-01", "2026-01-03", ["AUSDT", "BUSDT", "CUSDT", "DUSDT"])]
    )
    failing: dict[str, Exception] = {
        s: RetriesExhaustedError("429") for s in ("AUSDT", "BUSDT", "CUSDT")
    }
    client = _Binance(failing=failing)
    _install(monkeypatch, client)

    assert _sync_cli(tmp_path / "cache", str(universe)) == 1
    assert client.symbols_called() == ["AUSDT", "BUSDT", "CUSDT"]
    assert client.closed == 1


def test_cli_sync_symbol_without_candles_is_still_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Un par que Binance no tiene (BCC) no es un fallo: codigo 0."""
    universe = _universe_yaml(
        tmp_path / "u.yaml", [("2026-01-01", "2026-01-03", ["BCCUSDT", "AUSDT"])]
    )
    _install(monkeypatch, _Binance(available={"AUSDT": (0, 10**15)}))

    assert _sync_cli(tmp_path / "cache", str(universe)) == 0


def test_cli_sync_non_latin_symbol_end_to_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Un universo con `币安人生USDT` se sincroniza y se lee de la cache."""
    universe = _universe_yaml(tmp_path / "u.yaml", [("2026-01-01", "2026-01-03", ["币安人生USDT"])])
    _install(monkeypatch, _Binance())
    cache = tmp_path / "cache"

    assert _sync_cli(cache, str(universe), "--warmup-days", "0") == 0

    assert cache_bounds("币安人生USDT", INTERVAL, base_dir=cache) == (
        _ms("2026-01-01"),
        _ms("2026-01-02 23:00"),
    )


def _assert_invalid_arguments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *argv: str
) -> _Binance:
    """Comprueba codigo 2, error logueado y que no se crea cliente ni se escribe cache."""
    client = _Binance()
    factory = _install(monkeypatch, client)
    cache = tmp_path / "cache"

    assert cli.main(["sync", "--interval", INTERVAL, "--base-dir", str(cache), *argv]) == 2

    assert factory.created == 0
    assert client.calls == []
    assert not cache.exists()
    return client


def test_cli_sync_missing_universe_file_returns_two(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, logs: _ListHandler
) -> None:
    """Un fichero de universo que no existe da 2 antes de tocar la red."""
    _assert_invalid_arguments(
        monkeypatch, tmp_path, "--universe-files", str(tmp_path / "no-existe.yaml")
    )
    assert any("Universo no valido" in m for m in logs.errors)


def test_cli_sync_one_missing_file_among_valid_ones_returns_two(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, two_universes: tuple[Path, Path]
) -> None:
    """Si uno de varios ficheros no existe no se sincroniza nada."""
    files = f"{two_universes[0]},{tmp_path / 'no-existe.yaml'}"
    _assert_invalid_arguments(monkeypatch, tmp_path, "--universe-files", files)


@pytest.mark.parametrize("value", [",", " , ,", "   "])
def test_cli_sync_empty_file_list_returns_two(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, logs: _ListHandler, value: str
) -> None:
    """Una lista de ficheros sin ficheros reales da 2."""
    _assert_invalid_arguments(monkeypatch, tmp_path, f"--universe-files={value}")
    assert any("ningun fichero" in m for m in logs.errors)


def test_cli_sync_negative_warmup_returns_two(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    two_universes: tuple[Path, Path],
    logs: _ListHandler,
) -> None:
    """`--warmup-days -1` da 2."""
    _assert_invalid_arguments(
        monkeypatch, tmp_path, "--universe-files", str(two_universes[0]), "--warmup-days=-1"
    )
    assert any("warmup_days" in m for m in logs.errors)


def test_cli_sync_interval_without_fixed_duration_returns_two(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, two_universes: tuple[Path, Path]
) -> None:
    """Un intervalo sin duracion fija (`1M`) da 2."""
    client = _Binance()
    factory = _install(monkeypatch, client)

    code = cli.main(
        [
            "sync",
            "--universe-files",
            str(two_universes[0]),
            "--interval",
            "1M",
            "--base-dir",
            str(tmp_path),
        ]
    )

    assert code == 2
    assert factory.created == 0


def test_cli_sync_universe_without_pairs_returns_two(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, logs: _ListHandler
) -> None:
    """Un universo valido pero sin ningun par da 2: no hay nada que sincronizar."""
    universe = _universe_yaml(tmp_path / "vacio.yaml", [("2026-01-01", "2026-02-01", [])])

    _assert_invalid_arguments(monkeypatch, tmp_path, "--universe-files", str(universe))
    assert any("ningun par" in m for m in logs.errors)


@pytest.mark.parametrize("content", ["periods: [", "- 1\n- 2\n", "periods: 3\n", ""])
def test_cli_sync_malformed_universe_returns_two(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, content: str
) -> None:
    """Un YAML roto o con otra forma da 2, sin traceback."""
    broken = tmp_path / "roto.yaml"
    broken.write_text(content, encoding="utf-8")

    _assert_invalid_arguments(monkeypatch, tmp_path, "--universe-files", str(broken))


def test_cli_sync_invalid_dates_in_the_universe_return_two(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Fechas mal escritas en el universo dan 2."""
    universe = _universe_yaml(tmp_path / "u.yaml", [("2026-13-01", "2026-14-01", ["AUSDT"])])

    _assert_invalid_arguments(monkeypatch, tmp_path, "--universe-files", str(universe))


def test_cli_sync_non_integer_warmup_is_rejected_by_argparse() -> None:
    """Un `--warmup-days` que no es entero lo rechaza argparse con salida 2."""
    with pytest.raises(SystemExit) as info:
        cli.main(["sync", "--warmup-days", "siete"])
    assert info.value.code == 2


def test_cli_sync_prints_nothing_to_stdout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    two_universes: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Sin `print` (D-11): todo va por el logger."""
    _install(monkeypatch, _Binance())

    _sync_cli(tmp_path / "cache", str(two_universes[0]))
    cli.main(["sync", "--universe-files", str(tmp_path / "nope.yaml")])

    assert capsys.readouterr().out == ""
