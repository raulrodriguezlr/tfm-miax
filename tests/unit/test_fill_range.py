"""Tests de `fill_range` y de los nombres de cache no latinos: cliente falso, sin red."""

from __future__ import annotations

import logging
import os
import unicodedata
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from miax.ingest import cache_bounds, load_klines, save_klines
from miax.ingest import resume as resume_module
from miax.ingest.http import RetriesExhaustedError
from miax.ingest.klines import KLINE_COLUMNS, KLINE_DTYPES, MAX_LIMIT
from miax.ingest.resume import fill_range
from miax.ingest.store import _SAFE_SEGMENT, _validate_segment
from miax.utils.io import write_parquet

HOUR = 3_600_000
INTERVAL = "1h"
SYMBOL = "BTCUSDT"
UNICODE_SYMBOL = "币安人生USDT"


def _ms(text: str) -> int:
    """Milisegundos UTC de una fecha o fecha-hora ISO."""
    return int(pd.Timestamp(text, tz="UTC").timestamp() * 1000)


def _month_of(ms: int) -> str:
    """Mes UTC `YYYY-MM` de un instante."""
    return pd.to_datetime(ms, unit="ms", utc=True).strftime("%Y-%m")


def _row(open_time: int, step: int) -> list[Any]:
    """Una vela cruda de Binance determinista a partir de su `open_time`."""
    price = str(100.0 + open_time % 7)
    close_time = open_time + step - 1
    return [
        open_time,
        price,
        price,
        price,
        price,
        "1.5",
        close_time,
        "150.0",
        10,
        "0.7",
        "70.0",
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
    """Cliente falso: sirve velas de `first` a `last` (open_time) y anota cada llamada.

    Con `fail_after=N` lanza `RetriesExhaustedError` en la llamada N + 1 (corte a mitad).
    """

    def __init__(
        self,
        first: int = 0,
        last: int = 10**15,
        *,
        step: int = HOUR,
        fail_after: int | None = None,
    ) -> None:
        self.first, self.last, self.step, self.fail_after = first, last, step, fail_after
        self.calls: list[tuple[int, int]] = []

    def get(self, url: str, params: dict[str, Any] | None = None, weight: int = 1) -> _Response:
        """Registra `(startTime, endTime)` y devuelve el bloque que serviria Binance."""
        assert params is not None
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise RetriesExhaustedError("corte simulado")
        start, end, limit = params["startTime"], params["endTime"], params["limit"]
        self.calls.append((start, end))
        lowest = max(start, self.first)
        lowest += -lowest % self.step  # primera vela alineada >= lowest
        opens = range(lowest, min(end, self.last) + 1, self.step)
        return _Response([_row(t, self.step) for t in list(opens)[:limit]])

    def close(self) -> None:
        """No hace nada."""


def _fill(
    client: _Binance,
    base_dir: Path,
    start: int,
    end: int,
    *,
    symbol: str = SYMBOL,
    limit: int = MAX_LIMIT,
) -> int:
    """Ejecuta `fill_range` de `INTERVAL` con el falso y `base_dir`."""
    return fill_range(
        symbol,
        INTERVAL,
        start,
        end,
        client=client,  # type: ignore[arg-type]
        base_dir=base_dir,
        limit=limit,
    )


def _seed(base_dir: Path, first: int, last: int, *, symbol: str = SYMBOL, step: int = HOUR) -> None:
    """Deja en la cache las velas de `first` a `last` (open_time, inclusivos)."""
    frame = pd.DataFrame(
        [_row(t, step) for t in range(first, last + 1, step)], columns=KLINE_COLUMNS
    )
    save_klines(frame.astype(KLINE_DTYPES), symbol, INTERVAL, base_dir=base_dir)


def _opens(base_dir: Path, symbol: str = SYMBOL) -> list[int]:
    """Los `open_time` cacheados de `symbol`, ordenados."""
    return load_klines(symbol, INTERVAL, base_dir=base_dir)["open_time"].tolist()


def _expected(first: int, last: int, step: int = HOUR) -> list[int]:
    """Los `open_time` de una cache perfecta entre `first` y `last`."""
    return list(range(first, last + 1, step))


def _is_continuous(opens: list[int]) -> bool:
    """Dice si la serie no tiene huecos ni duplicados."""
    return bool(opens) and opens == _expected(opens[0], opens[-1])


class _ListHandler(logging.Handler):
    """Handler que guarda los mensajes formateados."""

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Guarda el mensaje ya formateado."""
        self.messages.append(record.getMessage())


@pytest.fixture
def resume_logs() -> Iterator[_ListHandler]:
    """Engancha un handler al logger de `resume` mientras dura el test."""
    handler = _ListHandler()
    resume_module.logger.addHandler(handler)
    yield handler
    resume_module.logger.removeHandler(handler)


# --- nombres de cache no latinos ---------------------------------------------------------


def test_unicode_symbol_round_trips_through_the_cache(tmp_path: Path) -> None:
    """`币安人生USDT` se guarda, se lee y se acota en la cache con su nombre real."""
    first, last = _ms("2026-07-10"), _ms("2026-07-10 23:00")
    _seed(tmp_path, first, last, symbol=UNICODE_SYMBOL)

    assert (tmp_path / UNICODE_SYMBOL / INTERVAL / "2026-07.parquet").is_file()
    loaded = load_klines(UNICODE_SYMBOL, INTERVAL, base_dir=tmp_path)
    assert loaded["open_time"].tolist() == _expected(first, last)
    assert cache_bounds(UNICODE_SYMBOL, INTERVAL, base_dir=tmp_path) == (first, last)


def test_unicode_symbol_can_be_filled_backwards_and_forwards(tmp_path: Path) -> None:
    """`fill_range` descarga y amplia por los dos lados un simbolo no latino."""
    client = _Binance()
    start, end = _ms("2026-07-08"), _ms("2026-07-12 23:00")
    _seed(tmp_path, _ms("2026-07-10"), _ms("2026-07-10 23:00"), symbol=UNICODE_SYMBOL)

    new = _fill(client, tmp_path, start, end, symbol=UNICODE_SYMBOL)

    assert _opens(tmp_path, UNICODE_SYMBOL) == _expected(start, end)
    assert new == len(_expected(start, end)) - 24
    assert cache_bounds(UNICODE_SYMBOL, INTERVAL, base_dir=tmp_path) == (start, end)


def test_unicode_symbol_downloads_without_previous_cache(tmp_path: Path) -> None:
    """Sin cache previa tambien se descarga con el nombre no latino."""
    start, end = _ms("2026-07-10"), _ms("2026-07-10 23:00")

    assert _fill(_Binance(), tmp_path, start, end, symbol=UNICODE_SYMBOL) == 24
    assert _opens(tmp_path, UNICODE_SYMBOL) == _expected(start, end)


UNSAFE_NAMES = [
    "../x",
    "..\\x",
    "a/b",
    "a\\b",
    "a:b",
    ".",
    "..",
    "",
    "a b",
    "a\nb",
    "a\n",
    "a\tb",
    "a\x00b",
    "a*b",
    "a?b",
    'a"b',
    "a<b",
    "a>b",
    "a|b",
    "/",
    "C:",
    # Parecidos Unicode a un separador: no son letras ni digitos, asi que no entran por `\w`.
    "a∕b",  # DIVISION SLASH
    "a／b",  # FULLWIDTH SOLIDUS
    "a＼b",  # FULLWIDTH REVERSE SOLIDUS
    "a﹨b",  # SMALL REVERSE SOLIDUS
    "a∖b",  # SET MINUS
    "a：b",  # FULLWIDTH COLON
]


@pytest.mark.parametrize("name", UNSAFE_NAMES, ids=[repr(n) for n in UNSAFE_NAMES])
def test_unsafe_segment_is_rejected_everywhere(name: str, tmp_path: Path) -> None:
    """Separadores, `:`, `..`, vacio y espacios se rechazan en validacion, lectura y descarga."""
    with pytest.raises(ValueError, match="no valido"):
        _validate_segment(name)
    with pytest.raises(ValueError):
        cache_bounds(name, INTERVAL, base_dir=tmp_path)
    with pytest.raises(ValueError):
        load_klines(name, INTERVAL, base_dir=tmp_path)
    client = _Binance()
    with pytest.raises(ValueError):
        _fill(client, tmp_path, _ms("2026-07-10"), _ms("2026-07-11"), symbol=name)
    assert client.calls == []
    assert list(tmp_path.iterdir()) == []


def test_unsafe_symbol_cannot_be_saved(tmp_path: Path) -> None:
    """`save_klines` rechaza un simbolo con separador y no escribe nada fuera de la cache."""
    frame = pd.DataFrame([_row(0, HOUR)], columns=KLINE_COLUMNS).astype(KLINE_DTYPES)
    with pytest.raises(ValueError):
        save_klines(frame, "../escape", INTERVAL, base_dir=tmp_path / "cache")
    assert not (tmp_path / "escape").exists()
    assert not (tmp_path / "cache").exists()


def test_no_single_code_point_with_path_semantics_is_accepted() -> None:
    """Ningun caracter suelto que admite la regex es un separador de ruta ni `:`."""
    accepted = [chr(c) for c in range(0x110000) if _SAFE_SEGMENT.fullmatch(chr(c))]
    assert accepted  # la regex no esta vacia
    for char in accepted:
        assert char not in "/\\:"
        # normpath no cambia el nombre ni lo parte en tramos: no es un separador en este SO.
        assert os.path.normpath("a" + char + "b") == "a" + char + "b"
        # Solo letras, numeros, marcas, conectores, guion y punto (nada de simbolos ni espacios).
        assert unicodedata.category(char)[0] in {"L", "N", "M"} or char in "_-."


# --- fill_range --------------------------------------------------------------------------


def test_without_cache_downloads_the_whole_range(tmp_path: Path) -> None:
    """Sin cache baja el rango entero y lo guarda mes a mes."""
    start, end = _ms("2024-01-30"), _ms("2024-02-02 23:00")
    client = _Binance()

    new = _fill(client, tmp_path, start, end)

    assert new == len(_expected(start, end))
    assert client.calls[0][0] == start
    assert _opens(tmp_path) == _expected(start, end)
    assert sorted(p.name for p in (tmp_path / SYMBOL / INTERVAL).glob("*.parquet")) == [
        "2024-01.parquet",
        "2024-02.parquet",
    ]


def test_range_already_cached_makes_no_calls(tmp_path: Path) -> None:
    """Un rango que la cache ya cubre no llama al cliente y no cambia los datos."""
    first, last = _ms("2024-01-30"), _ms("2024-02-02 23:00")
    _seed(tmp_path, first, last)
    before = load_klines(SYMBOL, INTERVAL, base_dir=tmp_path)
    client = _Binance()

    assert _fill(client, tmp_path, _ms("2024-01-31"), _ms("2024-02-01")) == 0
    assert _fill(client, tmp_path, first, last) == 0
    assert client.calls == []
    pd.testing.assert_frame_equal(load_klines(SYMBOL, INTERVAL, base_dir=tmp_path), before)


def test_backwards_into_a_partial_first_month_is_exact(tmp_path: Path) -> None:
    """Con la cache empezando a mitad de febrero, bajar enero y el inicio de febrero no trunca."""
    first, last = _ms("2024-02-10"), _ms("2024-02-13 23:00")
    start = _ms("2024-01-29")
    _seed(tmp_path, first, last)
    client = _Binance()

    new = _fill(client, tmp_path, start, last)

    assert new == len(_expected(start, first - HOUR))
    assert _opens(tmp_path) == _expected(start, last)
    # Nada de lo ya cacheado se vuelve a pedir: todo lo pedido acaba antes de la primera vela.
    assert all(end < first for _, end in client.calls)
    assert min(s for s, _ in client.calls) == start
    # El mes parcial conserva lo cacheado y suma lo nuevo (el fichero de febrero esta entero).
    february = pd.read_parquet(tmp_path / SYMBOL / INTERVAL / "2024-02.parquet")
    assert february["open_time"].tolist() == _expected(_ms("2024-02-01"), last)
    assert february["open_time"].is_unique


def test_both_sides_at_once(tmp_path: Path) -> None:
    """Amplia por detras y por delante en la misma llamada, sin tocar el centro."""
    first, last = _ms("2024-02-10"), _ms("2024-02-11 23:00")
    start, end = _ms("2024-01-25"), _ms("2024-03-05 23:00")
    _seed(tmp_path, first, last)
    client = _Binance()

    new = _fill(client, tmp_path, start, end)

    assert new == len(_expected(start, end)) - len(_expected(first, last))
    assert _opens(tmp_path) == _expected(start, end)
    # Ninguna peticion solapa con lo cacheado: o acaba antes de `first` o empieza tras `last`.
    assert all(e < first or s > last for s, e in client.calls)


def test_forwards_into_a_partial_month_keeps_the_cached_part(tmp_path: Path) -> None:
    """Hacia delante, el mes donde acaba la cache se fusiona y no se trunca."""
    first, last = _ms("2024-01-31"), _ms("2024-02-02 05:00")
    end = _ms("2024-03-02 23:00")
    _seed(tmp_path, first, last)

    _fill(_Binance(), tmp_path, first, end)

    assert _opens(tmp_path) == _expected(first, end)


def test_first_cached_month_exists_but_is_empty(tmp_path: Path) -> None:
    """Un fichero del primer mes vacio no cuenta como cache: los limites salen del primero lleno."""
    first, last = _ms("2024-02-10"), _ms("2024-02-11 23:00")
    _seed(tmp_path, first, last)
    empty = pd.DataFrame({c: pd.Series(dtype=KLINE_DTYPES[c]) for c in KLINE_COLUMNS})
    write_parquet(empty, tmp_path / SYMBOL / INTERVAL / "2024-01.parquet")
    assert cache_bounds(SYMBOL, INTERVAL, base_dir=tmp_path) == (first, last)
    start = _ms("2023-12-31")

    new = _fill(_Binance(), tmp_path, start, last)

    assert new == len(_expected(start, first - HOUR))
    assert _opens(tmp_path) == _expected(start, last)
    assert cache_bounds(SYMBOL, INTERVAL, base_dir=tmp_path) == (start, last)


def test_empty_backwards_keeps_the_cache_and_stops_at_the_first_empty_block(
    tmp_path: Path, resume_logs: _ListHandler
) -> None:
    """Si Binance no tiene nada antes de lo cacheado, 1 sola llamada y la cache queda intacta."""
    first, last = _ms("2024-02-10"), _ms("2024-02-11 23:00")
    _seed(tmp_path, first, last)
    before = load_klines(SYMBOL, INTERVAL, base_dir=tmp_path)
    client = _Binance(first=first)  # el par empieza exactamente donde empieza la cache

    # Cinco meses antes de lo cacheado: sin parar en el primer vacio serian 6 bloques.
    assert _fill(client, tmp_path, _ms("2023-09-01"), last) == 0

    assert len(client.calls) == 1
    pd.testing.assert_frame_equal(load_klines(SYMBOL, INTERVAL, base_dir=tmp_path), before)
    assert any("sin velas en Binance antes de lo cacheado" in m for m in resume_logs.messages)


def test_backwards_stops_at_the_first_empty_month_after_the_listing(tmp_path: Path) -> None:
    """Un par listado a mitad de camino baja hasta su listado y no pide los meses anteriores."""
    first, last = _ms("2024-02-10"), _ms("2024-02-11 23:00")
    listed = _ms("2024-01-20")
    _seed(tmp_path, first, last)
    client = _Binance(first=listed)

    new = _fill(client, tmp_path, _ms("2023-01-01"), last)

    assert _opens(tmp_path) == _expected(listed, last)
    assert new == len(_expected(listed, first - HOUR))
    # Febrero y enero (con velas) y diciembre (vacio, y ahi se para): nada de noviembre atras.
    assert [_month_of(s) for s, _ in client.calls] == [
        "2024-02",
        "2024-02",
        "2024-01",
        "2024-01",
        "2023-12",
    ]


def test_end_one_ms_before_a_candle_opens_makes_no_calls(tmp_path: Path) -> None:
    """Un fin como `period_end - 1 ms` que no abre vela no provoca ninguna peticion."""
    first, last = _ms("2024-02-10"), _ms("2024-02-11 23:00")
    _seed(tmp_path, first, last)
    client = _Binance()

    # Tras la ultima vela cacheada solo caben 59 min 59 s 999 ms: ninguna vela abre ahi.
    assert _fill(client, tmp_path, first, last + HOUR - 1) == 0
    assert client.calls == []


def test_end_before_the_next_slot_without_cache_makes_no_calls(tmp_path: Path) -> None:
    """Sin cache, un rango que no contiene ninguna vela no llama al cliente."""
    start = _ms("2024-02-10 00:00:00.001")
    client = _Binance()

    assert _fill(client, tmp_path, start, start + HOUR - 3) == 0
    assert client.calls == []


def test_start_after_end_does_nothing(tmp_path: Path) -> None:
    """`start > end` devuelve 0 sin llamadas, con y sin cache, y no crea carpetas."""
    client = _Binance()
    assert _fill(client, tmp_path, 10 * HOUR, 5 * HOUR) == 0
    assert list(tmp_path.iterdir()) == []

    _seed(tmp_path, _ms("2024-02-10"), _ms("2024-02-11"))
    before = load_klines(SYMBOL, INTERVAL, base_dir=tmp_path)
    assert _fill(client, tmp_path, _ms("2024-03-01"), _ms("2024-02-01")) == 0
    assert client.calls == []
    pd.testing.assert_frame_equal(load_klines(SYMBOL, INTERVAL, base_dir=tmp_path), before)


def test_repeating_a_fill_makes_no_calls(tmp_path: Path) -> None:
    """Idempotencia: repetir la misma peticion tras una descarga completa da 0 y no llama."""
    start, end = _ms("2024-01-20"), _ms("2024-03-05 23:00")
    _seed(tmp_path, _ms("2024-02-10"), _ms("2024-02-11"))
    assert _fill(_Binance(), tmp_path, start, end) > 0
    snapshot = load_klines(SYMBOL, INTERVAL, base_dir=tmp_path)
    again = _Binance()

    assert _fill(again, tmp_path, start, end) == 0
    assert again.calls == []
    pd.testing.assert_frame_equal(load_klines(SYMBOL, INTERVAL, base_dir=tmp_path), snapshot)


def test_never_listed_pair_returns_zero_without_files(
    tmp_path: Path, resume_logs: _ListHandler
) -> None:
    """Un par que Binance no tiene (BCC): 0, aviso, ningun fichero y ninguna excepcion."""
    client = _Binance(first=10**14, last=10**14 + HOUR)

    assert _fill(client, tmp_path, _ms("2024-01-01"), _ms("2024-03-31 23:00")) == 0

    assert cache_bounds(SYMBOL, INTERVAL, base_dir=tmp_path) is None
    assert not (tmp_path / SYMBOL).exists()
    assert any("sin velas en Binance para su rango" in m for m in resume_logs.messages)
    # Sin cache se recorre mes a mes (1 llamada vacia por mes), nunca mas de una por mes.
    assert len(client.calls) == 3


def test_range_entirely_before_the_cache_joins_it(tmp_path: Path) -> None:
    """Un rango anterior a la cache se baja hasta pegarse a ella: la cache sigue continua."""
    first, last = _ms("2024-03-10"), _ms("2024-03-11 23:00")
    start, end = _ms("2024-02-01"), _ms("2024-02-02 23:00")
    _seed(tmp_path, first, last)

    _fill(_Binance(), tmp_path, start, end)

    assert _opens(tmp_path) == _expected(start, last)


def test_range_entirely_after_the_cache_continues_from_the_last_candle(tmp_path: Path) -> None:
    """Un rango posterior a la cache arranca en la ultima vela + 1, no en `start`: sin huecos."""
    first, last = _ms("2024-02-01"), _ms("2024-02-02 23:00")
    end = _ms("2024-02-11 23:00")
    _seed(tmp_path, first, last)
    client = _Binance()

    _fill(client, tmp_path, _ms("2024-02-10"), end)

    assert client.calls[0][0] == last + 1
    assert _opens(tmp_path) == _expected(first, end)


def test_download_is_split_by_calendar_month(tmp_path: Path) -> None:
    """Un rango de 3 meses guarda 3 ficheros y ninguna peticion cruza un mes."""
    start, end = _ms("2024-01-01"), _ms("2024-03-31 23:00")
    client = _Binance()

    new = _fill(client, tmp_path, start, end)

    assert new == len(_expected(start, end))
    assert sorted(p.name for p in (tmp_path / SYMBOL / INTERVAL).glob("*.parquet")) == [
        "2024-01.parquet",
        "2024-02.parquet",
        "2024-03.parquet",
    ]
    # Cada peticion cae entera dentro de un unico mes UTC.
    assert all(_month_of(s) == _month_of(e) for s, e in client.calls)
    # El tope de cada bloque es el ultimo ms de su mes (o el fin pedido, en el ultimo).
    assert {e for _, e in client.calls} == {_ms("2024-02-01") - 1, _ms("2024-03-01") - 1, end}
    assert _opens(tmp_path) == _expected(start, end)
    for month in ("2024-01", "2024-02", "2024-03"):
        stored = pd.read_parquet(tmp_path / SYMBOL / INTERVAL / f"{month}.parquet")
        assert {_month_of(t) for t in stored["open_time"]} == {month}


def test_backwards_download_is_split_by_calendar_month_from_latest_to_earliest(
    tmp_path: Path,
) -> None:
    """Hacia atras tambien se va mes a mes y del mas reciente al mas antiguo."""
    first, last = _ms("2024-04-10"), _ms("2024-04-11 23:00")
    start = _ms("2024-01-05")
    _seed(tmp_path, first, last)
    client = _Binance()

    _fill(client, tmp_path, start, last)

    assert all(_month_of(s) == _month_of(e) for s, e in client.calls)
    # Abril, marzo, febrero y enero, en ese orden (dentro de un mes las paginas van hacia delante).
    assert list(dict.fromkeys(_month_of(s) for s, _ in client.calls)) == [
        "2024-04",
        "2024-03",
        "2024-02",
        "2024-01",
    ]
    assert _opens(tmp_path) == _expected(start, last)


def test_1m_candles_across_a_month_boundary(tmp_path: Path) -> None:
    """Con velas de 1m el cruce de mes tambien es exacto (varias paginas por bloque)."""
    minute = 60_000
    client = _Binance(step=minute)
    start, end = _ms("2024-01-31 22:00"), _ms("2024-02-01 02:00")

    new = fill_range(
        SYMBOL,
        "1m",
        start,
        end,
        client=client,  # type: ignore[arg-type]
        base_dir=tmp_path,
        limit=100,
    )

    expected = list(range(start, end + 1, minute))
    assert new == len(expected)
    assert load_klines(SYMBOL, "1m", base_dir=tmp_path)["open_time"].tolist() == expected
    assert all(_month_of(s) == _month_of(e) for s, e in client.calls)
    # Con limit=100 cada mes necesita varias paginas: el resultado no depende del tamano de pagina.
    assert len(client.calls) > 2


# --- cortes a mitad de descarga --------------------------------------------------------


@pytest.mark.parametrize("cut_after", [0, 1, 5, 8, 10, 15, 16, 22])
def test_forward_cut_leaves_a_continuous_cache_and_resumes_exactly(
    tmp_path: Path, cut_after: int
) -> None:
    """Un corte hacia delante deja la cache continua y al reanudar sale el rango exacto."""
    start, end = _ms("2024-01-01"), _ms("2024-03-31 23:00")
    broken = _Binance(fail_after=cut_after)

    with pytest.raises(RetriesExhaustedError):
        _fill(broken, tmp_path, start, end, limit=100)

    opens = _opens(tmp_path)
    if opens:
        assert _is_continuous(opens)
        assert opens[0] == start
    resumed = _Binance()
    _fill(resumed, tmp_path, start, end, limit=100)
    assert _opens(tmp_path) == _expected(start, end)
    if opens:
        # Reanuda tras la ultima vela cacheada: no repite nada de lo guardado.
        assert all(s > opens[-1] for s, _ in resumed.calls)


@pytest.mark.parametrize("cut_after", [0, 1, 3, 4, 7, 9, 12, 20, 25])
def test_backward_cut_leaves_a_continuous_cache_and_resumes_exactly(
    tmp_path: Path, cut_after: int
) -> None:
    """Un corte hacia atras deja la cache continua (empieza mas tarde) y reanuda el rango exacto."""
    first, last = _ms("2024-06-10"), _ms("2024-06-11 23:00")
    start = _ms("2024-03-01")
    _seed(tmp_path, first, last)
    broken = _Binance(fail_after=cut_after)

    with pytest.raises(RetriesExhaustedError):
        _fill(broken, tmp_path, start, last, limit=100)

    opens = _opens(tmp_path)
    assert _is_continuous(opens)
    assert opens[-1] == last
    assert opens[0] <= first
    resumed = _Binance()
    new = _fill(resumed, tmp_path, start, last, limit=100)
    assert _opens(tmp_path) == _expected(start, last)
    # Solo se pide lo que faltaba por detras de lo que habia al cortar.
    assert all(e < opens[0] for _, e in resumed.calls)
    assert new == len(_expected(start, opens[0] - HOUR))


def test_cut_in_the_middle_of_a_block_saves_nothing_of_that_block(tmp_path: Path) -> None:
    """Un corte entre paginas de un mes no guarda medio mes: o entra el bloque entero o ninguno."""
    start, end = _ms("2024-01-01"), _ms("2024-01-31 23:00")
    broken = _Binance(fail_after=3)  # 744 velas con limit=100: se corta en la cuarta pagina

    with pytest.raises(RetriesExhaustedError):
        _fill(broken, tmp_path, start, end, limit=100)

    assert cache_bounds(SYMBOL, INTERVAL, base_dir=tmp_path) is None
