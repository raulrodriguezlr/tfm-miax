"""Tests del remuestreo de 1m a 5m, 15m y 1h: tablas sinteticas y caches en `tmp_path`, sin red.

Las causas se escriben como cadenas, un caracter por minuto: V valid, T no_trades, N no_data.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from miax.features import (
    GapCause,
    bar_factor,
    build_time_grid,
    horizon_returns_from_cache,
    resample_fields,
    resample_from_cache,
    resample_validity,
    returns_from_cache,
    validity_mask,
)
from miax.ingest import save_klines
from miax.ingest.klines import KLINE_COLUMNS, KLINE_DTYPES

MIN_MS = 60_000
T0 = 1_767_225_600_000  # 2026-01-01T00:00:00Z, multiplo de 1h
LETTERS = {"V": GapCause.VALID, "T": GapCause.NO_TRADES, "N": GapCause.NO_DATA}


def _valid(specs: Mapping[str, str], start: int = T0) -> pd.DataFrame:
    """Mascara de validez de 1m `tiempo x simbolo` desde una cadena de causas por simbolo."""
    n = len(next(iter(specs.values())))
    index = build_time_grid(start, start + n * MIN_MS, "1m")
    data = {s: np.array([int(LETTERS[c]) for c in spec], dtype="int8") for s, spec in specs.items()}
    return validity_mask(pd.DataFrame(data, index=index))


def _frame(values: Mapping[str, list[float]], start: int = T0) -> pd.DataFrame:
    """Tabla `tiempo x simbolo` de 1m desde `start`."""
    n = len(next(iter(values.values())))
    index = build_time_grid(start, start + n * MIN_MS, "1m")
    return pd.DataFrame({s: np.asarray(v, dtype="float64") for s, v in values.items()}, index=index)


def _candles(minutes: list[int], close: list[float], volume: float = 1.0) -> pd.DataFrame:
    """Klines de 1m en los minutos dados (desde T0) con esos cierres y operaciones."""
    n = len(minutes)
    open_times = [T0 + k * MIN_MS for k in minutes]
    data = {
        "open_time": open_times,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": [volume] * n,
        "close_time": [t + MIN_MS - 1 for t in open_times],
        "quote_asset_volume": [10.0] * n,
        "num_trades": [7] * n,
        "taker_buy_base": [0.5] * n,
        "taker_buy_quote": [5.0] * n,
        "ignore": [0.0] * n,
    }
    return pd.DataFrame(data, columns=KLINE_COLUMNS).astype(KLINE_DTYPES)


def _cache(base_dir: Path, symbol: str, minutes: list[int], close: list[float]) -> None:
    """Guarda en `base_dir` una cache de 1m de `symbol`."""
    save_klines(_candles(minutes, close), symbol, "1m", base_dir=base_dir)


def _prices(n: int, seed: int = 0) -> list[float]:
    """Cierres deterministas, positivos y no triviales."""
    rng = np.random.default_rng(seed)
    return list(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n))))


# --- bar_factor ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("target", "factor"), [("1m", 1), ("5m", 5), ("15m", 15), ("1h", 60), ("60m", 60)]
)
def test_bar_factor_integer_multiples(target: str, factor: int) -> None:
    """Los multiplos enteros de 1m dan su factor; `1h` y `60m` son el mismo paso."""
    assert bar_factor("1m", target) == factor


def test_bar_factor_hour_and_sixty_minutes_are_equal() -> None:
    """`1h` y `60m` dan el mismo factor."""
    assert bar_factor("1m", "1h") == bar_factor("1m", "60m")


@pytest.mark.parametrize(
    ("base", "target"), [("5m", "1m"), ("5m", "7m"), ("1m", "1M"), ("1m", "0m"), ("1m", "1w")]
)
def test_bar_factor_rejects_invalid_pairs(base: str, target: str) -> None:
    """Objetivo mas corto, no multiplo, sin paso fijo o semanal es `ValueError`."""
    with pytest.raises(ValueError):
        bar_factor(base, target)


# --- Validez de barra ---------------------------------------------------------------------------


def _bar_valid(spec: str, **kwargs: float) -> list[bool]:
    """Validez de las barras de 5m de una sola serie."""
    out = resample_validity(_valid({"A": spec}), "1m", "5m", **kwargs)
    return out["A"].tolist()


def test_all_valid_bar_is_valid() -> None:
    """Una barra con todos los minutos validos es valida."""
    assert _bar_valid("VVVVV") == [True]


def test_invalid_last_minute_makes_bar_invalid() -> None:
    """Con el ultimo minuto invalido la barra es invalida aunque el resto sea valido."""
    assert _bar_valid("VVVVT") == [False]
    assert _bar_valid("VVVVN", min_valid_share=0.1) == [False]


def test_share_threshold_is_inclusive() -> None:
    """Exactamente 4 de 5 (0,8) es valida y 3 de 5 no."""
    assert _bar_valid("TVVVV") == [True]
    assert _bar_valid("TTVVV") == [False]


def test_strict_share_requires_every_minute() -> None:
    """Con `min_valid_share=1.0` basta un minuto invalido para tumbar la barra."""
    assert _bar_valid("TVVVV", min_valid_share=1.0) == [False]
    assert _bar_valid("VVVVV", min_valid_share=1.0) == [True]


def test_low_share_is_close_only_rule() -> None:
    """Con un umbral bajo solo manda que el ultimo minuto sea valido."""
    assert _bar_valid("TTTTV", min_valid_share=0.2) == [True]


@pytest.mark.parametrize("share", [0.0, -0.1, 1.5])
def test_share_out_of_range_is_rejected(share: float) -> None:
    """Un `min_valid_share` fuera de `(0, 1]` es `ValueError`."""
    with pytest.raises(ValueError):
        resample_validity(_valid({"A": "VVVVV"}), "1m", "5m", min_valid_share=share)


def test_validity_is_computed_per_symbol_and_per_bar() -> None:
    """Cada simbolo y cada barra se evaluan por separado; el indice es la apertura de la barra."""
    out = resample_validity(_valid({"A": "VVVVVTTTTT", "B": "TTTTTVVVVV"}), "1m", "5m")
    assert out["A"].tolist() == [True, False]
    assert out["B"].tolist() == [False, True]
    assert out.index.equals(build_time_grid(T0, T0 + 10 * MIN_MS, "5m"))
    assert out.index.freq is None
    assert str(out.index.dtype) == "datetime64[ms, UTC]"
    assert all(pd.api.types.is_bool_dtype(d) for d in out.dtypes)


def test_non_contiguous_index_is_rejected() -> None:
    """Un indice con un salto no es la rejilla contigua del origen."""
    valid = _valid({"A": "V" * 10})
    with pytest.raises(ValueError, match="contigua"):
        resample_validity(valid.drop(valid.index[3]).iloc[:9], "1m", "5m")


def test_index_not_starting_at_multiple_is_rejected() -> None:
    """Un indice que no empieza en multiplo del objetivo mezclaria horizontes."""
    with pytest.raises(ValueError, match="multiplo"):
        resample_validity(_valid({"A": "V" * 10}, start=T0 + MIN_MS), "1m", "5m")


def test_length_not_multiple_is_rejected() -> None:
    """Una longitud que no es multiplo del factor deja una barra incompleta."""
    with pytest.raises(ValueError, match="multiplo"):
        resample_validity(_valid({"A": "V" * 7}), "1m", "5m")


def test_validity_requires_bool() -> None:
    """Un `valid` que no es bool es `ValueError`."""
    with pytest.raises(ValueError, match="bool"):
        resample_validity(_frame({"A": [1.0] * 5}), "1m", "5m")


# --- Agregacion ---------------------------------------------------------------------------------


def _fields(n: int = 5) -> dict[str, pd.DataFrame]:
    """Campos sinteticos de `n` minutos, uno por tipo de agregacion."""
    base = np.arange(1, n + 1, dtype="float64")
    return {
        "open": _frame({"A": list(base)}),
        "high": _frame({"A": list(base * 10)}),
        "low": _frame({"A": list(100 - base)}),
        "close": _frame({"A": list(base + 0.5)}),
        "volume": _frame({"A": list(base)}),
        "num_trades": _frame({"A": [2.0] * n}),
    }


def test_each_field_uses_its_aggregation() -> None:
    """`open` primero, `high` maximo, `low` minimo, `close` ultimo y volumenes/trades suman."""
    out, valid = resample_fields(_fields(), _valid({"A": "VVVVV"}), "1m", "5m")
    assert valid["A"].tolist() == [True]
    assert out["open"]["A"].tolist() == [1.0]
    assert out["high"]["A"].tolist() == [50.0]
    assert out["low"]["A"].tolist() == [95.0]
    assert out["close"]["A"].tolist() == [5.5]
    assert out["volume"]["A"].tolist() == [15.0]
    assert out["num_trades"]["A"].tolist() == [10.0]


def test_internal_nan_is_ignored_without_rescaling() -> None:
    """Un `NaN` interno no cuenta: la suma es la de lo observado, sin reescalar."""
    fields = _fields()
    for frame in fields.values():
        frame.iloc[1, 0] = np.nan
    out, valid = resample_fields(fields, _valid({"A": "VTVVV"}), "1m", "5m")
    assert valid["A"].tolist() == [True]
    assert out["volume"]["A"].tolist() == [13.0]
    assert out["num_trades"]["A"].tolist() == [8.0]
    assert out["high"]["A"].tolist() == [50.0]
    assert out["low"]["A"].tolist() == [95.0]


def test_open_is_first_observed_minute() -> None:
    """Si falta el primer minuto, `open` es el de la primera vela observada."""
    fields = _fields()
    fields["open"].iloc[0, 0] = np.nan
    out, _ = resample_fields(fields, _valid({"A": "TVVVV"}), "1m", "5m")
    assert out["open"]["A"].tolist() == [2.0]


def _ohlc(
    open_: list[float], high: list[float], low: list[float], close: list[float]
) -> dict[str, pd.DataFrame]:
    """Campos de precio de una sola serie `A` desde listas de 1m."""
    return {
        "open": _frame({"A": open_}),
        "high": _frame({"A": high}),
        "low": _frame({"A": low}),
        "close": _frame({"A": close}),
    }


def test_no_trades_minute_at_start_does_not_set_open_or_low() -> None:
    """Un minuto sin operaciones al inicio trae el precio anterior: no es `open` ni `low`."""
    fields = _ohlc(
        [100.0, 101.0, 101.0, 101.0, 101.0],
        [100.0, 102.0, 103.0, 102.0, 102.0],
        [100.0, 101.0, 101.5, 101.2, 101.1],
        [100.0, 101.5, 102.0, 101.5, 101.5],
    )
    out, valid = resample_fields(fields, _valid({"A": "TVVVV"}), "1m", "5m")
    assert valid["A"].tolist() == [True]
    assert out["open"]["A"].tolist() == [101.0]
    assert out["low"]["A"].tolist() == [101.0]
    assert out["high"]["A"].tolist() == [103.0]
    assert out["close"]["A"].tolist() == [101.5]


def test_no_trades_minute_in_middle_does_not_enter_high_or_low() -> None:
    """Un minuto sin operaciones interno repite un precio rancio que no entra en `high` ni `low`."""
    fields = _ohlc(
        [100.0, 101.0, 500.0, 102.0, 103.0],
        [102.0, 103.0, 500.0, 104.0, 105.0],
        [99.0, 100.0, 500.0, 101.0, 102.0],
        [101.0, 102.0, 500.0, 103.0, 104.0],
    )
    fields["low"].iloc[2, 0] = 50.0  # precio rancio por debajo de todos los validos
    fields["high"].iloc[2, 0] = 900.0  # y otro por encima
    out, valid = resample_fields(fields, _valid({"A": "VVTVV"}), "1m", "5m")
    assert valid["A"].tolist() == [True]
    assert out["open"]["A"].tolist() == [100.0]
    assert out["high"]["A"].tolist() == [105.0]
    assert out["low"]["A"].tolist() == [99.0]
    assert out["close"]["A"].tolist() == [104.0]


def test_no_trades_minute_at_end_invalidates_bar() -> None:
    """Un minuto sin operaciones al final invalida la barra: todo `NaN` aunque tenga valores."""
    fields = _ohlc(
        [100.0, 101.0, 102.0, 103.0, 104.0],
        [101.0, 102.0, 103.0, 104.0, 105.0],
        [99.0, 100.0, 101.0, 102.0, 103.0],
        [100.5, 101.5, 102.5, 103.5, 103.5],
    )
    out, valid = resample_fields(fields, _valid({"A": "VVVVT"}), "1m", "5m")
    assert valid["A"].tolist() == [False]
    for name, frame in out.items():
        assert np.isnan(frame["A"].iloc[0]), name


def test_invalid_bar_is_nan_never_zero() -> None:
    """Una barra invalida deja `NaN` en todos los campos, nunca 0."""
    fields = _fields(10)
    out, valid = resample_fields(fields, _valid({"A": "VVVVVTTTTT"}), "1m", "5m")
    assert valid["A"].tolist() == [True, False]
    for name, frame in out.items():
        assert np.isnan(frame["A"].iloc[1]), name
        assert not np.isnan(frame["A"].iloc[0]), name
        assert (frame["A"].dropna() != 0).all(), name


def test_single_bar_and_all_invalid_bar() -> None:
    """Una sola barra se agrega bien y una barra con todo invalido da `NaN` sin avisos."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out, valid = resample_fields(_fields(), _valid({"A": "VVVVV"}), "1m", "5m")
        assert out["volume"].shape == (1, 1)
        assert out["volume"]["A"].tolist() == [15.0]
        out, valid = resample_fields(_fields(), _valid({"A": "TTTTT"}), "1m", "5m")
    assert valid["A"].tolist() == [False]
    for name, frame in out.items():
        assert np.isnan(frame["A"].iloc[0]), name


def test_strict_share_end_to_end() -> None:
    """Con `min_valid_share=1.0` la barra sale con todos los minutos validos o es `NaN`."""
    fields = _fields(10)
    out, valid = resample_fields(
        fields, _valid({"A": "VVVVVTVVVV"}), "1m", "5m", min_valid_share=1.0
    )
    assert valid["A"].tolist() == [True, False]
    assert out["volume"]["A"].iloc[0] == 15.0
    assert np.isnan(out["volume"]["A"].iloc[1])
    out, valid = resample_fields(
        fields, _valid({"A": "VVVVVVVVVV"}), "1m", "5m", min_valid_share=1.0
    )
    assert valid["A"].tolist() == [True, True]
    assert out["volume"]["A"].tolist() == [15.0, 40.0]


def test_four_of_five_boundary_with_share_08() -> None:
    """Con 0,8, 4 de 5 minutos validos da barra valida y su suma no incluye el minuto invalido."""
    fields = _fields()
    out, valid = resample_fields(fields, _valid({"A": "TVVVV"}), "1m", "5m", min_valid_share=0.8)
    assert valid["A"].tolist() == [True]
    assert out["open"]["A"].tolist() == [2.0]
    assert out["volume"]["A"].tolist() == [15.0]
    out, valid = resample_fields(fields, _valid({"A": "TTVVV"}), "1m", "5m", min_valid_share=0.8)
    assert valid["A"].tolist() == [False]


def test_all_nan_bar_makes_no_warnings() -> None:
    """Una barra entera de `NaN` no produce avisos de `numpy`."""
    fields = {
        "high": _frame({"A": [np.nan] * 5 + [1.0] * 5}),
        "volume": _frame({"A": [np.nan] * 5 + [1.0] * 5}),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out, _ = resample_fields(fields, _valid({"A": "NNNNNVVVVV"}), "1m", "5m")
    assert np.isnan(out["high"]["A"].iloc[0])
    assert np.isnan(out["volume"]["A"].iloc[0])
    assert out["volume"]["A"].iloc[1] == 5.0


def test_output_shape_dtype_and_index() -> None:
    """La salida es `float64`, con una fila por barra y el indice de aperturas sin `freq`."""
    fields = _fields(10)
    out, valid = resample_fields(fields, _valid({"A": "V" * 10}), "1m", "5m")
    for frame in out.values():
        assert frame.index.equals(valid.index)
        assert frame.index.freq is None
        assert str(frame.index.dtype) == "datetime64[ms, UTC]"
        assert (frame.dtypes == "float64").all()
        assert list(frame.columns) == ["A"]


def test_unknown_field_is_rejected() -> None:
    """Un campo que no se sabe agregar es `ValueError`."""
    with pytest.raises(ValueError, match="no remuestreables"):
        resample_fields({"ignore": _frame({"A": [1.0] * 5})}, _valid({"A": "V" * 5}), "1m", "5m")


def test_field_must_match_valid_index_and_be_float() -> None:
    """Un campo con otro indice o que no es float es `ValueError`."""
    valid = _valid({"A": "V" * 5})
    with pytest.raises(ValueError, match="coincidir"):
        resample_fields({"close": _frame({"B": [1.0] * 5})}, valid, "1m", "5m")
    with pytest.raises(ValueError, match="float"):
        resample_fields({"close": _frame({"A": [1.0] * 5}).astype("int64")}, valid, "1m", "5m")


# --- Mismo origen: aditividad entre horizontes --------------------------------------------------


def test_horizon_returns_are_additive_across_horizons(tmp_path: Path) -> None:
    """El retorno de 60 min es la suma de 12 de 5 min y de 60 de 1 min; el de 15 min, de 3 de 5."""
    n = 6 * 60
    _cache(tmp_path, "A", list(range(n)), _prices(n))
    start, end = T0 + 120 * MIN_MS, T0 + n * MIN_MS
    r1 = horizon_returns_from_cache(["A"], "1m", start, end, base_dir=tmp_path)["A"]
    r5 = horizon_returns_from_cache(["A"], "5m", start, end, base_dir=tmp_path)["A"]
    r15 = horizon_returns_from_cache(["A"], "15m", start, end, base_dir=tmp_path)["A"]
    r60 = horizon_returns_from_cache(["A"], "1h", start, end, base_dir=tmp_path)["A"]
    assert len(r60) == 4 and len(r15) == 16 and len(r5) == 48 and len(r1) == 240
    # La fila k de un horizonte H esta en start + k*H y suma los retornos menores que acaban en
    # (start + (k-1)*H, start + k*H]; la fila 0 necesita retornos previos al rango, de ahi `k >= 1`.
    for k in range(1, len(r60)):
        assert np.isclose(r60.iloc[k], r5.iloc[12 * k - 11 : 12 * k + 1].sum(), rtol=0, atol=1e-12)
        assert np.isclose(r60.iloc[k], r1.iloc[60 * k - 59 : 60 * k + 1].sum(), rtol=0, atol=1e-12)
    for k in range(1, len(r15)):
        assert np.isclose(r15.iloc[k], r5.iloc[3 * k - 2 : 3 * k + 1].sum(), rtol=0, atol=1e-12)


def test_sixty_minutes_equals_one_hour(tmp_path: Path) -> None:
    """Pedir `60m` o `1h` da exactamente lo mismo."""
    n = 4 * 60
    _cache(tmp_path, "A", list(range(n)), _prices(n))
    args = (["A"],)
    a = horizon_returns_from_cache(
        *args, "1h", T0 + 60 * MIN_MS, T0 + n * MIN_MS, base_dir=tmp_path
    )
    b = horizon_returns_from_cache(
        *args, "60m", T0 + 60 * MIN_MS, T0 + n * MIN_MS, base_dir=tmp_path
    )
    assert_frame_equal(a, b)


# --- Sin fuga -----------------------------------------------------------------------------------


def test_changing_later_minutes_does_not_alter_a_bar() -> None:
    """Cambiar los minutos posteriores al final de una barra no altera esa barra."""
    names = ("open", "high", "low", "close", "volume", "num_trades")
    spec = "V" * 10
    base = {n: _frame({"A": _prices(10, seed=i)}) for i, n in enumerate(names)}
    changed = {n: f.copy() for n, f in base.items()}
    for frame in changed.values():
        frame.iloc[5:, 0] = frame.iloc[5:, 0] * 3.0 + 1.0
    valid_changed = _valid({"A": "VVVVVTTVTT"})
    out_a, valid_a = resample_fields(base, _valid({"A": spec}), "1m", "5m")
    out_b, valid_b = resample_fields(changed, valid_changed, "1m", "5m")
    assert valid_a["A"].iloc[0] == valid_b["A"].iloc[0]
    for name in names:
        assert out_a[name]["A"].iloc[0] == out_b[name]["A"].iloc[0], name


def test_changing_future_candles_does_not_alter_past_returns(tmp_path: Path) -> None:
    """Los retornos conocidos antes de un instante no cambian al cambiar velas posteriores."""
    n = 3 * 60
    prices = _prices(n)
    other = prices[:90] + [p * 2 for p in prices[90:]]
    dir_a, dir_b = tmp_path / "a", tmp_path / "b"
    _cache(dir_a, "A", list(range(n)), prices)
    _cache(dir_b, "A", list(range(n)), other)
    start, end = T0 + 15 * MIN_MS, T0 + n * MIN_MS
    ra = horizon_returns_from_cache(["A"], "15m", start, end, base_dir=dir_a)
    rb = horizon_returns_from_cache(["A"], "15m", start, end, base_dir=dir_b)
    # Las velas cambiadas empiezan en el minuto 90: lo conocido hasta T0 + 90 min no varia.
    cutoff = pd.Timestamp(T0 + 90 * MIN_MS, unit="ms", tz="UTC")
    assert_frame_equal(ra[ra.index <= cutoff], rb[rb.index <= cutoff])
    assert not ra.equals(rb)


def test_returns_are_indexed_by_open_time_plus_horizon(tmp_path: Path) -> None:
    """La fila `tau` es el retorno de la barra que abre en `tau - H`."""
    n = 60
    prices = _prices(n)
    _cache(tmp_path, "A", list(range(n)), prices)
    out = horizon_returns_from_cache(
        ["A"], "5m", T0 + 10 * MIN_MS, T0 + n * MIN_MS, base_dir=tmp_path
    )
    assert out.index.equals(build_time_grid(T0 + 10 * MIN_MS, T0 + n * MIN_MS, "5m"))
    # Barra de T0+10..T0+14 (cierre prices[14]) conocida en T0+15; la anterior cierra en prices[9].
    tau = pd.Timestamp(T0 + 15 * MIN_MS, unit="ms", tz="UTC")
    assert np.isclose(out.loc[tau, "A"], np.log(prices[14] / prices[9]), rtol=0, atol=1e-12)
    # El primer slot del rango (T0+10) tiene retorno: el de la barra anterior (pasado).
    assert not np.isnan(out["A"].iloc[0])


def test_resample_from_cache_indexes_fields_by_open_time(tmp_path: Path) -> None:
    """Los campos agregados van por apertura de barra y la validez usa solo el rango pedido."""
    prices = _prices(10)
    _cache(tmp_path, "A", list(range(10)), prices)
    fields, valid = resample_from_cache(
        ["A"],
        "1m",
        "5m",
        T0,
        T0 + 10 * MIN_MS,
        fields=("open", "close", "volume"),
        base_dir=tmp_path,
    )
    assert fields["close"].index.equals(build_time_grid(T0, T0 + 10 * MIN_MS, "5m"))
    assert valid["A"].tolist() == [True, True]
    assert fields["close"]["A"].tolist() == [prices[4], prices[9]]
    assert fields["open"]["A"].tolist() == [prices[0], prices[5]]
    assert fields["volume"]["A"].tolist() == [5.0, 5.0]


def test_resample_from_cache_with_five_minute_base(tmp_path: Path) -> None:
    """Con base `5m` y objetivo `15m` cada barra agrupa tres velas de 5m de la cache."""
    n = 6
    step = 5 * MIN_MS
    close = _prices(n)
    candles = _candles(list(range(n)), close)
    candles["open_time"] = [T0 + k * step for k in range(n)]
    candles["close_time"] = [T0 + k * step + step - 1 for k in range(n)]
    save_klines(candles.astype(KLINE_DTYPES), "A", "5m", base_dir=tmp_path)
    fields, valid = resample_from_cache(
        ["A"],
        "5m",
        "15m",
        T0,
        T0 + n * step,
        fields=("close", "volume"),
        base_dir=tmp_path,
    )
    assert valid["A"].tolist() == [True, True]
    assert fields["close"]["A"].tolist() == [close[2], close[5]]
    assert fields["volume"]["A"].tolist() == [3.0, 3.0]
    assert fields["close"].index.equals(build_time_grid(T0, T0 + n * step, "15m"))


def test_symbol_without_cache_is_all_nan(tmp_path: Path) -> None:
    """Un simbolo sin cache da una columna `NaN` y no rompe al resto."""
    _cache(tmp_path, "A", list(range(10)), _prices(10))
    fields, valid = resample_from_cache(
        ["A", "B"], "1m", "5m", T0, T0 + 10 * MIN_MS, fields=("close",), base_dir=tmp_path
    )
    assert valid["A"].tolist() == [True, True]
    assert valid["B"].tolist() == [False, False]
    assert fields["close"]["B"].isna().all()
    assert fields["close"]["A"].notna().all()


def test_cache_gap_makes_bar_invalid_and_returns_nan(tmp_path: Path) -> None:
    """Un hueco que deja la barra sin minuto final valido la invalida y su retorno es `NaN`."""
    n = 30
    minutes = [m for m in range(n) if m != 14]  # falta el ultimo minuto de la 3.a barra de 5m
    _cache(tmp_path, "A", minutes, [p for m, p in zip(range(n), _prices(n)) if m != 14])
    out = horizon_returns_from_cache(
        ["A"], "5m", T0 + 5 * MIN_MS, T0 + n * MIN_MS, base_dir=tmp_path
    )
    # Barra 2 (T0+10..14) invalida: su retorno (fila T0+15) y el siguiente (T0+20) son NaN.
    by_time = out["A"]
    assert np.isnan(by_time.loc[pd.Timestamp(T0 + 15 * MIN_MS, unit="ms", tz="UTC")])
    assert np.isnan(by_time.loc[pd.Timestamp(T0 + 20 * MIN_MS, unit="ms", tz="UTC")])
    assert not np.isnan(by_time.loc[pd.Timestamp(T0 + 25 * MIN_MS, unit="ms", tz="UTC")])
    assert not (by_time == 0).any()


# --- Identidad con returns_from_cache -----------------------------------------------------------


def test_one_minute_horizon_equals_returns_from_cache(tmp_path: Path) -> None:
    """Con `1m` sobre `1m` el resultado es exactamente el de `returns_from_cache`."""
    n = 120
    minutes = [m for m in range(n) if m not in (20, 21, 70)]
    prices = _prices(n)
    _cache(tmp_path, "A", minutes, [prices[m] for m in minutes])
    _cache(tmp_path, "B", list(range(n)), _prices(n, seed=1))
    start, end = T0 + 10 * MIN_MS, T0 + n * MIN_MS
    expected = returns_from_cache(["A", "B"], "1m", start, end, base_dir=tmp_path)
    got = horizon_returns_from_cache(["A", "B"], "1m", start, end, base_dir=tmp_path)
    assert_frame_equal(got, expected)


# --- Rangos -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (T0 + MIN_MS, T0 + 30 * MIN_MS),  # inicio no multiplo
        (T0, T0 + 31 * MIN_MS),  # fin no multiplo
        (T0 + 30 * MIN_MS, T0 + 30 * MIN_MS),  # vacio
        (T0 + 30 * MIN_MS, T0),  # invertido
    ],
)
def test_bad_ranges_are_rejected(tmp_path: Path, start: int, end: int) -> None:
    """Un rango con extremos que no son multiplo del objetivo, vacio o invertido es `ValueError`."""
    with pytest.raises(ValueError):
        horizon_returns_from_cache(["A"], "15m", start, end, base_dir=tmp_path)
    with pytest.raises(ValueError):
        resample_from_cache(["A"], "1m", "15m", start, end, base_dir=tmp_path)


def test_invalid_parameters_fail_before_reading_cache(tmp_path: Path) -> None:
    """Un objetivo o un umbral invalidos fallan sin tocar la cache."""
    with pytest.raises(ValueError):
        horizon_returns_from_cache(["A"], "1M", T0, T0 + 70 * MIN_MS, base_dir=tmp_path / "x")
    with pytest.raises(ValueError):
        horizon_returns_from_cache(
            ["A"], "5m", T0, T0 + 10 * MIN_MS, min_valid_share=0.0, base_dir=tmp_path / "x"
        )
    assert not (tmp_path / "x").exists()
