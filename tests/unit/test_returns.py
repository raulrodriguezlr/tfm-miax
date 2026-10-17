"""Tests de los retornos logaritmicos: tablas sinteticas y caches en `tmp_path`, sin red.

Las causas se escriben como cadenas, un caracter por slot de 1 minuto:
V valid, T no_trades, L not_listed, H halted, O outage, D delisted, N no_data.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from miax.features import (
    GapCause,
    align_klines,
    build_time_grid,
    classify_klines,
    log_returns,
    return_validity,
    returns_from_cache,
    validity_mask,
)
from miax.ingest import save_klines
from miax.ingest.klines import KLINE_COLUMNS, KLINE_DTYPES

MIN_MS = 60_000
T0 = 1_767_225_600_000  # 2026-01-01T00:00:00Z
LETTERS = {
    "V": GapCause.VALID,
    "T": GapCause.NO_TRADES,
    "L": GapCause.NOT_LISTED,
    "H": GapCause.HALTED,
    "O": GapCause.OUTAGE,
    "D": GapCause.DELISTED,
    "N": GapCause.NO_DATA,
}


def _valid(specs: Mapping[str, str]) -> pd.DataFrame:
    """Mascara de validez `tiempo x simbolo` desde una cadena de causas por simbolo."""
    n = len(next(iter(specs.values())))
    index = build_time_grid(T0, T0 + n * MIN_MS, "1m")
    data = {s: np.array([int(LETTERS[c]) for c in spec], dtype="int8") for s, spec in specs.items()}
    return validity_mask(pd.DataFrame(data, index=index))


def _close(prices: Mapping[str, list[float]]) -> pd.DataFrame:
    """Tabla de cierres `tiempo x simbolo` sobre la rejilla de 1m desde T0."""
    n = len(next(iter(prices.values())))
    index = build_time_grid(T0, T0 + n * MIN_MS, "1m")
    return pd.DataFrame({s: np.asarray(p, dtype="float64") for s, p in prices.items()}, index=index)


def _candles(
    minutes: list[int], close: list[float], num_trades: list[int] | None = None
) -> pd.DataFrame:
    """Klines de 1m en los minutos dados (desde T0) con esos cierres y operaciones."""
    n = len(minutes)
    open_times = [T0 + k * MIN_MS for k in minutes]
    trades = [7] * n if num_trades is None else num_trades
    data = {
        "open_time": open_times,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": [1.0] * n,
        "close_time": [t + MIN_MS - 1 for t in open_times],
        "quote_asset_volume": [10.0] * n,
        "num_trades": trades,
        "taker_buy_base": [0.5] * n,
        "taker_buy_quote": [5.0] * n,
        "ignore": [0.0] * n,
    }
    return pd.DataFrame(data, columns=KLINE_COLUMNS).astype(KLINE_DTYPES)


def _cache(
    base_dir: Path,
    symbol: str,
    minutes: list[int],
    close: list[float],
    num_trades: list[int] | None = None,
) -> None:
    """Guarda en `base_dir` una cache de 1m de `symbol`."""
    save_klines(_candles(minutes, close, num_trades), symbol, "1m", base_dir=base_dir)


def _prices(n: int) -> list[float]:
    """Cierres deterministas, positivos y no triviales."""
    rng = np.random.default_rng(0)
    return list(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n))))


# --- Aditividad (criterio de hecho) ------------------------------------------------------------


def test_returns_are_additive_over_any_window() -> None:
    """La suma de las filas en `(t0, t1]` es `ln(P(t1)/P(t0))`, con `P(t)` el cierre conocido."""
    prices = _prices(40)
    out = log_returns(_close({"A": prices}), _valid({"A": "V" * 40}), "1m")
    # La vela k (cierre prices[k]) se conoce en out.index[k] = open_time_k + 1 minuto.
    for a, b in ((0, 5), (3, 20), (10, 39), (0, 39)):
        t0, t1 = out.index[a], out.index[b]
        total = out["A"][(out.index > t0) & (out.index <= t1)].sum()
        assert np.isclose(total, np.log(prices[b] / prices[a]), rtol=0, atol=1e-12)


def test_sum_by_segments_equals_total_sum() -> None:
    """Sumar por tramos contiguos da lo mismo que sumar todo de una vez."""
    prices = _prices(40)
    out = log_returns(_close({"A": prices}), _valid({"A": "V" * 40}), "1m")["A"]
    parts = out.iloc[1:11].sum() + out.iloc[11:25].sum() + out.iloc[25:40].sum()
    assert np.isclose(parts, out.iloc[1:40].sum(), rtol=0, atol=1e-12)
    assert np.isclose(parts, np.log(prices[39] / prices[0]), rtol=0, atol=1e-12)


def test_known_values() -> None:
    """Valores calculados a mano: ln(110/100) y ln(99/110)."""
    out = log_returns(_close({"A": [100.0, 110.0, 99.0]}), _valid({"A": "VVV"}), "1m")
    assert np.isnan(out["A"].iloc[0])
    assert np.isclose(out["A"].iloc[1], np.log(1.1))
    assert np.isclose(out["A"].iloc[2], np.log(99.0 / 110.0))


# --- Validez -----------------------------------------------------------------------------------


@pytest.mark.parametrize("cause", ["T", "H", "O", "N", "L", "D"])
def test_invalid_slot_gives_nan_in_its_return_and_the_next(cause: str) -> None:
    """Un slot invalido (sin vela o NO_TRADES) anula su retorno y el siguiente, y nada mas."""
    spec = "VVV" + cause + "VVV"
    out = log_returns(
        _close({"A": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0]}), _valid({"A": spec}), "1m"
    )
    assert out["A"].isna().tolist() == [True, False, False, True, True, False, False]


def test_nan_price_in_invalid_slot_is_kept_as_nan() -> None:
    """Un hueco sin vela llega como NaN en el precio: es legitimo y da NaN, no error."""
    close = _close({"A": [100.0, 101.0, np.nan, 103.0, 104.0]})
    out = log_returns(close, _valid({"A": "VVHVV"}), "1m")
    assert out["A"].isna().tolist() == [True, False, True, True, False]


def test_real_zero_return_with_trades_is_kept_as_zero() -> None:
    """Un precio plano con operaciones es un retorno 0.0 real y se conserva."""
    out = log_returns(_close({"A": [5.0, 5.0, 5.0, 5.0]}), _valid({"A": "VVVV"}), "1m")
    assert out["A"].iloc[1:].tolist() == [0.0, 0.0, 0.0]


def test_stale_price_without_trades_is_not_a_zero() -> None:
    """Un precio rancio sin operaciones (NO_TRADES) no se convierte en un 0 artificial."""
    out = log_returns(_close({"A": [5.0, 5.0, 5.0, 5.0, 5.0]}), _valid({"A": "VVTVV"}), "1m")
    assert out["A"].isna().tolist() == [True, False, True, True, False]
    assert (out["A"] == 0).sum() == 2  # solo los ceros reales: t=1 y t=4, no el rancio


def test_no_nan_is_turned_into_zero() -> None:
    """Todos los NaN de la entrada y de la mascara siguen siendo NaN, nunca 0."""
    valid = _valid({"A": "VVTVVHVV", "B": "TVVVVVVV"})
    close = _close({"A": [1.0, 2.0, 2.0, 3.0, 4.0, np.nan, 5.0, 6.0], "B": [1.0] + [2.0] * 7})
    out = log_returns(close, valid, "1m")
    assert not np.isinf(out.to_numpy()).any()  # solo NaN o finitos, nunca inf
    # Posiciones con retorno invalido segun la mascara de retornos: todas NaN.
    rv = return_validity(valid).to_numpy()
    assert np.isnan(out.to_numpy()[~rv]).all()
    assert not np.isnan(out.to_numpy()[rv]).any()


def test_first_row_is_nan() -> None:
    """La primera fila no tiene `t-1`: es NaN en todas las columnas."""
    out = log_returns(
        _close({"A": [1.0, 2.0, 3.0], "B": [3.0, 2.0, 1.0]}),
        _valid({"A": "VVV", "B": "VVV"}),
        "1m",
    )
    assert out.iloc[0].isna().all()


def test_output_keeps_columns_and_float64_and_does_not_mutate_input() -> None:
    """La salida conserva las columnas, es float64 y no modifica la entrada."""
    close = _close({"A": [1.0, 2.0, 4.0], "B": [3.0, 3.0, 3.0]})
    snapshot = close.copy()
    out = log_returns(close, _valid({"A": "VVV", "B": "VVV"}), "1m")
    assert out.shape == close.shape
    assert out.columns.equals(close.columns)
    assert all(dtype == "float64" for dtype in out.dtypes)
    assert_frame_equal(close, snapshot)


def test_output_index_is_input_index_plus_one_step() -> None:
    """El indice de salida es el de entrada mas un paso, `datetime64[ms, UTC]` y sin `freq`."""
    close = _close({"A": [1.0, 2.0, 4.0, 8.0]})
    out = log_returns(close, _valid({"A": "VVVV"}), "1m")
    assert out.index.equals(close.index + pd.Timedelta(minutes=1))
    assert out.index.dtype == "datetime64[ms, UTC]"
    assert out.index.freq is None
    assert out.index[0] == pd.Timestamp(T0 + MIN_MS, unit="ms", tz="UTC")


def test_output_index_step_follows_the_interval() -> None:
    """El desplazamiento es el paso del intervalo dado (5m = 5 minutos)."""
    index = build_time_grid(T0, T0 + 15 * MIN_MS, "5m")
    close = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=index)
    valid = pd.DataFrame({"A": [True, True, True]}, index=index)
    out = log_returns(close, valid, "5m")
    assert out.index.equals(index + pd.Timedelta(minutes=5))
    assert out.index.dtype == "datetime64[ms, UTC]"
    assert out.index.freq is None


def test_interval_without_fixed_step_raises() -> None:
    """Un intervalo sin paso fijo (`1M`) lanza `ValueError`."""
    close = _close({"A": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="duracion fija"):
        log_returns(close, _valid({"A": "VVV"}), "1M")


def test_literal_example_candle_opening_at_1000_is_in_row_1001() -> None:
    """Con velas de apertura 09:59 y 10:00, `ln(close 10:00 / close 09:59)` va en la fila 10:01."""
    ten = T0 + 600 * MIN_MS
    index = build_time_grid(ten - MIN_MS, ten + MIN_MS, "1m")  # aperturas 09:59 y 10:00
    close = pd.DataFrame({"A": [100.0, 103.0]}, index=index)
    valid = pd.DataFrame({"A": [True, True]}, index=index)
    out = log_returns(close, valid, "1m")
    row_1001 = pd.Timestamp(ten + MIN_MS, unit="ms", tz="UTC")
    assert out.index.tolist() == [pd.Timestamp(ten, unit="ms", tz="UTC"), row_1001]
    assert np.isclose(out.loc[row_1001, "A"], np.log(103.0 / 100.0))
    assert np.isnan(out.loc[pd.Timestamp(ten, unit="ms", tz="UTC"), "A"])


def test_columns_are_independent() -> None:
    """Lo que pasa en una columna no afecta a la otra."""
    close = _close({"A": [1.0, 2.0, 3.0, 4.0], "B": [1.0, 2.0, 3.0, 4.0]})
    out = log_returns(close, _valid({"A": "VVVV", "B": "VTVV"}), "1m")
    assert out["A"].iloc[1:].notna().all()
    assert out["B"].isna().tolist() == [True, True, True, False]


def test_single_row_is_all_nan() -> None:
    """Con una sola fila no hay retorno."""
    out = log_returns(_close({"A": [1.0]}), _valid({"A": "V"}), "1m")
    assert out.shape == (1, 1)
    assert out.isna().all().all()


def test_all_invalid_and_no_columns_and_no_rows() -> None:
    """Todo invalido da todo NaN; sin columnas o sin filas conserva la forma."""
    out = log_returns(_close({"A": [1.0, 2.0, 3.0]}), _valid({"A": "TTT"}), "1m")
    assert out.isna().all().all()
    index = build_time_grid(T0, T0 + 3 * MIN_MS, "1m")
    empty_cols = log_returns(pd.DataFrame(index=index), pd.DataFrame(index=index), "1m")
    assert empty_cols.shape == (3, 0)
    empty_rows = log_returns(
        pd.DataFrame({"A": np.array([], dtype="float64")}, index=index[:0]),
        pd.DataFrame({"A": np.array([], dtype=bool)}, index=index[:0]),
        "1m",
    )
    assert empty_rows.shape == (0, 1)


# --- No-fuga -----------------------------------------------------------------------------------


def test_no_leak_future_prices_do_not_change_past_returns() -> None:
    """Cambiar los precios desde la fila k no altera los retornos de las filas anteriores a k."""
    k = 8
    prices = _prices(15)
    changed = prices[:k] + [p * 7.0 for p in prices[k:]]
    spec = "VVVTVVVVVVVVVVV"
    r0 = log_returns(_close({"A": prices}), _valid({"A": spec}), "1m")
    r1 = log_returns(_close({"A": changed}), _valid({"A": spec}), "1m")
    assert_frame_equal(r0.iloc[:k], r1.iloc[:k])
    assert not r0.iloc[k:].equals(r1.iloc[k:])  # el cambio si afecta a su propio tramo


def test_no_leak_log_returns_row_does_not_depend_on_candle_opening_at_or_after_it() -> None:
    """La fila `tau` no cambia al tocar el close de la vela que abre en `tau`; la siguiente si."""
    k = 6
    prices = _prices(12)
    changed = list(prices)
    changed[k] *= 3.0
    valid = _valid({"A": "V" * 12})
    r0 = log_returns(_close({"A": prices}), valid, "1m")
    r1 = log_returns(_close({"A": changed}), valid, "1m")
    # La vela k abre en T0 + k min; su fila (conocida en +1) ya la usa, la fila k es de la vela k-1.
    tau = pd.Timestamp(T0 + k * MIN_MS, unit="ms", tz="UTC")
    assert r0.loc[tau, "A"] == r1.loc[tau, "A"]
    assert_series_equal(r0.loc[:tau, "A"], r1.loc[:tau, "A"])
    assert r0.loc[tau + pd.Timedelta(minutes=1), "A"] != r1.loc[tau + pd.Timedelta(minutes=1), "A"]


def test_no_leak_future_validity_does_not_change_past_returns() -> None:
    """Cambiar la validez desde la fila k tampoco altera los retornos anteriores a k."""
    k = 6
    prices = _prices(10)
    r0 = log_returns(_close({"A": prices}), _valid({"A": "VVTVVVVVVV"}), "1m")
    r1 = log_returns(_close({"A": prices}), _valid({"A": "VVTVVVTTTT"}), "1m")
    assert_frame_equal(r0.iloc[:k], r1.iloc[:k])


def test_returns_from_cache_right_trim_does_not_change_common_returns(tmp_path: Path) -> None:
    """Recortar el rango por la derecha no cambia los retornos de las filas comunes."""
    n = 30
    trades = [7] * n
    trades[12] = 0
    _cache(tmp_path, "A", list(range(n)), _prices(n), trades)
    _cache(tmp_path, "B", list(range(n)), _prices(n)[::-1])
    full = returns_from_cache(
        ["A", "B"], "1m", T0 + 5 * MIN_MS, T0 + 28 * MIN_MS, base_dir=tmp_path
    )
    short = returns_from_cache(
        ["A", "B"], "1m", T0 + 5 * MIN_MS, T0 + 15 * MIN_MS, base_dir=tmp_path
    )
    assert_frame_equal(full.loc[short.index], short)


# --- Validaciones ------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_corrupt_price_in_valid_slot_raises_naming_symbol(bad: float) -> None:
    """Un precio 0, negativo, NaN o inf en un slot valido lanza `ValueError` con el simbolo."""
    close = _close({"OKUSDT": [1.0, 2.0, 3.0], "BADUSDT": [1.0, bad, 3.0]})
    valid = _valid({"OKUSDT": "VVV", "BADUSDT": "VVV"})
    with pytest.raises(ValueError, match="BADUSDT") as info:
        log_returns(close, valid, "1m")
    assert "OKUSDT" not in str(info.value)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf, -np.inf])
@pytest.mark.filterwarnings("error")
def test_same_value_in_invalid_slot_does_not_raise(bad: float) -> None:
    """El mismo valor en un slot invalido no lanza ni avisa: se enmascara."""
    close = _close({"A": [1.0, bad, 3.0, 4.0]})
    out = log_returns(close, _valid({"A": "VTVV"}), "1m")
    assert out["A"].isna().tolist() == [True, True, True, False]


@pytest.mark.filterwarnings("error")
def test_no_runtime_warning_in_normal_and_edge_cases() -> None:
    """Ni los casos normales ni los bordes emiten `RuntimeWarning` (log de 0, NaN, etc.)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        log_returns(_close({"A": [1.0, np.nan, 0.0, 2.0, 2.0]}), _valid({"A": "VHTVV"}), "1m")
        log_returns(_close({"A": [1.0, 1.0, 1.0]}), _valid({"A": "VVV"}), "1m")


@pytest.mark.parametrize("bad", [0.0, -1.0, np.inf])
def test_corrupt_price_raises_without_runtime_warning(bad: float) -> None:
    """El error por dato corrupto sale antes del `log`: ningun `RuntimeWarning` previo."""
    close = _close({"A": [1.0, bad, 3.0]})
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ValueError, match="A"):
            log_returns(close, _valid({"A": "VVV"}), "1m")


def test_different_index_or_columns_raise() -> None:
    """Indices o columnas distintos entre `close` y `valid` lanzan `ValueError`."""
    close = _close({"A": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="mismo indice"):
        log_returns(close, _valid({"B": "VVV"}), "1m")
    with pytest.raises(ValueError, match="mismo indice"):
        log_returns(close, _valid({"A": "VV"}), "1m")
    shifted = _valid({"A": "VVV"})
    shifted.index = shifted.index + pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="mismo indice"):
        log_returns(close, shifted, "1m")


def test_non_bool_valid_raises() -> None:
    """Un `valid` con NaN, float u object lanza `ValueError`: un NaN no es `False`."""
    close = _close({"A": [1.0, 2.0]})
    for column in ([True, np.nan], [1.0, 0.0], [True, None]):
        valid = pd.DataFrame({"A": column}, index=close.index)
        with pytest.raises(ValueError, match="bool"):
            log_returns(close, valid, "1m")


def test_non_float_close_raises() -> None:
    """Un `close` entero lanza `ValueError`."""
    index = build_time_grid(T0, T0 + 2 * MIN_MS, "1m")
    with pytest.raises(ValueError, match="float"):
        log_returns(pd.DataFrame({"A": [1, 2]}, index=index), _valid({"A": "VV"}), "1m")


# --- returns_from_cache ------------------------------------------------------------------------


def _ts(minute: int) -> pd.Timestamp:
    """Instante UTC del minuto `minute` desde T0."""
    return pd.Timestamp(T0 + minute * MIN_MS, unit="ms", tz="UTC")


def test_cache_first_return_uses_candles_opening_at_start_minus_one_and_two_steps(
    tmp_path: Path,
) -> None:
    """La fila `start` es `ln(close(start-1) / close(start-2))`: velas anteriores al rango."""
    close = [100.0 + k * k for k in range(20)]
    _cache(tmp_path, "A", list(range(20)), close)
    out = returns_from_cache(["A"], "1m", T0 + 5 * MIN_MS, T0 + 10 * MIN_MS, base_dir=tmp_path)
    assert out.index[0] == _ts(5)
    assert np.isclose(out["A"].iloc[0], np.log(close[4] / close[3]))
    assert not np.isclose(out["A"].iloc[0], np.log(close[5] / close[4]))
    assert out["A"].notna().all()
    # Y la ultima fila (end - 1 paso) es la de la vela que abre en end - 2 pasos.
    assert np.isclose(out["A"].iloc[-1], np.log(close[8] / close[7]))


def test_cache_range_starting_at_cache_start_has_two_nan_rows(tmp_path: Path) -> None:
    """Sin velas anteriores al inicio de la cache, las dos primeras filas son NaN y el resto no."""
    _cache(tmp_path, "A", list(range(20)), _prices(20))
    out = returns_from_cache(["A"], "1m", T0, T0 + 6 * MIN_MS, base_dir=tmp_path)
    # La fila 0 pide las velas -1 y -2; la fila 1, la vela 0 y la -1: faltan.
    assert out["A"].isna().tolist() == [True, True, False, False, False, False]


@pytest.mark.parametrize("missing", ["absent", "no_trades"])
def test_cache_candle_at_start_minus_two_steps_not_valid_gives_nan_first_row(
    tmp_path: Path, missing: str
) -> None:
    """Si la vela de `start - 2 pasos` no existe o no es valida, la primera fila es NaN."""
    close = _prices(20)
    if missing == "absent":
        _cache(tmp_path, "A", [k for k in range(20) if k != 3], close[:3] + close[4:])
    else:
        trades = [7] * 20
        trades[3] = 0
        _cache(tmp_path, "A", list(range(20)), close, trades)
    out = returns_from_cache(["A"], "1m", T0 + 5 * MIN_MS, T0 + 10 * MIN_MS, base_dir=tmp_path)
    assert np.isnan(out["A"].iloc[0])
    # La vela 3 solo interviene en la fila 5 (la de la vela 4): el resto es valido.
    assert out["A"].iloc[1:].notna().all()


def test_cache_candle_at_start_minus_one_step_not_valid_gives_two_nan_rows(
    tmp_path: Path,
) -> None:
    """Una vela sin operaciones en `start - 1 paso` anula la fila `start` y la siguiente."""
    trades = [7] * 20
    trades[4] = 0
    _cache(tmp_path, "A", list(range(20)), _prices(20), trades)
    out = returns_from_cache(["A"], "1m", T0 + 5 * MIN_MS, T0 + 10 * MIN_MS, base_dir=tmp_path)
    assert out["A"].isna().tolist() == [True, True, False, False, False]


def test_cache_month_boundary_first_row_uses_previous_month_candles(tmp_path: Path) -> None:
    """En el primer minuto de un mes, la primera fila usa las velas 23:59 y 23:58 del anterior."""
    first_of_feb = 31 * 24 * 60  # minutos desde T0 (2026-01-01) hasta 2026-02-01
    minutes = list(range(first_of_feb - 5, first_of_feb + 10))
    close = [100.0 + k * k for k in range(len(minutes))]
    _cache(tmp_path, "A", minutes, close)
    start, end = T0 + first_of_feb * MIN_MS, T0 + (first_of_feb + 5) * MIN_MS
    out = returns_from_cache(["A"], "1m", start, end, base_dir=tmp_path)
    assert out.index[0] == _ts(first_of_feb)
    # Velas 23:59 (posicion 4) y 23:58 (posicion 3) de enero.
    assert np.isclose(out["A"].iloc[0], np.log(close[4] / close[3]))
    assert out["A"].notna().all()


def test_cache_no_leak_row_tau_does_not_depend_on_candle_opening_at_or_after_tau(
    tmp_path: Path,
) -> None:
    """Cambiar el close (o la liquidez) de la vela que abre en `tau` no cambia la fila `tau`."""
    n = 30
    start, end = T0 + 10 * MIN_MS, T0 + 25 * MIN_MS
    close = _prices(n)
    changed = list(close)
    changed[15] *= 5.0
    no_trades = [7] * n
    no_trades[15] = 0
    _cache(tmp_path / "base", "A", list(range(n)), close)
    _cache(tmp_path / "price", "A", list(range(n)), changed)
    _cache(tmp_path / "liquidity", "A", list(range(n)), close, no_trades)
    r0 = returns_from_cache(["A"], "1m", start, end, base_dir=tmp_path / "base")
    r_price = returns_from_cache(["A"], "1m", start, end, base_dir=tmp_path / "price")
    r_liq = returns_from_cache(["A"], "1m", start, end, base_dir=tmp_path / "liquidity")
    tau, step = _ts(15), pd.Timedelta(minutes=1)
    for other in (r_price, r_liq):
        # Todo hasta la fila tau incluida es identico; la de tau + paso ya cambia.
        assert_series_equal(r0.loc[:tau, "A"], other.loc[:tau, "A"])
        assert r0.loc[tau + step, "A"] != other.loc[tau + step, "A"]
    # El precio de la vela 15 tambien entra en la fila de tau + 2 pasos (vela 16 sobre vela 15).
    assert r0.loc[tau + 2 * step, "A"] != r_price.loc[tau + 2 * step, "A"]
    assert np.isnan(r_liq.loc[tau + step, "A"])  # la vela 15 no es valida


def test_cache_output_is_exactly_the_requested_grid(tmp_path: Path) -> None:
    """El indice de salida es `build_time_grid(start, end)`, con su dtype y sin la vela extra."""
    _cache(tmp_path, "A", list(range(30)), _prices(30))
    start, end = T0 + 5 * MIN_MS, T0 + 20 * MIN_MS
    out = returns_from_cache(["A"], "1m", start, end, base_dir=tmp_path)
    grid = build_time_grid(start, end, "1m")
    assert out.index.equals(grid)
    assert out.index.dtype == grid.dtype
    assert len(out) == 15


def test_cache_unaligned_start_and_end(tmp_path: Path) -> None:
    """`start_ms`/`end_ms` no alineados dan la rejilla de `build_time_grid` y el retorno bien."""
    close = _prices(30)
    _cache(tmp_path, "A", list(range(30)), close)
    start, end = T0 + 5 * MIN_MS + 1, T0 + 15 * MIN_MS - 1
    out = returns_from_cache(["A"], "1m", start, end, base_dir=tmp_path)
    grid = build_time_grid(start, end, "1m")
    assert out.index.equals(grid)
    assert grid[0] == _ts(6)
    assert grid[-1] == _ts(14)
    assert np.isclose(out["A"].iloc[0], np.log(close[5] / close[4]))


def test_cache_symbol_without_cache_is_all_nan_column(tmp_path: Path) -> None:
    """Un simbolo sin cache es una columna de NaN y no estropea a los demas."""
    _cache(tmp_path, "A", list(range(20)), _prices(20))
    out = returns_from_cache(
        ["A", "GHOST"], "1m", T0 + 5 * MIN_MS, T0 + 10 * MIN_MS, base_dir=tmp_path
    )
    assert out.columns.tolist() == ["A", "GHOST"]
    assert out["GHOST"].isna().all()
    assert out["A"].notna().all()


def test_cache_duplicate_symbols_are_collapsed(tmp_path: Path) -> None:
    """Un simbolo repetido sale una sola vez, en el orden de la primera aparicion."""
    _cache(tmp_path, "A", list(range(20)), _prices(20))
    _cache(tmp_path, "B", list(range(20)), _prices(20)[::-1])
    out = returns_from_cache(
        ["B", "A", "B"], "1m", T0 + 5 * MIN_MS, T0 + 10 * MIN_MS, base_dir=tmp_path
    )
    assert out.columns.tolist() == ["B", "A"]


def test_cache_gap_inside_range_is_nan_and_not_bridged(tmp_path: Path) -> None:
    """Un minuto sin vela anula la fila de esa vela (conocida un paso despues) y la siguiente."""
    minutes = [k for k in range(20) if k != 10]
    _cache(tmp_path, "A", minutes, _prices(19))
    out = returns_from_cache(["A"], "1m", T0 + 5 * MIN_MS, T0 + 15 * MIN_MS, base_dir=tmp_path)
    nan_minutes = [(ts.value // 10**6 - T0) // MIN_MS for ts in out.index[out["A"].isna()]]
    # La vela 10 falta: su retorno se conoce en el minuto 11 y el de la vela 11, en el 12.
    assert nan_minutes == [11, 12]


def test_cache_empty_range_raises(tmp_path: Path) -> None:
    """Un rango sin slots (o vacio o invertido) lanza `ValueError`."""
    _cache(tmp_path, "A", list(range(20)), _prices(20))
    with pytest.raises(ValueError):
        returns_from_cache(["A"], "1m", T0 + 1, T0 + MIN_MS - 1, base_dir=tmp_path)
    with pytest.raises(ValueError):
        returns_from_cache(["A"], "1m", T0 + 5 * MIN_MS, T0 + 5 * MIN_MS, base_dir=tmp_path)
    with pytest.raises(ValueError):
        returns_from_cache(["A"], "1m", T0 + 6 * MIN_MS, T0 + 5 * MIN_MS, base_dir=tmp_path)


def test_cache_interval_without_fixed_step_raises(tmp_path: Path) -> None:
    """`1M` no tiene paso fijo: `ValueError`."""
    with pytest.raises(ValueError, match="paso fijo|duracion fija"):
        returns_from_cache(["A"], "1M", T0, T0 + 60 * MIN_MS, base_dir=tmp_path)


def test_cache_result_equals_manual_chain(tmp_path: Path) -> None:
    """Identico a encadenar `align`/`classify` en `[start-2p, end-p)`, `log_returns` y quitar 1."""
    n = 40
    trades_a = [7] * n
    trades_a[15] = 0
    _cache(tmp_path, "A", [k for k in range(n) if k != 22], _prices(n - 1), trades_a[: n - 1])
    _cache(tmp_path, "B", list(range(n)), _prices(n)[::-1])
    start, end = T0 + 10 * MIN_MS, T0 + 35 * MIN_MS
    symbols = ["A", "B", "GHOST"]
    out = returns_from_cache(symbols, "1m", start, end, base_dir=tmp_path)

    wide_start, wide_end = start - 2 * MIN_MS, end - MIN_MS
    aligned = align_klines(
        symbols, "1m", wide_start, wide_end, fields=("close",), base_dir=tmp_path
    )
    causes = classify_klines(symbols, "1m", wide_start, wide_end, base_dir=tmp_path)
    manual = log_returns(aligned["close"], validity_mask(causes), "1m").iloc[1:]
    assert_frame_equal(out, manual)


def test_cache_is_deterministic(tmp_path: Path) -> None:
    """Dos llamadas iguales dan exactamente el mismo resultado."""
    _cache(tmp_path, "A", list(range(30)), _prices(30))
    args = (["A"], "1m", T0 + 5 * MIN_MS, T0 + 25 * MIN_MS)
    assert_frame_equal(
        returns_from_cache(*args, base_dir=tmp_path), returns_from_cache(*args, base_dir=tmp_path)
    )
