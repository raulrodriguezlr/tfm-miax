"""Tests de la volatilidad realizada: retornos sinteticos y caches en `tmp_path`, sin red."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.lib.stride_tricks import sliding_window_view
from pandas.testing import assert_frame_equal

from miax.features import (
    build_time_grid,
    realized_volatility,
    realized_volatility_from_cache,
    returns_from_cache,
)
from miax.ingest import save_klines
from miax.ingest.klines import KLINE_COLUMNS, KLINE_DTYPES

MIN_MS = 60_000
T0 = 1_767_225_600_000  # 2026-01-01T00:00:00Z
NAN = float("nan")


def _rets(data: Mapping[str, Sequence[float]]) -> pd.DataFrame:
    """Tabla de retornos `tiempo x simbolo` sobre la rejilla de 1m desde T0."""
    n = len(next(iter(data.values())))
    index = build_time_grid(T0, T0 + n * MIN_MS, "1m")
    return pd.DataFrame({s: np.asarray(v, dtype="float64") for s, v in data.items()}, index=index)


def _rv(values: Sequence[float], **kwargs: float) -> np.ndarray:
    """Volatilidad de una sola serie como vector de numpy."""
    return realized_volatility(_rets({"A": values}), **kwargs)["A"].to_numpy()


def _exact(values: np.ndarray, window: int, min_valid_share: float) -> np.ndarray:
    """Referencia ingenua: `sliding_window_view` sobre la serie rellena con NaN por delante."""
    padded = np.concatenate([np.full(window - 1, np.nan), values])
    windows = sliding_window_view(padded, window)
    n = np.sum(~np.isnan(windows), axis=1)
    total = np.nansum(windows**2, axis=1)
    out = np.full(len(values), np.nan)
    enough = n / window >= min_valid_share
    out[enough] = np.sqrt(window / n[enough] * total[enough])
    return out


# --- Valores -----------------------------------------------------------------------------------


@pytest.mark.parametrize("window", [1, 2, 5, 60])
def test_constant_plus_minus_returns_give_a_sqrt_window(window: int) -> None:
    """Retornos constantes `±a` dan `a·sqrt(W)` en cuanto la ventana se llena."""
    a = 0.003
    values = [a if k % 3 else -a for k in range(window + 20)]
    out = _rv(values, window=window)
    assert np.allclose(out[window - 1 :], a * np.sqrt(window), rtol=0, atol=1e-15)


def test_known_values_computed_by_hand() -> None:
    """W=3: [0.1, -0.2, 0.2] da 0.3 y [-0.2, 0.2, 0.0] da sqrt(0.08)."""
    out = _rv([0.1, -0.2, 0.2, 0.0], window=3)
    assert np.isnan(out[:1]).all()  # 1 de 3 validos: por debajo del 80 %
    assert np.isnan(out[1])  # 2 de 3 = 0,667 < 0,8
    assert np.isclose(out[2], 0.3, rtol=0, atol=1e-15)
    assert np.isclose(out[3], np.sqrt(0.08), rtol=0, atol=1e-15)


def test_window_is_inclusive_of_tau_and_ends_at_tau() -> None:
    """La fila `tau` usa las `W` filas `tau-W+1 .. tau`: la que sale de la ventana ya no pesa."""
    values = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
    out = _rv(values, window=3)
    assert np.isclose(out[2], 0.5)  # ventana filas 0..2, con el 0.5
    assert out[3] == 0.0  # ventana filas 1..3: el 0.5 ya salio


# --- Reescalado --------------------------------------------------------------------------------


def test_one_nan_in_window_rescales_by_w_over_w_minus_one() -> None:
    """Con un NaN en la ventana, `RV = sqrt(W/(W-1) · Σ)` sobre los presentes."""
    window = 10
    values = [0.01] * 9 + [NAN, 0.02] + [0.01] * 9
    out = _rv(values, window=window)
    # Fila 10: ventana de las filas 1..10, con un NaN en la 9.
    row = 10
    window_values = np.array(values[row - window + 1 : row + 1])
    valid = window_values[~np.isnan(window_values)]
    assert len(valid) == window - 1
    expected = np.sqrt(window / (window - 1) * np.sum(valid**2))
    assert np.isclose(out[row], expected, rtol=0, atol=1e-15)
    # Y es mayor que la suma sin reescalar.
    assert out[row] > np.sqrt(np.sum(valid**2))


def test_rescaling_makes_missing_returns_equivalent_to_average_variance() -> None:
    """Con retornos constantes `±a` y NaN dispersos, el reescalado devuelve `a·sqrt(W)`."""
    window, a = 20, 0.004
    values = np.full(60, a)
    values[[25, 31, 44]] = np.nan  # 3 NaN, nunca mas de 3 por ventana (cobertura >= 85 %)
    out = _rv(values.tolist(), window=window)
    assert np.allclose(out[window - 1 :], a * np.sqrt(window), rtol=0, atol=1e-15)


# --- Umbral ------------------------------------------------------------------------------------


def test_exact_threshold_gives_value_w5() -> None:
    """`n/W == 0.8` exacto (4 de 5) da valor, no NaN: el umbral es inclusivo."""
    out = _rv([0.1, 0.1, NAN, 0.1, 0.1, 0.1], window=5)
    # Fila 4: ventana 0..4 con un NaN = 4 validos -> valor reescalado.
    assert np.isclose(out[4], np.sqrt(5 / 4 * 4 * 0.01), rtol=0, atol=1e-15)


def test_one_valid_less_than_threshold_gives_nan_w5() -> None:
    """3 de 5 validos (0.6) es NaN."""
    out = _rv([0.1, 0.1, NAN, NAN, 0.1, 0.1], window=5)
    assert np.isnan(out[4])


def test_exact_threshold_gives_value_w60_and_one_less_gives_nan() -> None:
    """W=60: 48 validos (0.8) da valor y 47 da NaN."""
    base = np.full(60, 0.001)
    at_threshold = base.copy()
    at_threshold[:12] = np.nan  # 48 validos
    out = _rv(at_threshold.tolist(), window=60)
    assert np.isclose(out[59], 0.001 * np.sqrt(60), rtol=0, atol=1e-15)
    below = base.copy()
    below[:13] = np.nan  # 47 validos
    assert np.isnan(_rv(below.tolist(), window=60)[59])


@pytest.mark.parametrize("share", [0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
def test_inclusive_threshold_for_other_shares(share: float) -> None:
    """Con `W=10` y `share = k/10`, `k` validos dan valor y `k-1` dan NaN (sin errores de float)."""
    window = 10
    k = round(share * window)
    for valid_count, expect_value in ((k, True), (k - 1, False)):
        values = np.full(window, 0.01)
        values[: window - valid_count] = np.nan
        out = _rv(values.tolist(), window=window, min_valid_share=share)
        assert bool(np.isfinite(out[window - 1])) is expect_value, (share, valid_count)


def test_all_nan_window_is_nan_never_zero() -> None:
    """`n == 0` da NaN, nunca 0, ni con `min_valid_share` minimo."""
    out = _rv([0.1] * 5 + [NAN] * 8, window=3, min_valid_share=1e-9)
    assert np.isnan(out[7:]).all()  # filas 7..12: ventana de 3 filas toda NaN (filas 5..)
    assert not (out == 0.0).any()


def test_zero_returns_that_are_valid_give_zero_not_nan() -> None:
    """Un retorno 0.0 valido cuenta: la volatilidad es 0 y no NaN."""
    out = _rv([0.0] * 6, window=3)
    assert np.isnan(out[:1]).all()
    assert out[2:].tolist() == [0.0] * 4


def test_first_rows_have_incomplete_window_counted_as_missing() -> None:
    """Con W=10 y todo valido, hay valor desde la fila 7 (8 de 10); antes, NaN."""
    out = _rv([0.01] * 20, window=10)
    assert np.isnan(out[:7]).all()
    assert np.isfinite(out[7:]).all()
    # La fila 7 tiene 8 retornos: reescala por 10/8.
    assert np.isclose(out[7], np.sqrt(10 / 8 * 8 * 0.01**2), rtol=0, atol=1e-15)
    assert np.isclose(out[7], 0.01 * np.sqrt(10), rtol=0, atol=1e-15)


def test_first_rows_with_stricter_share_start_later() -> None:
    """Con `min_valid_share=1.0` no hay valor hasta que la ventana esta completa."""
    out = _rv([0.01] * 12, window=5, min_valid_share=1.0)
    assert np.isnan(out[:4]).all()
    assert np.isfinite(out[4:]).all()


def test_window_one_is_absolute_value_of_return() -> None:
    """`window=1`: `RV = |r|` si hay retorno y NaN si no."""
    values = [0.1, -0.2, NAN, 0.0, 0.3]
    out = _rv(values, window=1)
    assert np.allclose(out[[0, 1, 3, 4]], [0.1, 0.2, 0.0, 0.3], rtol=0, atol=1e-15)
    assert np.isnan(out[2])


# --- No-fuga -----------------------------------------------------------------------------------


def test_no_leak_changing_returns_after_tau_does_not_change_rv_up_to_tau() -> None:
    """Cambiar los retornos desde la fila k no altera la RV de las filas anteriores a k."""
    rng = np.random.default_rng(1)
    base = rng.normal(0.0, 0.01, 60)
    base[[10, 33]] = np.nan
    k = 40
    changed = base.copy()
    changed[k:] = changed[k:] * 9.0 + 0.5
    a = realized_volatility(_rets({"A": base}), window=10)
    b = realized_volatility(_rets({"A": changed}), window=10)
    assert_frame_equal(a.iloc[:k], b.iloc[:k])
    assert not a.iloc[k:].equals(b.iloc[k:])


def test_no_leak_nan_after_tau_does_not_change_rv_up_to_tau() -> None:
    """Cubrir de NaN el futuro tampoco cambia el pasado."""
    rng = np.random.default_rng(2)
    base = rng.normal(0.0, 0.01, 50)
    masked = base.copy()
    masked[30:] = np.nan
    a = realized_volatility(_rets({"A": base}), window=8)
    b = realized_volatility(_rets({"A": masked}), window=8)
    assert_frame_equal(a.iloc[:30], b.iloc[:30])


def test_rv_tau_changes_when_r_tau_changes() -> None:
    """La ventana incluye `tau`: cambiar `r_tau` cambia `RV_tau` (y no el de `tau-1`)."""
    rng = np.random.default_rng(3)
    base = rng.normal(0.0, 0.01, 30)
    tau = 20
    changed = base.copy()
    changed[tau] += 0.05
    a = realized_volatility(_rets({"A": base}), window=10)["A"]
    b = realized_volatility(_rets({"A": changed}), window=10)["A"]
    assert a.iloc[tau] != b.iloc[tau]
    assert a.iloc[tau - 1] == b.iloc[tau - 1]


def test_right_trim_does_not_change_common_values() -> None:
    """Recortar por la derecha deja idénticas las filas comunes."""
    rng = np.random.default_rng(4)
    values = rng.normal(0.0, 0.01, 80)
    values[[5, 6, 50]] = np.nan
    full = realized_volatility(_rets({"A": values}), window=12)
    short = realized_volatility(_rets({"A": values[:45]}), window=12)
    assert_frame_equal(full.iloc[:45], short)


def test_left_trim_only_changes_the_first_window_rows() -> None:
    """Recortar por la izquierda solo cambia las `W-1` primeras filas del resto (ventana corta)."""
    rng = np.random.default_rng(5)
    values = rng.normal(0.0, 0.01, 60)
    window, cut = 10, 20
    full = realized_volatility(_rets({"A": values}), window=window)["A"].to_numpy()
    tail = _rv(values[cut:].tolist(), window=window)
    assert np.allclose(full[cut + window - 1 :], tail[window - 1 :], rtol=0, atol=1e-15)


# --- Precision ---------------------------------------------------------------------------------


def test_spike_leaving_the_window_leaves_exact_zero() -> None:
    """Tras un pico de 0.8 y ceros validos, al salir el pico `RV` vale 0 exacto (sin residuo)."""
    window = 60
    values = [0.8] + [0.0] * 200
    out = _rv(values, window=window)
    assert out[window - 1] == 0.8  # w/n = 1 con 60 validos; la raiz de 0.64 es 0.8
    assert (out[window:] == 0.0).all()


def test_spike_among_tiny_returns_leaves_no_residue() -> None:
    """Un pico grande seguido de retornos diminutos no deja residuo de redondeo al salir."""
    window = 60
    values = [0.8] + [1e-6] * 200
    out = _rv(values, window=window)
    assert np.allclose(out[window:], 1e-6 * np.sqrt(window), rtol=1e-9, atol=0)


def test_matches_exact_computation_on_random_case() -> None:
    """Coincide con `sliding_window_view` + `nansum` en un caso aleatorio con huecos."""
    rng = np.random.default_rng(42)
    values = rng.standard_t(3, 3000) * 0.002
    values[rng.random(3000) < 0.15] = np.nan
    values[1000:1030] = np.nan  # un tramo largo sin datos
    for window, share in ((60, 0.8), (7, 0.5), (1, 0.8), (30, 1.0)):
        got = _rv(values.tolist(), window=window, min_valid_share=share)
        want = _exact(values, window, share)
        assert np.array_equal(np.isnan(got), np.isnan(want)), (window, share)
        assert np.allclose(got, want, rtol=1e-9, atol=1e-15, equal_nan=True), (window, share)


def test_constant_series_after_large_values_keeps_relative_precision() -> None:
    """Valores positivos pequenos tras retornos grandes mantienen precision relativa 1e-9."""
    rng = np.random.default_rng(7)
    values = np.concatenate([rng.normal(0.0, 0.5, 100), rng.normal(0.0, 1e-4, 300)])
    got = _rv(values.tolist(), window=60)
    want = _exact(values, 60, 0.8)
    assert np.allclose(got[200:], want[200:], rtol=1e-9, atol=0)


# --- Estructura --------------------------------------------------------------------------------


def test_columns_are_independent() -> None:
    """Lo que pasa en una columna no afecta a la otra."""
    a = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    b = [NAN, NAN, NAN, 0.1, 0.1, 0.1]
    both = realized_volatility(_rets({"A": a, "B": b}), window=3)
    only_a = realized_volatility(_rets({"A": a}), window=3)
    only_b = realized_volatility(_rets({"B": b}), window=3)
    assert_frame_equal(both[["A"]], only_a)
    assert_frame_equal(both[["B"]], only_b)


def test_output_shape_index_columns_dtype_and_input_unchanged() -> None:
    """Misma forma, indice y columnas, `float64`, y la entrada no se modifica."""
    returns = _rets({"B": [0.1, NAN, 0.2, 0.3], "A": [0.0, 0.1, 0.1, 0.1]})
    snapshot = returns.copy()
    out = realized_volatility(returns, window=3)
    assert out.shape == returns.shape
    assert out.index.equals(returns.index)
    assert out.index.dtype == returns.index.dtype
    assert out.columns.tolist() == ["B", "A"]
    assert all(dtype == "float64" for dtype in out.dtypes)
    assert_frame_equal(returns, snapshot)
    assert out is not returns


def test_float32_input_gives_float64_output() -> None:
    """Una entrada `float32` es valida y la salida es `float64`."""
    returns = _rets({"A": [0.1, 0.2, 0.3, 0.4]}).astype("float32")
    out = realized_volatility(returns, window=2)
    assert out["A"].dtype == "float64"


def test_is_deterministic() -> None:
    """Dos llamadas iguales dan exactamente el mismo resultado."""
    rng = np.random.default_rng(8)
    returns = _rets({"A": rng.normal(0.0, 0.01, 100).tolist()})
    assert_frame_equal(realized_volatility(returns), realized_volatility(returns))


def test_defaults_are_window_60_and_share_0_8() -> None:
    """Los valores por defecto son `window=60` y `min_valid_share=0.8`."""
    returns = _rets({"A": [0.01] * 100})
    assert_frame_equal(
        realized_volatility(returns),
        realized_volatility(returns, window=60, min_valid_share=0.8),
    )
    assert np.isnan(realized_volatility(returns)["A"].iloc[46])
    assert np.isfinite(realized_volatility(returns)["A"].iloc[47])


def test_window_longer_than_data() -> None:
    """Con ventana mayor que los datos, solo hay valor si los datos cubren el 80 % de `W`."""
    out = _rv([0.01] * 10, window=100)
    assert np.isnan(out).all()
    out = _rv([0.01] * 10, window=12)
    assert np.isnan(out[:9]).all()
    assert np.isclose(out[9], 0.01 * np.sqrt(12), rtol=0, atol=1e-15)


# --- Avisos ------------------------------------------------------------------------------------


def test_no_runtime_warning_in_normal_and_edge_cases() -> None:
    """Ni los casos normales ni los bordes emiten `RuntimeWarning`."""
    index = build_time_grid(T0, T0 + 5 * MIN_MS, "1m")
    cases = [
        _rets({"A": [0.1, -0.1, 0.2, 0.0, 0.1]}),
        _rets({"A": [NAN] * 6, "B": [NAN, 0.1, NAN, NAN, 0.2, NAN]}),
        _rets({"A": [NAN] * 3}),
        pd.DataFrame({"A": np.array([], dtype="float64")}, index=index[:0]),
        pd.DataFrame(index=index),
        pd.DataFrame(index=index[:0]),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for returns in cases:
            for window in (1, 3, 60):
                realized_volatility(returns, window=window)
            realized_volatility(returns, window=3, min_valid_share=1.0)


def test_all_nan_gives_all_nan() -> None:
    """Todo NaN da todo NaN."""
    out = realized_volatility(_rets({"A": [NAN] * 5, "B": [NAN] * 5}), window=3)
    assert out.shape == (5, 2)
    assert out.isna().all().all()


def test_no_rows_and_no_columns_keep_shape() -> None:
    """Sin filas o sin columnas conserva la forma y el dtype."""
    index = build_time_grid(T0, T0 + 3 * MIN_MS, "1m")
    no_cols = realized_volatility(pd.DataFrame(index=index), window=3)
    assert no_cols.shape == (3, 0)
    assert no_cols.index.equals(index)
    no_rows = realized_volatility(
        pd.DataFrame({"A": np.array([], dtype="float64")}, index=index[:0]), window=3
    )
    assert no_rows.shape == (0, 1)
    assert no_rows["A"].dtype == "float64"


def test_single_row() -> None:
    """Con una sola fila: NaN si `W` es mayor que 1, y `|r|` si `W=1`."""
    assert np.isnan(_rv([0.1], window=3)).all()
    assert _rv([0.1], window=1).tolist() == [0.1]


# --- Validaciones ------------------------------------------------------------------------------


@pytest.mark.parametrize("window", [0, -1, -60])
def test_window_below_one_raises(window: int) -> None:
    """`window < 1` lanza `ValueError`."""
    with pytest.raises(ValueError, match="window"):
        realized_volatility(_rets({"A": [0.1, 0.2]}), window=window)


@pytest.mark.parametrize("share", [0.0, -0.1, 1.0000001, 2.0, np.nan, np.inf, -np.inf])
def test_min_valid_share_outside_range_raises(share: float) -> None:
    """`min_valid_share` fuera de `(0, 1]` (incluidos 0, NaN e inf) lanza `ValueError`."""
    with pytest.raises(ValueError, match="min_valid_share"):
        realized_volatility(_rets({"A": [0.1, 0.2]}), min_valid_share=share)


@pytest.mark.parametrize("share", [1e-9, 0.5, 1.0])
def test_min_valid_share_inside_range_is_accepted(share: float) -> None:
    """Los extremos validos `(0, 1]` no lanzan."""
    realized_volatility(_rets({"A": [0.1, 0.2]}), window=2, min_valid_share=share)


def test_non_float_column_raises() -> None:
    """Una columna entera, bool u object lanza `ValueError`, aunque las demas sean float."""
    index = build_time_grid(T0, T0 + 2 * MIN_MS, "1m")
    for bad in ([1, 2], [True, False], ["a", "b"]):
        frame = pd.DataFrame({"A": [0.1, 0.2], "B": bad}, index=index)
        with pytest.raises(ValueError, match="float"):
            realized_volatility(frame)


def test_non_increasing_index_raises() -> None:
    """Un indice repetido o decreciente lanza `ValueError`."""
    returns = _rets({"A": [0.1, 0.2, 0.3]})
    duplicated = returns.copy()
    duplicated.index = returns.index[[0, 1, 1]]
    with pytest.raises(ValueError, match="indice"):
        realized_volatility(duplicated)
    decreasing = returns.iloc[::-1]
    with pytest.raises(ValueError, match="indice"):
        realized_volatility(decreasing)
    unsorted = returns.iloc[[0, 2, 1]]
    with pytest.raises(ValueError, match="indice"):
        realized_volatility(unsorted)


def test_validation_does_not_depend_on_data() -> None:
    """Los parametros se validan tambien con una tabla vacia."""
    empty = pd.DataFrame(index=build_time_grid(T0, T0 + 3 * MIN_MS, "1m"))
    with pytest.raises(ValueError, match="window"):
        realized_volatility(empty, window=0)


# --- realized_volatility_from_cache ------------------------------------------------------------


def _candles(
    minutes: list[int],
    close: list[float],
    num_trades: list[int] | None = None,
    *,
    step_ms: int = MIN_MS,
) -> pd.DataFrame:
    """Klines en los slots dados (multiplos de `step_ms` desde T0) con esos cierres."""
    n = len(minutes)
    open_times = [T0 + k * step_ms for k in minutes]
    trades = [7] * n if num_trades is None else num_trades
    data = {
        "open_time": open_times,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": [1.0] * n,
        "close_time": [t + step_ms - 1 for t in open_times],
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
    *,
    interval: str = "1m",
    step_ms: int = MIN_MS,
) -> None:
    """Guarda en `base_dir` una cache de `symbol`."""
    save_klines(
        _candles(minutes, close, num_trades, step_ms=step_ms),
        symbol,
        interval,
        base_dir=base_dir,
    )


def _prices(n: int, seed: int = 0) -> list[float]:
    """Cierres deterministas, positivos y no triviales."""
    rng = np.random.default_rng(seed)
    return list(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n))))


def test_cache_first_row_window_uses_returns_before_the_range(tmp_path: Path) -> None:
    """La RV de la primera fila usa los `W-1` retornos anteriores a `start`, calculada a mano."""
    close = [100.0 + k * k for k in range(60)]
    _cache(tmp_path, "A", list(range(60)), close)
    window, start_min = 5, 20
    out = realized_volatility_from_cache(
        ["A"],
        "1m",
        T0 + start_min * MIN_MS,
        T0 + 30 * MIN_MS,
        window=window,
        base_dir=tmp_path,
    )

    def ret(m: int) -> float:  # retorno de la fila m: vela m-1 sobre vela m-2
        return float(np.log(close[m - 1] / close[m - 2]))

    first = np.sqrt(sum(ret(m) ** 2 for m in range(start_min - window + 1, start_min + 1)))
    assert np.isclose(out["A"].iloc[0], first, rtol=0, atol=1e-15)
    only_in_range = np.sqrt(window / 1 * ret(start_min) ** 2)
    assert not np.isclose(out["A"].iloc[0], only_in_range)  # no solo la fila start
    last = np.sqrt(sum(ret(m) ** 2 for m in range(29 - window + 1, 30)))
    assert np.isclose(out["A"].iloc[-1], last, rtol=0, atol=1e-15)


def test_cache_output_is_exactly_the_requested_grid(tmp_path: Path) -> None:
    """El indice de salida es `build_time_grid(start, end)`, con su dtype y sin las filas extra."""
    _cache(tmp_path, "A", list(range(60)), _prices(60))
    start, end = T0 + 20 * MIN_MS, T0 + 40 * MIN_MS
    out = realized_volatility_from_cache(["A"], "1m", start, end, window=10, base_dir=tmp_path)
    grid = build_time_grid(start, end, "1m")
    assert out.index.equals(grid)
    assert out.index.dtype == grid.dtype
    assert out.index.freq is None
    assert len(out) == 20
    assert all(dtype == "float64" for dtype in out.dtypes)


def test_cache_unaligned_start_and_end(tmp_path: Path) -> None:
    """`start_ms` y `end_ms` no alineados dan la rejilla de `build_time_grid`."""
    _cache(tmp_path, "A", list(range(60)), _prices(60))
    start, end = T0 + 20 * MIN_MS + 1, T0 + 40 * MIN_MS - 1
    out = realized_volatility_from_cache(["A"], "1m", start, end, window=10, base_dir=tmp_path)
    assert out.index.equals(build_time_grid(start, end, "1m"))


def test_cache_symbol_without_cache_is_all_nan_column(tmp_path: Path) -> None:
    """Un simbolo sin cache es una columna de NaN y no estropea a los demas."""
    _cache(tmp_path, "A", list(range(60)), _prices(60))
    out = realized_volatility_from_cache(
        ["A", "GHOST"], "1m", T0 + 20 * MIN_MS, T0 + 30 * MIN_MS, window=5, base_dir=tmp_path
    )
    assert out.columns.tolist() == ["A", "GHOST"]
    assert out["GHOST"].isna().all()
    assert out["A"].notna().all()


def test_cache_window_longer_than_range(tmp_path: Path) -> None:
    """Con ventana mayor que el rango, la ventana de cada fila usa retornos de antes del rango."""
    _cache(tmp_path, "A", list(range(100)), _prices(100))
    start, end = T0 + 50 * MIN_MS, T0 + 60 * MIN_MS
    out = realized_volatility_from_cache(["A"], "1m", start, end, window=30, base_dir=tmp_path)
    assert len(out) == 10
    assert out["A"].notna().all()
    wide = realized_volatility_from_cache(
        ["A"], "1m", T0 + 20 * MIN_MS, end, window=30, base_dir=tmp_path
    )
    assert_frame_equal(wide.loc[out.index], out)


def test_cache_range_starting_at_cache_start_is_nan_until_coverage(tmp_path: Path) -> None:
    """Sin datos antes del inicio de la cache, hay NaN hasta que la ventana cubre el 80 %."""
    _cache(tmp_path, "A", list(range(40)), _prices(40))
    out = realized_volatility_from_cache(
        ["A"], "1m", T0, T0 + 12 * MIN_MS, window=5, base_dir=tmp_path
    )
    # Retornos validos desde la fila 2 (las filas 0 y 1 son NaN, como en `returns_from_cache`);
    # la fila 5 es la primera con 4 de 5 validos (filas 2..5).
    assert out["A"].isna().tolist()[:7] == [True] * 5 + [False, False]
    assert out["A"].iloc[5:].notna().all()


def test_cache_interval_without_fixed_step_raises(tmp_path: Path) -> None:
    """`1M` no tiene paso fijo: `ValueError`."""
    with pytest.raises(ValueError, match="duracion fija"):
        realized_volatility_from_cache(["A"], "1M", T0, T0 + 60 * MIN_MS, base_dir=tmp_path)


def test_cache_empty_range_raises(tmp_path: Path) -> None:
    """Un rango sin slots (vacio o invertido) lanza `ValueError`."""
    _cache(tmp_path, "A", list(range(30)), _prices(30))
    with pytest.raises(ValueError, match="vaci"):
        realized_volatility_from_cache(["A"], "1m", T0 + 1, T0 + MIN_MS - 1, base_dir=tmp_path)
    with pytest.raises(ValueError, match="vaci"):
        realized_volatility_from_cache(
            ["A"], "1m", T0 + 5 * MIN_MS, T0 + 5 * MIN_MS, base_dir=tmp_path
        )
    with pytest.raises(ValueError, match="vaci"):
        realized_volatility_from_cache(
            ["A"], "1m", T0 + 6 * MIN_MS, T0 + 5 * MIN_MS, base_dir=tmp_path
        )


def test_cache_invalid_params_raise(tmp_path: Path) -> None:
    """`window` y `min_valid_share` invalidos lanzan `ValueError` antes de leer la cache."""
    with pytest.raises(ValueError, match="window"):
        realized_volatility_from_cache(
            ["A"], "1m", T0, T0 + 10 * MIN_MS, window=0, base_dir=tmp_path
        )
    with pytest.raises(ValueError, match="min_valid_share"):
        realized_volatility_from_cache(
            ["A"], "1m", T0, T0 + 10 * MIN_MS, min_valid_share=0.0, base_dir=tmp_path
        )


def test_cache_result_equals_manual_chain(tmp_path: Path) -> None:
    """Identico a `returns_from_cache` en `[start-(W-1)p, end)`, `realized_volatility` y recorte."""
    n = 80
    trades_a = [7] * n
    trades_a[30] = 0
    trades_a[31] = 0
    _cache(tmp_path, "A", [k for k in range(n) if k != 45], _prices(n - 1), trades_a[: n - 1])
    _cache(tmp_path, "B", list(range(n)), _prices(n, seed=1))
    start, end = T0 + 20 * MIN_MS, T0 + 70 * MIN_MS
    symbols = ["A", "B", "GHOST"]
    for window, share in ((10, 0.8), (1, 0.8), (25, 0.5)):
        out = realized_volatility_from_cache(
            symbols,
            "1m",
            start,
            end,
            window=window,
            min_valid_share=share,
            base_dir=tmp_path,
        )
        returns = returns_from_cache(
            symbols, "1m", start - (window - 1) * MIN_MS, end, base_dir=tmp_path
        )
        manual = realized_volatility(returns, window=window, min_valid_share=share).iloc[
            window - 1 :
        ]
        assert_frame_equal(out, manual)


def test_cache_right_trim_does_not_change_common_values(tmp_path: Path) -> None:
    """Recortar el rango por la derecha no cambia los valores de las filas comunes."""
    _cache(tmp_path, "A", list(range(100)), _prices(100))
    full = realized_volatility_from_cache(
        ["A"], "1m", T0 + 30 * MIN_MS, T0 + 90 * MIN_MS, window=10, base_dir=tmp_path
    )
    short = realized_volatility_from_cache(
        ["A"], "1m", T0 + 30 * MIN_MS, T0 + 50 * MIN_MS, window=10, base_dir=tmp_path
    )
    assert_frame_equal(full.loc[short.index], short)


def test_cache_no_leak_candles_after_tau_do_not_change_rv_up_to_tau(tmp_path: Path) -> None:
    """Cambiar los cierres desde la vela k no altera la RV de las filas conocidas antes de k+1."""
    n, k = 80, 50
    close = _prices(n)
    changed = close[:k] + [p * 3.0 for p in close[k:]]
    _cache(tmp_path / "base", "A", list(range(n)), close)
    _cache(tmp_path / "changed", "A", list(range(n)), changed)
    start, end = T0 + 20 * MIN_MS, T0 + 70 * MIN_MS
    a = realized_volatility_from_cache(
        ["A"], "1m", start, end, window=10, base_dir=tmp_path / "base"
    )
    b = realized_volatility_from_cache(
        ["A"], "1m", start, end, window=10, base_dir=tmp_path / "changed"
    )
    # La vela k abre en el minuto k y se conoce en el k+1: las filas <= k no la usan.
    cut = pd.Timestamp(T0 + k * MIN_MS, unit="ms", tz="UTC")
    assert_frame_equal(a.loc[:cut], b.loc[:cut])
    assert not a.loc[cut + pd.Timedelta(minutes=1) :].equals(b.loc[cut + pd.Timedelta(minutes=1) :])


def test_cache_step_follows_the_interval(tmp_path: Path) -> None:
    """Con velas de 5m, los `W-1` retornos extra son `W-1` pasos de 5 minutos."""
    step = 5 * MIN_MS
    close = _prices(60)
    _cache(tmp_path, "A", list(range(60)), close, interval="5m", step_ms=step)
    start, end = T0 + 20 * step, T0 + 30 * step
    out = realized_volatility_from_cache(["A"], "5m", start, end, window=4, base_dir=tmp_path)
    assert out.index.equals(build_time_grid(start, end, "5m"))
    returns = returns_from_cache(["A"], "5m", start - 3 * step, end, base_dir=tmp_path)
    manual = realized_volatility(returns, window=4).iloc[3:]
    assert_frame_equal(out, manual)
    assert out["A"].notna().all()


def test_cache_is_deterministic(tmp_path: Path) -> None:
    """Dos llamadas iguales dan exactamente el mismo resultado."""
    _cache(tmp_path, "A", list(range(60)), _prices(60))
    args = (["A"], "1m", T0 + 20 * MIN_MS, T0 + 50 * MIN_MS)
    assert_frame_equal(
        realized_volatility_from_cache(*args, window=10, base_dir=tmp_path),
        realized_volatility_from_cache(*args, window=10, base_dir=tmp_path),
    )
