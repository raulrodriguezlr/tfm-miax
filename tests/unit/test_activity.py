"""Tests del volumen relativo al propio pasado: series sinteticas y caches en `tmp_path`."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from miax.features import (
    align_klines,
    build_time_grid,
    classify_klines,
    log_relative_to_past,
    relative_volume_from_cache,
    shift_to_close,
    validity_mask,
)
from miax.features.volatility import check_window_params
from miax.ingest import save_klines
from miax.ingest.klines import KLINE_COLUMNS, KLINE_DTYPES

MIN_MS = 60_000
T0 = 1_767_225_600_000  # 2026-01-01T00:00:00Z
NAN = float("nan")
LN2 = float(np.log(2.0))


def _grid(n: int, interval: str = "1m") -> pd.DatetimeIndex:
    """Rejilla de `n` aperturas desde T0."""
    step = {"1m": MIN_MS, "5m": 5 * MIN_MS}[interval]
    return build_time_grid(T0, T0 + n * step, interval)


def _vol(data: Mapping[str, Sequence[float]]) -> pd.DataFrame:
    """Tabla de volumenes `tiempo x simbolo` sobre la rejilla de 1m desde T0."""
    n = len(next(iter(data.values())))
    return pd.DataFrame(
        {s: np.asarray(v, dtype="float64") for s, v in data.items()}, index=_grid(n)
    )


def _valid_of(frame: pd.DataFrame, invalid: Sequence[int] = ()) -> pd.DataFrame:
    """Mascara de validez: valido donde hay dato, salvo en las filas `invalid`."""
    valid = frame.notna()
    valid.iloc[list(invalid)] = False
    return valid.astype(bool)


def _rel(values: Sequence[float], invalid: Sequence[int] = (), **kwargs: float) -> np.ndarray:
    """Volumen relativo de una sola serie como vector de numpy."""
    frame = _vol({"A": values})
    return log_relative_to_past(frame, _valid_of(frame, invalid), "1m", **kwargs)["A"].to_numpy()


def _random_volume(n: int, seed: int = 0) -> np.ndarray:
    """Volumenes deterministas, positivos y con dispersion."""
    rng = np.random.default_rng(seed)
    return np.exp(rng.normal(10.0, 1.0, n))


# --- Valores -----------------------------------------------------------------------------------


@pytest.mark.parametrize("window", [1, 2, 5, 30])
def test_constant_volume_gives_zero(window: int) -> None:
    """Volumen constante: en cuanto hay base, `rel` es 0."""
    out = _rel([123.0] * (window + 10), window=window, min_valid_share=1.0)
    assert np.isnan(out[:window]).all()
    assert np.allclose(out[window:], 0.0, rtol=0, atol=1e-12)


def test_one_candle_with_double_volume_gives_ln2() -> None:
    """Una vela con el doble del volumen constante da `ln 2` y las vecinas siguen en 0."""
    values = [50.0] * 12
    values[8] = 100.0
    out = _rel(values, window=4, min_valid_share=1.0)
    assert np.isclose(out[8], LN2, rtol=0, atol=1e-12)
    assert np.allclose(out[4:8], 0.0, rtol=0, atol=1e-12)


def test_known_values_computed_by_hand_w3() -> None:
    """W=3 sobre [1,2,4,8,16,32]: la base es la media de 3 logs anteriores (calculado a mano)."""
    out = _rel([1, 2, 4, 8, 16, 32], window=3, min_valid_share=1.0)
    assert np.isnan(out[:3]).all()
    # Fila 3: ln 8 - media(ln 1, ln 2, ln 4) = 3 ln2 - 1 ln2 = 2 ln2 = ln 4.
    assert np.allclose(out[3:], 2 * LN2, rtol=0, atol=1e-12)
    # Lo mismo, escrito explicitamente con la media de los logs.
    logs = np.log([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    for tau in (3, 4, 5):
        assert np.isclose(out[tau], logs[tau] - logs[tau - 3 : tau].mean(), rtol=0, atol=1e-12)


def test_known_values_with_an_invalid_slot_in_the_base() -> None:
    """W=3, share 2/3, slot 2 invalido (con basura): la base usa solo los validos de la ventana."""
    out = _rel([1, 2, 999.0, 8, 16, 32], invalid=[2], window=3, min_valid_share=2 / 3)
    assert np.isnan(out[0])  # sin base
    assert np.isnan(out[1])  # 1 valido de 3 < 2/3
    assert np.isnan(out[2])  # el slot no es valido
    assert np.isclose(out[3], 3 * LN2 - 0.5 * LN2, rtol=0, atol=1e-12)  # base {ln1, ln2}
    assert np.isclose(out[4], 4 * LN2 - 2.0 * LN2, rtol=0, atol=1e-12)  # base {ln2, ln8}
    assert np.isclose(out[5], 5 * LN2 - 3.5 * LN2, rtol=0, atol=1e-12)  # base {ln8, ln16}


# --- Comparabilidad ----------------------------------------------------------------------------


def test_scale_does_not_matter_two_assets_x1000_give_the_same_rel() -> None:
    """Criterio de hecho: dos activos con la misma forma y escala x1000 dan el mismo `rel`."""
    small = _random_volume(300, seed=1)
    small[[40, 41, 120]] = NAN
    frame = _vol({"SMALL": small.tolist(), "BIG": (small * 1000.0).tolist()})
    out = log_relative_to_past(frame, frame.notna(), "1m", window=50, min_valid_share=0.5)
    assert out["SMALL"].notna().sum() > 200
    assert out["SMALL"].isna().equals(out["BIG"].isna())
    assert np.allclose(out["SMALL"], out["BIG"], rtol=0, atol=1e-12, equal_nan=True)


def test_rel_is_scale_invariant_for_any_positive_factor() -> None:
    """Multiplicar el volumen por una constante deja `rel` igual."""
    base = _random_volume(200, seed=2)
    for factor in (1e-6, 0.37, 1e9):
        assert np.allclose(
            _rel(base.tolist(), window=30),
            _rel((base * factor).tolist(), window=30),
            rtol=0,
            atol=1e-9,
            equal_nan=True,
        )


# --- Validez -----------------------------------------------------------------------------------


def test_invalid_slot_gives_nan_even_with_a_volume_value() -> None:
    """Un slot no valido (NO_TRADES, sin vela) da NaN en su fila."""
    values = [10.0] * 10
    values[6] = 0.0  # NO_TRADES: volumen 0
    values[7] = NAN  # sin vela
    out = _rel(values, invalid=[6, 7], window=4, min_valid_share=0.5)
    assert np.isnan(out[6]) and np.isnan(out[7])
    assert np.isfinite(out[[4, 5, 8, 9]]).all()


def test_invalid_slots_do_not_enter_the_base() -> None:
    """Cambiar el valor de un slot no valido no cambia nada; lo mismo que quitarlo de la base."""
    values = _random_volume(60, seed=3)
    values[20] = NAN
    values[35] = 0.0
    other = values.copy()
    other[20] = 5e9
    other[35] = 1e-9
    a = _rel(values.tolist(), invalid=[20, 35], window=10, min_valid_share=0.5)
    b = _rel(other.tolist(), invalid=[20, 35], window=10, min_valid_share=0.5)
    assert np.array_equal(a, b, equal_nan=True)


def test_base_is_the_mean_of_valid_logs_only() -> None:
    """Con un hueco en la ventana, la base es la media de los logs de los validos (no de W)."""
    values = _random_volume(40, seed=4)
    out = _rel(values.tolist(), invalid=[10], window=8, min_valid_share=0.5)
    tau = 14
    window_rows = [r for r in range(tau - 8, tau) if r != 10]
    want = np.log(values[tau]) - np.log(values[window_rows]).mean()
    assert np.isclose(out[tau], want, rtol=0, atol=1e-12)


def test_coverage_threshold_is_inclusive_w3() -> None:
    """W=3, share 2/3: 2 validos dan valor y 1 valido da NaN."""
    two = _rel([1, 2, 999.0, 8, 4], invalid=[2], window=3, min_valid_share=2 / 3)
    assert np.isfinite(two[3])  # ventana filas 0..2: validos 0 y 1 = 2
    one = _rel([1, 999.0, 999.0, 8, 4], invalid=[1, 2], window=3, min_valid_share=2 / 3)
    assert np.isnan(one[3])  # ventana filas 0..2: solo el 0 = 1


@pytest.mark.parametrize("share", [0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
def test_inclusive_threshold_for_other_shares(share: float) -> None:
    """Con `W=10` y `share = k/10`, `k` validos dan valor y `k-1` dan NaN."""
    window = 10
    k = round(share * window)
    for valid_count, expect_value in ((k, True), (k - 1, False)):
        values = [3.0] * (window + 1)
        invalid = list(range(window - valid_count))  # las primeras filas de la base
        out = _rel(values, invalid=invalid, window=window, min_valid_share=share)
        assert bool(np.isfinite(out[window])) is expect_value, (share, valid_count)


def test_never_minus_inf_nor_zero_for_lack_of_data() -> None:
    """Sin datos o con volumen 0 no valido nunca sale -inf ni 0: sale NaN."""
    frame = _vol({"A": [0.0] * 8, "B": [NAN] * 8, "C": [0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0, 5.0]})
    valid = frame.notna() & (frame > 0)
    out = log_relative_to_past(frame, valid, "1m", window=3, min_valid_share=1e-9)
    assert not np.isinf(out.to_numpy()).any()
    assert out["A"].isna().all() and out["B"].isna().all()
    # C: las tres primeras filas no son validas; el primer valido (fila 3) no tiene base.
    assert out["C"].isna().iloc[:4].all()
    # Fila 4: base = solo ln 5 (1 valido de 3 >= 1e-9): rel 0 por volumen igual, un valor real.
    assert np.isclose(out["C"].iloc[4], 0.0, rtol=0, atol=1e-12)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("-inf"), float("nan")])
def test_bad_value_in_a_valid_slot_raises_with_the_symbol(bad: float) -> None:
    """0, negativo, inf o NaN en un slot valido es dato corrupto: `ValueError` con el simbolo."""
    frame = _vol({"OK": [5.0] * 6, "BADSYM": [5.0, 5.0, bad, 5.0, 5.0, 5.0]})
    valid = pd.DataFrame(True, index=frame.index, columns=frame.columns)
    with pytest.raises(ValueError, match="BADSYM") as info:
        log_relative_to_past(frame, valid, "1m", window=3)
    assert "OK" not in str(info.value)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
def test_same_values_in_an_invalid_slot_do_not_raise(bad: float) -> None:
    """Los mismos valores en un slot no valido son un hueco legitimo: no lanzan."""
    frame = _vol({"A": [5.0, 5.0, bad, 5.0, 5.0, 5.0]})
    valid = pd.DataFrame(True, index=frame.index, columns=frame.columns)
    valid.iloc[2] = False
    out = log_relative_to_past(frame, valid, "1m", window=3, min_valid_share=0.5)
    assert np.isnan(out["A"].iloc[2])
    assert out.shape == frame.shape


# --- No-fuga -----------------------------------------------------------------------------------


def test_no_leak_changing_future_volume_does_not_change_past_rel() -> None:
    """Cambiar los volumenes desde la fila k no altera `rel` de las filas anteriores a k."""
    base = _random_volume(80, seed=5)
    base[[10, 33]] = NAN
    k = 50
    changed = base.copy()
    changed[k:] = changed[k:] * 7.0 + 1e4
    a = log_relative_to_past(_vol({"A": base}), _vol({"A": base}).notna(), "1m", window=12)
    cb = _vol({"A": changed})
    b = log_relative_to_past(cb, cb.notna(), "1m", window=12)
    assert_frame_equal(a.iloc[:k], b.iloc[:k], rtol=0, atol=1e-12)
    assert not np.allclose(a.iloc[k:], b.iloc[k:], equal_nan=True)


def test_no_leak_nan_future_does_not_change_past_rel() -> None:
    """Volver NaN el futuro tampoco cambia el pasado."""
    base = _random_volume(60, seed=6)
    masked = base.copy()
    masked[35:] = NAN
    fa, fb = _vol({"A": base}), _vol({"A": masked})
    a = log_relative_to_past(fa, fa.notna(), "1m", window=8)
    b = log_relative_to_past(fb, fb.notna(), "1m", window=8)
    assert_frame_equal(a.iloc[:35], b.iloc[:35], rtol=0, atol=1e-12)


def test_base_of_tau_excludes_its_own_candle() -> None:
    """Cambiar `V_tau` por un factor `f` mueve `rel_tau` en `ln f` (solo el numerador)."""
    window, tau, factor = 10, 30, 5.0
    base = _random_volume(60, seed=7)
    changed = base.copy()
    changed[tau] *= factor
    a = _rel(base.tolist(), window=window)
    b = _rel(changed.tolist(), window=window)
    assert np.isclose(b[tau] - a[tau], np.log(factor), rtol=0, atol=1e-12)
    # El pasado no cambia; el siguiente si (entra en su base con peso 1/W); desde tau+W+1, no.
    assert np.array_equal(a[:tau], b[:tau], equal_nan=True)
    assert np.isclose(b[tau + 1] - a[tau + 1], -np.log(factor) / window, rtol=0, atol=1e-12)
    assert np.allclose(a[tau + window + 1 :], b[tau + window + 1 :], rtol=0, atol=1e-12)


def test_right_trim_does_not_change_common_values() -> None:
    """Recortar por la derecha deja identicas las filas comunes."""
    values = _random_volume(100, seed=8)
    values[[5, 6, 60]] = NAN
    full, short = _vol({"A": values}), _vol({"A": values[:55]})
    a = log_relative_to_past(full, full.notna(), "1m", window=12)
    b = log_relative_to_past(short, short.notna(), "1m", window=12)
    assert_frame_equal(a.iloc[:55], b, rtol=0, atol=1e-12)


# --- Indice ------------------------------------------------------------------------------------


def test_output_index_is_shifted_one_step_to_the_close() -> None:
    """La fila `tau` de la salida es la vela que abre en `tau - paso`: indice `+ paso` (D-19)."""
    frame = _vol({"A": [1.0, 2.0, 3.0, 4.0, 5.0]})
    out = log_relative_to_past(frame, frame.notna(), "1m", window=2)
    assert out.index.equals(frame.index + pd.Timedelta(minutes=1))
    assert out.index[0] == pd.Timestamp(T0 + MIN_MS, unit="ms", tz="UTC")
    assert str(out.index.dtype) == "datetime64[ms, UTC]"
    assert out.index.freq is None
    # El valor de la fila 2 (vela de apertura T0+2) esta bajo el indice T0+3.
    assert np.isclose(out["A"].iloc[2], np.log(3.0) - np.log([1.0, 2.0]).mean(), atol=1e-12)


def test_output_index_follows_the_interval() -> None:
    """Con 5m, el desplazamiento es de 5 minutos."""
    index = _grid(6, "5m")
    frame = pd.DataFrame({"A": np.arange(1.0, 7.0)}, index=index)
    out = log_relative_to_past(frame, frame.notna(), "5m", window=2)
    assert out.index.equals(index + pd.Timedelta(minutes=5))


# --- Estructura --------------------------------------------------------------------------------


def test_output_shape_columns_dtype_and_input_unchanged() -> None:
    """Misma forma y columnas, `float64`, y las entradas no se modifican."""
    frame = _vol({"B": [1.0, NAN, 2.0, 3.0], "A": [4.0, 5.0, 6.0, 7.0]})
    valid = frame.notna()
    snap_f, snap_v = frame.copy(), valid.copy()
    out = log_relative_to_past(frame, valid, "1m", window=2)
    assert out.shape == frame.shape
    assert out.columns.tolist() == ["B", "A"]
    assert all(dtype == "float64" for dtype in out.dtypes)
    assert_frame_equal(frame, snap_f)
    assert_frame_equal(valid, snap_v)


def test_columns_are_independent() -> None:
    """Lo que pasa en una columna no afecta a la otra."""
    a = _random_volume(30, seed=9).tolist()
    b = _random_volume(30, seed=10).tolist()
    b[4] = NAN
    both = _vol({"A": a, "B": b})
    only_a, only_b = _vol({"A": a}), _vol({"B": b})
    out = log_relative_to_past(both, both.notna(), "1m", window=5)
    assert_frame_equal(out[["A"]], log_relative_to_past(only_a, only_a.notna(), "1m", window=5))
    assert_frame_equal(out[["B"]], log_relative_to_past(only_b, only_b.notna(), "1m", window=5))


def test_is_deterministic() -> None:
    """Dos llamadas iguales dan exactamente el mismo resultado."""
    frame = _vol({"A": _random_volume(100, seed=11).tolist()})
    args = (frame, frame.notna(), "1m")
    assert_frame_equal(log_relative_to_past(*args), log_relative_to_past(*args))


def test_defaults_are_window_10080_and_share_0_5() -> None:
    """Valores por defecto: `window=10080` y `min_valid_share=0.5` (umbral inclusivo)."""
    frame = _vol({"A": [4.0] * 10_100})
    valid = frame.notna()
    default = log_relative_to_past(frame, valid, "1m")
    explicit = log_relative_to_past(frame, valid, "1m", window=10080, min_valid_share=0.5)
    assert_frame_equal(default, explicit)
    # 5040 de 10080 = 0,5 exacto: la primera fila con base suficiente es la 5040.
    assert default["A"].iloc[:5040].isna().all()
    assert np.isfinite(default["A"].iloc[5040:]).all()


def test_window_longer_than_data_is_all_nan() -> None:
    """Con ventana mayor que los datos no hay base suficiente: todo NaN."""
    assert np.isnan(_rel([1.0] * 10, window=100)).all()


def test_all_nan_and_single_row() -> None:
    """Todo NaN da todo NaN; una sola fila da NaN."""
    assert np.isnan(_rel([NAN] * 6, window=3)).all()
    assert np.isnan(_rel([2.0], window=3)).all()
    assert np.isnan(_rel([2.0], window=1)).all()


def test_window_one_uses_only_the_previous_candle() -> None:
    """`window=1`: `rel` es el log del cociente con la vela anterior valida."""
    out = _rel([1.0, 2.0, 8.0, 8.0], window=1)
    assert np.isnan(out[0])
    assert np.allclose(out[1:], [LN2, 2 * LN2, 0.0], rtol=0, atol=1e-12)


def test_no_runtime_warning_in_normal_and_edge_cases() -> None:
    """Ni los casos normales ni los bordes emiten `RuntimeWarning` (ni ningun aviso)."""
    index = _grid(5)
    cases = [
        _vol({"A": [1.0, 2.0, 3.0, 4.0, 5.0]}),
        _vol({"A": [NAN] * 5, "B": [NAN, 2.0, NAN, NAN, 3.0]}),
        pd.DataFrame({"A": np.array([], dtype="float64")}, index=index[:0]),
        pd.DataFrame(index=index),
        pd.DataFrame(index=index[:0]),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for frame in cases:
            valid = frame.notna().astype(bool)
            for window in (1, 3, 60):
                log_relative_to_past(frame, valid, "1m", window=window)
            log_relative_to_past(frame, valid, "1m", window=3, min_valid_share=1.0)
        # Volumen 0 no valido (log 0 se enmascara antes de calcular).
        zeros = _vol({"A": [0.0, 0.0, 3.0, 3.0, 3.0]})
        log_relative_to_past(zeros, zeros > 0, "1m", window=2)


def test_no_rows_and_no_columns_keep_shape() -> None:
    """Sin filas o sin columnas conserva la forma, el indice desplazado y el dtype."""
    index = _grid(3)
    no_cols = pd.DataFrame(index=index)
    out = log_relative_to_past(no_cols, no_cols.astype(bool), "1m", window=3)
    assert out.shape == (3, 0)
    assert out.index.equals(index + pd.Timedelta(minutes=1))
    empty = pd.DataFrame({"A": np.array([], dtype="float64")}, index=index[:0])
    out = log_relative_to_past(empty, empty.astype(bool), "1m", window=3)
    assert out.shape == (0, 1)
    assert out["A"].dtype == "float64"


# --- Validaciones ------------------------------------------------------------------------------


@pytest.mark.parametrize("window", [0, -1, -60])
def test_window_below_one_raises(window: int) -> None:
    """`window < 1` lanza `ValueError`."""
    frame = _vol({"A": [1.0, 2.0]})
    with pytest.raises(ValueError, match="window"):
        log_relative_to_past(frame, frame.notna(), "1m", window=window)


@pytest.mark.parametrize("share", [0.0, -0.1, 1.0000001, 2.0, np.nan, np.inf, -np.inf])
def test_min_valid_share_outside_range_raises(share: float) -> None:
    """`min_valid_share` fuera de `(0, 1]` lanza `ValueError`."""
    frame = _vol({"A": [1.0, 2.0]})
    with pytest.raises(ValueError, match="min_valid_share"):
        log_relative_to_past(frame, frame.notna(), "1m", min_valid_share=share)


def test_non_float_values_raise() -> None:
    """Una columna entera, bool u object lanza `ValueError`."""
    index = _grid(2)
    for bad in ([1, 2], [True, False], ["a", "b"]):
        frame = pd.DataFrame({"A": [1.0, 2.0], "B": bad}, index=index)
        with pytest.raises(ValueError, match="float"):
            log_relative_to_past(frame, pd.DataFrame(True, index=index, columns=["A", "B"]), "1m")


def test_non_bool_valid_raises() -> None:
    """Una mascara no bool lanza `ValueError`."""
    frame = _vol({"A": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="bool"):
        log_relative_to_past(frame, frame.astype("float64"), "1m")


def test_mismatched_valid_raises() -> None:
    """Una mascara con otro indice o columnas lanza `ValueError`."""
    frame = _vol({"A": [1.0, 2.0, 3.0]})
    valid = frame.notna().rename(columns={"A": "Z"})
    with pytest.raises(ValueError, match="mismo indice"):
        log_relative_to_past(frame, valid, "1m")


def test_non_increasing_index_raises() -> None:
    """Un indice repetido, decreciente o desordenado lanza `ValueError`."""
    frame = _vol({"A": [1.0, 2.0, 3.0]})
    duplicated = frame.copy()
    duplicated.index = frame.index[[0, 1, 1]]
    decreasing = frame.iloc[::-1]
    unsorted = frame.iloc[[0, 2, 1]]
    for bad in (duplicated, decreasing, unsorted):
        with pytest.raises(ValueError, match="indice"):
            log_relative_to_past(bad, bad.notna(), "1m")


def test_interval_without_fixed_step_raises() -> None:
    """`1M` no tiene paso fijo: `ValueError`."""
    frame = _vol({"A": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="duracion fija"):
        log_relative_to_past(frame, frame.notna(), "1M")


def test_validation_does_not_depend_on_data() -> None:
    """Los parametros se validan tambien con una tabla vacia."""
    empty = pd.DataFrame(index=_grid(3))
    with pytest.raises(ValueError, match="window"):
        log_relative_to_past(empty, empty.astype(bool), "1m", window=0)


# --- shift_to_close ----------------------------------------------------------------------------


def test_shift_to_close_moves_one_step() -> None:
    """Desplaza el indice un paso del intervalo."""
    index = _grid(4)
    out = shift_to_close(index, "1m")
    assert out.equals(index + pd.Timedelta(minutes=1))
    assert shift_to_close(_grid(3, "5m"), "5m").equals(_grid(3, "5m") + pd.Timedelta(minutes=5))


def test_shift_to_close_keeps_utc_unit_and_has_no_freq() -> None:
    """Conserva UTC y `datetime64[ms]`, y no tiene `freq`."""
    out = shift_to_close(_grid(4), "1m")
    assert str(out.dtype) == "datetime64[ms, UTC]"
    assert str(out.tz) == "UTC"
    assert out.freq is None


def test_shift_to_close_with_other_timezone_gives_utc() -> None:
    """Un indice en otra zona horaria sale en UTC con el mismo instante desplazado."""
    index = _grid(3).tz_convert("Europe/Madrid")
    out = shift_to_close(index, "1m")
    assert str(out.tz) == "UTC"
    assert out.equals(_grid(3).tz_convert("UTC") + pd.Timedelta(minutes=1))


def test_shift_to_close_empty_index() -> None:
    """Un indice vacio da un indice vacio UTC."""
    out = shift_to_close(_grid(3)[:0], "1m")
    assert len(out) == 0
    assert str(out.dtype) == "datetime64[ms, UTC]"


def test_shift_to_close_interval_without_fixed_step_raises() -> None:
    """`1M` lanza `ValueError`."""
    with pytest.raises(ValueError, match="duracion fija"):
        shift_to_close(_grid(3), "1M")


# --- check_window_params -----------------------------------------------------------------------


def test_check_window_params_accepts_valid_values() -> None:
    """Los valores validos no lanzan y devuelve `None`."""
    assert check_window_params(1, 1.0) is None
    assert check_window_params(10080, 0.5) is None
    assert check_window_params(5, 1e-9) is None


@pytest.mark.parametrize("window", [0, -1])
def test_check_window_params_rejects_bad_window(window: int) -> None:
    """`window < 1` lanza `ValueError` que nombra `window`."""
    with pytest.raises(ValueError, match="window"):
        check_window_params(window, 0.5)


@pytest.mark.parametrize("share", [0.0, -0.5, 1.01, np.nan, np.inf])
def test_check_window_params_rejects_bad_share(share: float) -> None:
    """`min_valid_share` fuera de `(0, 1]` lanza `ValueError` que lo nombra."""
    with pytest.raises(ValueError, match="min_valid_share"):
        check_window_params(5, share)


# --- relative_volume_from_cache ----------------------------------------------------------------


def _candles(
    minutes: list[int],
    quote_volume: list[float],
    num_trades: list[int] | None = None,
    *,
    step_ms: int = MIN_MS,
) -> pd.DataFrame:
    """Klines en los slots dados (multiplos de `step_ms` desde T0) con ese `quote_asset_volume`."""
    n = len(minutes)
    open_times = [T0 + k * step_ms for k in minutes]
    trades = [7] * n if num_trades is None else num_trades
    data = {
        "open_time": open_times,
        "open": [100.0] * n,
        "high": [100.0] * n,
        "low": [100.0] * n,
        "close": [100.0] * n,
        "volume": [1.0] * n,
        "close_time": [t + step_ms - 1 for t in open_times],
        "quote_asset_volume": quote_volume,
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
    quote_volume: list[float],
    num_trades: list[int] | None = None,
    *,
    interval: str = "1m",
    step_ms: int = MIN_MS,
) -> None:
    """Guarda en `base_dir` una cache de `symbol`."""
    save_klines(
        _candles(minutes, quote_volume, num_trades, step_ms=step_ms),
        symbol,
        interval,
        base_dir=base_dir,
    )


def test_cache_first_slot_base_uses_data_before_the_range(tmp_path: Path) -> None:
    """La base del primer slot son las `W` velas anteriores a su vela, que caen antes del rango."""
    volume = [float(10 + k * k) for k in range(60)]
    _cache(tmp_path, "A", list(range(60)), volume)
    window, start_min, end_min = 5, 20, 30
    out = relative_volume_from_cache(
        ["A"],
        "1m",
        T0 + start_min * MIN_MS,
        T0 + end_min * MIN_MS,
        window=window,
        base_dir=tmp_path,
    )
    logs = np.log(volume)

    def want(candle: int) -> float:
        return float(logs[candle] - logs[candle - window : candle].mean())

    # La fila `start` es la vela de apertura start - 1 min, con base en las 5 anteriores a ella.
    assert np.isclose(out["A"].iloc[0], want(start_min - 1), rtol=0, atol=1e-12)
    assert np.isclose(out["A"].iloc[-1], want(end_min - 2), rtol=0, atol=1e-12)
    assert out["A"].notna().all()


def test_cache_first_slot_is_nan_without_previous_data(tmp_path: Path) -> None:
    """Si la cache empieza en el rango, la base del primer slot no existe y da NaN (no 0)."""
    _cache(tmp_path, "A", list(range(30, 60)), [5.0] * 30)
    out = relative_volume_from_cache(
        ["A"],
        "1m",
        T0 + 30 * MIN_MS,
        T0 + 45 * MIN_MS,
        window=4,
        min_valid_share=1.0,
        base_dir=tmp_path,
    )
    # Cache desde el minuto 30. Fila 0 = vela 29 (inexistente); fila k = vela 29+k.
    # Con share 1 hace falta la base completa: la fila 5 (vela 34, base 30..33) es la primera.
    assert out["A"].isna().tolist()[:5] == [True] * 5
    assert np.allclose(out["A"].iloc[5:], 0.0, rtol=0, atol=1e-12)


def test_cache_output_is_exactly_the_requested_grid(tmp_path: Path) -> None:
    """El indice de salida es `build_time_grid(start, end)`, con su dtype y sin freq."""
    _cache(tmp_path, "A", list(range(60)), _random_volume(60).tolist())
    start, end = T0 + 20 * MIN_MS, T0 + 40 * MIN_MS
    out = relative_volume_from_cache(["A"], "1m", start, end, window=10, base_dir=tmp_path)
    grid = build_time_grid(start, end, "1m")
    assert out.index.equals(grid)
    assert out.index.dtype == grid.dtype
    assert out.index.freq is None
    assert len(out) == 20
    assert all(dtype == "float64" for dtype in out.dtypes)


def test_cache_unaligned_start_and_end(tmp_path: Path) -> None:
    """`start_ms` y `end_ms` no alineados dan la rejilla de `build_time_grid`."""
    _cache(tmp_path, "A", list(range(60)), _random_volume(60).tolist())
    start, end = T0 + 20 * MIN_MS + 1, T0 + 40 * MIN_MS - 1
    out = relative_volume_from_cache(["A"], "1m", start, end, window=10, base_dir=tmp_path)
    assert out.index.equals(build_time_grid(start, end, "1m"))


def test_cache_symbol_without_cache_is_all_nan_column(tmp_path: Path) -> None:
    """Un simbolo sin cache es una columna de NaN y no estropea a los demas."""
    _cache(tmp_path, "A", list(range(60)), _random_volume(60).tolist())
    out = relative_volume_from_cache(
        ["A", "GHOST"], "1m", T0 + 20 * MIN_MS, T0 + 30 * MIN_MS, window=5, base_dir=tmp_path
    )
    assert out.columns.tolist() == ["A", "GHOST"]
    assert out["GHOST"].isna().all()
    assert out["A"].notna().all()


def test_cache_empty_range_raises(tmp_path: Path) -> None:
    """Un rango sin slots (vacio o invertido) lanza `ValueError`."""
    _cache(tmp_path, "A", list(range(30)), _random_volume(30).tolist())
    for start, end in (
        (T0 + 1, T0 + MIN_MS - 1),
        (T0 + 5 * MIN_MS, T0 + 5 * MIN_MS),
        (T0 + 6 * MIN_MS, T0 + 5 * MIN_MS),
    ):
        with pytest.raises(ValueError, match="vaci"):
            relative_volume_from_cache(["A"], "1m", start, end, window=3, base_dir=tmp_path)


def test_cache_interval_without_fixed_step_raises(tmp_path: Path) -> None:
    """`1M` no tiene paso fijo: `ValueError`."""
    with pytest.raises(ValueError, match="duracion fija"):
        relative_volume_from_cache(["A"], "1M", T0, T0 + 60 * MIN_MS, base_dir=tmp_path)


def test_cache_invalid_params_raise(tmp_path: Path) -> None:
    """`window` y `min_valid_share` invalidos lanzan `ValueError` antes de leer la cache."""
    with pytest.raises(ValueError, match="window"):
        relative_volume_from_cache(["A"], "1m", T0, T0 + 10 * MIN_MS, window=0, base_dir=tmp_path)
    with pytest.raises(ValueError, match="min_valid_share"):
        relative_volume_from_cache(
            ["A"], "1m", T0, T0 + 10 * MIN_MS, min_valid_share=0.0, base_dir=tmp_path
        )


def test_cache_result_equals_manual_chain(tmp_path: Path) -> None:
    """Identico a `align_klines` + `classify_klines` en `[start-(W+1)p, end-p)` y a mano."""
    n = 90
    trades_a = [7] * (n - 1)
    trades_a[30] = 0  # NO_TRADES
    trades_a[31] = 0
    volume_a = _random_volume(n - 1, seed=1).tolist()
    volume_a[30] = volume_a[31] = 0.0
    # Un hueco (falta la vela 45) en A; B completa; C tiene cuotas distintas; GHOST no existe.
    _cache(tmp_path, "A", [k for k in range(n) if k != 45], volume_a, trades_a)
    _cache(tmp_path, "B", list(range(n)), (_random_volume(n, seed=2) * 1000).tolist())
    _cache(tmp_path, "C", list(range(10, n)), _random_volume(n - 10, seed=3).tolist())
    start, end = T0 + 40 * MIN_MS, T0 + 80 * MIN_MS
    symbols = ["A", "B", "C", "GHOST"]
    for window, share in ((10, 0.5), (1, 0.5), (25, 0.8)):
        out = relative_volume_from_cache(
            symbols, "1m", start, end, window=window, min_valid_share=share, base_dir=tmp_path
        )
        wide_start, wide_end = start - (window + 1) * MIN_MS, end - MIN_MS
        volume = align_klines(
            symbols, "1m", wide_start, wide_end, fields=("quote_asset_volume",), base_dir=tmp_path
        )["quote_asset_volume"]
        causes = classify_klines(symbols, "1m", wide_start, wide_end, base_dir=tmp_path)
        manual = log_relative_to_past(
            volume, validity_mask(causes), "1m", window=window, min_valid_share=share
        ).iloc[window:]
        assert_frame_equal(out, manual, rtol=0, atol=1e-12)
        assert out.index.equals(build_time_grid(start, end, "1m"))
        assert out["GHOST"].isna().all()


def test_cache_no_trades_and_missing_candles_are_nan(tmp_path: Path) -> None:
    """Las velas con 0 operaciones y las que faltan dan NaN en su fila, nunca 0 ni -inf."""
    n = 40
    volume = [10.0] * n
    trades = [7] * n
    volume[20] = 0.0
    trades[20] = 0
    minutes = [k for k in range(n) if k != 25]
    del volume[25]
    del trades[25]
    # Tras quitar la 25, la 20 sigue en la posicion 20.
    _cache(tmp_path, "A", minutes, volume, trades)
    out = relative_volume_from_cache(
        ["A"], "1m", T0 + 10 * MIN_MS, T0 + 35 * MIN_MS, window=5, base_dir=tmp_path
    )
    # Fila con indice T0+m corresponde a la vela m-1.
    row = lambda m: out["A"].loc[pd.Timestamp(T0 + m * MIN_MS, unit="ms", tz="UTC")]
    assert np.isnan(row(21))  # vela 20: NO_TRADES
    assert np.isnan(row(26))  # vela 25: sin vela
    assert not np.isinf(out["A"].to_numpy()).any()
    assert np.isclose(
        row(22), 0.0, rtol=0, atol=1e-12
    )  # la 20 no entra en la base: log 10 - log 10


def test_cache_right_trim_does_not_change_common_values(tmp_path: Path) -> None:
    """Recortar el rango por la derecha no cambia los valores de las filas comunes."""
    _cache(tmp_path, "A", list(range(100)), _random_volume(100).tolist())
    full = relative_volume_from_cache(
        ["A"], "1m", T0 + 30 * MIN_MS, T0 + 90 * MIN_MS, window=10, base_dir=tmp_path
    )
    short = relative_volume_from_cache(
        ["A"], "1m", T0 + 30 * MIN_MS, T0 + 50 * MIN_MS, window=10, base_dir=tmp_path
    )
    assert_frame_equal(full.loc[short.index], short, rtol=0, atol=1e-12)


def test_cache_window_longer_than_range(tmp_path: Path) -> None:
    """Con ventana mayor que el rango, la base de cada fila sale de antes del rango."""
    _cache(tmp_path, "A", list(range(100)), _random_volume(100).tolist())
    start, end = T0 + 50 * MIN_MS, T0 + 60 * MIN_MS
    out = relative_volume_from_cache(["A"], "1m", start, end, window=30, base_dir=tmp_path)
    assert len(out) == 10
    assert out["A"].notna().all()
    wide = relative_volume_from_cache(
        ["A"], "1m", T0 + 20 * MIN_MS, end, window=30, base_dir=tmp_path
    )
    assert_frame_equal(wide.loc[out.index], out, rtol=0, atol=1e-12)


def test_cache_no_leak_candles_after_tau_do_not_change_rel_up_to_tau(tmp_path: Path) -> None:
    """Cambiar el volumen desde la vela k no altera las filas conocidas hasta k (indice <= T0+k)."""
    n, k = 80, 50
    volume = _random_volume(n)
    changed = np.concatenate([volume[:k], volume[k:] * 9.0])
    _cache(tmp_path / "base", "A", list(range(n)), volume.tolist())
    _cache(tmp_path / "changed", "A", list(range(n)), changed.tolist())
    start, end = T0 + 20 * MIN_MS, T0 + 70 * MIN_MS
    a = relative_volume_from_cache(["A"], "1m", start, end, window=10, base_dir=tmp_path / "base")
    b = relative_volume_from_cache(
        ["A"], "1m", start, end, window=10, base_dir=tmp_path / "changed"
    )
    # La vela k abre en el minuto k y se conoce en el k+1: las filas con indice <= k no la usan.
    cut = pd.Timestamp(T0 + k * MIN_MS, unit="ms", tz="UTC")
    after = cut + pd.Timedelta(minutes=1)
    assert_frame_equal(a.loc[:cut], b.loc[:cut], rtol=0, atol=1e-12)
    assert not np.allclose(a.loc[after:], b.loc[after:])


def test_cache_step_follows_the_interval(tmp_path: Path) -> None:
    """Con velas de 5m, la base son las `W` velas de 5 minutos anteriores."""
    step = 5 * MIN_MS
    volume = _random_volume(60).tolist()
    _cache(tmp_path, "A", list(range(60)), volume, interval="5m", step_ms=step)
    start, end = T0 + 20 * step, T0 + 30 * step
    out = relative_volume_from_cache(["A"], "5m", start, end, window=4, base_dir=tmp_path)
    assert out.index.equals(build_time_grid(start, end, "5m"))
    logs = np.log(volume)
    assert np.isclose(out["A"].iloc[0], logs[19] - logs[15:19].mean(), rtol=0, atol=1e-12)


def test_cache_is_deterministic(tmp_path: Path) -> None:
    """Dos llamadas iguales dan exactamente el mismo resultado."""
    _cache(tmp_path, "A", list(range(60)), _random_volume(60).tolist())
    args = (["A"], "1m", T0 + 20 * MIN_MS, T0 + 50 * MIN_MS)
    assert_frame_equal(
        relative_volume_from_cache(*args, window=10, base_dir=tmp_path),
        relative_volume_from_cache(*args, window=10, base_dir=tmp_path),
    )
