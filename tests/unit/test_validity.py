"""Tests de la mascara de validez: tablas sinteticas, sin red ni disco.

Las causas se escriben como cadenas, un caracter por slot de 1 minuto:
V valid, T no_trades, L not_listed, H halted, O outage, D delisted, N no_data.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import pytest

from miax.features import (
    GapCause,
    build_time_grid,
    classify_slots,
    mask_invalid,
    return_validity,
    validity_mask,
)

MIN_MS = 60_000
START = 1_767_225_600_000  # 2026-01-01T00:00:00Z
LETTERS = {
    "V": GapCause.VALID,
    "T": GapCause.NO_TRADES,
    "L": GapCause.NOT_LISTED,
    "H": GapCause.HALTED,
    "O": GapCause.OUTAGE,
    "D": GapCause.DELISTED,
    "N": GapCause.NO_DATA,
}


def _causes(specs: Mapping[str, str]) -> pd.DataFrame:
    """Tabla `tiempo x simbolo` de causas (`int8`) desde una cadena por simbolo."""
    n = len(next(iter(specs.values())))
    index = build_time_grid(START, START + n * MIN_MS, "1m")
    data = {s: np.array([int(LETTERS[c]) for c in spec], dtype="int8") for s, spec in specs.items()}
    return pd.DataFrame(data, index=index)


@pytest.mark.parametrize("cause", list(GapCause))
def test_mask_only_valid_is_true(cause: GapCause) -> None:
    """Solo `VALID` es verdadero, para cada causa de `GapCause`."""
    causes = _causes({"A": "VV"})
    causes.iloc[1, 0] = int(cause)
    mask = validity_mask(causes)
    assert mask["A"].tolist() == [True, cause == GapCause.VALID]


def test_mask_dtype_index_columns() -> None:
    """La mascara es `bool` y conserva indice y columnas."""
    causes = _causes({"A": "VTVH", "B": "VVNO"})
    mask = validity_mask(causes)
    assert all(dtype == bool for dtype in mask.dtypes)
    assert mask.index.equals(causes.index)
    assert mask.columns.equals(causes.columns)


def test_mask_unknown_value_raises() -> None:
    """Un valor que no es un `GapCause` lanza `ValueError`."""
    causes = _causes({"A": "VVV"})
    causes.iloc[1, 0] = 99
    with pytest.raises(ValueError, match="GapCause"):
        validity_mask(causes)


def test_stale_price_slot_is_invalid() -> None:
    """Un slot con vela pero sin operaciones (precio rancio) es invalido."""
    # Vela presente (con close rancio) pero sin operaciones: NO_TRADES, inválido.
    mask = validity_mask(_causes({"A": "VTV"}))
    assert mask["A"].tolist() == [True, False, True]


def test_return_validity_run_of_L_gives_L_minus_1() -> None:
    """Una racha de L slots validos da L-1 retornos validos."""
    valid = validity_mask(_causes({"A": "HVVVVH"}))
    out = return_validity(valid)
    assert out["A"].tolist() == [False, False, True, True, True, False]
    assert out["A"].sum() == 3  # racha de 4 slots validos -> 3 retornos


def test_return_validity_single_invalid_kills_two() -> None:
    """Un slot invalido suelto invalida su retorno y el siguiente."""
    valid = validity_mask(_causes({"A": "VVVTVVV"}))
    out = return_validity(valid)
    assert out["A"].tolist() == [False, True, True, False, False, True, True]


def test_return_validity_first_row_false_and_bool() -> None:
    """La primera fila es `False`, sin NaN y con dtype `bool`."""
    out = return_validity(validity_mask(_causes({"A": "VVV", "B": "TVV"})))
    assert not out.iloc[0].any()
    assert all(dtype == bool for dtype in out.dtypes)
    assert not out.isna().any().any()


def test_no_leak_future_changes_do_not_alter_past() -> None:
    """Cambiar las causas desde la fila k no altera la validez de las filas anteriores."""
    k = 6
    base = _causes({"A": "VVTVVVVVVV", "B": "VVVVHVVVVV"})
    changed = _causes({"A": "VVTVVV" + "TTTT", "B": "VVVVHV" + "HTHT"})
    m0, m1 = validity_mask(base), validity_mask(changed)
    assert m0.iloc[:k].equals(m1.iloc[:k])
    r0, r1 = return_validity(m0), return_validity(m1)
    assert r0.iloc[:k].equals(r1.iloc[:k])


def test_d02_no_new_zeros_around_gaps() -> None:
    """Enmascarar deja NaN alrededor del hueco y ningun cero nuevo (D-02)."""
    # Precio constante con operaciones (retorno 0 real) en los slots 1-2; hueco T en el 4.
    spec = "VVVVTVVV"
    causes = _causes({"A": spec})
    close = pd.DataFrame(
        {"A": [100.0, 100.0, 100.0, 101.0, 101.0, 101.0, 102.0, 102.0]}, index=causes.index
    )
    returns = np.log(close).diff()
    masked = mask_invalid(returns, return_validity(validity_mask(causes)))
    assert masked["A"].isna().tolist() == [True, False, False, False, True, True, False, False]
    # Ceros esperados: retornos planos con operaciones en t y t-1 (slots 1, 2 y 7).
    assert int((masked["A"] == 0).sum()) == 3
    # Sin el enmascarado, el slot 4 (vela rancia) habria sido un 0 artificial.
    assert int((returns["A"] == 0).sum()) == 5


def test_real_zero_return_is_kept() -> None:
    """Un retorno 0 con operaciones es una observacion real y se conserva."""
    causes = _causes({"A": "VVVV"})
    close = pd.DataFrame({"A": [5.0, 5.0, 5.0, 5.0]}, index=causes.index)
    returns = np.log(close).diff()
    masked = mask_invalid(returns, return_validity(validity_mask(causes)))
    assert masked["A"].iloc[1:].tolist() == [0.0, 0.0, 0.0]
    assert np.isnan(masked["A"].iloc[0])


def test_mask_invalid_values_and_nan_never_zero() -> None:
    """Lo valido queda intacto, lo invalido es NaN y la entrada no se modifica."""
    index = build_time_grid(START, START + 3 * MIN_MS, "1m")
    frame = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=index)
    valid = pd.DataFrame({"A": [True, False, True]}, index=index)
    out = mask_invalid(frame, valid)
    assert out["A"].tolist()[0] == 1.0 and out["A"].tolist()[2] == 3.0
    assert np.isnan(out["A"].iloc[1])
    assert frame["A"].tolist() == [1.0, 2.0, 3.0]  # no modifica la entrada


def test_mask_invalid_shape_mismatch_raises() -> None:
    """Indices o columnas distintos lanzan `ValueError`."""
    index = build_time_grid(START, START + 3 * MIN_MS, "1m")
    frame = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=index)
    with pytest.raises(ValueError, match="mismo indice"):
        mask_invalid(frame, pd.DataFrame({"B": [True] * 3}, index=index))
    with pytest.raises(ValueError, match="mismo indice"):
        mask_invalid(frame, pd.DataFrame({"A": [True] * 2}, index=index[:2]))


def test_mask_invalid_non_float_raises() -> None:
    """Un frame entero lanza `ValueError`: no admite NaN."""
    index = build_time_grid(START, START + 2 * MIN_MS, "1m")
    frame = pd.DataFrame({"A": [1, 2]}, index=index)
    valid = pd.DataFrame({"A": [True, True]}, index=index)
    with pytest.raises(ValueError, match="float"):
        mask_invalid(frame, valid)


def test_integration_classify_to_return_validity() -> None:
    """`classify_slots` -> `validity_mask` -> `return_validity` encajan."""
    index = build_time_grid(START, START + 6 * MIN_MS, "1m")
    num_trades = pd.DataFrame(
        {"A": [5.0, 5.0, 0.0, 5.0, 5.0, 5.0], "B": [5.0, 5.0, 5.0, np.nan, 5.0, 5.0]}, index=index
    )
    first = {"A": START, "B": START}
    last = {"A": START + 5 * MIN_MS, "B": START + 5 * MIN_MS}
    causes = classify_slots(num_trades, first_ms=first, last_ms=last)
    valid = validity_mask(causes)
    assert valid["A"].tolist() == [True, True, False, True, True, True]
    assert valid["B"].tolist() == [True, True, True, False, True, True]
    out = return_validity(valid)
    assert out["A"].tolist() == [False, True, False, False, True, True]
    assert out["B"].tolist() == [False, True, True, False, False, True]


def test_mask_bool_input_raises() -> None:
    """Un DataFrame `bool` (una mascara pasada por error como causas) lanza `ValueError`."""
    index = build_time_grid(START, START + 2 * MIN_MS, "1m")
    with pytest.raises(ValueError, match="bool"):
        validity_mask(pd.DataFrame({"A": [True, False]}, index=index))


def test_mask_invalid_non_bool_valid_raises() -> None:
    """Un `valid` con NaN, object o float lanza `ValueError`: un NaN no es `False`."""
    index = build_time_grid(START, START + 2 * MIN_MS, "1m")
    frame = pd.DataFrame({"A": [1.0, 2.0]}, index=index)
    for column in ([True, np.nan], [1.0, 0.0], [True, None]):
        valid = pd.DataFrame({"A": column}, index=index)
        with pytest.raises(ValueError, match="bool"):
            mask_invalid(frame, valid)


def test_return_validity_single_row_is_false_bool() -> None:
    """Con una sola fila el retorno no es valido y el dtype es `bool`."""
    out = return_validity(validity_mask(_causes({"A": "V", "B": "V"})))
    assert out.shape == (1, 2)
    assert not out.any().any()
    assert all(dtype == bool for dtype in out.dtypes)


def test_empty_frames_keep_bool_dtype() -> None:
    """Sin filas o sin columnas, las salidas siguen siendo `bool` y conservan la forma."""
    index = build_time_grid(START, START + 3 * MIN_MS, "1m")
    no_rows = pd.DataFrame({"A": np.array([], dtype="int8")}, index=index[:0])
    no_cols = pd.DataFrame(index=index)
    for causes, shape in ((no_rows, (0, 1)), (no_cols, (3, 0))):
        mask = validity_mask(causes)
        out = return_validity(mask)
        assert mask.shape == shape and out.shape == shape
        assert all(dtype == bool for dtype in mask.dtypes)
        assert all(dtype == bool for dtype in out.dtypes)


def test_all_valid_and_all_invalid() -> None:
    """Todo valido da retornos validos salvo la primera fila; todo invalido, ninguno."""
    valid = return_validity(validity_mask(_causes({"A": "VVVV"})))
    assert valid["A"].tolist() == [False, True, True, True]
    invalid = return_validity(validity_mask(_causes({"A": "TTTT"})))
    assert not invalid.any().any()


def test_mask_invalid_unordered_valid_raises() -> None:
    """Un `valid` con el indice desordenado lanza `ValueError`."""
    index = build_time_grid(START, START + 3 * MIN_MS, "1m")
    frame = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=index)
    valid = pd.DataFrame({"A": [True, False, True]}, index=index).iloc[::-1]
    with pytest.raises(ValueError, match="mismo indice"):
        mask_invalid(frame, valid)


def test_mask_invalid_keeps_previous_nan() -> None:
    """Los `NaN` previos de `frame` se conservan aunque el slot sea valido."""
    index = build_time_grid(START, START + 3 * MIN_MS, "1m")
    frame = pd.DataFrame({"A": [np.nan, 2.0, 3.0]}, index=index)
    valid = pd.DataFrame({"A": [True, True, False]}, index=index)
    out = mask_invalid(frame, valid)
    assert out["A"].isna().tolist() == [True, False, True]
    assert out["A"].iloc[1] == 2.0
