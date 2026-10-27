"""Suite reutilizable del contrato de `FactorModel` (garantias 1-7) para los modelos concretos."""

from __future__ import annotations

import copy
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from miax.features.factor import FactorModel, NotFittedError, _same

SYMBOLS = ("AAA", "BBB", "CCC", "DDD")


def make_returns(
    n: int = 60,
    symbols: tuple[str, ...] = SYMBOLS,
    seed: int = 0,
    start: str = "2024-01-01",
    nan_frac: float = 0.0,
) -> pd.DataFrame:
    """Retornos sinteticos UTC float64, con NaN dispersos si `nan_frac > 0`."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="h", tz="UTC")
    data = rng.normal(0.0, 0.01, size=(n, len(symbols)))
    if nan_frac > 0:
        data[rng.random(data.shape) < nan_frac] = np.nan
    return pd.DataFrame(data, index=idx, columns=list(symbols), dtype="float64")


def make_train_test(
    nan_frac: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Par train/test contiguos, con NaN en ambos."""
    train = make_returns(80, seed=1, nan_frac=nan_frac)
    test = make_returns(40, seed=2, start="2024-01-04 08:00", nan_frac=nan_frac)
    return train, test


def _assert_out(out: pd.DataFrame, ref: pd.DataFrame, columns: list[str] | None = None) -> None:
    assert isinstance(out, pd.DataFrame)
    assert out.index.equals(ref.index)
    if columns is not None:
        assert list(out.columns) == columns
    assert all(dt == np.float64 for dt in out.dtypes)


def check_factor_model(
    make_model: Callable[[], FactorModel],
    returns_train: pd.DataFrame,
    returns_test: pd.DataFrame,
) -> None:
    """Comprueba el contrato de `FactorModel`; `make_model` debe dar un modelo nuevo cada vez."""
    cols = list(returns_train.columns)
    assert list(returns_test.columns) == cols

    # 1. NotFittedError antes de fit
    m = make_model()
    assert not m.is_fitted
    for call in (m.transform, m.factors, lambda _r: m.symbols):
        with pytest.raises(NotFittedError):
            call(returns_test)

    # fit devuelve self
    assert m.fit(returns_train) is m
    assert m.is_fitted
    assert m.symbols == tuple(cols)

    # 2. salida con el indice, columnas y dtype esperados
    res = m.transform(returns_test)
    fac = m.factors(returns_test)
    _assert_out(res, returns_test, cols)
    _assert_out(fac, returns_test)
    assert fac.shape[0] == returns_test.shape[0]

    # 3. el orden de las columnas de entrada se respeta
    perm = cols[::-1]
    res_p = m.transform(returns_test[perm])
    assert list(res_p.columns) == perm
    pd.testing.assert_frame_equal(res_p[cols], res)
    pd.testing.assert_frame_equal(m.factors(returns_test[perm]), fac)

    # 4. simbolos distintos: ValueError (faltan, sobran)
    with pytest.raises(ValueError, match="simbolos distintos"):
        m.transform(returns_test[cols[:-1]])
    extra = returns_test.copy()
    extra["ZZZ"] = 0.0
    with pytest.raises(ValueError, match="simbolos distintos"):
        m.factors(extra)

    # 5. NaN de entrada -> NaN de salida, nunca 0
    inp = returns_test.copy()
    inp.iloc[3, 1] = np.nan
    inp.iloc[10, 0] = np.nan
    out = m.transform(inp)
    assert np.isnan(out.iloc[3, 1])
    assert np.isnan(out.iloc[10, 0])
    assert out.isna().to_numpy()[inp.isna().to_numpy()].all()

    # 6. NaN en el factor de una fila -> residuo NaN en toda la fila
    fac_in = m.factors(inp)
    rows = fac_in.isna().any(axis=1).to_numpy()
    assert out.isna().to_numpy()[rows].all()

    # entrada no mutada
    before = inp.copy()
    m.transform(inp)
    m.factors(inp)
    pd.testing.assert_frame_equal(inp, before)
    train_before = returns_train.copy()
    make_model().fit(returns_train)
    pd.testing.assert_frame_equal(returns_train, train_before)

    # 7. get_state igual antes y despues de transform / factors
    state = copy.deepcopy(m.get_state())
    m.transform(returns_test)
    m.factors(returns_test)
    assert _same(state, m.get_state())

    # fit_transform == fit + transform
    a = make_model().fit_transform(returns_train)
    b = make_model().fit(returns_train).transform(returns_train)
    pd.testing.assert_frame_equal(a, b)

    # determinismo: mismo ajuste, mismo resultado
    m2 = make_model().fit(returns_train)
    pd.testing.assert_frame_equal(m2.transform(returns_test), res)
    assert _same(m2.get_state(), m.get_state())
