"""Tests de la interfaz `FactorModel`: contrato, validacion de entrada y guardas de fuga."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from miax.features.factor import FactorModel, NoFactor, NotFittedError, _same
from tests.unit.factor_contract import check_factor_model, make_returns, make_train_test


class ToyFactor(FactorModel):
    """Factor = media de las 2 primeras columnas; residuo = retorno - factor."""

    def __init__(self) -> None:
        self.means: np.ndarray | None = None

    def get_state(self) -> dict[str, object]:
        """Estado ajustado del modelo de juguete."""
        return {"means": self.means}

    def _fit(self, returns: pd.DataFrame) -> None:
        self.means = returns.mean().to_numpy()

    def _factors(self, returns: pd.DataFrame) -> pd.DataFrame:
        f = returns.iloc[:, :2].mean(axis=1, skipna=False)
        return f.to_frame("f")

    def _residuals(self, returns: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
        return returns.sub(factors["f"], axis=0)


class CheatingFactor(ToyFactor):
    """Cambia su estado al proyectar: debe detectarse."""

    def __init__(self, where: str) -> None:
        super().__init__()
        self.where = where

    def _factors(self, returns: pd.DataFrame) -> pd.DataFrame:
        if self.where == "factors":
            self.means = returns.mean().to_numpy()
        return super()._factors(returns)

    def _residuals(self, returns: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
        if self.where == "residuals":
            self.means = returns.mean().to_numpy()
        return super()._residuals(returns, factors)


class BadOutput(ToyFactor):
    """Salidas defectuosas configurables."""

    def __init__(self, mode: str) -> None:
        super().__init__()
        self.mode = mode

    def _factors(self, returns: pd.DataFrame) -> pd.DataFrame:
        f = super()._factors(returns)
        if self.mode == "factors_inf":
            f.iloc[0, 0] = np.inf
        elif self.mode == "factors_index":
            f.index = f.index + pd.Timedelta("1s")
        return f

    def _residuals(self, returns: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
        r = super()._residuals(returns, factors)
        if self.mode == "res_inf":
            r.iloc[0, 0] = -np.inf
        elif self.mode == "res_index":
            r.index = r.index + pd.Timedelta("1s")
        elif self.mode == "res_columns":
            r.columns = [f"x{i}" for i in range(r.shape[1])]
        return r


@pytest.mark.parametrize("factory", [NoFactor, ToyFactor], ids=["NoFactor", "ToyFactor"])
def test_contract_suite(factory: type[FactorModel]) -> None:
    """Contrato completo sobre NoFactor y el modelo de juguete."""
    train, test = make_train_test()
    check_factor_model(factory, train, test)


def test_contract_suite_without_nan() -> None:
    """Contrato con datos sin NaN."""
    train, test = make_train_test(nan_frac=0.0)
    check_factor_model(ToyFactor, train, test)


def test_nofactor_residual_equals_returns() -> None:
    """NoFactor devuelve el retorno y un factor vacio."""
    train, test = make_train_test()
    m = NoFactor().fit(train)
    pd.testing.assert_frame_equal(m.transform(test), test)
    assert m.factors(test).shape == (len(test), 0)
    assert m.get_state() == {}


def test_toy_values() -> None:
    """Valores del modelo de juguete contra el calculo directo."""
    train, test = make_train_test(nan_frac=0.0)
    m = ToyFactor().fit(train)
    f = m.factors(test)["f"]
    pd.testing.assert_series_equal(f, test.iloc[:, :2].mean(axis=1), check_names=False)
    pd.testing.assert_frame_equal(m.transform(test), test.sub(f, axis=0))


def test_factor_nan_masks_whole_row() -> None:
    """Factor NaN en una fila enmascara toda la fila del residuo."""
    train, test = make_train_test(nan_frac=0.0)
    test = test.copy()
    test.iloc[5, 0] = np.nan  # factor NaN en la fila 5
    out = ToyFactor().fit(train).transform(test)
    assert out.iloc[5].isna().all()
    assert not out.drop(out.index[5]).isna().any().any()


def test_abstract_get_state_required() -> None:
    """Sin get_state no se puede instanciar."""

    class NoState(FactorModel):
        def _fit(self, returns: pd.DataFrame) -> None:
            return None

        def _factors(self, returns: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame(index=returns.index)

        def _residuals(self, returns: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
            return returns

    with pytest.raises(TypeError):
        NoState()  # type: ignore[abstract]


@pytest.mark.parametrize("where", ["factors", "residuals"])
def test_state_change_detected(where: str) -> None:
    """Cambiar el estado al proyectar lanza RuntimeError."""
    train, test = make_train_test()
    m = CheatingFactor(where).fit(train)
    with pytest.raises(RuntimeError, match="fuga de datos"):
        m.transform(test)


def test_state_change_detected_in_factors_call() -> None:
    """Igual en la llamada directa a factors."""
    train, test = make_train_test()
    m = CheatingFactor("factors").fit(train)
    with pytest.raises(RuntimeError, match="fuga de datos"):
        m.factors(test)


def _bad_frames() -> dict[str, object]:
    ok = make_returns(10)
    utc = ok.index
    inf = ok.copy()
    inf.iloc[2, 1] = np.inf
    ninf = ok.copy()
    ninf.iloc[2, 1] = -np.inf
    int_cols = ok.copy()
    int_cols.columns = [0, 1, 2, 3]
    dup_cols = ok.copy()
    dup_cols.columns = ["A", "A", "B", "C"]
    return {
        "not_df": ok.to_numpy(),
        "series": ok.iloc[:, 0],
        "range_index": ok.reset_index(drop=True),
        "naive": ok.set_axis(utc.tz_localize(None)),
        "not_utc": ok.set_axis(utc.tz_convert("Europe/Madrid")),
        "decreasing": ok.iloc[::-1],
        "duplicated": ok.set_axis(utc[[0, 0, *range(2, 10)]]),
        "float32": ok.astype("float32"),
        "int": (ok.fillna(0) * 100).astype("int64"),
        "inf": inf,
        "neg_inf": ninf,
        "int_columns": int_cols,
        "dup_columns": dup_cols,
    }


BAD = _bad_frames()


@pytest.mark.parametrize("name", list(BAD))
def test_fit_rejects_bad_input(name: str) -> None:
    """Fit rechaza entradas invalidas."""
    with pytest.raises(ValueError):
        ToyFactor().fit(BAD[name])  # type: ignore[arg-type]


@pytest.mark.parametrize("name", list(BAD))
def test_transform_rejects_bad_input(name: str) -> None:
    """Transform y factors rechazan entradas invalidas."""
    m = ToyFactor().fit(make_returns(20))
    for call in (m.transform, m.factors):
        with pytest.raises(ValueError):
            call(BAD[name])  # type: ignore[arg-type]


def test_fit_rejects_empty() -> None:
    """Fit rechaza 0 filas o 0 columnas."""
    ok = make_returns(10)
    with pytest.raises(ValueError, match="0 filas"):
        ToyFactor().fit(ok.iloc[:0])
    with pytest.raises(ValueError, match="0 filas"):
        ToyFactor().fit(ok.iloc[:, :0])


@pytest.mark.parametrize("mode", ["factors_inf", "factors_index"])
def test_bad_factors_output(mode: str) -> None:
    """Factores con inf o indice distinto lanzan ValueError."""
    m = BadOutput(mode).fit(make_returns(20))
    with pytest.raises(ValueError):
        m.factors(make_returns(10))
    with pytest.raises(ValueError):
        m.transform(make_returns(10))


@pytest.mark.parametrize("mode", ["res_inf", "res_index", "res_columns"])
def test_bad_residuals_output(mode: str) -> None:
    """Residuos con inf, indice o columnas distintos lanzan ValueError."""
    m = BadOutput(mode).fit(make_returns(20))
    with pytest.raises(ValueError):
        m.transform(make_returns(10))


def test_failed_refit_leaves_model_unfitted() -> None:
    """Un reajuste fallido deja el modelo sin ajustar."""
    m = ToyFactor().fit(make_returns(20))
    assert m.is_fitted
    with pytest.raises(ValueError):
        m.fit(BAD["inf"])  # type: ignore[arg-type]
    assert not m.is_fitted
    with pytest.raises(NotFittedError):
        m.transform(make_returns(10))


def test_refit_with_other_symbols() -> None:
    """Reajustar cambia los simbolos."""
    m = ToyFactor().fit(make_returns(20))
    m.fit(make_returns(20, symbols=("X", "Y")))
    assert m.symbols == ("X", "Y")


def test_fit_does_not_mutate_input() -> None:
    """Fit no modifica la entrada."""
    train = make_returns(20, nan_frac=0.1)
    before = train.copy()
    ToyFactor().fit(train)
    pd.testing.assert_frame_equal(train, before)


def test_same_nan_float32_equal() -> None:
    """_same trata NaN como iguales, tambien en float32."""
    assert _same(np.float32("nan"), np.float32("nan"))
    assert _same(float("nan"), float("nan"))
    assert not _same(np.float32(1.0), np.float32("nan"))
    assert _same(np.array([1.0, np.nan], dtype="float32"), np.array([1.0, np.nan], dtype="float32"))
    assert not _same(np.array([1.0, np.nan]), np.array([1.0, 2.0]))
    assert _same({"a": [1, (2.0, np.nan)]}, {"a": [1, (2.0, np.nan)]})
    assert not _same({"a": 1}, {"b": 1})
    assert not _same([1], (1,))


def test_same_dataframe_and_object_arrays() -> None:
    """_same con DataFrame y arrays de objetos."""
    df = make_returns(5, nan_frac=0.3)
    assert _same(df, df.copy())
    assert not _same(df, df.iloc[:, 0])
    assert _same(np.array(["a", "b"], dtype=object), np.array(["a", "b"], dtype=object))


def test_same_unsupported_type_raises() -> None:
    """_same lanza TypeError con tipos no soportados."""
    with pytest.raises(TypeError, match="no soportado"):
        _same(object(), object())
    with pytest.raises(TypeError):
        _same({"a": {1, 2}}, {"a": {1, 2}})
