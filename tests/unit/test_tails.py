"""Tests de las colas de los retornos: datos sinteticos, sin red."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from miax.features import (
    GapCause,
    build_time_grid,
    clip_returns,
    fit_clip_bounds,
    log_returns,
    tail_report,
    validity_mask,
)

MIN_MS = 60_000
T0 = 1_767_225_600_000  # 2026-01-01T00:00:00Z


def _frame(columns: dict[str, list[float]]) -> pd.DataFrame:
    """Tabla `tiempo x simbolo` de float64 sobre la rejilla de 1m desde T0."""
    n = len(next(iter(columns.values())))
    index = build_time_grid(T0, T0 + n * MIN_MS, "1m")
    return pd.DataFrame(
        {s: np.asarray(v, dtype="float64") for s, v in columns.items()}, index=index
    )


def test_log_returns_keeps_a_fifty_percent_jump_unclipped() -> None:
    """La politica por defecto es no recortar: un salto del 50 % llega entero a los retornos."""
    close = _frame({"A": [100.0, 100.0, 150.0, 150.0]})
    causes = pd.DataFrame(
        int(GapCause.VALID), index=close.index, columns=close.columns, dtype="int8"
    )
    valid = validity_mask(causes)
    out = log_returns(close, valid, "1m")
    assert out["A"].max() == pytest.approx(np.log(1.5))


def test_tail_report_normal_sample_has_near_zero_kurtosis() -> None:
    """Una muestra normal tiene curtosis en exceso cercana a 0."""
    rng = np.random.default_rng(0)
    returns = _frame({"A": rng.normal(0.0, 0.001, 50_000).tolist()})
    report = tail_report(returns)
    assert abs(report.loc["A", "kurtosis"]) < 0.2
    assert report.loc["A", "n"] == 50_000


def test_tail_report_injected_tail_raises_kurtosis_and_counts_sigma_exactly() -> None:
    """Una cola inyectada sube la curtosis y el conteo por sigma es exacto."""
    rng = np.random.default_rng(1)
    base = rng.normal(0.0, 0.001, 10_000)
    fat = base.copy()
    fat[[10, 20, 30]] = [0.5, -0.5, 0.5]
    report = tail_report(_frame({"base": base.tolist(), "fat": fat.tolist()}))
    assert report.loc["fat", "kurtosis"] > report.loc["base", "kurtosis"] + 50
    assert report.loc["fat", "n_gt_5sd"] == 3
    assert report.loc["fat", "n_gt_10sd"] == 3
    assert report.loc["base", "n_gt_10sd"] == 0


def test_tail_report_max_abs_z_uses_absolute_value() -> None:
    """El maximo de z usa el valor absoluto, tambien para retornos negativos."""
    x = [0.0, 0.0, 0.0, 0.0, -10.0, 1.0]
    report = tail_report(_frame({"A": x}))
    values = np.asarray(x)
    expected = 10.0 / values.std(ddof=1)
    assert report.loc["A", "max_abs_z"] == pytest.approx(expected)


def test_tail_report_top_var_share_by_hand() -> None:
    """La fraccion de varianza del extremo coincide con el calculo a mano."""
    # Cuadrados 1, 4, 9, 16 (suma 30); la fraccion 0,25 mas grande es 1 valor: 16/30.
    report = tail_report(_frame({"A": [1.0, -2.0, 3.0, -4.0]}), top_share=0.25)
    assert report.loc["A", "top_var_share"] == pytest.approx(16 / 30)
    report_half = tail_report(_frame({"A": [1.0, -2.0, 3.0, -4.0]}), top_share=0.5)
    assert report_half.loc["A", "top_var_share"] == pytest.approx(25 / 30)


def test_tail_report_ignores_nan_and_keeps_columns() -> None:
    """Los NaN no cuentan y cada simbolo conserva su fila."""
    rng = np.random.default_rng(2)
    a = rng.normal(0.0, 0.001, 1_000)
    a_nan = a.copy()
    a_nan[::10] = np.nan
    report = tail_report(_frame({"A": a_nan.tolist(), "B": a.tolist()}))
    assert list(report.index) == ["A", "B"]
    assert report.loc["A", "n"] == 900
    assert report.loc["B", "n"] == 1_000
    assert report.loc["A", "std"] == pytest.approx(np.nanstd(a_nan, ddof=1))
    assert {"n", "std", "kurtosis", "q0.001", "q0.999", "max_abs_z", "top_var_share"} <= set(
        report.columns
    )


def test_tail_report_custom_levels_name_columns() -> None:
    """Los cuantiles y niveles de sigma pedidos dan nombre a las columnas."""
    rng = np.random.default_rng(3)
    returns = _frame({"A": rng.normal(0.0, 1.0, 500).tolist()})
    report = tail_report(returns, quantiles=(0.01,), sigma_levels=(3.0,))
    assert "q0.01" in report.columns
    assert "n_gt_3sd" in report.columns
    assert "q0.001" not in report.columns


@pytest.mark.parametrize(
    "kwargs",
    [
        {"quantiles": (0.0, 0.5)},
        {"quantiles": (0.5, 1.0)},
        {"sigma_levels": (0.0,)},
        {"sigma_levels": (-1.0,)},
        {"top_share": 0.0},
        {"top_share": 1.5},
    ],
)
def test_tail_report_invalid_parameters_raise(kwargs: dict[str, object]) -> None:
    """Parametros fuera de rango lanzan ValueError."""
    with pytest.raises(ValueError):
        tail_report(_frame({"A": [1.0, 2.0, 3.0]}), **kwargs)  # type: ignore[arg-type]


def test_tail_report_constant_series_has_no_z() -> None:
    """Una serie constante no tiene z: NaN, sin dividir por cero."""
    report = tail_report(_frame({"A": [0.0, 0.0, 0.0, 0.0, 0.0]}))
    assert np.isnan(report.loc["A", "max_abs_z"])
    assert np.isnan(report.loc["A", "kurtosis"])
    assert report.loc["A", "n_gt_5sd"] == 0
    assert np.isnan(report.loc["A", "top_var_share"])


def test_fit_clip_bounds_exact_quantiles_on_arange() -> None:
    """Los limites son los cuantiles exactos de la muestra."""
    train = _frame({"A": np.arange(101, dtype="float64").tolist()})
    bounds = fit_clip_bounds(train, lower_q=0.1, upper_q=0.9)
    assert list(bounds.index) == ["lower", "upper"]
    assert bounds.loc["lower", "A"] == pytest.approx(10.0)
    assert bounds.loc["upper", "A"] == pytest.approx(90.0)


def test_fit_clip_bounds_asymmetric_negative_sample_matches_numpy() -> None:
    """Con cola izquierda larga y negativos, los limites son cuantiles con signo, no de |x|."""
    values = np.concatenate([np.arange(-100.0, 0.0), np.arange(0.0, 11.0)])
    bounds = fit_clip_bounds(_frame({"A": values.tolist()}), lower_q=0.1, upper_q=0.9)
    expected_lower = float(np.quantile(values, 0.1))
    expected_upper = float(np.quantile(values, 0.9))
    assert expected_upper != pytest.approx(float(np.quantile(np.abs(values), 0.9)))
    assert bounds.loc["lower", "A"] == pytest.approx(expected_lower)
    assert bounds.loc["upper", "A"] == pytest.approx(expected_upper)


def test_fit_clip_bounds_ignores_nan() -> None:
    """Los NaN no entran en los limites."""
    values = np.arange(101, dtype="float64")
    with_nan = np.concatenate([values, np.full(50, np.nan)])
    bounds = fit_clip_bounds(_frame({"A": with_nan.tolist()}), lower_q=0.1, upper_q=0.9)
    assert bounds.loc["lower", "A"] == pytest.approx(10.0)
    assert bounds.loc["upper", "A"] == pytest.approx(90.0)


@pytest.mark.parametrize(
    ("lower_q", "upper_q"), [(0.9, 0.1), (0.5, 0.5), (0.0, 0.9), (0.1, 1.0), (-0.1, 0.9)]
)
def test_fit_clip_bounds_invalid_quantiles_raise(lower_q: float, upper_q: float) -> None:
    """Cuantiles invertidos o fuera de (0, 1) lanzan ValueError."""
    with pytest.raises(ValueError):
        fit_clip_bounds(_frame({"A": [1.0, 2.0, 3.0]}), lower_q=lower_q, upper_q=upper_q)


def test_fit_clip_bounds_depends_only_on_its_input_contract() -> None:
    """Test de contrato, no de fuga real (ese flujo train/test es de MIAX-135).

    `fit_clip_bounds` solo depende de lo que recibe y no guarda estado entre llamadas.
    """
    rng = np.random.default_rng(4)
    values = np.concatenate([rng.normal(0.0, 0.001, 2_000), rng.normal(0.0, 0.01, 2_000)])
    full = _frame({"A": values.tolist()})
    train, other = full.iloc[:2_000].copy(), full.iloc[2_000:].copy()
    bounds_train = fit_clip_bounds(train)
    # (a) Una llamada intermedia con otro frame no altera el resultado de la siguiente.
    fit_clip_bounds(other)
    assert_frame_equal(fit_clip_bounds(train), bounds_train)
    # (b) Mutar train despues de ajustar no cambia los limites ya devueltos.
    snapshot = bounds_train.copy()
    train.iloc[:, 0] = 999.0
    assert_frame_equal(bounds_train, snapshot)
    # (c) Contraste: train+test SI da limites distintos; por eso se ajusta solo con train.
    contaminated = fit_clip_bounds(full)
    assert contaminated.loc["upper", "A"] > snapshot.loc["upper", "A"] * 2
    assert contaminated.loc["lower", "A"] < snapshot.loc["lower", "A"] * 2


def test_clip_returns_clips_extremes_and_keeps_interior() -> None:
    """Los extremos se recortan y el interior queda intacto."""
    returns = _frame({"A": [-5.0, -1.0, 0.0, 1.0, 5.0]})
    bounds = pd.DataFrame({"A": [-2.0, 2.0]}, index=["lower", "upper"])
    out = clip_returns(returns, bounds)
    assert out["A"].tolist() == [-2.0, -1.0, 0.0, 1.0, 2.0]


def test_clip_returns_keeps_nan_and_never_makes_zero() -> None:
    """El NaN sigue NaN y el recorte nunca produce 0."""
    returns = _frame({"A": [np.nan, 5.0, np.nan, 0.5]})
    bounds = pd.DataFrame({"A": [-1.0, 1.0]}, index=["lower", "upper"])
    out = clip_returns(returns, bounds)
    assert out["A"].isna().tolist() == [True, False, True, False]
    assert out["A"].tolist()[1] == 1.0
    assert not (out["A"] == 0.0).any()


def test_clip_returns_does_not_modify_input_and_keeps_index_and_dtype() -> None:
    """No modifica la entrada y conserva indice y dtype."""
    returns = _frame({"A": [-5.0, 5.0], "B": [3.0, np.nan]})
    original = returns.copy()
    bounds = pd.DataFrame({"A": [-1.0, 1.0], "B": [-2.0, 2.0]}, index=["lower", "upper"])
    out = clip_returns(returns, bounds)
    assert_frame_equal(returns, original)
    assert out.index.equals(returns.index)
    assert (out.dtypes == "float64").all()
    assert out.loc[out.index[0], "B"] == 2.0
    assert out is not returns


def test_clip_returns_uses_each_columns_own_bounds() -> None:
    """Cada columna usa sus propios limites."""
    returns = _frame({"A": [10.0], "B": [10.0]})
    bounds = pd.DataFrame({"A": [-1.0, 1.0], "B": [-5.0, 5.0]}, index=["lower", "upper"])
    out = clip_returns(returns, bounds)
    assert out.iloc[0].tolist() == [1.0, 5.0]


def test_clip_returns_mismatched_columns_raise() -> None:
    """Columnas distintas entre retornos y limites lanzan ValueError."""
    returns = _frame({"A": [1.0, 2.0]})
    bounds = pd.DataFrame({"B": [-1.0, 1.0]}, index=["lower", "upper"])
    with pytest.raises(ValueError):
        clip_returns(returns, bounds)


def test_fit_then_clip_projects_train_bounds_onto_test() -> None:
    """Ajustado con train y proyectado sobre test, el pico de test se recorta al limite de train."""
    train = _frame({"A": np.linspace(-1.0, 1.0, 1_001).tolist()})
    test = _frame({"A": [0.0, 50.0, -50.0]})
    bounds = fit_clip_bounds(train, lower_q=0.01, upper_q=0.99)
    out = clip_returns(test, bounds)
    assert out["A"].tolist() == pytest.approx(
        [0.0, bounds.loc["upper", "A"], bounds.loc["lower", "A"]]
    )


def test_tail_report_top_share_floor_is_robust_to_float_error() -> None:
    """int(100 * 0.29) da 28; el k correcto es 29."""
    values = [float(i) for i in range(1, 101)]
    report = tail_report(_frame({"A": values}), top_share=0.29)
    squares = np.sort(np.asarray(values) ** 2)
    assert report.loc["A", "top_var_share"] == pytest.approx(squares[-29:].sum() / squares.sum())


def test_tail_report_all_nan_column_is_nan_with_zero_count() -> None:
    """Una columna toda NaN tiene n = 0 y el resto de estadisticos NaN."""
    report = tail_report(_frame({"A": [np.nan, np.nan, np.nan]}))
    assert report.loc["A", "n"] == 0
    for col in ("std", "kurtosis", "q0.001", "max_abs_z", "top_var_share"):
        assert np.isnan(report.loc["A", col])
    assert report.loc["A", "n_gt_5sd"] == 0


@pytest.mark.parametrize("values", [[1.0], [1.0, 2.0, 4.0]])
def test_tail_report_short_series_have_nan_kurtosis(values: list[float]) -> None:
    """Con n < 4 la curtosis no esta definida: NaN, sin error."""
    report = tail_report(_frame({"A": values}))
    assert report.loc["A", "n"] == len(values)
    assert np.isnan(report.loc["A", "kurtosis"])
    if len(values) == 1:
        assert np.isnan(report.loc["A", "std"])


def test_tail_report_empty_frame_has_no_rows() -> None:
    """Un frame sin columnas da un informe vacio."""
    assert len(tail_report(pd.DataFrame())) == 0


@pytest.mark.parametrize("bad", [np.inf, -np.inf])
def test_tail_report_infinite_values_raise_naming_column(bad: float) -> None:
    """Un infinito es dato corrupto: falla alto nombrando la columna."""
    with pytest.raises(ValueError, match="B"):
        tail_report(_frame({"A": [1.0, 2.0, 3.0], "B": [1.0, bad, 3.0]}))


def test_fit_clip_bounds_infinite_values_raise() -> None:
    """Un infinito en train falla alto."""
    with pytest.raises(ValueError, match="A"):
        fit_clip_bounds(_frame({"A": [1.0, np.inf, 3.0]}))


def test_fit_clip_bounds_accepts_quantiles_near_the_edges() -> None:
    """Los cuantiles validos mas cercanos a 0 y 1 dan casi los extremos de la muestra."""
    train = _frame({"A": np.arange(101, dtype="float64").tolist()})
    bounds = fit_clip_bounds(train, lower_q=1e-9, upper_q=1 - 1e-9)
    assert bounds.loc["lower", "A"] == pytest.approx(0.0, abs=1e-6)
    assert bounds.loc["upper", "A"] == pytest.approx(100.0, abs=1e-6)


def test_fit_clip_bounds_empty_frame_gives_empty_bounds() -> None:
    """Un frame sin columnas da limites sin columnas."""
    assert fit_clip_bounds(pd.DataFrame()).shape[1] == 0


def test_clip_returns_inverted_bounds_raise() -> None:
    """Limites con lower > upper lanzan ValueError en vez de devolver basura."""
    returns = _frame({"A": [-5.0, 0.0, 5.0]})
    bounds = pd.DataFrame({"A": [2.0, -2.0]}, index=["lower", "upper"])
    with pytest.raises(ValueError, match="lower > upper"):
        clip_returns(returns, bounds)


def test_clip_returns_nan_bounds_leave_column_unclipped() -> None:
    """Una columna con limites NaN no se recorta."""
    returns = _frame({"A": [-5.0, 5.0]})
    bounds = pd.DataFrame({"A": [np.nan, np.nan]}, index=["lower", "upper"])
    assert clip_returns(returns, bounds)["A"].tolist() == [-5.0, 5.0]


def test_clip_returns_empty_frame_returns_empty() -> None:
    """Un frame sin columnas se devuelve vacio."""
    bounds = pd.DataFrame(index=["lower", "upper"])
    assert clip_returns(pd.DataFrame(), bounds).shape[1] == 0


def test_duplicate_columns_raise() -> None:
    """Columnas duplicadas en retornos, limites o train lanzan ValueError claro."""
    dup = pd.DataFrame([[1.0, 2.0]], columns=["A", "A"])
    good = pd.DataFrame({"A": [-1.0, 1.0]}, index=["lower", "upper"])
    with pytest.raises(ValueError, match="duplicadas"):
        clip_returns(dup, good)
    with pytest.raises(ValueError, match="duplicadas"):
        clip_returns(pd.DataFrame({"A": [1.0]}), pd.concat([good, good], axis=1))
    with pytest.raises(ValueError, match="duplicadas"):
        tail_report(dup)
    with pytest.raises(ValueError, match="duplicadas"):
        fit_clip_bounds(dup)
