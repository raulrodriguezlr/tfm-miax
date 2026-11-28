"""Test antifuga de la residualizacion: inyectar futuro hace fallar a los tramposos (MIAX-058)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from miax.eval import beta_study
from miax.eval.beta_study import load_fold, project_fixed, rolling_residuals
from miax.features.factor import FactorModel
from miax.features.factor_btc import BtcFactor
from miax.features.factor_mean import MeanFactor
from miax.features.factor_pca import PcaFactor
from miax.features.panel import Panel
from miax.viz.factor_comparison import compare_factors, split_period
from tests.unit.factor_contract import make_returns
from tests.unit.leakage import assert_no_lookahead, perturb_after

MIN_PERIODS = 10
SYMS = ("AAA", "BBB", "CCC", "DDD")
BTC_SYMS = ("BTCUSDT", "AAA", "BBB", "CCC")
N_TRAIN, N_TEST = 80, 40


def make_factor_returns(
    n: int,
    symbols: tuple[str, ...],
    *,
    seed: int,
    start: str = "2024-01-01",
    factor_col: str | None = None,
) -> pd.DataFrame:
    """Retornos horarios con un factor comun y NaN dispersos; `factor_col` es solo el factor."""
    base = make_returns(n, symbols, seed=seed, start=start, nan_frac=0.02)
    factor = np.random.default_rng(seed + 1000).normal(0.0, 0.01, n)
    out = base + factor[:, None] * np.linspace(0.5, 1.5, len(symbols))
    if factor_col is not None:
        out[factor_col] = pd.Series(factor, index=base.index).where(base[factor_col].notna())
    return out


# kind -> (fabrica del modelo honesto, simbolos, columna del factor)
KINDS: dict[str, tuple[Callable[[], FactorModel], tuple[str, ...], str | None]] = {
    "btc": (lambda: BtcFactor(min_periods=MIN_PERIODS), BTC_SYMS, "BTCUSDT"),
    "mean": (lambda: MeanFactor(min_periods=MIN_PERIODS), SYMS, None),
    "pca": (lambda: PcaFactor(min_periods=MIN_PERIODS), SYMS, None),
}


def _data(kind: str) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Panel train+test contiguo (80 + 40 horas) y el primer instante de test."""
    _, symbols, factor_col = KINDS[kind]
    data = make_factor_returns(N_TRAIN + N_TEST, symbols, seed=7, factor_col=factor_col)
    return data, data.index[N_TRAIN]


# Inicio de test (segunda fila), mitad y ultimo tramo.
TEST_CUTS = [1, N_TEST // 2, N_TEST - 1]


def _pipeline(
    data: pd.DataFrame,
    test_start: pd.Timestamp,
    make_model: Callable[[], FactorModel],
    *,
    fit_with_test: bool = False,
) -> pd.DataFrame:
    """Fit con train y transform de test; `fit_with_test` activa la trampa de ajustar con ambos."""
    # Train = filas < test_start: el corte es exclusivo.
    train = data[data.index < test_start]
    test = data[data.index >= test_start]
    model = make_model()
    model.fit(pd.concat([train, test]) if fit_with_test else train)
    return model.transform(test)


def _fit_state(
    data: pd.DataFrame, test_start: pd.Timestamp, make_model: Callable[[], FactorModel], shift: int
) -> dict[str, object]:
    """Estado ajustado (betas o cargas) al hacer fit con las filas < test_start (+ `shift`)."""
    end = data.index[data.index.get_loc(test_start) + shift]
    return make_model().fit(data[data.index < end]).get_state()


# ---------- el ayudante ----------


def test_perturb_after_changes_only_the_future() -> None:
    """`perturb_after` no toca nada antes del corte, cambia lo posterior y conserva los NaN."""
    data, test_start = _data("mean")
    out = perturb_after(data, test_start)
    pd.testing.assert_frame_equal(out[out.index < test_start], data[data.index < test_start])
    assert not np.allclose(
        out[out.index >= test_start].fillna(0.0), data[data.index >= test_start].fillna(0.0)
    )
    assert out.isna().equals(data.isna())
    pd.testing.assert_frame_equal(out, perturb_after(data, test_start))  # determinista


def test_perturb_after_without_future_raises() -> None:
    """Un corte posterior al ultimo dato no deja futuro que perturbar."""
    data, _ = _data("mean")
    with pytest.raises(ValueError, match="futuro"):
        perturb_after(data, data.index[-1] + pd.Timedelta(hours=1))


def test_assert_no_lookahead_rejects_empty_checks() -> None:
    """Una salida sin filas previas al corte es un error, no un aprobado vacio."""
    data, _ = _data("mean")
    with pytest.raises(ValueError, match="no se compara nada"):
        assert_no_lookahead(lambda d: d.iloc[:0], data, data.index[10])


# ---------- A. los caminos reales no miran al futuro ----------


@pytest.mark.parametrize("move_nans", [False, True])
@pytest.mark.parametrize("kind", list(KINDS))
@pytest.mark.parametrize("cut_pos", TEST_CUTS)
def test_real_transform_and_factors_ignore_the_future(
    kind: str, cut_pos: int, move_nans: bool
) -> None:
    """Con fit(train), residuos y factor de las filas < corte no cambian al perturbar >= corte."""
    make_model, _, _ = KINDS[kind]
    data, test_start = _data(kind)
    cut = data.index[N_TRAIN + cut_pos]
    assert_no_lookahead(
        lambda d: _pipeline(d, test_start, make_model),
        data,
        cut,
        move_nans=move_nans,
        what=f"{kind}: transform",
    )

    def factors(d: pd.DataFrame) -> pd.DataFrame:
        model = make_model().fit(d[d.index < test_start])
        return model.factors(d[d.index >= test_start])

    assert_no_lookahead(factors, data, cut, move_nans=move_nans, what=f"{kind}: factors")


@pytest.mark.parametrize("kind", list(KINDS))
def test_real_fitted_state_ignores_test(kind: str) -> None:
    """El estado ajustado (betas o cargas) con fit(train) no cambia si se perturba todo el test."""
    make_model, _, _ = KINDS[kind]
    data, test_start = _data(kind)
    for pos in (0, N_TEST // 2, N_TEST - 1):
        assert_no_lookahead(
            lambda d: _fit_state(d, test_start, make_model, 0),
            data,
            data.index[N_TRAIN + pos],
            what=f"{kind}: estado de fit(train)",
        )


@pytest.mark.parametrize("move_nans", [False, True])
@pytest.mark.parametrize("kind", ["btc", "mean", "pca"])
@pytest.mark.parametrize("cut_pos", TEST_CUTS)
def test_real_project_fixed_ignores_the_future(kind: str, cut_pos: int, move_nans: bool) -> None:
    """`project_fixed`: perturbar el test no cambia betas ni residuos anteriores al corte."""
    make_model, _, _ = KINDS[kind]
    data, test_start = _data(kind)
    cut = data.index[N_TRAIN + cut_pos]

    def run(d: pd.DataFrame) -> pd.DataFrame:
        return project_fixed(
            make_model, d[d.index < test_start], d[d.index >= test_start]
        ).residuals

    assert_no_lookahead(run, data, cut, move_nans=move_nans, what=f"{kind}: project_fixed residuos")

    def betas(d: pd.DataFrame) -> object:
        return project_fixed(make_model, d[d.index < test_start], d[d.index >= test_start]).betas

    assert_no_lookahead(
        betas, data, test_start, move_nans=move_nans, what=f"{kind}: project_fixed betas"
    )


# 14 dias horarios; test desde el dia 6; ventana de 3 dias y bloques de 2: 6-8, 8-10, 10-12, ...
ROLL_START = pd.Timestamp("2024-01-06", tz="UTC")
ROLL_END = pd.Timestamp("2024-01-15", tz="UTC")
ROLL_BLOCK_STARTS = ["2024-01-08 00:00", "2024-01-10 00:00", "2024-01-14 00:00"]
ROLL_MID_BLOCK = ["2024-01-07 06:00", "2024-01-09 12:00", "2024-01-11 23:00", "2024-01-14 06:00"]


def _roll_data(kind: str) -> pd.DataFrame:
    """Catorce dias horarios con factor comun para los tests de ventana movil."""
    _, symbols, factor_col = KINDS[kind]
    return make_factor_returns(14 * 24, symbols, seed=9, factor_col=factor_col)


@pytest.mark.parametrize("move_nans", [False, True])
@pytest.mark.parametrize("kind", ["btc", "mean"])
@pytest.mark.parametrize("cut_text", ROLL_BLOCK_STARTS + ROLL_MID_BLOCK)
def test_real_rolling_residuals_ignore_the_future(
    kind: str, cut_text: str, move_nans: bool
) -> None:
    """`rolling_residuals`: perturbar desde un corte (inicio o mitad de bloque) no cambia antes."""
    make_model, _, _ = KINDS[kind]
    data = _roll_data(kind)

    def run(d: pd.DataFrame) -> pd.DataFrame:
        fit = rolling_residuals(
            make_model, d, ROLL_START, ROLL_END, window_days=3, step_days=2, min_valid_share=0.5
        )
        return fit.residuals

    # Precondicion solo con los datos reales: con `move_nans` el futuro pierde celdas y un bloque
    # posterior puede fallar legitimamente (<50 % validas); eso no es fuga, y se compara igual.
    baseline = rolling_residuals(
        make_model, data, ROLL_START, ROLL_END, window_days=3, step_days=2, min_valid_share=0.5
    )
    assert baseline.n_failed == 0
    assert_no_lookahead(
        run,
        data,
        pd.Timestamp(cut_text, tz="UTC"),
        move_nans=move_nans,
        what=f"{kind}: rolling {cut_text}",
    )


@pytest.mark.parametrize("kind", ["btc", "mean", "pca"])
def test_real_split_period_and_compare_factors_ignore_the_future(kind: str) -> None:
    """`split_period` y el ajuste de `compare_factors` solo usan train: el 3er mes no los cambia."""
    make_model, symbols, factor_col = KINDS[kind]
    data = make_factor_returns(24 * 90, symbols, seed=11, factor_col=factor_col)
    cut = data.index[0] + pd.DateOffset(months=2)  # fin del train de split_period

    train_rows = lambda d: split_period(d)[0]
    assert_no_lookahead(train_rows, data, cut, what=f"{kind}: split_period train")

    def fitted_state(d: pd.DataFrame) -> object:
        seen: list[FactorModel] = []

        def recorder() -> FactorModel:
            seen.append(make_model())
            return seen[-1]

        compare_factors(d, models={"m": recorder}, min_periods=100)
        return seen[0].get_state()

    assert_no_lookahead(fitted_state, data, cut, what=f"{kind}: compare_factors estado")
    train, test = split_period(data)
    assert train.index.max() < test.index.min() == cut


# ---------- load_fold con cached_panel sustituido ----------

FOLD_START = pd.Timestamp("2024-01-01", tz="UTC")
FOLD_SYMS = ("AAAUSDT", "BBBUSDT", "CCCUSDT", "SPARSEUSDT")


def _fold_data() -> pd.DataFrame:
    """Dos meses horarios (diciembre train, enero test) con una moneda casi vacia en train."""
    data = make_factor_returns(62 * 24, FOLD_SYMS, seed=13, start="2023-12-01")
    sparse = (np.random.default_rng(5).random(len(data)) < 0.7) & (data.index < FOLD_START)
    data.loc[sparse, "SPARSEUSDT"] = np.nan
    return data


def _fake_load_fold(monkeypatch: pytest.MonkeyPatch, d: pd.DataFrame) -> beta_study.FoldData:
    """Ejecuta `load_fold` con un panel falso hecho con `d` en lugar de la cache de disco."""

    def fake_cached_panel(*_args: object, **_kwargs: object) -> Panel:
        return Panel(
            features={"return": d}, valid=d.notna(), interval="1m", symbols=tuple(d.columns)
        )

    monkeypatch.setattr(beta_study, "cached_panel", fake_cached_panel)
    universe = pd.DataFrame(
        {
            "period_start": [FOLD_START] * len(FOLD_SYMS),
            "period_end": [pd.Timestamp("2024-02-01", tz="UTC")] * len(FOLD_SYMS),
            "symbol": list(FOLD_SYMS),
            "category": "l1",
        }
    )
    return load_fold(FOLD_START, universe=universe, train_months=1, test_months=1)


@pytest.mark.parametrize("cut_text", ["2024-01-01 00:00", "2024-01-15 12:00"])
def test_real_load_fold_ignores_the_test_period(
    monkeypatch: pytest.MonkeyPatch, cut_text: str
) -> None:
    """`load_fold`: cambiar retornos y NaN del test no cambia `returns_train` ni `train_valid`."""
    data = _fold_data()
    cut = pd.Timestamp(cut_text, tz="UTC")
    fold = _fake_load_fold(monkeypatch, data)
    assert fold.kept == ["AAAUSDT", "BBBUSDT", "CCCUSDT"]  # la casi vacia queda fuera

    assert_no_lookahead(
        lambda d: _fake_load_fold(monkeypatch, d).returns_train,
        data,
        FOLD_START,
        move_nans=True,
        what="load_fold: returns_train",
    )
    assert_no_lookahead(
        lambda d: _fake_load_fold(monkeypatch, d).train_valid,
        data,
        FOLD_START,
        move_nans=True,
        what="load_fold: train_valid",
    )
    assert_no_lookahead(
        lambda d: _fake_load_fold(monkeypatch, d).kept,
        data,
        cut,
        move_nans=True,
        what="load_fold: kept",
    )


# ---------- B. tramposos: inyectar el futuro los hace fallar ----------


class _MeanLooksAhead(MeanFactor):
    """Fuga: el factor de la fila t es la media de la fila t+1 (`shift(-1)`)."""

    def _mean_factor(self, returns: pd.DataFrame) -> pd.Series:
        return super()._mean_factor(returns).shift(-1)


class _MeanLooksAtFutureValidity(MeanFactor):
    """Fuga solo por el patron de NaN: el factor se escala con el nº de monedas validas en t+1."""

    def _mean_factor(self, returns: pd.DataFrame) -> pd.Series:
        future_count = returns.notna().sum(axis=1).shift(-1)
        return super()._mean_factor(returns) * (1 + 1e-3 * future_count)


class _MeanTinyLeak(MeanFactor):
    """Fuga atenuada: suma 1e-9 veces el retorno de la primera moneda en t+1 al factor."""

    def _mean_factor(self, returns: pd.DataFrame) -> pd.Series:
        return super()._mean_factor(returns) + 1e-9 * returns.iloc[:, 0].shift(-1)


class _BtcCenteredSmoothing(BtcFactor):
    """Fuga: el factor es BTC suavizado con una ventana centrada (usa t-1, t y t+1)."""

    def _factors(self, returns: pd.DataFrame) -> pd.DataFrame:
        smooth = returns[self.symbol].rolling(3, center=True, min_periods=1).mean()
        return smooth.rename("btc").to_frame()


class _PcaStdOfAllPanel(PcaFactor):
    """Fuga: las cargas se reescalan con la desviacion tipica de todo el panel (train + test)."""

    def __init__(self, whole_panel: pd.DataFrame, *, min_periods: int = MIN_PERIODS) -> None:
        super().__init__(min_periods=min_periods)
        self._whole_panel = whole_panel

    def _fit(self, returns: pd.DataFrame) -> None:
        super()._fit(returns)
        assert self._loadings is not None
        scaled = self._loadings * (self._whole_panel.std() / returns.std())
        self._loadings = scaled / scaled.mean()


def _leaky_rolling_residuals(
    make_model: Callable[[], FactorModel],
    returns: pd.DataFrame,
    *,
    window_days: int,
    step_days: int,
) -> pd.DataFrame:
    """Copia de `rolling_residuals` con la fuga: la ventana llega hasta el fin del propio bloque."""
    index = returns.index
    out = pd.DataFrame(
        np.nan, index=index[(index >= ROLL_START) & (index < ROLL_END)], columns=returns.columns
    )
    block_start = ROLL_START
    while block_start < ROLL_END:
        block_end = min(block_start + pd.Timedelta(days=step_days), ROLL_END)
        # FUGA: `< block_end` en vez de `< block_start`; el bloque proyectado entra en el ajuste.
        in_window = (index >= block_start - pd.Timedelta(days=window_days)) & (index < block_end)
        block = returns[(index >= block_start) & (index < block_end)]
        model = make_model().fit(returns[in_window])
        out.loc[block.index] = model.transform(block)
        block_start += pd.Timedelta(days=step_days)
    return out


def _mid_test_cuts() -> list[int]:
    """Posiciones de corte dentro del test: mitad y ultima fila."""
    return [N_TEST // 2, N_TEST - 1]


@pytest.mark.parametrize("cut_pos", _mid_test_cuts())
def test_cheater_mean_shift_minus_one_fails(cut_pos: int) -> None:
    """Un MeanFactor que usa la fila t+1 hace fallar el ayudante; el MeanFactor real no."""
    data, test_start = _data("mean")
    cut = data.index[N_TRAIN + cut_pos]
    honest = lambda d: _pipeline(d, test_start, lambda: MeanFactor(min_periods=MIN_PERIODS))
    cheater = lambda d: _pipeline(d, test_start, lambda: _MeanLooksAhead(min_periods=MIN_PERIODS))
    assert_no_lookahead(honest, data, cut, what="control mean")
    with pytest.raises(AssertionError, match="D-03"):
        assert_no_lookahead(cheater, data, cut, what="mean shift(-1)")


@pytest.mark.parametrize("cut_pos", _mid_test_cuts())
def test_cheater_mean_future_validity_fails(cut_pos: int) -> None:
    """Una fuga que viaja solo por el patron de NaN del futuro solo se caza con `move_nans`."""
    data, test_start = _data("mean")
    cut = data.index[N_TRAIN + cut_pos]
    honest = lambda d: _pipeline(d, test_start, lambda: MeanFactor(min_periods=MIN_PERIODS))
    cheater = lambda d: _pipeline(
        d, test_start, lambda: _MeanLooksAtFutureValidity(min_periods=MIN_PERIODS)
    )
    assert_no_lookahead(honest, data, cut, move_nans=True, what="control mean validez")
    with pytest.raises(AssertionError, match="D-03"):
        assert_no_lookahead(cheater, data, cut, move_nans=True, what="mean validez del futuro")


@pytest.mark.parametrize("cut_pos", _mid_test_cuts())
def test_cheater_mean_tiny_leak_fails(cut_pos: int) -> None:
    """Una fuga de 1e-9 falla porque la comparacion es exacta; el MeanFactor real es identico."""
    data, test_start = _data("mean")
    cut = data.index[N_TRAIN + cut_pos]
    honest = lambda d: _pipeline(d, test_start, lambda: MeanFactor(min_periods=MIN_PERIODS))
    cheater = lambda d: _pipeline(d, test_start, lambda: _MeanTinyLeak(min_periods=MIN_PERIODS))
    assert_no_lookahead(honest, data, cut, what="control mean exacto")
    with pytest.raises(AssertionError, match="D-03"):
        assert_no_lookahead(cheater, data, cut, what="mean fuga atenuada")


@pytest.mark.parametrize("cut_pos", _mid_test_cuts())
def test_cheater_btc_centered_window_fails(cut_pos: int) -> None:
    """Un BtcFactor con el factor suavizado centrado falla; el BtcFactor real no."""
    data, test_start = _data("btc")
    cut = data.index[N_TRAIN + cut_pos]
    honest = lambda d: _pipeline(d, test_start, lambda: BtcFactor(min_periods=MIN_PERIODS))
    cheater = lambda d: _pipeline(
        d, test_start, lambda: _BtcCenteredSmoothing(min_periods=MIN_PERIODS)
    )
    assert_no_lookahead(honest, data, cut, what="control btc")
    with pytest.raises(AssertionError, match="D-03"):
        assert_no_lookahead(cheater, data, cut, what="btc ventana centrada")


@pytest.mark.parametrize("kind", ["btc", "mean", "pca"])
@pytest.mark.parametrize("cut_pos", _mid_test_cuts())
def test_cheater_fit_with_train_and_test_fails(kind: str, cut_pos: int) -> None:
    """Ajustar con train + test hace fallar el ayudante; ajustar solo con train no."""
    make_model, _, _ = KINDS[kind]
    data, test_start = _data(kind)
    cut = data.index[N_TRAIN + cut_pos]
    assert_no_lookahead(
        lambda d: _pipeline(d, test_start, make_model), data, cut, what=f"control {kind}"
    )
    with pytest.raises(AssertionError, match="D-03"):
        assert_no_lookahead(
            lambda d: _pipeline(d, test_start, make_model, fit_with_test=True),
            data,
            cut,
            what=f"{kind} fit con train+test",
        )


@pytest.mark.parametrize("cut_pos", _mid_test_cuts())
def test_cheater_pca_std_of_whole_panel_fails(cut_pos: int) -> None:
    """Un PCA que estandariza con la desviacion de todo el panel falla; el PcaFactor real no."""
    data, test_start = _data("pca")
    cut = data.index[N_TRAIN + cut_pos]
    honest = lambda d: _pipeline(d, test_start, lambda: PcaFactor(min_periods=MIN_PERIODS))
    cheater = lambda d: _pipeline(d, test_start, lambda: _PcaStdOfAllPanel(d))
    assert_no_lookahead(honest, data, cut, what="control pca")
    with pytest.raises(AssertionError, match="D-03"):
        assert_no_lookahead(cheater, data, cut, what="pca std de todo el panel")


@pytest.mark.parametrize("kind", ["btc", "mean", "pca"])
def test_cheater_split_shifted_one_row_fails(kind: str) -> None:
    """Un corte desplazado una fila (train incluye el primer minuto de test) falla; el exacto no."""
    make_model, _, _ = KINDS[kind]
    data, test_start = _data(kind)
    # Se perturba justo desde test_start: la unica fila de test que el tramposo ve en train.
    assert_no_lookahead(
        lambda d: _fit_state(d, test_start, make_model, 0), data, test_start, what="control corte"
    )
    with pytest.raises(AssertionError, match="D-03"):
        assert_no_lookahead(
            lambda d: _fit_state(d, test_start, make_model, 1),
            data,
            test_start,
            what=f"{kind} corte desplazado",
        )


@pytest.mark.parametrize("cut_text", ROLL_MID_BLOCK)
def test_cheater_rolling_window_including_the_block_fails(cut_text: str) -> None:
    """Una ventana movil que incluye el bloque proyectado falla a mitad de bloque; la real no."""
    make_model, _, _ = KINDS["mean"]
    data = _roll_data("mean")
    cut = pd.Timestamp(cut_text, tz="UTC")
    # Solo a mitad de bloque: con la ventana hasta `block_end`, un corte justo al inicio de un
    # bloque no altera nada anterior.

    def honest(d: pd.DataFrame) -> pd.DataFrame:
        return rolling_residuals(
            make_model, d, ROLL_START, ROLL_END, window_days=3, step_days=2
        ).residuals

    def cheater(d: pd.DataFrame) -> pd.DataFrame:
        return _leaky_rolling_residuals(make_model, d, window_days=3, step_days=2)

    assert_no_lookahead(honest, data, cut, what="control rolling")
    with pytest.raises(AssertionError, match="D-03"):
        assert_no_lookahead(cheater, data, cut, what="rolling con el bloque en la ventana")
