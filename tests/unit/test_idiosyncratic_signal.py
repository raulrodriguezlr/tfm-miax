"""Tests de la senal idiosincratica (MIAX-057): correlacion con retardo, nulo, decision y CLI."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import miax.eval.__main__ as cli
import miax.eval.idiosyncratic_signal as sig
from miax.eval import (
    SignalResult,
    contemporaneous_structure,
    decision_table,
    fold_symbols,
    fold_windows,
    lagged_cross_correlation,
    load_fold,
    signal_statistics,
    study_signal,
    write_signal_tables,
)
from miax.eval.beta_study import FoldData
from miax.features.factor import FactorModel
from miax.features.factor_mean import MeanFactor
from miax.features.factor_pca import PcaFactor

DAY = 1440


def _index(n: int, start: str = "2024-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="min", tz="UTC")


def _noise(n: int, k: int, seed: int, start: str = "2024-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = [f"S{i}USDT" for i in range(k)]
    return pd.DataFrame(rng.normal(size=(n, k)), index=_index(n, start), columns=cols)


def _leadlag(n: int = 3000, lag: int = 2, seed: int = 3, k: int = 5) -> pd.DataFrame:
    """`S1(t) = 0.8 * S0(t - lag) + ruido`: S0 adelanta a S1; el resto es ruido."""
    frame = _noise(n, k, seed)
    x = frame["S0USDT"].to_numpy()
    y = np.zeros(n)
    y[lag:] = 0.8 * x[:-lag]
    frame["S1USDT"] = y + 0.6 * frame["S1USDT"].to_numpy()
    return frame


# --- lagged_cross_correlation ---


def _loop_corr(frame: pd.DataFrame, lag: int, min_obs: int) -> np.ndarray:
    values = frame.to_numpy()
    n = values.shape[1]
    out = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            a, b = values[:-lag, i], values[lag:, j]
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() >= min_obs and a[ok].std() > 0 and b[ok].std() > 0:
                out[i, j] = np.corrcoef(a[ok], b[ok])[0, 1]
    return out


@pytest.mark.parametrize("lag", [1, 2, 5])
def test_lagged_correlation_matches_pairwise_loop_with_nans(lag: int) -> None:
    """Coincide con un bucle de `np.corrcoef` por pares completos con un 5 % de NaN."""
    frame = _leadlag(1500, k=6)
    mask = np.random.default_rng(9).random(frame.shape) < 0.05
    frame = frame.mask(mask)
    got = lagged_cross_correlation(frame, lag, min_obs=100)
    np.testing.assert_allclose(got, _loop_corr(frame, lag, 100), atol=1e-9, equal_nan=True)


def test_lagged_correlation_nan_cells() -> None:
    """Menos de `min_obs` comunes o varianza cero dan NaN solo en las casillas afectadas."""
    frame = _noise(500, 3, seed=1)
    frame["S2USDT"] = 0.0  # varianza cero
    frame.iloc[:450, 1] = np.nan  # 50 datos: por debajo de min_obs
    got = lagged_cross_correlation(frame, 1, min_obs=100)
    assert np.isfinite(got[0, 0])
    assert np.isnan(got[:, 2]).all() and np.isnan(got[2, :]).all()
    assert np.isnan(got[:, 1]).all() and np.isnan(got[1, :]).all()


def test_lagged_correlation_rejects_bad_input() -> None:
    """Lag < 1, indice irregular o sin filas suficientes dan `ValueError`."""
    frame = _noise(100, 2, seed=1)
    with pytest.raises(ValueError, match="lag"):
        lagged_cross_correlation(frame, 0, min_obs=10)
    with pytest.raises(ValueError, match="regular"):
        lagged_cross_correlation(frame.iloc[[0, 1, 2, 5, 6, 7, 20, 30]], 1, min_obs=2)
    with pytest.raises(ValueError, match="regular"):
        lagged_cross_correlation(frame.reset_index(drop=True), 1, min_obs=10)
    with pytest.raises(ValueError):
        lagged_cross_correlation(frame, 100, min_obs=10)


def test_known_lead_lag_peaks_in_the_right_direction() -> None:
    """X adelanta a y en 2 barras: pico en C_2[x, y] y no en C_2[y, x] ni en otros retardos."""
    frame = _leadlag()
    c2 = lagged_cross_correlation(frame, 2, min_obs=100)
    assert c2[0, 1] > 0.7
    assert abs(c2[1, 0]) < 0.1
    assert abs(lagged_cross_correlation(frame, 1, min_obs=100)[0, 1]) < 0.1
    assert abs(lagged_cross_correlation(frame, 3, min_obs=100)[0, 1]) < 0.1


# --- signal_statistics ---

_KW = {"min_shift_rows": 50, "max_lag": 3, "n_shifts": 99, "min_obs": 100}


def test_signal_statistics_detects_injected_lead_lag() -> None:
    """Con lead-lag inyectado, p_cross = 0,01 (nada del nulo lo alcanza) y la asimetria tambien."""
    stats = signal_statistics(_leadlag(), **_KW)
    assert stats["p_cross"] == pytest.approx(0.01)
    assert stats["p_asym"] == pytest.approx(0.01)
    assert stats["s_cross"] > stats["null_q95_cross"] > 0
    assert stats["ratio_cross"] > 5
    assert stats["n_coins"] == 5 and stats["n_rows"] == 3000


def test_signal_statistics_pure_noise_is_not_significant_in_most_seeds() -> None:
    """Ruido i.i.d. con un factor comun ya quitado: p_cross < 0,05 en a lo sumo 2 de 10 semillas."""
    hits = 0
    for seed in range(10):
        stats = signal_statistics(_noise(2000, 8, seed=100 + seed), seed=seed, **_KW)
        hits += stats["p_cross"] < 0.05
    assert hits <= 2


def test_signal_statistics_is_deterministic_and_seed_sensitive() -> None:
    """Misma semilla, mismos resultados; otra semilla, otro nulo."""
    frame = _noise(1500, 5, seed=4)
    a = signal_statistics(frame, seed=5, **_KW)
    b = signal_statistics(frame, seed=5, **_KW)
    c = signal_statistics(frame, seed=6, **_KW)
    assert a == b
    assert a["null_median_cross"] != c["null_median_cross"]
    assert a["s_cross"] == c["s_cross"]  # el observado no depende de la semilla


def test_observed_statistic_never_pairs_with_future_rows() -> None:
    """Cambiar filas posteriores a K no altera la correlacion calculada con las filas <= K."""
    frame = _leadlag(2000)
    cut = 1200
    changed = frame.copy()
    changed.iloc[cut:] = np.random.default_rng(0).normal(size=changed.iloc[cut:].shape) * 50
    for lag in (1, 2, 4):
        np.testing.assert_array_equal(
            lagged_cross_correlation(frame.iloc[:cut], lag, min_obs=100),
            lagged_cross_correlation(changed.iloc[:cut], lag, min_obs=100),
        )


def test_future_only_dependence_gives_no_forward_signal() -> None:
    """Si y(t) = x(t + 2) (x depende del futuro de y), x no 'predice' a y: el pico va de y a x."""
    frame = _noise(3000, 4, seed=8)
    frame["S1USDT"] = frame["S0USDT"].shift(-2).fillna(0.0)
    c2 = lagged_cross_correlation(frame, 2, min_obs=100)
    assert abs(c2[0, 1]) < 0.1
    assert c2[1, 0] > 0.9


def test_null_shifts_are_within_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Los desfases del nulo nunca son 0 ni < min_shift_rows, ni > T - max_lag - min_shift_rows."""
    frame = _noise(1000, 3, seed=2)
    seen: list[int] = []
    real = sig._correlation_matrix

    def spy(blocks: sig._LagBlocks, shift: int, min_obs: int) -> object:
        if blocks.length == len(frame) - 1:  # retardo 1: el modulo no cambia el desfase
            seen.append(shift)
        return real(blocks, shift, min_obs)

    monkeypatch.setattr(sig, "_correlation_matrix", spy)
    signal_statistics(frame, min_shift_rows=60, max_lag=3, n_shifts=300, min_obs=100)
    assert seen[0] == 0 and len(seen) == 301
    nulls = np.array(seen[1:])
    assert nulls.min() >= 60 and nulls.max() <= 1000 - 3 - 60
    assert (nulls > 0).all()
    assert nulls.max() > 800 and nulls.min() < 120  # recorre el rango, no una esquina


@pytest.mark.parametrize(
    ("rows", "min_shift", "max_lag"),
    [(1000, 3, 3), (1000, 2, 3), (100, 60, 3), (120, 60, 3)],
)
def test_signal_statistics_rejects_impossible_shifts(
    rows: int, min_shift: int, max_lag: int
) -> None:
    """min_shift_rows <= max_lag, o T que no da para el desfase minimo, dan `ValueError`."""
    with pytest.raises(ValueError):
        signal_statistics(
            _noise(rows, 3, seed=1),
            min_shift_rows=min_shift,
            max_lag=max_lag,
            n_shifts=5,
            min_obs=10,
        )


def test_signal_statistics_rejects_bad_parameters() -> None:
    """Parametros fuera de rango dan `ValueError`."""
    frame = _noise(500, 3, seed=1)
    for bad in ({"max_lag": 0}, {"n_shifts": 0}, {"min_shift_rows": 0}, {"min_obs": 1}):
        kwargs = {**_KW, "n_shifts": 5, **bad}
        with pytest.raises(ValueError):
            signal_statistics(frame, **kwargs)


# --- contemporaneous_structure ---


def test_structure_detects_sector_factor() -> None:
    """Un factor sectorial inyectado deja al menos un autovalor sobre el borde de MP."""
    frame = _noise(2000, 20, seed=5)
    sector = np.random.default_rng(6).normal(size=2000)
    frame.iloc[:, :8] += 1.5 * sector[:, None]
    out = contemporaneous_structure(frame, min_obs=100)
    assert out["n_above_mp"] >= 1
    assert out["first_eigenvalue_share"] > 0.15
    assert out["mp_edge"] == pytest.approx((1 + np.sqrt(20 / 2000)) ** 2)


def test_structure_pure_noise_has_no_eigenvalue_above_mp() -> None:
    """Ruido puro: ningun autovalor sobre el borde y borde = (1 + sqrt(N/T))^2."""
    out = contemporaneous_structure(_noise(5000, 20, seed=7), min_obs=100)
    assert out["n_above_mp"] == 0
    assert out["mp_edge"] == pytest.approx((1 + np.sqrt(20 / 5000)) ** 2)


def test_structure_without_enough_coins_is_nan() -> None:
    """Con menos de dos monedas usables todo es NaN."""
    frame = _noise(500, 1, seed=1)
    out = contemporaneous_structure(frame, min_obs=100)
    assert all(np.isnan(v) for v in out.values())


# --- decision_table ---


def _by_fold(
    p: dict[tuple[str, str], list[float]],
) -> pd.DataFrame:
    rows = [
        {"fold": f"f{i}", "factor": factor, "horizon": horizon, "p_cross": value}
        for (factor, horizon), values in p.items()
        for i, value in enumerate(values)
    ]
    return pd.DataFrame(rows)


def _wins(n_sig: int, n: int = 12) -> list[float]:
    return [0.01] * n_sig + [0.5] * (n - n_sig)


def _final(table: pd.DataFrame) -> str:
    return str(table.iloc[-1]["verdict"])


def test_decision_signal_pivot_inconclusive() -> None:
    """Los tres veredictos con 12 folds."""
    signal = decision_table(_by_fold({("mean", "1m"): _wins(9), ("pca", "1m"): _wins(12)}))
    assert _final(signal) == "signal"
    pivot = decision_table(_by_fold({("mean", "1m"): _wins(12), ("pca", "1m"): _wins(3)}))
    assert _final(pivot) == "pivot"
    middle = decision_table(_by_fold({("mean", "1m"): _wins(9), ("pca", "1m"): _wins(6)}))
    assert _final(middle) == "inconclusive"
    edge = decision_table(_by_fold({("mean", "1m"): _wins(8), ("pca", "1m"): _wins(12)}))
    assert _final(edge) == "inconclusive"
    edge4 = decision_table(_by_fold({("mean", "1m"): _wins(4), ("pca", "1m"): _wins(12)}))
    assert _final(edge4) == "inconclusive"


def test_decision_nan_is_not_a_signal_fold() -> None:
    """Un NaN no cuenta como fold con senal ni como fold valido."""
    p = _wins(9, 9) + [float("nan")] * 3
    table = decision_table(_by_fold({("mean", "1m"): p, ("pca", "1m"): _wins(12)}))
    mean = table[(table["factor"] == "mean") & (table["horizon"] == "1m")].iloc[0]
    assert (mean["n_folds"], mean["n_significant"]) == (9, 9)
    assert mean["verdict"] == "inconclusive"  # 9 validos < 12 exigidos
    assert _final(table) == "inconclusive"


def test_decision_too_few_valid_folds_is_never_pivot() -> None:
    """Menos de 12 folds validos es inconclusive aunque haya 0 significativos."""
    table = decision_table(_by_fold({("mean", "1m"): _wins(0, 4), ("pca", "1m"): _wins(0, 4)}))
    assert _final(table) == "inconclusive"
    assert set(table.iloc[:-1]["verdict"]) == {"inconclusive"}
    ok = decision_table(
        _by_fold({("mean", "1m"): _wins(0, 4), ("pca", "1m"): _wins(0, 4)}),
        min_wins=4,
        n_folds_required=4,
    )
    assert _final(ok) == "pivot"


def test_decision_missing_deciding_factor_is_inconclusive() -> None:
    """Si falta un factor que decide (PCA), el veredicto no es signal."""
    table = decision_table(_by_fold({("mean", "1m"): _wins(12)}))
    assert _final(table) == "inconclusive"


def test_decision_reports_btc_and_non_primary_horizons() -> None:
    """BTC y los horizontes distintos de 1m son `report` y no deciden."""
    table = decision_table(
        _by_fold(
            {
                ("mean", "1m"): _wins(12),
                ("pca", "1m"): _wins(12),
                ("btc", "1m"): _wins(0),
                ("mean", "5m"): _wins(0),
                ("pca", "15m"): _wins(0),
            }
        )
    )
    verdicts = {(r.factor, r.horizon): r.verdict for r in table.itertuples()}
    assert verdicts[("btc", "1m")] == "report"
    assert verdicts[("mean", "5m")] == "report" and verdicts[("pca", "15m")] == "report"
    assert _final(table) == "signal"


def test_decision_verdict_row_has_na_counts() -> None:
    """La fila `verdict` es la ultima y n_folds / n_significant son NA."""
    table = decision_table(_by_fold({("mean", "1m"): _wins(12), ("pca", "1m"): _wins(12)}))
    last = table.iloc[-1]
    assert last["factor"] == "verdict" and last["horizon"] == "1m"
    assert pd.isna(last["n_folds"]) and pd.isna(last["n_significant"])
    assert list(table.columns) == sig.DECISION_COLUMNS


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": 0.0},
        {"alpha": 1.0},
        {"min_wins": 3, "max_wins_pivot": 3},
        {"min_wins": 5, "max_wins_pivot": 6},
        {"n_folds_required": 0},
        {"n_folds_required": 8},
    ],
)
def test_decision_rejects_bad_parameters(kwargs: dict[str, float]) -> None:
    """alpha, min_wins, max_wins_pivot y n_folds_required invalidos dan `ValueError`."""
    with pytest.raises(ValueError):
        decision_table(_by_fold({("mean", "1m"): _wins(12)}), **kwargs)


def test_decision_alpha_changes_the_count() -> None:
    """`alpha` se aplica con desigualdad estricta."""
    by_fold = _by_fold({("mean", "1m"): [0.05] * 12, ("pca", "1m"): [0.05] * 12})
    assert _final(decision_table(by_fold, alpha=0.05)) == "pivot"
    assert _final(decision_table(by_fold, alpha=0.06)) == "signal"


# --- fold_symbols y load_fold ---


def _universe() -> pd.DataFrame:
    rows = [("2024-01-01", "2024-02-01", s, "l1") for s in ("AAAUSDT", "BBBUSDT", "AAAUSDT")]
    frame = pd.DataFrame(rows, columns=["period_start", "period_end", "symbol", "category"])
    for column in ("period_start", "period_end"):
        frame[column] = pd.to_datetime(frame[column], utc=True).astype("datetime64[us, UTC]")
    return frame


def test_fold_symbols_public_dedup_and_errors() -> None:
    """Devuelve las monedas del trimestre sin repetir y falla si no esta o acaba en otro dia."""
    start = pd.Timestamp("2024-01-01", tz="UTC")
    assert fold_symbols(_universe(), start, pd.Timestamp("2024-02-01", tz="UTC")) == [
        "AAAUSDT",
        "BBBUSDT",
    ]
    with pytest.raises(ValueError, match="no esta en el universo"):
        fold_symbols(_universe(), pd.Timestamp("2025-01-01", tz="UTC"), start)
    with pytest.raises(ValueError, match="no acaba"):
        fold_symbols(_universe(), start, pd.Timestamp("2024-03-01", tz="UTC"))


def test_load_fold_public_splits_train_and_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`load_fold` parte train y test sin solape y descarta monedas poco validas en train."""
    import miax.eval.beta_study as study

    index = _index(40 * DAY, "2023-12-01")
    returns = pd.DataFrame(
        np.random.default_rng(1).normal(size=(len(index), 2)),
        index=index,
        columns=["AAAUSDT", "BBBUSDT"],
    )
    valid = pd.DataFrame(True, index=index, columns=returns.columns)
    valid.loc[index < "2024-01-01", "BBBUSDT"] = False

    panel = SimpleNamespace(features={"return": returns}, valid=valid)
    monkeypatch.setattr(study, "cached_panel", lambda *a, **k: panel)
    fold = load_fold(
        pd.Timestamp("2024-01-01", tz="UTC"),
        universe=_universe(),
        train_months=1,
        test_months=1,
        base_dir=tmp_path,
        cache_dir=tmp_path,
    )
    assert isinstance(fold, FoldData)
    assert fold.label == "2024-01-01" and fold.symbols == ["AAAUSDT", "BBBUSDT"]
    assert fold.kept == ["AAAUSDT"]
    assert fold.returns_train.index.max() < fold.test_start == fold.returns_test.index.min()
    assert fold.returns_test.index.max() < fold.test_end
    assert list(fold.returns_test.columns) == ["AAAUSDT"]


# --- study_signal de extremo a extremo con datos sinteticos ---

_SYMS = ("S0USDT", "S1USDT", "S2USDT", "S3USDT")


def _fake_fold_world(
    monkeypatch: pytest.MonkeyPatch, *, test_rows: int, train_rows: int = 3000, seed: int = 0
) -> None:
    """Sustituye `load_fold` y `horizon_returns_from_cache` por retornos sinteticos."""
    store: dict[pd.Timestamp, pd.DataFrame] = {}

    def make(start: pd.Timestamp, n: int, s: int) -> pd.DataFrame:
        rng = np.random.default_rng(s)
        factor = rng.normal(0.0, 0.001, n)
        data = {
            sym: (0.5 + 0.3 * k) * factor + rng.normal(0.0, 0.001, n) for k, sym in enumerate(_SYMS)
        }
        return pd.DataFrame(data, index=pd.date_range(start, periods=n, freq="min"))

    def fake_load(test_start: pd.Timestamp, **kwargs: object) -> FoldData:
        train_start, start, end = fold_windows(test_start)
        train = make(start - pd.Timedelta(minutes=train_rows), train_rows, seed + start.month)
        test = make(start, test_rows, seed + 100 + start.month)
        store[start] = pd.concat([train, test])
        return FoldData(
            f"{start:%Y-%m-%d}", list(_SYMS), list(_SYMS), pd.Series(1.0, index=list(_SYMS)),
            train, test, train_start, start, end,
        )  # fmt: skip

    def fake_horizon(
        kept: list[str], horizon: str, start_ms: int, end_ms: int, **kwargs: object
    ) -> pd.DataFrame:
        frame = pd.concat(store.values())
        lo, hi = (pd.Timestamp(m, unit="ms", tz="UTC") for m in (start_ms, end_ms))
        frame = frame[(frame.index >= lo) & (frame.index < hi)][kept]
        return frame.groupby(frame.index.floor(horizon.replace("m", "min"))).sum()

    monkeypatch.setattr(sig, "load_fold", fake_load)
    monkeypatch.setattr(sig, "horizon_returns_from_cache", fake_horizon)


def _mean() -> FactorModel:
    return MeanFactor(min_periods=100)


def _pca() -> FactorModel:
    return PcaFactor(min_periods=100)


_STARTS = [pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-04-01", tz="UTC")]
_STUDY = {
    "universe": pd.DataFrame(),
    "factors": {"mean": _mean, "pca": _pca},
    "max_lag": 2,
    "n_shifts": 5,
    "min_obs": 100,
}


def test_study_signal_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """Una fila por fold, factor y horizonte, con las columnas esperadas y estadisticos finitos."""
    _fake_fold_world(monkeypatch, test_rows=4000)
    result = study_signal(_STARTS, horizons=("1m", "5m"), **_STUDY)
    by_fold = result.by_fold
    assert list(by_fold.columns) == sig.BY_FOLD_COLUMNS
    assert len(by_fold) == 2 * 2 * 2
    assert set(by_fold["fold"]) == {"2024-01-01", "2024-04-01"}
    assert set(by_fold["factor"]) == {"mean", "pca"} and set(by_fold["horizon"]) == {"1m", "5m"}
    assert (by_fold["error"] == "").all()
    assert by_fold[["s_cross", "p_cross", "n_above_mp"]].notna().all().all()
    assert ((by_fold["p_cross"] >= 1 / 6) & (by_fold["p_cross"] <= 1)).all()
    assert list(result.decision.columns) == sig.DECISION_COLUMNS
    assert _final(result.decision) == "inconclusive"  # 2 folds < 12


def test_study_signal_failed_factor_fit_gives_error_row(monkeypatch: pytest.MonkeyPatch) -> None:
    """Un ajuste que falla da fila con `error` y NaN, y el estudio sigue con los demas."""
    _fake_fold_world(monkeypatch, test_rows=4000)
    factors = {"mean": _mean, "pca": partial(PcaFactor, min_periods=10**7)}
    result = study_signal(_STARTS, horizons=("1m",), **{**_STUDY, "factors": factors})
    by_fold = result.by_fold
    pca, mean = by_fold[by_fold["factor"] == "pca"], by_fold[by_fold["factor"] == "mean"]
    assert (pca["error"] == "ajuste del factor fallido").all() and pca["p_cross"].isna().all()
    assert (mean["error"] == "").all() and mean["p_cross"].notna().all()


def test_study_signal_signal_statistics_value_error_is_a_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pocas filas para el desfase minimo: fila con `error` y el estudio sigue."""
    _fake_fold_world(monkeypatch, test_rows=2000)  # 1m exige 1440 de desfase minimo
    result = study_signal(_STARTS, horizons=("1m",), **_STUDY)
    assert len(result.by_fold) == 4
    assert (result.by_fold["error"] != "").all()
    assert result.by_fold["p_cross"].isna().all()
    assert _final(result.decision) == "inconclusive"


def test_study_signal_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dos ejecuciones con la misma semilla dan tablas identicas."""
    _fake_fold_world(monkeypatch, test_rows=4000)
    a = study_signal(_STARTS, horizons=("1m",), **_STUDY).by_fold
    b = study_signal(_STARTS, horizons=("1m",), **_STUDY).by_fold
    pd.testing.assert_frame_equal(a, b)


def test_study_signal_rejects_bad_share() -> None:
    """`min_train_valid_share` fuera de (0, 1] da `ValueError`."""
    with pytest.raises(ValueError, match="min_train_valid_share"):
        study_signal(_STARTS, min_train_valid_share=0.0, universe=pd.DataFrame())


# --- write_signal_tables ---


def test_write_signal_tables_format_order_and_determinism(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """CSV con `%.10g`, columnas fijas y bytes identicos en dos ejecuciones."""
    _fake_fold_world(monkeypatch, test_rows=4000)
    first = write_signal_tables(
        study_signal(_STARTS, horizons=("1m",), **_STUDY), out_dir=tmp_path / "a"
    )
    second = write_signal_tables(
        study_signal(_STARTS, horizons=("1m",), **_STUDY), out_dir=tmp_path / "b"
    )
    assert [p.name for p in first] == ["idio_signal_by_fold.csv", "idio_signal_decision.csv"]
    for a, b in zip(first, second, strict=True):
        assert a.read_bytes() == b.read_bytes() and b"\r" not in a.read_bytes()
    lines = first[0].read_text(encoding="utf-8").splitlines()
    assert lines[0] == ",".join(sig.BY_FOLD_COLUMNS)
    assert first[1].read_text(encoding="utf-8").splitlines()[0] == ",".join(sig.DECISION_COLUMNS)
    assert first[1].read_text(encoding="utf-8").splitlines()[-1].startswith("verdict,1m,,,")


def test_write_signal_tables_number_format(tmp_path: Path) -> None:
    """Los flotantes salen con 10 cifras significativas."""
    by_fold = pd.DataFrame({c: [np.nan] for c in sig.BY_FOLD_COLUMNS})
    by_fold["s_cross"] = 1 / 3
    decision = pd.DataFrame({c: [None] for c in sig.DECISION_COLUMNS})
    paths = write_signal_tables(SignalResult(by_fold, decision), out_dir=tmp_path)
    assert "0.3333333333" in paths[0].read_text(encoding="utf-8")


# --- CLI ---


def _fake_result() -> SignalResult:
    return SignalResult(pd.DataFrame(), pd.DataFrame({"verdict": ["report", "inconclusive"]}))


def test_registry_has_all_subcommands() -> None:
    """El registro de subcomandos incluye `beta-study`, `idio-signal` y `market-lag`."""
    assert set(cli._COMMANDS) == {"beta-study", "idio-signal", "market-lag"}


def test_cli_idio_signal_is_a_thin_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parsea, llama a `study_signal` y `write_signal_tables` y devuelve 0."""
    calls: dict[str, object] = {}

    def fake_study(starts: list[pd.Timestamp], **kwargs: object) -> SignalResult:
        calls["starts"], calls["kwargs"] = starts, kwargs
        return _fake_result()

    monkeypatch.setattr(cli, "study_signal", fake_study)
    monkeypatch.setattr(cli, "write_signal_tables", lambda result: calls.setdefault("w", result))
    code = cli.main(
        ["idio-signal", "--tests", "2024-01-01", "--horizons", "1m", "5m", "--max-lag", "3"]
        + ["--n-shifts", "7", "--rebuild"]
    )
    assert code == 0 and "w" in calls
    assert calls["starts"] == [pd.Timestamp("2024-01-01", tz="UTC")]
    kwargs = calls["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["horizons"] == ["1m", "5m"] and kwargs["max_lag"] == 3
    assert kwargs["n_shifts"] == 7 and kwargs["rebuild"] is True


@pytest.mark.parametrize(
    "argv",
    [
        ["idio-signal", "--tests", "2023-02-01"],
        ["idio-signal", "--tests", "no-es-fecha"],
        ["idio-signal", "--horizons", "7x"],
        ["idio-signal", "--max-lag", "0"],
        ["idio-signal", "--n-shifts", "0"],
        ["idio-signal", "--max-lag", "abc"],
    ],
)
def test_cli_idio_signal_invalid_arguments_exit_2(
    argv: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Argumentos invalidos salen con 2 sin llegar a ejecutar el estudio."""

    def must_not_run(*args: object, **kwargs: object) -> SignalResult:
        raise AssertionError("no debe ejecutarse")

    monkeypatch.setattr(cli, "study_signal", must_not_run)
    assert cli.main(argv) == 2


def test_cli_idio_signal_job_failure_exits_1(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Un `ValueError` del trabajo sale con 1 y sin traceback."""

    def boom(starts: list[pd.Timestamp], **kwargs: object) -> SignalResult:
        raise ValueError("sin cache")

    monkeypatch.setattr(cli, "study_signal", boom)
    assert cli.main(["idio-signal", "--tests", "2024-01-01"]) == 1
    assert "Traceback" not in caplog.text
