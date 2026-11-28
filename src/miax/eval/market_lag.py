"""Residuo sin el retraso respecto al mercado (MIAX-216): con y sin quitar f(t-1..t-L), en test."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from miax.eval.beta_study import TABLES_DIR, FoldData, _corr, _first_column, load_fold
from miax.eval.idiosyncratic_signal import decision_table, study_signal
from miax.features.factor import FactorModel
from miax.features.factor_btc import BtcFactor
from miax.features.factor_lagged import DEFAULT_BAR, LaggedFactor, lag_matrix
from miax.features.factor_mean import MeanFactor
from miax.features.factor_pca import PcaFactor
from miax.features.panel_store import PANEL_DIR
from miax.ingest.store import DEFAULT_BASE_DIR
from miax.ingest.universe import load_universe
from miax.utils.io import resolve_repo_path
from miax.utils.logging import get_logger

logger = get_logger(__name__)

_NUMBER_FORMAT = "%.10g"
DEFAULT_LAGS = 5
LAG_SUFFIX = "_lag"
# Factores del contraste de senal; btc solo de referencia. El orden fija la semilla de cada
# factor en study_signal: btc, mean y pca conservan los indices de MIAX-057 (p-valores comparables).
SIGNAL_FACTORS: tuple[str, ...] = ("btc", "mean", "pca", "mean_lag", "pca_lag")
LAG_DECIDERS: tuple[str, ...] = ("mean_lag", "pca_lag")
_SIGNAL_RESERVED = frozenset(
    ("factors", "horizons", "universe", "min_obs", "n_folds_required", "rebuild")
    + ("base_dir", "cache_dir")
)


def build_makers(lags: int = DEFAULT_LAGS) -> dict[str, Callable[[], FactorModel]]:
    """Los tres modelos base y sus versiones con retardos (`LaggedFactor`) de `lags` retardos."""

    def lagged(base: Callable[[], FactorModel]) -> Callable[[], FactorModel]:
        return lambda: LaggedFactor(base(), lags=lags)

    return {
        "btc": BtcFactor,
        "mean": MeanFactor,
        "pca": PcaFactor,
        "btc_lag": lagged(BtcFactor),
        "mean_lag": lagged(MeanFactor),
        "pca_lag": lagged(PcaFactor),
    }


LAG_MAKERS: dict[str, Callable[[], FactorModel]] = build_makers(DEFAULT_LAGS)


def diagnostic_columns(lags: int = DEFAULT_LAGS) -> list[str]:
    """Columnas fijas de la tabla de diagnostico para `lags` retardos."""
    steps = range(1, lags + 1)
    return [
        "fold",
        "model",
        "n_coins",
        "var_removed",
        "var_removed_extra",
        *[f"corr_lag{k}" for k in steps],
        *[f"abs_corr_lag{k}" for k in steps],
        *[f"gamma_lag{k}" for k in steps],
        "error",
    ]


@dataclass(frozen=True)
class MarketLagResult:
    """Diagnostico por fold y modelo, y (si se ejecuta) la senal por fold y su decision."""

    diagnostics: pd.DataFrame
    signal_by_fold: pd.DataFrame | None
    signal_decision: pd.DataFrame | None


def lag_diagnostics(
    returns: pd.DataFrame,
    residuals: pd.DataFrame,
    factor: pd.Series,
    *,
    lags: int = DEFAULT_LAGS,
    bar: pd.Timedelta = DEFAULT_BAR,
    min_obs: int = 1000,
) -> pd.DataFrame:
    """Por moneda: `corr(e(t), f(t-l*bar))` con signo para l=1..L y varianza quitada del factor."""
    if not returns.index.equals(residuals.index) or not returns.index.equals(factor.index):
        raise ValueError("returns, residuals y factor deben compartir indice")
    if lags < 1 or min_obs < 2:
        raise ValueError(f"lags debe ser >= 1 y min_obs >= 2: {lags}, {min_obs}")
    # f(t-l) es factor pasado tomado por tiempo: diagnostico, no entra en ningun ajuste.
    lagged = lag_matrix(factor, returns.index, lags=lags, bar=bar)
    names = [f"corr_lag{k}" for k in range(1, lags + 1)]
    rows: dict[str, dict[str, float]] = {}
    for symbol in returns.columns:
        r = returns[symbol].to_numpy(dtype="float64")
        e = residuals[symbol].to_numpy(dtype="float64")
        row = dict.fromkeys(["var_removed", *names], np.nan)
        common = ~np.isnan(r) & ~np.isnan(e)
        if common.sum() >= min_obs:
            denominator = float((r[common] ** 2).sum())
            if denominator > 0:
                row["var_removed"] = 1.0 - float((e[common] ** 2).sum()) / denominator
        for k, name in enumerate(names):
            ok = ~np.isnan(e) & ~np.isnan(lagged[:, k])
            if ok.sum() >= min_obs:
                row[name] = _corr(e[ok], lagged[ok, k])
        rows[str(symbol)] = row
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "symbol"
    return table[["var_removed", *names]]


def _median(values: pd.Series) -> float:
    """Mediana de los valores finitos; NaN si no hay ninguno."""
    return float(values.median()) if values.notna().any() else float("nan")


def _model_row(
    fold: str,
    name: str,
    make_model: Callable[[], FactorModel],
    fold_data: FoldData,
    *,
    lags: int,
    min_obs: int,
) -> dict[str, object]:
    """Fila de diagnostico de un modelo en un fold: ajuste en train y metricas en test."""
    row: dict[str, object] = dict.fromkeys(diagnostic_columns(lags), np.nan)
    row.update(fold=fold, model=name, error="")
    try:
        # El modelo se ajusta solo con train y se proyecta en test sin reajustar (D-03).
        model = make_model().fit(fold_data.returns_train)
        residuals = model.transform(fold_data.returns_test)
        factor = _first_column(model.factors(fold_data.returns_test))
    except ValueError as exc:
        row["error"] = str(exc)
        logger.warning("Fold %s modelo %s sin ajuste: %s", fold, name, exc)
        return row
    table = lag_diagnostics(fold_data.returns_test, residuals, factor, lags=lags, min_obs=min_obs)
    row["n_coins"] = int(table["var_removed"].notna().sum())
    row["var_removed"] = _median(table["var_removed"])
    for k in range(1, lags + 1):
        row[f"corr_lag{k}"] = _median(table[f"corr_lag{k}"])
        row[f"abs_corr_lag{k}"] = _median(table[f"corr_lag{k}"].abs())
    if isinstance(model, LaggedFactor):
        gammas = model.gammas
        for k in range(1, lags + 1):
            row[f"gamma_lag{k}"] = _median(gammas[f"lag_{k}"])
    return row


def _lag_decision(by_fold: pd.DataFrame, n_folds_required: int) -> pd.DataFrame:
    """Decision de 057 con las versiones sin retraso como deciders, mas el veredicto de las base."""
    base = decision_table(by_fold, n_folds_required=n_folds_required)
    lagged = decision_table(by_fold, deciding=LAG_DECIDERS, n_folds_required=n_folds_required)
    base_verdict = base.iloc[-1:].copy()
    base_verdict["factor"] = "verdict_base"
    lag_verdict = lagged.iloc[-1:].copy()
    lag_verdict["factor"] = "verdict_lag"
    return pd.concat([lagged.iloc[:-1], base_verdict, lag_verdict], ignore_index=True)


def compare_market_lag(
    tests: Sequence[str],
    *,
    universe: pd.DataFrame | None = None,
    makers: Mapping[str, Callable[[], FactorModel]] | None = None,
    lags: int = DEFAULT_LAGS,
    run_signal: bool = True,
    signal_kwargs: Mapping[str, object] | None = None,
    min_obs: int = 1000,
    n_folds_required: int = 12,
    rebuild: bool = False,
    base_dir: str | Path = DEFAULT_BASE_DIR,
    cache_dir: str | Path = PANEL_DIR,
) -> MarketLagResult:
    """Compara el residuo con y sin retardos del mercado en cada fold y, si se pide, la senal."""
    if lags < 1:
        raise ValueError(f"lags debe ser >= 1, es {lags}")
    if not tests:
        raise ValueError("hace falta al menos un trimestre de prueba")
    starts = sorted({pd.Timestamp(text, tz="UTC") for text in tests})
    models = dict(build_makers(lags) if makers is None else makers)
    frame = load_universe() if universe is None else universe
    rows: list[dict[str, object]] = []
    for start in starts:
        fold = load_fold(
            start, universe=frame, rebuild=rebuild, base_dir=base_dir, cache_dir=cache_dir
        )
        fold_rows = [
            _model_row(fold.label, name, make, fold, lags=lags, min_obs=min_obs)
            for name, make in models.items()
        ]
        by_model = {str(r["model"]): r for r in fold_rows}
        for name, row in by_model.items():
            base_row = by_model.get(name.removesuffix(LAG_SUFFIX))
            if name.endswith(LAG_SUFFIX) and base_row is not None:
                row["var_removed_extra"] = float(row["var_removed"]) - float(  # type: ignore[arg-type]
                    base_row["var_removed"]  # type: ignore[arg-type]
                )
            logger.info("Fold %s %s: abs_corr_lag1=%s", fold.label, name, row.get("abs_corr_lag1"))
        rows.extend(fold_rows)
    diagnostics = pd.DataFrame(rows, columns=diagnostic_columns(lags))
    if not run_signal:
        return MarketLagResult(diagnostics, None, None)
    extra = dict(signal_kwargs or {})
    clash = sorted(set(extra) & _SIGNAL_RESERVED)
    if clash:
        raise ValueError(f"signal_kwargs no puede incluir claves ya fijadas: {clash}")
    # Se respeta el orden de SIGNAL_FACTORS, no el de `models`: de el depende la semilla.
    signal_models = {k: models[k] for k in SIGNAL_FACTORS if k in models}
    if not signal_models:
        raise ValueError(f"ningun modelo de {SIGNAL_FACTORS} para el contraste de senal")
    signal = study_signal(
        starts,
        universe=frame,
        factors=signal_models,
        horizons=("1m",),
        min_obs=min_obs,
        n_folds_required=n_folds_required,
        rebuild=rebuild,
        base_dir=base_dir,
        cache_dir=cache_dir,
        **extra,  # type: ignore[arg-type]
    )
    return MarketLagResult(
        diagnostics, signal.by_fold, _lag_decision(signal.by_fold, n_folds_required)
    )


def write_market_lag_tables(
    result: MarketLagResult, *, out_dir: str | Path = TABLES_DIR
) -> list[Path]:
    """Escribe las tablas del estudio en CSV con orden y formato fijos; devuelve las rutas."""
    directory = resolve_repo_path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    tables = [("diagnostics", result.diagnostics)]
    if result.signal_by_fold is not None and result.signal_decision is not None:
        tables += [
            ("signal_by_fold", result.signal_by_fold),
            ("signal_decision", result.signal_decision),
        ]
    paths: list[Path] = []
    for name, table in tables:
        path = directory / f"market_lag_{name}.csv"
        table.to_csv(path, index=False, float_format=_NUMBER_FORMAT, lineterminator="\n")
        paths.append(path)
    logger.info("Tablas del retraso respecto al mercado: %s", ", ".join(str(p) for p in paths))
    return paths
