"""Factor sin la propia moneda (MIAX-215): estudio descriptivo por folds, control D-39 (c).

Compara los residuos de media y PCA (D-35, D-36) con sus versiones leave-one-out en los folds de
D-37, solo en test. No cambia el pipeline ni elige un factor: mide que arregla y que no.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from miax.eval.beta_study import TABLES_DIR, load_fold, residual_diagnostics
from miax.features.correlation import (
    explained_variance,
    off_diagonal,
    raw_correlation,
    symbols_without_data,
)
from miax.features.factor import FactorModel
from miax.features.factor_loo import LooMeanFactor, LooPcaFactor
from miax.features.factor_mean import MeanFactor
from miax.features.factor_metrics import correlation_drop
from miax.features.factor_pca import PcaFactor
from miax.features.panel_store import PANEL_DIR
from miax.ingest.store import DEFAULT_BASE_DIR
from miax.ingest.universe import load_universe
from miax.utils.io import resolve_repo_path
from miax.utils.logging import get_logger

logger = get_logger(__name__)

_NUMBER_FORMAT = "%.10g"

# Los 12 trimestres de prueba de D-37 (2023T1 a 2025T4).
DEFAULT_TESTS = tuple(
    f"{year}-{month:02d}-01" for year in (2023, 2024, 2025) for month in (1, 4, 7, 10)
)

LOO_MAKERS: dict[str, Callable[[], FactorModel]] = {
    "mean": MeanFactor,
    "pca": PcaFactor,
    "loo_mean": LooMeanFactor,
    "loo_pca": LooPcaFactor,
}
# Cada version LOO se compara con la estandar de la misma familia.
_REFERENCE = {"loo_mean": "mean", "loo_pca": "pca"}

METRIC_COLUMNS = [
    "n_symbols",
    "floor",
    "pair_mean",
    "pair_abs_mean",
    "correlation_drop",
    "min_eigenvalue",
    "min_eigenvalue_pairwise",
    "sum_ratio",
    "n_complete_rows",
    "market_exposure",
    "beta_mean",
    "matrix_change",
]
FOLD_COLUMNS = ["fold", "model", *METRIC_COLUMNS, "error"]
SUMMARY_COLUMNS = ["model", "n_folds", *METRIC_COLUMNS]


class LooResult(NamedTuple):
    """Tabla por fold y modelo, y su resumen (mediana entre folds por modelo)."""

    folds: pd.DataFrame
    summary: pd.DataFrame


def _usable_correlation(residuals: pd.DataFrame, min_obs: int) -> pd.DataFrame:
    """Correlacion por pares de los residuos sin las monedas sin datos suficientes."""
    corr = raw_correlation(residuals, min_periods=min_obs)
    without = set(symbols_without_data(corr))
    usable = [s for s in corr.columns if s not in without]
    return corr.loc[usable, usable]


def _model_metrics(
    model: FactorModel,
    returns_test: pd.DataFrame,
    *,
    min_obs: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Metricas de un modelo ya ajustado, medidas en test, y su matriz de correlacion residual."""
    # El modelo se ajusto solo con train: aqui solo se proyecta test, sin reajustar (D-03).
    residuals = model.transform(returns_test)
    market = model.factors(returns_test).iloc[:, 0].rename("factor")
    corr = _usable_correlation(residuals, min_obs)
    n = len(corr)
    if n < 2:
        raise ValueError(f"menos de 2 monedas con residuo utilizable ({n})")
    pairs = off_diagonal(corr)
    out: dict[str, float] = dict.fromkeys(METRIC_COLUMNS, np.nan)
    out["n_symbols"] = float(n)
    out["floor"] = -1.0 / (n - 1)
    out["pair_mean"] = float(np.nanmean(pairs))
    out["pair_abs_mean"] = float(np.nanmean(np.abs(pairs)))
    out["correlation_drop"] = float(
        correlation_drop(returns_test, residuals, min_periods=min_obs)["correlation_drop"]
    )
    try:
        # Version por pares: no es semidefinida positiva con huecos distintos por moneda.
        out["min_eigenvalue_pairwise"] = float(explained_variance(corr, require_psd=False)[-1] * n)
    except ValueError as exc:
        logger.warning("Sin autovalor minimo por pares de los residuos (%s)", exc)
    complete = residuals[list(corr.columns)].dropna()
    out["n_complete_rows"] = float(len(complete))
    if len(complete) >= min_obs:
        # Desviacion de la suma de residuos frente a la de residuos independientes: ~0 si suman 0.
        total_std = float(complete.sum(axis=1).std())
        independent_std = float(np.sqrt(complete.var().sum()))
        if independent_std > 0:
            out["sum_ratio"] = total_std / independent_std
        # Principal con filas completas: es PSD; por pares, los huecos la vuelven negativa.
        if bool((complete.std() > 0).all()):
            eigenvalues = np.linalg.eigvalsh(complete.corr().to_numpy())
            out["min_eigenvalue"] = float(eigenvalues[0])
    else:
        logger.warning(
            "Menos de %d filas completas (%d): sin sum_ratio ni min_eigenvalue",
            min_obs,
            len(complete),
        )
    diagnostics = residual_diagnostics(returns_test, residuals, market, min_obs=min_obs)
    out["market_exposure"] = float(diagnostics["corr_lag0"].median())
    betas = getattr(model, "betas", None)
    if betas is not None:
        out["beta_mean"] = float(betas.mean())
    return out, corr


def _matrix_change(loo: pd.DataFrame, reference: pd.DataFrame) -> float:
    """Media de |rho_LOO - rho_estandar| por pares, en las monedas comunes a las dos matrices."""
    common = [s for s in loo.columns if s in set(reference.columns)]
    if len(common) < 2:
        return float("nan")
    diff = np.abs(
        off_diagonal(loo.loc[common, common]) - off_diagonal(reference.loc[common, common])
    )
    return float(np.nanmean(diff)) if np.isfinite(diff).any() else float("nan")


def _summarize(folds: pd.DataFrame, models: Sequence[str]) -> pd.DataFrame:
    """Mediana entre folds validos (sin error) de cada metrica, por modelo, y su numero de folds."""
    rows: list[dict[str, object]] = []
    for name in models:
        valid = folds[(folds["model"] == name) & (folds["error"] == "")]
        row: dict[str, object] = {"model": name, "n_folds": len(valid)}
        for column in METRIC_COLUMNS:
            row[column] = float(valid[column].median()) if len(valid) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def compare_loo(
    tests: Sequence[str] = DEFAULT_TESTS,
    *,
    universe: pd.DataFrame | None = None,
    makers: Mapping[str, Callable[[], FactorModel]] | None = None,
    train_months: int = 12,
    test_months: int = 3,
    min_train_valid_share: float = 0.5,
    min_obs: int = 1000,
    rebuild: bool = False,
    base_dir: str | Path = DEFAULT_BASE_DIR,
    cache_dir: str | Path = PANEL_DIR,
) -> LooResult:
    """Ajusta media, PCA y sus versiones LOO en train de cada fold y las mide en test.

    Con menos de min_obs filas completas, sum_ratio y min_eigenvalue son NaN (ver n_complete_rows).
    """
    if min_obs < 2:
        raise ValueError(f"min_obs debe ser >= 2: {min_obs}")
    chosen = dict(LOO_MAKERS if makers is None else makers)
    starts = sorted(dict.fromkeys(pd.Timestamp(text, tz="UTC") for text in tests))
    frame = load_universe() if universe is None else universe
    rows: list[dict[str, object]] = []
    for test_start in starts:
        fold = load_fold(
            test_start,
            universe=frame,
            train_months=train_months,
            test_months=test_months,
            min_train_valid_share=min_train_valid_share,
            rebuild=rebuild,
            base_dir=base_dir,
            cache_dir=cache_dir,
        )
        matrices: dict[str, pd.DataFrame] = {}
        fold_rows: list[dict[str, object]] = []
        for name, make_model in chosen.items():
            row: dict[str, object] = dict.fromkeys(FOLD_COLUMNS, np.nan)
            row.update(fold=fold.label, model=name, error="")
            try:
                # Se ajusta solo con train, que acaba antes de test_start (D-37, D-03).
                model = make_model().fit(fold.returns_train)
                metrics, matrices[name] = _model_metrics(model, fold.returns_test, min_obs=min_obs)
                row.update(metrics)
            except ValueError as exc:
                # Deliberado: un RuntimeError (fuga al proyectar o NotFittedError)
                # debe tumbar el estudio, asi que no se captura.
                row["error"] = str(exc)
                logger.warning("Fold %s %s sin metricas: %s", fold.label, name, exc)
            fold_rows.append(row)
        for row in fold_rows:
            reference = _REFERENCE.get(str(row["model"]))
            if reference in matrices and row["model"] in matrices:
                row["matrix_change"] = _matrix_change(
                    matrices[str(row["model"])], matrices[reference]
                )
            logger.info(
                "Fold %s %s: pair_mean=%s min_eigenvalue=%s",
                row["fold"],
                row["model"],
                row["pair_mean"],
                row["min_eigenvalue"],
            )
        rows.extend(fold_rows)
    folds = pd.DataFrame(rows, columns=FOLD_COLUMNS)
    return LooResult(folds, _summarize(folds, list(chosen)))


def write_loo_tables(result: LooResult, *, out_dir: str | Path = TABLES_DIR) -> list[Path]:
    """Escribe las dos tablas en CSV con orden y formato numerico fijos; devuelve las rutas."""
    directory = resolve_repo_path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, table in (("folds", result.folds), ("summary", result.summary)):
        path = directory / f"loo_factor_{name}.csv"
        table.to_csv(path, index=False, float_format=_NUMBER_FORMAT, lineterminator="\n")
        paths.append(path)
    logger.info("Tablas de factor leave-one-out: %s", ", ".join(str(p) for p in paths))
    return paths
