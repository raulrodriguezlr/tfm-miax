"""Senal idiosincratica (MIAX-057): el residuo pasado de unas monedas predice el de otras.

Contraste global por fold, factor y horizonte, solo en test (fuera de muestra). No construye
aristas ni p-valores por par; solo dice si queda algo que modelar (punto de decision del proyecto).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from miax.eval.beta_study import (
    BASE_HORIZON,
    FACTOR_MAKERS,
    TABLES_DIR,
    load_fold,
    project_fixed,
)
from miax.features.correlation import explained_variance, raw_correlation, symbols_without_data
from miax.features.factor import FactorModel
from miax.features.panel_store import PANEL_DIR
from miax.features.resample import bar_factor, horizon_returns_from_cache
from miax.ingest.store import DEFAULT_BASE_DIR
from miax.ingest.universe import load_universe, to_ms
from miax.utils.io import resolve_repo_path
from miax.utils.logging import get_logger

logger = get_logger(__name__)

_NUMBER_FORMAT = "%.10g"
_MINUTES_PER_DAY = 1440
_VARIANCE_EPS = 1e-12
_STATS = ("cross", "own", "asym")

BY_FOLD_COLUMNS = [
    "fold",
    "factor",
    "horizon",
    "n_coins",
    "n_rows",
    "s_cross",
    "s_own",
    "s_asym",
    "null_median_cross",
    "null_median_own",
    "null_median_asym",
    "null_q95_cross",
    "null_q95_own",
    "null_q95_asym",
    "p_cross",
    "p_own",
    "p_asym",
    "ratio_cross",
    "ratio_own",
    "ratio_asym",
    "first_eigenvalue_share",
    "n_above_mp",
    "mp_edge",
    "error",
]
DECISION_COLUMNS = ["factor", "horizon", "n_folds", "n_significant", "verdict"]


@dataclass(frozen=True)
class SignalResult:
    """Tabla por fold (un estadistico por fila) y tabla de decision."""

    by_fold: pd.DataFrame
    decision: pd.DataFrame


def _check_regular(index: pd.Index) -> None:
    """`ValueError` si el indice no es temporal, creciente y de paso constante."""
    steps = np.diff(index.asi8) if isinstance(index, pd.DatetimeIndex) else None
    if steps is None or (len(steps) and (steps[0] <= 0 or (steps != steps[0]).any())):
        raise ValueError("el indice de los residuos debe ser temporal, creciente y regular")


def _standardize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Columnas centradas y de desviacion 1 con 0 donde falta, y su mascara de datos validos.

    Solo mejora el condicionamiento: la correlacion por pares no cambia con esta escala.
    """
    finite = np.isfinite(values)
    count = finite.sum(axis=0)
    filled = np.where(finite, values, 0.0)
    mean = np.divide(filled.sum(axis=0), count, out=np.zeros(values.shape[1]), where=count > 0)
    centred = np.where(finite, values - mean, 0.0)
    var = np.divide(
        (centred**2).sum(axis=0), count - 1, out=np.zeros(values.shape[1]), where=count > 1
    )
    ok = var > 0
    std = np.sqrt(np.where(ok, var, 1.0))
    mask = finite & ok
    return np.where(mask, centred / std, 0.0), mask.astype("float64")


class _LagBlocks:
    """Bloques pasado `e(t-lag)` y presente `e(t)` de un retardo, listos para los productos."""

    def __init__(self, z: np.ndarray, mask: np.ndarray, lag: int) -> None:
        # El bloque pasado son las filas 0..T-lag-1 y el presente las lag..T-1: cada `t` solo se
        # empareja con `t-lag` (anterior), nunca con una fila posterior.
        self.past = z[: len(z) - lag]
        self.past_mask = mask[: len(z) - lag]
        now, now_mask = z[lag:], mask[lag:]
        self.now_cat = np.concatenate([now, now * now, now_mask], axis=1)
        self.now_mask = now_mask
        self.now = now
        self.length = len(self.past)


def _correlation_matrix(
    blocks: _LagBlocks, shift: int, min_obs: int
) -> tuple[np.ndarray, np.ndarray]:
    """`C[i, j] = corr(e_i(t-lag), e_j(t))` por pares completos; `NaN` sin datos o sin varianza.

    Con `shift > 0` el bloque pasado de todas las monedas se desplaza a la vez de forma circular.
    """
    past, past_mask = blocks.past, blocks.past_mask
    if shift:
        # Desplazamiento circular del pasado: rompe su relacion temporal con el presente y
        # conserva la dinamica de cada serie y la correlacion contemporanea entre monedas.
        past = np.roll(past, shift, axis=0)
        past_mask = np.roll(past_mask, shift, axis=0)
    n_cols = past.shape[1]
    from_mask = past_mask.T @ blocks.now_cat
    sy, syy, n = (from_mask[:, k * n_cols : (k + 1) * n_cols] for k in range(3))
    sx = past.T @ blocks.now_mask
    sxx = (past * past).T @ blocks.now_mask
    sxy = past.T @ blocks.now
    cov = n * sxy - sx * sy
    var_x = n * sxx - sx * sx
    var_y = n * syy - sy * sy
    valid = (n >= min_obs) & (var_x > _VARIANCE_EPS * n * n) & (var_y > _VARIANCE_EPS * n * n)
    denominator = np.sqrt(np.where(valid, var_x * var_y, 1.0))
    corr = np.where(valid, np.clip(cov / denominator, -1.0, 1.0), np.nan)
    return corr, valid


def lagged_cross_correlation(
    residuals: pd.DataFrame, lag: int, *, min_obs: int = 1000
) -> np.ndarray:
    """Matriz N x N con `corr(e_i(t-lag), e_j(t))` por pares completos; `NaN` si no hay datos."""
    if lag < 1:
        raise ValueError(f"lag debe ser >= 1: {lag}")
    if min_obs < 2:
        raise ValueError(f"min_obs debe ser >= 2: {min_obs}")
    _check_regular(residuals.index)
    if len(residuals) <= lag:
        raise ValueError(f"hay {len(residuals)} filas y el retardo es {lag}")
    z, mask = _standardize(residuals.to_numpy(dtype="float64"))
    return _correlation_matrix(_LagBlocks(z, mask, lag), 0, min_obs)[0]


def _squared_sums(corr: np.ndarray) -> np.ndarray:
    """Sumas y cuentas de los tres estadisticos de una matriz: `[sum, count]` por estadistico."""
    squared = corr * corr
    cross_mask = ~np.eye(len(corr), dtype=bool)
    diff = (corr - corr.T) ** 2
    upper = np.triu(np.ones_like(cross_mask), k=1)
    out = np.zeros((3, 2))
    for k, (values, selector) in enumerate(
        ((squared, cross_mask), (squared, ~cross_mask), (diff, upper))
    ):
        picked = values[selector]
        picked = picked[np.isfinite(picked)]
        out[k] = picked.sum(), len(picked)
    return out


def _p_value(observed: float, null: np.ndarray) -> float:
    """`(1 + #{nulo >= observado}) / (n + 1)` con los nulos finitos; `NaN` si el observado lo es."""
    null = null[np.isfinite(null)]
    if not np.isfinite(observed):
        return float("nan")
    return float((1 + (null >= observed).sum()) / (len(null) + 1))


def signal_statistics(
    residuals: pd.DataFrame,
    *,
    min_shift_rows: int,
    max_lag: int = 5,
    n_shifts: int = 99,
    seed: int | Sequence[int] = 57,
    min_obs: int = 1000,
) -> dict[str, float]:
    """Estadisticos cruzado, propio y de asimetria con su nulo de desplazamiento circular.

    El desfase es entero y uniforme entre `min_shift_rows` y lo que deja el bloque pasado menos
    `min_shift_rows`; los tres estadisticos de una fila usan los mismos desfases.
    """
    if max_lag < 1 or n_shifts < 1 or min_shift_rows < 1 or min_obs < 2:
        raise ValueError("max_lag, n_shifts y min_shift_rows deben ser >= 1 y min_obs >= 2")
    _check_regular(residuals.index)
    n_rows = len(residuals)
    upper = n_rows - max_lag - min_shift_rows
    if min_shift_rows <= max_lag or upper < min_shift_rows:
        raise ValueError(
            f"{n_rows} filas no dan para un desplazamiento minimo de {min_shift_rows} filas "
            f"con {max_lag} retardos"
        )
    # Los desfases no dependen de los datos y se fijan antes de calcular nada (semilla fija).
    shifts = np.concatenate(
        [[0], np.random.default_rng(seed).integers(min_shift_rows, upper + 1, size=n_shifts)]
    )
    z, mask = _standardize(residuals.to_numpy(dtype="float64"))
    totals = np.zeros((len(shifts), 3, 2))
    seen = np.zeros(z.shape[1], dtype=bool)
    for lag in range(1, max_lag + 1):
        blocks = _LagBlocks(z, mask, lag)
        for k, shift in enumerate(shifts):
            # El desfase se aplica al bloque pasado `e(t-lag)`; el presente `e(t)` no se mueve.
            corr, _ = _correlation_matrix(blocks, int(shift) % blocks.length, min_obs)
            totals[k] += _squared_sums(corr)
            if k == 0:
                finite = np.isfinite(corr)
                seen |= finite.any(axis=0) | finite.any(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        stats = np.where(totals[..., 1] > 0, totals[..., 0] / totals[..., 1], np.nan)
    out: dict[str, float] = {"n_coins": float(seen.sum()), "n_rows": float(n_rows)}
    for k, name in enumerate(_STATS):
        observed, null = float(stats[0, k]), stats[1:, k]
        finite_null = null[np.isfinite(null)]
        median = float(np.median(finite_null)) if len(finite_null) else float("nan")
        out[f"s_{name}"] = observed
        out[f"null_median_{name}"] = median
        out[f"null_q95_{name}"] = (
            float(np.quantile(finite_null, 0.95)) if len(finite_null) else float("nan")
        )
        out[f"p_{name}"] = _p_value(observed, null)
        out[f"ratio_{name}"] = observed / median if median > 0 else float("nan")
    return out


def contemporaneous_structure(residuals: pd.DataFrame, *, min_obs: int = 1000) -> dict[str, float]:
    """Primer autovalor (fraccion de N) y autovalores sobre el borde de Marchenko-Pastur.

    Solo se reporta: dice cuanta estructura comun contemporanea deja el factor en los residuos.
    """
    out = {"first_eigenvalue_share": np.nan, "n_above_mp": np.nan, "mp_edge": np.nan}
    corr = raw_correlation(residuals, min_periods=min_obs)
    usable = [s for s in corr.columns if s not in set(symbols_without_data(corr))]
    if len(usable) < 2:
        return out
    corr = corr.loc[usable, usable]
    try:
        shares = explained_variance(corr, require_psd=False)
    except ValueError as exc:
        logger.warning("Sin autovalores de los residuos (%s)", exc)
        return out
    n = len(usable)
    edge = (1.0 + np.sqrt(n / len(residuals))) ** 2
    out["first_eigenvalue_share"] = float(shares[0])
    # Los autovalores en unidades de correlacion (sin dividir por N) se comparan con el borde.
    out["n_above_mp"] = float((shares * n > edge).sum())
    out["mp_edge"] = float(edge)
    return out


def study_signal(
    test_starts: Sequence[pd.Timestamp],
    *,
    universe: pd.DataFrame | None = None,
    factors: Mapping[str, Callable[[], FactorModel]] | None = None,
    horizons: Sequence[str] = ("1m", "5m", "15m"),
    max_lag: int = 5,
    n_shifts: int = 99,
    seed: int = 57,
    train_months: int = 12,
    test_months: int = 3,
    min_train_valid_share: float = 0.5,
    min_obs: int = 1000,
    n_folds_required: int = 12,
    rebuild: bool = False,
    base_dir: str | Path = DEFAULT_BASE_DIR,
    cache_dir: str | Path = PANEL_DIR,
) -> SignalResult:
    """Contraste de senal idiosincratica por fold, factor y horizonte, y su tabla de decision."""
    if not 0 < min_train_valid_share <= 1:
        raise ValueError(f"min_train_valid_share debe estar en (0, 1]: {min_train_valid_share}")
    horizon_list = list(dict.fromkeys(horizons))
    day_rows = {h: _MINUTES_PER_DAY // bar_factor(BASE_HORIZON, h) for h in horizon_list}
    makers = dict(FACTOR_MAKERS if factors is None else factors)
    frame = load_universe() if universe is None else universe
    rows: list[dict[str, object]] = []
    for i_fold, test_start in enumerate(sorted(dict.fromkeys(test_starts))):
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
        for i_horizon, horizon in enumerate(horizon_list):
            if horizon == BASE_HORIZON:
                train_h, test_h = fold.returns_train, fold.returns_test
            else:
                # Los retornos de train cierran antes de test_start (barra completa, D-19).
                train_h = horizon_returns_from_cache(
                    fold.kept,
                    horizon,
                    to_ms(fold.train_start),
                    to_ms(fold.test_start),
                    base_dir=base_dir,
                )
                test_h = horizon_returns_from_cache(
                    fold.kept,
                    horizon,
                    to_ms(fold.test_start),
                    to_ms(fold.test_end),
                    base_dir=base_dir,
                )
            for i_factor, (name, make_model) in enumerate(makers.items()):
                row: dict[str, object] = dict.fromkeys(BY_FOLD_COLUMNS, np.nan)
                row.update(fold=fold.label, factor=name, horizon=horizon, error="")
                fit = project_fixed(make_model, train_h, test_h)
                if fit.n_failed:
                    row["error"] = "ajuste del factor fallido"
                else:
                    try:
                        stats = signal_statistics(
                            fit.residuals,
                            min_shift_rows=day_rows[horizon],
                            max_lag=max_lag,
                            n_shifts=n_shifts,
                            seed=[seed, i_fold, i_factor, i_horizon],
                            min_obs=min_obs,
                        )
                        structure = contemporaneous_structure(fit.residuals, min_obs=min_obs)
                    except ValueError as exc:
                        row["error"] = str(exc)
                        logger.warning(
                            "Fold %s %s %s sin estadisticos: %s", fold.label, name, horizon, exc
                        )
                    else:
                        row.update(stats)
                        row.update(structure)
                logger.info("Fold %s %s %s: p_cross=%s", fold.label, name, horizon, row["p_cross"])
                rows.append(row)
    by_fold = pd.DataFrame(rows, columns=BY_FOLD_COLUMNS)
    return SignalResult(by_fold, decision_table(by_fold, n_folds_required=n_folds_required))


def decision_table(
    by_fold: pd.DataFrame,
    *,
    alpha: float = 0.05,
    min_wins: int = 9,
    max_wins_pivot: int = 3,
    deciding: Sequence[str] = ("mean", "pca"),
    primary: str = "1m",
    n_folds_required: int = 12,
) -> pd.DataFrame:
    """Folds con `p_cross < alpha` por factor y horizonte, y un veredicto signal/pivot/inconclusive.

    La regla se fija antes de ver los datos: signal si todos los factores que deciden llegan a
    `min_wins` folds, pivot si alguno se queda en `max_wins_pivot` o menos, y si no, inconclusive.
    Un factor con menos de `n_folds_required` folds validos no puede dar signal ni pivot.
    """
    if not 0 < alpha < 1 or max_wins_pivot >= min_wins:
        raise ValueError("alpha debe estar en (0, 1) y max_wins_pivot debe ser < min_wins")
    if n_folds_required < 1 or n_folds_required < min_wins:
        raise ValueError("n_folds_required debe ser >= 1 y >= min_wins")
    rows: list[dict[str, object]] = []
    verdicts: dict[str, str] = {}
    keys = by_fold[["factor", "horizon"]].drop_duplicates().itertuples(index=False)
    for factor, horizon in keys:
        sel = by_fold[(by_fold["factor"] == factor) & (by_fold["horizon"] == horizon)]
        n_significant = int((sel["p_cross"] < alpha).sum())
        n_valid = int(sel["p_cross"].notna().sum())
        verdict = "report"
        if horizon == primary and factor in deciding:
            if n_valid < n_folds_required:
                logger.warning(
                    "Factor %s: %d folds validos, se exigen %d; veredicto inconclusive",
                    factor,
                    n_valid,
                    n_folds_required,
                )
                verdict = "inconclusive"
            elif n_significant >= min_wins:
                verdict = "signal"
            elif n_significant <= max_wins_pivot:
                verdict = "pivot"
            else:
                verdict = "inconclusive"
            verdicts[str(factor)] = verdict
        rows.append(
            {
                "factor": factor,
                "horizon": horizon,
                "n_folds": n_valid,
                "n_significant": n_significant,
                "verdict": verdict,
            }
        )
    states = [verdicts.get(f, "inconclusive") for f in deciding]
    if "pivot" in states:
        final = "pivot"
    elif all(s == "signal" for s in states):
        final = "signal"
    else:
        final = "inconclusive"
    rows.append(
        {
            "factor": "verdict",
            "horizon": primary,
            "n_folds": None,
            "n_significant": None,
            "verdict": final,
        }
    )
    table = pd.DataFrame(rows, columns=DECISION_COLUMNS)
    # Enteros anulables: la fila del veredicto global no tiene recuento propio (queda vacia).
    table[["n_folds", "n_significant"]] = table[["n_folds", "n_significant"]].astype("Int64")
    return table


def write_signal_tables(result: SignalResult, *, out_dir: str | Path = TABLES_DIR) -> list[Path]:
    """Escribe las dos tablas en CSV con orden y formato numerico fijos; devuelve las rutas."""
    directory = resolve_repo_path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, table in (("by_fold", result.by_fold), ("decision", result.decision)):
        path = directory / f"idio_signal_{name}.csv"
        table.to_csv(path, index=False, float_format=_NUMBER_FORMAT, lineterminator="\n")
        paths.append(path)
    logger.info("Tablas de senal idiosincratica: %s", ", ".join(str(p) for p in paths))
    return paths
