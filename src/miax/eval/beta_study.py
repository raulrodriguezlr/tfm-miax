"""Estudio de betas (MIAX-053): beta movil o por fold, y beta a 1 min o a 5 y 15 min.

Es descriptivo: no cambia ningun modelo de factor ni el pipeline. Todas las metricas son de test
(fuera de muestra) y el residuo que no se puede calcular queda en `NaN`, nunca en 0 (D-02).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from miax.features.factor import FactorModel
from miax.features.factor_btc import BtcFactor
from miax.features.factor_mean import MeanFactor
from miax.features.factor_pca import PcaFactor
from miax.features.panel_store import PANEL_DIR, cached_panel
from miax.features.resample import bar_factor, horizon_returns_from_cache
from miax.ingest.store import DEFAULT_BASE_DIR
from miax.ingest.universe import load_universe, to_ms
from miax.utils.io import resolve_repo_path
from miax.utils.logging import get_logger

logger = get_logger(__name__)

TABLES_DIR = "reports/tables"
BASE_HORIZON = "1m"
FOLD_SCHEME = "fold"
# Factores con beta OLS (la carga del PCA va normalizada a media 1 y su cociente no vale).
OLS_FACTORS: tuple[str, ...] = ("btc", "mean")
FACTOR_MAKERS: dict[str, Callable[[], FactorModel]] = {
    "btc": BtcFactor,
    "mean": MeanFactor,
    "pca": PcaFactor,
}
_NUMBER_FORMAT = "%.10g"
_QUARTER_MONTHS = (1, 4, 7, 10)

DIAGNOSTIC_COLUMNS = [
    "fold",
    "scheme",
    "factor",
    "n_universe",
    "n_coins",
    "var_removed",
    "corr_lag0",
    "corr_lag1",
    "abs_corr_lag1",
    "n_blocks",
    "n_failed",
]
BIAS_COLUMNS = [
    "fold",
    "factor",
    "horizon",
    "n_coins",
    "spearman",
    "median_ratio",
    "median_ratio_liquid",
    "median_ratio_mid",
    "median_ratio_illiquid",
]
DECISION_COLUMNS = [
    "scheme",
    "factor",
    "question",
    "n_folds",
    "n_wins",
    "min_wins",
    "passes",
    "n_ratio_grows",
]


class ResidualFit(NamedTuple):
    """Residuos y factor de un esquema en test, con los bloques que no ajustaron y las betas."""

    residuals: pd.DataFrame
    factor: pd.Series
    n_blocks: int
    n_failed: int
    betas: pd.Series | None


@dataclass(frozen=True)
class StudyResult:
    """Tablas largas del estudio: metricas por esquema, sesgo por horizonte y decision."""

    diagnostics: pd.DataFrame
    bias: pd.DataFrame
    decision: pd.DataFrame


def fold_windows(
    test_start: pd.Timestamp, *, train_months: int = 12, test_months: int = 3
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Da (inicio de train, inicio de test, fin de test) de un fold; train acaba donde empieza test.

    `ValueError` si `test_start` no es la medianoche UTC del primer dia de un trimestre.
    """
    if train_months < 1 or test_months < 1:
        raise ValueError(
            f"train_months y test_months deben ser >= 1: {train_months}, {test_months}"
        )
    if test_start.tzinfo is None or str(test_start.tz) != "UTC":
        raise ValueError(f"test_start debe estar en UTC: {test_start}")
    if (
        test_start.day != 1
        or test_start.month not in _QUARTER_MONTHS
        or test_start != test_start.normalize()
    ):
        raise ValueError(f"test_start debe ser el inicio de un trimestre (UTC): {test_start}")
    # Train acaba justo en test_start (limite exclusivo): ningun minuto de test entra en train.
    train_start = test_start - pd.DateOffset(months=train_months)
    test_end = test_start + pd.DateOffset(months=test_months)
    return train_start, test_start, test_end


def _first_column(factors: pd.DataFrame) -> pd.Series:
    """Serie del factor; el estudio solo admite modelos con un factor (los tres actuales)."""
    if factors.shape[1] != 1:
        raise ValueError(f"el estudio exige un solo factor, el modelo da {factors.shape[1]}")
    return factors.iloc[:, 0].rename("factor")


def _failed(
    index: pd.DatetimeIndex, columns: pd.Index, n_blocks: int, n_failed: int
) -> ResidualFit:
    """Resultado todo `NaN` (nunca 0, D-02) para cuando no hay ajuste posible."""
    return ResidualFit(
        residuals=pd.DataFrame(np.nan, index=index, columns=columns, dtype="float64"),
        factor=pd.Series(np.nan, index=index, dtype="float64", name="factor"),
        n_blocks=n_blocks,
        n_failed=n_failed,
        betas=None,
    )


def project_fixed(
    make_model: Callable[[], FactorModel], fit_returns: pd.DataFrame, test: pd.DataFrame
) -> ResidualFit:
    """Ajusta con `fit_returns` y proyecta en `test`; un fallo de ajuste deja el residuo en `NaN`.

    `fit_returns` puede tener otra frecuencia que `test` (betas de 5 min sobre retornos de 1 min).
    """
    # fit_returns acaba antes que test (lo garantiza quien llama) y transform no reajusta (D-03).
    model = make_model()
    try:
        model.fit(fit_returns)
        residuals = model.transform(test)
        factor = _first_column(model.factors(test))
    except ValueError as exc:
        logger.warning("Ajuste fallido (%s): residuo en NaN", exc)
        return _failed(test.index, test.columns, 1, 1)
    betas: pd.Series | None = getattr(model, "betas", None)
    symbol = getattr(model, "symbol", None)
    if betas is not None and symbol is not None:
        betas = betas.drop(index=symbol, errors="ignore")
    return ResidualFit(residuals, factor, 1, 0, betas)


def rolling_residuals(
    make_model: Callable[[], FactorModel],
    returns: pd.DataFrame,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    *,
    window_days: int,
    step_days: int = 7,
    min_valid_share: float = 0.5,
) -> ResidualFit:
    """Residuos de `[test_start, test_end)` con `fit(ventana previa).transform(bloque)` por bloque.

    Cada `step_days` se reajusta un modelo nuevo con los `window_days` previos al bloque; las
    monedas con menos de `min_valid_share` de datos en la ventana quedan fuera (residuo `NaN`).
    """
    if window_days < 1 or step_days < 1:
        raise ValueError(f"window_days y step_days deben ser >= 1: {window_days}, {step_days}")
    if not 0 < min_valid_share <= 1:
        raise ValueError(f"min_valid_share debe estar en (0, 1]: {min_valid_share}")
    if test_start >= test_end:
        raise ValueError(f"rango de test vacio o invertido: [{test_start}, {test_end})")
    index = returns.index
    test_index = index[(index >= test_start) & (index < test_end)]
    out = pd.DataFrame(np.nan, index=test_index, columns=returns.columns, dtype="float64")
    factor = pd.Series(np.nan, index=test_index, dtype="float64", name="factor")
    window = pd.Timedelta(days=window_days)
    step = pd.Timedelta(days=step_days)
    n_blocks = n_failed = 0
    first_error: str | None = None
    block_start = test_start
    while block_start < test_end:
        block_end = min(block_start + step, test_end)
        n_blocks += 1
        # La ventana usa solo filas con indice < block_start: nunca el primer minuto del bloque
        # ni nada posterior, asi que lo que se proyecta no entra en el ajuste (D-03).
        in_window = (index >= block_start - window) & (index < block_start)
        in_block = (index >= block_start) & (index < block_end)
        train = returns[in_window]
        block = returns[in_block]
        kept = train.columns[(train.notna().mean() >= min_valid_share).to_numpy()]
        try:
            model = make_model().fit(train[kept])
            res = model.transform(block[kept])
            fac = _first_column(model.factors(block[kept]))
        except ValueError as exc:
            n_failed += 1
            first_error = first_error or str(exc)
            logger.warning("Bloque %s sin ajuste (%s): residuo en NaN", block_start.date(), exc)
        else:
            out.loc[block.index, kept] = res
            factor.loc[block.index] = fac
        block_start = block_start + step
    if n_failed == n_blocks:
        logger.error("Ningun bloque de %d dias ajusto; primer fallo: %s", window_days, first_error)
    return ResidualFit(out, factor, n_blocks, n_failed, None)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson de dos vectores sin NaN; `NaN` si hay menos de 2 datos o alguno es constante."""
    if len(a) < 2 or a.max() == a.min() or b.max() == b.min():
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def residual_diagnostics(
    returns: pd.DataFrame, residuals: pd.DataFrame, factor: pd.Series, *, min_obs: int = 1000
) -> pd.DataFrame:
    """Por moneda, solo en test: varianza que quita el factor y correlacion del residuo con `f`.

    `corr_lag0` es `|corr(e(t), f(t))|`; `corr_lag1` es `corr(e(t), f(t-1))` con signo.
    """
    if not returns.index.equals(residuals.index) or not returns.index.equals(factor.index):
        raise ValueError("returns, residuals y factor deben compartir indice")
    if min_obs < 2:
        raise ValueError(f"min_obs debe ser >= 2: {min_obs}")
    # f(t-1) es el factor del minuto anterior (pasado): diagnostico, no entra en ningun ajuste.
    lagged = factor.shift(1)
    rows: dict[str, dict[str, float]] = {}
    for symbol in returns.columns:
        r, e = returns[symbol], residuals[symbol]
        common = (r.notna() & e.notna()).to_numpy()
        with_f0 = common & factor.notna().to_numpy()
        with_f1 = e.notna().to_numpy() & lagged.notna().to_numpy()
        row = dict.fromkeys(("var_removed", "corr_lag0", "corr_lag1", "abs_corr_lag1"), np.nan)
        row["n_obs"] = float(common.sum())
        if common.sum() >= min_obs:
            denominator = float((r.to_numpy()[common] ** 2).sum())
            if denominator > 0:
                row["var_removed"] = 1.0 - float((e.to_numpy()[common] ** 2).sum()) / denominator
        if with_f0.sum() >= min_obs:
            row["corr_lag0"] = abs(_corr(e.to_numpy()[with_f0], factor.to_numpy()[with_f0]))
        if with_f1.sum() >= min_obs:
            lag1 = _corr(e.to_numpy()[with_f1], lagged.to_numpy()[with_f1])
            row["corr_lag1"] = lag1
            row["abs_corr_lag1"] = abs(lag1)
        rows[str(symbol)] = row
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "symbol"
    return table[["var_removed", "corr_lag0", "corr_lag1", "abs_corr_lag1", "n_obs"]]


def beta_horizon_bias(
    betas_by_horizon: Mapping[str, pd.Series],
    illiquidity: pd.Series,
    *,
    base: str = BASE_HORIZON,
) -> pd.DataFrame:
    """Cociente `beta_h / beta_base` por moneda frente a su iliquidez (Spearman y tercil).

    Solo cuentan las monedas con ambas betas finitas y `beta_base > 0` (si no, no vale).
    """
    if base not in betas_by_horizon:
        raise ValueError(f"falta la beta del horizonte base {base!r}")
    reference = betas_by_horizon[base]
    rows: list[dict[str, float | str | int]] = []
    for horizon, betas in betas_by_horizon.items():
        if horizon == base:
            continue
        frame = pd.DataFrame(
            {"base": reference, "other": betas, "illiq": illiquidity}, dtype="float64"
        ).dropna()
        frame = frame[frame["base"] > 0]
        ratio = frame["other"] / frame["base"]
        row: dict[str, float | str | int] = {
            "horizon": horizon,
            "n_coins": len(frame),
            "spearman": np.nan,
            "median_ratio": float(ratio.median()) if len(ratio) else np.nan,
            "median_ratio_liquid": np.nan,
            "median_ratio_mid": np.nan,
            "median_ratio_illiquid": np.nan,
        }
        if len(frame) >= 3:
            # Spearman con empates: Pearson de rangos promediados, independiente del orden.
            avg = frame["illiq"].rank(method="average")
            row["spearman"] = float(avg.corr(ratio.rank(method="average")))
            # Terciles: desempate determinista por iliquidez y, a igualdad, simbolo alfabetico.
            order = np.lexsort((frame.index.astype(str).to_numpy(), frame["illiq"].to_numpy()))
            positions = pd.Series(np.empty(len(frame)), index=frame.index)
            positions.iloc[order] = np.arange(1, len(frame) + 1)
            tercile = pd.qcut(positions, 3, labels=False)
            for k, name in enumerate(("liquid", "mid", "illiquid")):
                row[f"median_ratio_{name}"] = float(ratio[tercile == k].median())
        rows.append(row)
    return pd.DataFrame(rows, columns=["horizon", *BIAS_COLUMNS[3:]])


def _summarize(
    fit: ResidualFit, test: pd.DataFrame, *, min_obs: int
) -> tuple[dict[str, float], int]:
    """Medianas entre monedas de las metricas de un esquema y numero de monedas con metrica."""
    table = residual_diagnostics(test, fit.residuals, fit.factor, min_obs=min_obs)
    medians = {
        name: float(table[name].median()) if table[name].notna().any() else np.nan
        for name in ("var_removed", "corr_lag0", "corr_lag1", "abs_corr_lag1")
    }
    return medians, int(table["var_removed"].notna().sum())


def fold_symbols(
    universe: pd.DataFrame, test_start: pd.Timestamp, test_end: pd.Timestamp
) -> list[str]:
    """Monedas del trimestre de test en el universo; `ValueError` si no esta o acaba en otro dia."""
    starts = universe["period_start"].map(lambda moment: to_ms(pd.Timestamp(moment)))
    rows = universe[starts == to_ms(test_start)]
    if rows.empty:
        known = sorted({pd.Timestamp(m).strftime("%Y-%m-%d") for m in universe["period_start"]})
        raise ValueError(f"el trimestre {test_start:%Y-%m-%d} no esta en el universo; hay {known}")
    ends = {to_ms(pd.Timestamp(m)) for m in rows["period_end"]}
    if ends != {to_ms(test_end)}:
        raise ValueError(f"el trimestre {test_start:%Y-%m-%d} no acaba en {test_end:%Y-%m-%d}")
    return list(dict.fromkeys(rows["symbol"]))


class FoldData(NamedTuple):
    """Datos de un fold: monedas, validez en train y retornos de 1 min de train y de test."""

    label: str
    symbols: list[str]
    kept: list[str]
    train_valid: pd.Series
    returns_train: pd.DataFrame
    returns_test: pd.DataFrame
    train_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def load_fold(
    test_start: pd.Timestamp,
    *,
    universe: pd.DataFrame,
    train_months: int = 12,
    test_months: int = 3,
    min_train_valid_share: float = 0.5,
    rebuild: bool = False,
    base_dir: str | Path = DEFAULT_BASE_DIR,
    cache_dir: str | Path = PANEL_DIR,
) -> FoldData:
    """Carga los retornos de 1 min de train y test de un fold, solo con monedas validas en train."""
    train_start, start, test_end = fold_windows(
        test_start, train_months=train_months, test_months=test_months
    )
    symbols = fold_symbols(universe, start, test_end)
    label = f"{start:%Y-%m-%d}"
    logger.info("Fold %s: %d monedas en el universo", label, len(symbols))
    panel = cached_panel(
        symbols,
        "1m",
        to_ms(train_start),
        to_ms(test_end),
        base_dir=base_dir,
        cache_dir=cache_dir,
        rebuild=rebuild,
    )
    index = panel.features["return"].index
    in_train = (index >= train_start) & (index < start)
    in_test = (index >= start) & (index < test_end)
    # Train = [train_start, test_start) y test = [test_start, test_end): disjuntos y contiguos.
    train_valid = panel.valid[in_train].mean()
    kept = [s for s in symbols if train_valid[s] >= min_train_valid_share]
    logger.info("Fold %s: %d monedas con datos suficientes en train", label, len(kept))
    returns = panel.features["return"][kept]
    return FoldData(
        label,
        symbols,
        kept,
        train_valid,
        returns[in_train],
        returns[in_test],
        train_start,
        start,
        test_end,
    )


def _check_horizons(horizons: Sequence[str]) -> list[str]:
    """Horizontes distintos del base, sin repetir; `ValueError` si no son multiplo de 1 min."""
    for horizon in horizons:
        bar_factor(BASE_HORIZON, horizon)
    return [h for h in dict.fromkeys(horizons) if h != BASE_HORIZON]


def _fold_rows(
    fold: str,
    universe_symbols: list[str],
    train_valid: pd.Series,
    returns_train: pd.DataFrame,
    returns_test: pd.DataFrame,
    horizon_train: Mapping[str, pd.DataFrame],
    factors: Mapping[str, Callable[[], FactorModel]],
    windows: Sequence[int],
    test_bounds: tuple[pd.Timestamp, pd.Timestamp],
    *,
    step_days: int,
    min_valid_share: float,
    min_obs: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Filas de diagnostico y de sesgo de un fold, para todos los factores y esquemas."""
    diagnostics: list[dict[str, object]] = []
    bias: list[dict[str, object]] = []
    both = pd.concat([returns_train, returns_test])
    illiquidity = 1.0 - train_valid
    for name, make_model in factors.items():
        schemes: list[tuple[str, ResidualFit]] = [
            (FOLD_SCHEME, project_fixed(make_model, returns_train, returns_test))
        ]
        schemes += [
            (
                f"roll{w}",
                rolling_residuals(
                    make_model,
                    both,
                    *test_bounds,
                    window_days=w,
                    step_days=step_days,
                    min_valid_share=min_valid_share,
                ),
            )
            for w in windows
        ]
        betas = {BASE_HORIZON: schemes[0][1].betas}
        for horizon, fit_returns in horizon_train.items():
            fit = project_fixed(make_model, fit_returns, returns_test)
            schemes.append((f"h{horizon}", fit))
            betas[horizon] = fit.betas
        for scheme, fit in schemes:
            medians, n_coins = _summarize(fit, returns_test, min_obs=min_obs)
            diagnostics.append(
                {
                    "fold": fold,
                    "scheme": scheme,
                    "factor": name,
                    "n_universe": len(universe_symbols),
                    "n_coins": n_coins,
                    **medians,
                    "n_blocks": fit.n_blocks,
                    "n_failed": fit.n_failed,
                }
            )
        if name in OLS_FACTORS:
            missing = [h for h, b in betas.items() if b is None]
            if missing:
                logger.warning(
                    "Fold %s, factor %s: sin ajuste en %s; se omite la tabla de sesgo "
                    "(n_ratio_grows cuenta un fold menos)",
                    fold,
                    name,
                    ", ".join(missing),
                )
            else:
                table = beta_horizon_bias(betas, illiquidity)  # type: ignore[arg-type]
                for record in table.to_dict("records"):
                    bias.append({"fold": fold, "factor": name, **record})
    return diagnostics, bias


def _decision_table(
    diagnostics: pd.DataFrame, bias: pd.DataFrame, *, min_wins: int
) -> pd.DataFrame:
    """Folds ganados por cada esquema alternativo frente al de beta fija, por factor y en total."""
    base = diagnostics[diagnostics["scheme"] == FOLD_SCHEME].set_index(["fold", "factor"])
    rows: list[dict[str, object]] = []
    schemes = list(dict.fromkeys(s for s in diagnostics["scheme"] if s != FOLD_SCHEME))
    for scheme in schemes:
        alt = diagnostics[diagnostics["scheme"] == scheme].set_index(["fold", "factor"])
        question = "A" if scheme.startswith("roll") else "B"
        per_factor: list[dict[str, object]] = []
        for factor in dict.fromkeys(diagnostics["factor"]):
            a, b = alt.xs(factor, level="factor"), base.xs(factor, level="factor")
            b = b.reindex(a.index)
            if question == "A":
                # Mejora = quita mas varianza Y deja menos mercado instantaneo (los dos a la vez).
                win = (a["var_removed"] > b["var_removed"]) & (a["corr_lag0"] < b["corr_lag0"])
            else:
                win = a["abs_corr_lag1"] < b["abs_corr_lag1"]
            grows = np.nan
            if question == "B" and factor in OLS_FACTORS and len(bias):
                sel = bias[(bias["factor"] == factor) & (bias["horizon"] == scheme[1:])]
                grows = int((sel["spearman"] > 0).sum())
            per_factor.append(
                {
                    "scheme": scheme,
                    "factor": factor,
                    "question": question,
                    "n_folds": len(a),
                    "n_wins": int(win.sum()),
                    "min_wins": min_wins,
                    "passes": int(win.sum()) >= min_wins,
                    "n_ratio_grows": grows,
                }
            )
        rows.extend(per_factor)
        wins = [int(r["n_wins"]) for r in per_factor]  # type: ignore[call-overload]
        rows.append(
            {
                "scheme": scheme,
                "factor": "all",
                "question": question,
                "n_folds": per_factor[0]["n_folds"] if per_factor else 0,
                "n_wins": min(wins) if wins else 0,
                "min_wins": min_wins,
                "passes": bool(per_factor) and all(bool(r["passes"]) for r in per_factor),
                "n_ratio_grows": np.nan,
            }
        )
    return pd.DataFrame(rows, columns=DECISION_COLUMNS)


def compare_schemes(
    test_starts: Sequence[pd.Timestamp],
    *,
    universe: pd.DataFrame | None = None,
    factors: Mapping[str, Callable[[], FactorModel]] | None = None,
    windows: Sequence[int] = (30, 90),
    horizons: Sequence[str] = ("1m", "5m", "15m"),
    step_days: int = 7,
    train_months: int = 12,
    test_months: int = 3,
    min_train_valid_share: float = 0.5,
    min_valid_share: float = 0.5,
    min_obs: int = 1000,
    min_wins: int = 9,
    rebuild: bool = False,
    base_dir: str | Path = DEFAULT_BASE_DIR,
    cache_dir: str | Path = PANEL_DIR,
) -> StudyResult:
    """Compara beta por fold, beta movil y beta a 5 y 15 min en cada fold, para cada factor.

    El esquema `fold` (beta de train a 1 min) es la referencia y coincide con el horizonte 1m.
    """
    if any(w < 1 for w in windows):
        raise ValueError(f"las ventanas deben ser >= 1 dia: {list(windows)}")
    if not 0 < min_train_valid_share <= 1:
        raise ValueError(f"min_train_valid_share debe estar en (0, 1]: {min_train_valid_share}")
    other_horizons = _check_horizons(horizons)
    makers = dict(FACTOR_MAKERS if factors is None else factors)
    frame = load_universe() if universe is None else universe
    diagnostics: list[dict[str, object]] = []
    bias: list[dict[str, object]] = []
    for test_start in sorted(dict.fromkeys(test_starts)):
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
        # Los retornos de 5/15 min de train cierran antes de test_start (barra completa, D-19).
        horizon_train = {
            h: horizon_returns_from_cache(
                fold.kept,
                h,
                to_ms(fold.train_start),
                to_ms(fold.test_start),
                base_dir=base_dir,
            )
            for h in other_horizons
        }
        fold_diag, fold_bias = _fold_rows(
            fold.label,
            fold.symbols,
            fold.train_valid[fold.kept],
            fold.returns_train,
            fold.returns_test,
            horizon_train,
            makers,
            windows,
            (fold.test_start, fold.test_end),
            step_days=step_days,
            min_valid_share=min_valid_share,
            min_obs=min_obs,
        )
        diagnostics.extend(fold_diag)
        bias.extend(fold_bias)
    diag_table = pd.DataFrame(diagnostics, columns=DIAGNOSTIC_COLUMNS)
    bias_table = pd.DataFrame(bias, columns=BIAS_COLUMNS)
    decision = _decision_table(diag_table, bias_table, min_wins=min_wins)
    return StudyResult(diag_table, bias_table, decision)


def write_tables(result: StudyResult, *, out_dir: str | Path = TABLES_DIR) -> list[Path]:
    """Escribe las tres tablas en CSV con orden y formato numerico fijos; devuelve las rutas."""
    directory = resolve_repo_path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, table in (
        ("diagnostics", result.diagnostics),
        ("bias", result.bias),
        ("decision", result.decision),
    ):
        path = directory / f"beta_study_{name}.csv"
        table.to_csv(path, index=False, float_format=_NUMBER_FORMAT, lineterminator="\n")
        paths.append(path)
    logger.info("Tablas del estudio de betas: %s", ", ".join(str(p) for p in paths))
    return paths
