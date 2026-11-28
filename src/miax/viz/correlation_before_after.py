"""Figura de memoria: correlacion entre activos antes y despues de residualizar (MIAX-060).

Es descriptiva del trimestre que se pinta: NO es una feature ni un filtro. Cada factor se ajusta en
los primeros meses del trimestre y todo lo que se pinta es del resto (fuera de muestra).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from matplotlib.image import AxesImage

from miax.features.correlation import (
    explained_variance,
    off_diagonal,
    raw_correlation,
    symbols_without_data,
)
from miax.features.factor import FactorModel
from miax.features.panel_store import PANEL_DIR, cached_panel
from miax.ingest.store import DEFAULT_BASE_DIR
from miax.ingest.universe import load_universe, to_ms
from miax.utils.io import resolve_repo_path
from miax.utils.logging import get_logger
from miax.viz.factor_comparison import (
    FACTOR_MODELS,
    RESERVED_FROM,
    factor_comparison_by_period,
    split_period,
    usable_quarters,
)
from miax.viz.raw_correlation import (
    _NUMBER_FORMAT,
    FIGURES_DIR,
    MISSING_COLOR,
    TABLES_DIR,
    category_groups,
    order_by_category,
    save_figure,
)

logger = get_logger(__name__)

# Orden fijo de las matrices: el crudo primero y luego los tres factores.
MATRIX_NAMES: tuple[str, ...] = ("raw", "btc", "mean", "pca")
_COLORS = {"raw": "#222222", "btc": "#e8710a", "mean": "#1f77b4", "pca": "#2ca02c"}
_PANEL_TITLES = {
    "raw": "(a) Crudo",
    "btc": "(b) Residuo, factor BTC",
    "mean": "(c) Residuo, factor media",
    "pca": "(d) Residuo, factor PCA",
}
_LEGEND_NAMES = {
    "raw": "Crudo",
    "btc": "Residuo BTC",
    "mean": "Residuo media",
    "pca": "Residuo PCA",
}
_FACTOR_NAMES = {"btc": "BTC", "mean": "media", "pca": "PCA"}
_FACTOR_ROWS = ("btc", "mean", "pca")
# Hueco entre el mapa y su titulo para que quepan las etiquetas de categoria de arriba.
_TITLE_PAD = 46
_CSV_COLUMNS: tuple[str, ...] = (
    "matrix",
    "period_start",
    "n_symbols",
    "n_pairs",
    "pair_abs_mean",
    "pair_mean",
    "first_eigenvalue_share",
    "correlation_drop",
)


class CorrelationFigurePaths(NamedTuple):
    """Ficheros que escribe `build_correlation_before_after_report`."""

    png: Path
    pdf: Path
    csv: Path


def _comma(value: float, digits: int = 2) -> str:
    """Numero con coma decimal, como se escribe en la memoria."""
    return f"{value:.{digits}f}".replace(".", ",")


def residual_correlations(
    returns: pd.DataFrame,
    *,
    train_months: int = 2,
    models: Mapping[str, Callable[[], FactorModel]] | None = None,
    min_periods: int = 1000,
) -> dict[str, pd.DataFrame]:
    """Correlacion del test cruda y tras quitar cada factor, con las mismas filas y columnas.

    Un factor que falla deja su matriz entera en `NaN` y un aviso: no tumba la figura.
    """
    factories = FACTOR_MODELS if models is None else models
    train, test = split_period(returns, train_months=train_months)
    result: dict[str, pd.DataFrame] = {"raw": raw_correlation(test, min_periods=min_periods)}
    columns = list(returns.columns)
    for name, factory in factories.items():
        try:
            # Ajuste solo con el train; el test se transforma con betas ya fijadas: sin fuga.
            residuals = factory().fit(train).transform(test)
            result[name] = raw_correlation(
                residuals.reindex(columns=columns), min_periods=min_periods
            )
        except ValueError as exc:
            logger.warning("Factor %s sin resultado: %s", name, exc)
            result[name] = pd.DataFrame(np.nan, index=columns, columns=columns)
    return result


def _pairs(matrix: pd.DataFrame) -> np.ndarray:
    """Correlaciones fuera de la diagonal, sin los pares sin valor."""
    pairs = off_diagonal(matrix)
    return pairs[~np.isnan(pairs)]


def _abs_mean(matrix: pd.DataFrame) -> float:
    """|rho| medio fuera de la diagonal; `NaN` si no hay ningun par."""
    pairs = _pairs(matrix)
    return float(np.abs(pairs).mean()) if pairs.size else float("nan")


def _first_eigenvalue_share(matrix: pd.DataFrame, *, max_nan_share: float = 0.05) -> float:
    """Primer autovalor como fraccion de N sobre las monedas con datos; `NaN` si no se puede."""
    kept = matrix.drop(index=symbols_without_data(matrix), columns=symbols_without_data(matrix))
    try:
        return float(explained_variance(kept, max_nan_share=max_nan_share, require_psd=False)[0])
    except ValueError:
        return float("nan")


def _plot_heatmap(
    ax: Axes,
    matrix: pd.DataFrame,
    cmap: Colormap,
    groups: list[tuple[str, int, int]],
    *,
    labels: list[str] | None,
) -> AxesImage:
    """Mapa de calor con separadores de categoria; etiquetas solo si se pasan."""
    n = len(matrix)
    image = ax.imshow(
        matrix.to_numpy(dtype="float64"), cmap=cmap, vmin=-1.0, vmax=1.0, interpolation="nearest"
    )
    for _, _, end in groups[:-1]:
        ax.axhline(end - 0.5, color="black", linewidth=0.6)
        ax.axvline(end - 0.5, color="black", linewidth=0.6)
    if labels is None:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.set_xticks(range(n), labels, rotation=90, fontsize=5)
        ax.set_yticks(range(n), labels, fontsize=5)
    # Categorias arriba en todos los mapas: el hueco sobre el titulo es igual y esta alineado.
    top = ax.secondary_xaxis("top")
    top.set_xticks(
        [(a + b - 1) / 2 for _, a, b in groups], [g[0] for g in groups], fontsize=7, rotation=90
    )
    top.tick_params(length=0)
    return image


def plot_correlation_before_after(
    matrices: Mapping[str, pd.DataFrame],
    categories: Mapping[str, str],
    by_period: pd.DataFrame,
    *,
    period_label: str,
    title: str,
    factor: str = "BTCUSDT",
    floor: float | None = None,
    highlight: str | None = None,
    drops: Mapping[str, float] | None = None,
    min_periods: int = 1000,
    interval: str = "1m",
    train_months: int = 2,
) -> Figure:
    """Cuatro mapas de calor (crudo y tres factores), sus histogramas y |rho| medio por trimestre.

    `by_period` es el formato largo de `factor_comparison_by_period` ya filtrado; `highlight` es
    el `period_start` (YYYY-MM-DD) del trimestre que se marca en (f).
    """
    missing = [name for name in MATRIX_NAMES if name not in matrices]
    if missing:
        raise ValueError(f"faltan matrices: {missing}")
    order = list(matrices["raw"].index)
    groups = category_groups(order, categories, factor)
    short = [s.removesuffix("USDT") for s in order]
    cmap = colormaps["RdBu_r"].with_extremes(bad=MISSING_COLOR)

    fig = Figure(figsize=(18, 10))
    FigureCanvasAgg(fig)
    grid = fig.add_gridspec(
        2,
        5,
        width_ratios=[1, 1, 1, 1, 0.05],
        height_ratios=[1.3, 1],
        left=0.05,
        right=0.97,
        top=0.84,
        bottom=0.12,
        wspace=0.12,
        hspace=0.38,
    )
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.99)

    # Fila 1: los cuatro mapas con la misma escala fija, el mismo orden y una sola barra de color.
    image = None
    for column, name in enumerate(MATRIX_NAMES):
        ax = fig.add_subplot(grid[0, column])
        image = _plot_heatmap(
            ax, matrices[name], cmap, groups, labels=short if name == "raw" else None
        )
        mean_abs = _abs_mean(matrices[name])
        shown = "sin dato" if np.isnan(mean_abs) else _comma(mean_abs)
        panel = _PANEL_TITLES[name]
        if name == "btc":
            n_btc = len(matrices[name]) - len(symbols_without_data(matrices[name]))
            panel = f"(b) Residuo, factor BTC (sin BTC, {n_btc} monedas)"
        ax.set_title(f"{panel}\n|ρ| medio {shown}", fontsize=10, pad=_TITLE_PAD)
    cax = fig.add_subplot(grid[0, 4])
    fig.colorbar(image, cax=cax, label="correlacion")

    # Fila 2 con su propio reparto: mas hueco entre (e) y (f) para la etiqueta del eje de (f).
    row2 = grid[1, :].subgridspec(1, 2, width_ratios=[2, 3.05], wspace=0.16)

    # (e) Distribucion de las correlaciones por pares, fuera de la diagonal.
    ax_e = fig.add_subplot(row2[0])
    bins = np.linspace(-1.0, 1.0, 201)
    for name in MATRIX_NAMES:
        pairs = _pairs(matrices[name])
        if pairs.size:
            ax_e.hist(
                pairs,
                bins=bins,
                histtype="step",
                color=_COLORS[name],
                linewidth=1.4,
                label=_LEGEND_NAMES[name],
            )
    if floor is None:
        usable = len(order) - len(symbols_without_data(matrices["raw"]))
        floor = -1.0 / (usable - 1) if usable > 1 else None
    if floor is not None:
        ax_e.axvline(floor, color="gray", linestyle=":", linewidth=1.2)
        ax_e.text(
            floor,
            0.97,
            f" suelo -1/(N-1) = {_comma(floor, 3)}",
            transform=ax_e.get_xaxis_transform(),
            fontsize=7,
            color="gray",
            va="top",
        )
    ax_e.set_xlim(-0.3, 1.0)
    ax_e.set_xlabel("correlacion por pares (fuera de la diagonal)")
    ax_e.set_ylabel("numero de pares")
    ax_e.set_title("(e) Distribucion de las correlaciones por pares", fontsize=11)
    ax_e.legend(fontsize=8, loc="upper right")

    # (f) |rho| medio trimestre a trimestre: el trimestre pintado no es uno escogido.
    ax_f = fig.add_subplot(row2[1])
    _plot_by_period(ax_f, by_period, highlight)
    ax_f.set_title("(f) |ρ| medio por trimestre", fontsize=11)

    note = (
        f"Metodo: cada factor se ajusta en los {train_months} primeros meses del trimestre y todo "
        f"lo pintado es del resto (fuera de muestra); Pearson por pares sobre retornos log de "
        f"{interval}, "
        f"al menos {min_periods} minutos comunes; gris = sin dato. Trimestre {period_label}."
    )
    if drops:
        parts = ", ".join(
            f"{_FACTOR_NAMES.get(name, name)} {_comma(value * 100, 1)} %"
            for name, value in drops.items()
            if not np.isnan(value)
        )
        if parts:
            note += f" Caida del |ρ| medio sobre las mismas monedas que cada residuo: {parts}."
    fig.text(0.05, 0.02, note, fontsize=8, wrap=True)
    return fig


def _plot_by_period(ax: Axes, by_period: pd.DataFrame, highlight: str | None) -> None:
    """Lineas del |rho| medio crudo y de cada factor por trimestre, con uno marcado."""
    ax.set_ylabel("|ρ| medio fuera de la diagonal")
    if by_period.empty:
        ax.text(0.5, 0.5, "sin trimestres validos", ha="center", transform=ax.transAxes)
        return
    quarters = sorted(by_period["period_start"].unique())
    index = {q: i for i, q in enumerate(quarters)}
    # El crudo es el de la fila "mean": no enmascara monedas y coincide con el mapa (a).
    mean_rows = by_period[by_period["factor"] == "mean"]
    raw = mean_rows.set_index("period_start")["raw_pair_abs_mean"].reindex(quarters)
    series = {"raw": raw}
    for name in _FACTOR_ROWS:
        part = by_period[by_period["factor"] == name].set_index("period_start")
        series[name] = part["residual_pair_abs_mean"].reindex(quarters)
    xs = np.arange(len(quarters))
    for name in MATRIX_NAMES:
        ax.plot(
            xs,
            series[name].to_numpy(dtype="float64"),
            marker="o",
            markersize=4,
            color=_COLORS[name],
            label=_LEGEND_NAMES[name],
        )
    if highlight is not None and highlight not in index:
        logger.warning("El trimestre %s no esta entre los trimestres de (f)", highlight)
    if highlight in index:
        ax.axvspan(index[highlight] - 0.4, index[highlight] + 0.4, color="#fde68a", alpha=0.5)
    labels = [f"{pd.Timestamp(q).year}T{pd.Timestamp(q).quarter}" for q in quarters]
    ax.set_xticks(xs, labels, rotation=90, fontsize=8)
    top = max(0.75, float(np.nanmax([s.to_numpy(dtype="float64") for s in series.values()])) * 1.08)
    ax.set_ylim(0.0, top)
    # Abajo a la derecha, por encima de las lineas de los residuos media y PCA (|ρ| < 0,1): la
    # franja de 0,1 a 0,2 queda sin datos y la leyenda no tapa ninguna linea.
    ax.legend(fontsize=8, loc="lower right", bbox_to_anchor=(0.995, 0.13), ncol=2)


def _period_rows(
    universe: pd.DataFrame, start: pd.Timestamp, reserved_from: pd.Timestamp, factor: str
) -> pd.DataFrame:
    """Filas del universo del periodo; falla si no existe, no tiene el factor o es reservado."""
    starts = universe["period_start"].map(lambda moment: to_ms(pd.Timestamp(moment)))
    rows = universe[starts == to_ms(start)]
    if rows.empty:
        known = sorted({pd.Timestamp(m).strftime("%Y-%m-%d") for m in universe["period_start"]})
        raise ValueError(f"el periodo {start:%Y-%m-%d} no esta en el universo; hay {known}")
    if rows["period_end"].nunique() != 1:
        raise ValueError(f"el periodo {start:%Y-%m-%d} tiene varios period_end distintos")
    if pd.Timestamp(rows["period_end"].iloc[0]) > reserved_from:
        raise ValueError(
            f"el periodo {start:%Y-%m-%d} toca lo reservado desde {reserved_from:%Y-%m-%d} (D-37)"
        )
    if factor not in set(rows["symbol"]):
        raise ValueError(f"{factor} no esta en el universo del periodo {start:%Y-%m-%d}")
    return rows


def _matrix_table(
    matrices: Mapping[str, pd.DataFrame], period: str, drops: Mapping[str, float]
) -> pd.DataFrame:
    """Una fila por matriz con sus cifras escalares, en orden fijo."""
    rows = []
    for name in MATRIX_NAMES:
        matrix = matrices[name]
        pairs = _pairs(matrix)
        rows.append(
            {
                "matrix": name,
                "period_start": period,
                "n_symbols": len(matrix) - len(symbols_without_data(matrix)),
                "n_pairs": int(pairs.size),
                "pair_abs_mean": _abs_mean(matrix),
                "pair_mean": float(pairs.mean()) if pairs.size else float("nan"),
                "first_eigenvalue_share": _first_eigenvalue_share(matrix),
                "correlation_drop": drops.get(name, float("nan")),
            }
        )
    return pd.DataFrame(rows, columns=list(_CSV_COLUMNS))


def build_correlation_before_after_report(
    period_start: pd.Timestamp | str = "2025-01-01",
    *,
    interval: str = "1m",
    universe: pd.DataFrame | None = None,
    train_months: int = 2,
    min_periods: int = 1000,
    min_symbols: int = 30,
    factor: str = "BTCUSDT",
    reserved_from: pd.Timestamp = RESERVED_FROM,
    rebuild: bool = False,
    base_dir: str | Path = DEFAULT_BASE_DIR,
    cache_dir: str | Path = PANEL_DIR,
    out_dir: str | Path = FIGURES_DIR,
    tables_dir: str | Path = TABLES_DIR,
) -> CorrelationFigurePaths:
    """Genera la figura y el CSV de correlacion antes y despues de residualizar en un trimestre.

    Es descriptivo del trimestre: no debe usarse como filtro ni como entrada de un modelo.
    """
    start = (
        pd.Timestamp(period_start, tz="UTC")
        if isinstance(period_start, str)
        else pd.Timestamp(period_start)
    )
    if start.tzinfo is None:
        raise ValueError(f"period_start debe estar en UTC: {period_start}")
    if reserved_from.tzinfo is None:
        raise ValueError(f"reserved_from debe estar en UTC: {reserved_from}")
    frame = load_universe() if universe is None else universe
    rows = _period_rows(frame, start, reserved_from, factor)
    symbols = list(dict.fromkeys(rows["symbol"]))
    categories = dict(zip(rows["symbol"], rows["category"], strict=True))
    order = order_by_category(symbols, categories, first=factor)

    panel = cached_panel(
        symbols,
        interval,
        to_ms(start),
        to_ms(pd.Timestamp(rows["period_end"].iloc[0])),
        base_dir=base_dir,
        cache_dir=cache_dir,
        rebuild=rebuild,
    )
    matrices = residual_correlations(
        panel.features["return"][order], train_months=train_months, min_periods=min_periods
    )
    del panel

    by_period = factor_comparison_by_period(
        frame,
        interval=interval,
        train_months=train_months,
        reserved_from=reserved_from,
        base_dir=base_dir,
        cache_dir=cache_dir,
    )
    period = f"{start:%Y-%m-%d}"
    this = by_period[by_period["period_start"] == period].set_index("factor")
    drops = {
        name: float(this.loc[name, "correlation_drop"]) if name in this.index else float("nan")
        for name in _FACTOR_ROWS
    }
    shown = usable_quarters(by_period, min_symbols=min_symbols)

    title = (
        f"Correlacion entre {len(order)} criptoactivos antes y despues de quitar el factor comun, "
        f"trimestre desde {period} ({interval})"
    )
    figure = plot_correlation_before_after(
        matrices,
        categories,
        shown,
        period_label=period,
        title=title,
        factor=factor,
        highlight=period,
        drops=drops,
        min_periods=min_periods,
        interval=interval,
        train_months=train_months,
    )
    label = f"{period}_{interval}"
    png, pdf = save_figure(figure, f"correlation_before_after_{label}", out_dir=out_dir)
    tables = resolve_repo_path(tables_dir)
    tables.mkdir(parents=True, exist_ok=True)
    csv = tables / f"correlation_before_after_{label}.csv"
    _matrix_table(matrices, period, drops).to_csv(
        csv, index=False, float_format=_NUMBER_FORMAT, lineterminator="\n"
    )
    logger.info("Figura de correlacion antes y despues %s: %s, %s", label, png, csv)
    return CorrelationFigurePaths(png=png, pdf=pdf, csv=csv)
