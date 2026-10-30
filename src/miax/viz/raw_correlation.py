"""Correlacion cruda entre activos: la foto del "antes" de residualizar (MIAX-046).

Todo es descriptivo de la muestra que recibe (el periodo entero): NO debe usarse como filtro ni como
entrada de un modelo, porque miraria al futuro. No recorta nada (D-22) ni rellena con 0 (D-02).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from miax.features.panel_store import PANEL_DIR, cached_panel
from miax.features.tails import tail_report
from miax.ingest.store import DEFAULT_BASE_DIR
from miax.ingest.universe import load_universe, to_ms
from miax.utils.io import resolve_repo_path
from miax.utils.logging import get_logger

logger = get_logger(__name__)

FIGURES_DIR = "reports/figures"
TABLES_DIR = "reports/tables"
DPI = 200
_NUMBER_FORMAT = "%.10g"
_NEGATIVE_EIGENVALUE_TOLERANCE = 1e-8
_METHOD_NOTE = (
    "Metodo: correlacion de Pearson por pares sobre retornos log de 1 minuto validos, "
    "sin recortar; "
    "solo observaciones comunes a cada par. Orden: BTC y luego categoria (alfabetico). "
    "Descriptivo del periodo completo: no es una feature ni un filtro."
)
_MISSING_COLOR = "#d9d9d9"
_MISSING_NOTE = " Gris = sin dato."


class ReportPaths(NamedTuple):
    """Ficheros que escribe `build_raw_correlation_report`."""

    png: Path
    pdf: Path
    summary_csv: Path
    eda_csv: Path


def raw_correlation(returns: pd.DataFrame, *, min_periods: int = 1000) -> pd.DataFrame:
    """Pearson por pares con observaciones comunes; `NaN` si un par tiene menos de `min_periods`.

    Una serie constante, toda `NaN` o con menos de `min_periods` datos queda con su fila, columna y
    diagonal en `NaN`: no se inventa una correlacion de 1 consigo misma.
    """
    if min_periods < 2:
        raise ValueError(f"min_periods debe ser al menos 2: {min_periods}")
    if not returns.columns.is_unique:
        raise ValueError("returns tiene columnas duplicadas")
    # Solo pares de observaciones del mismo instante: sin retardos, no se mira otra fila.
    corr = returns.astype("float64").corr(method="pearson", min_periods=min_periods)
    values = corr.to_numpy(copy=True)
    # Solo se fija a 1 la diagonal que pandas pudo calcular (sin varianza o sin datos es NaN).
    diagonal = np.diag(values).copy()
    np.fill_diagonal(values, np.where(np.isnan(diagonal), np.nan, 1.0))
    return pd.DataFrame(values, index=corr.index, columns=corr.columns)


def symbols_without_data(corr: pd.DataFrame) -> list[str]:
    """Simbolos con la diagonal en `NaN` (sin datos, constantes o con menos de `min_periods`).

    Mantiene el orden de la matriz; `raw_correlation` deja asi a esas series.
    """
    diagonal = np.diag(corr.to_numpy(dtype="float64"))
    return [
        str(symbol) for symbol, value in zip(corr.index, diagonal, strict=True) if np.isnan(value)
    ]


def _off_diagonal(corr: pd.DataFrame) -> np.ndarray:
    """Triangulo superior sin la diagonal, como vector (incluye los `NaN`)."""
    rows, cols = np.triu_indices(len(corr), k=1)
    return corr.to_numpy(dtype="float64")[rows, cols]


def count_missing_pairs(corr: pd.DataFrame) -> int:
    """Pares distintos (triangulo superior, sin diagonal) de la matriz sin valor."""
    return int(np.isnan(_off_diagonal(corr)).sum())


def explained_variance(corr: pd.DataFrame, *, max_nan_share: float = 0.05) -> np.ndarray:
    """Autovalores de la matriz de correlacion, de mayor a menor, como fraccion de N.

    Los pares sin valor se sustituyen por la media de los pares con valor; `ValueError` si faltan
    mas del `max_nan_share` de los pares. El relleno no garantiza una matriz semidefinida positiva:
    si algun autovalor sale negativo (mas alla de `-1e-8`) lanza `ValueError`.
    """
    if not 0.0 <= max_nan_share < 1.0:
        raise ValueError(f"max_nan_share debe estar en [0, 1): {max_nan_share}")
    n = len(corr)
    if n < 2 or corr.shape[1] != n:
        raise ValueError("la matriz de correlacion debe ser cuadrada y de al menos 2x2")
    values = corr.to_numpy(dtype="float64", copy=True)
    pairs = _off_diagonal(corr)
    missing = np.isnan(pairs)
    if missing.mean() > max_nan_share:
        raise ValueError(
            f"faltan {missing.mean():.1%} de los pares, mas que max_nan_share={max_nan_share:.1%}"
        )
    if missing.any():
        # Rellenar con 0 mentiria (D-02): se usa la media de los pares observados.
        fill = float(np.nanmean(pairs))
        values[np.isnan(values)] = fill
    np.fill_diagonal(values, 1.0)
    values = (values + values.T) / 2
    eigenvalues = np.linalg.eigvalsh(values)[::-1]
    if eigenvalues[-1] < -_NEGATIVE_EIGENVALUE_TOLERANCE:
        raise ValueError(
            f"la matriz de correlacion no es semidefinida positiva: autovalor minimo "
            f"{eigenvalues[-1]:.3g} (el relleno de pares sin valor no lo garantiza)"
        )
    return eigenvalues / n


def correlation_summary(
    corr: pd.DataFrame,
    *,
    factor: str = "BTCUSDT",
    high: float = 0.5,
    max_nan_share: float = 0.05,
) -> dict[str, float | int | str]:
    """Cifras escalares de la matriz: pares, correlacion con `factor` y primeros autovalores."""
    if factor not in corr.columns:
        raise ValueError(f"{factor} no esta en la matriz de correlacion")
    pairs = _off_diagonal(corr)
    pairs = pairs[~np.isnan(pairs)]
    if pairs.size == 0:
        raise ValueError("no hay ningun par con valor")
    with_factor = corr[factor].drop(index=factor).dropna()
    if with_factor.empty:
        raise ValueError(f"{factor} no tiene correlacion con ningun otro simbolo")
    shares = explained_variance(corr, max_nan_share=max_nan_share)
    n = len(corr)
    return {
        "n_symbols": n,
        "n_pairs": int(pairs.size),
        # Pares que `explained_variance` rellena con la media (sin diagonal ni duplicados).
        "n_missing_pairs": count_missing_pairs(corr),
        "pair_mean": float(pairs.mean()),
        "pair_median": float(np.median(pairs)),
        "pair_min": float(pairs.min()),
        "pair_max": float(pairs.max()),
        "high_threshold": float(high),
        # Estricto: un par igual a `high` no cuenta.
        "pairs_above_high_pct": float((pairs > high).mean() * 100),
        "factor": factor,
        "factor_median": float(with_factor.median()),
        "factor_min": float(with_factor.min()),
        "factor_min_symbol": str(with_factor.idxmin()),
        "factor_max": float(with_factor.max()),
        "factor_max_symbol": str(with_factor.idxmax()),
        "first_eigenvalue": float(shares[0] * n),
        "first_eigenvalue_share": float(shares[0]),
        "second_eigenvalue": float(shares[1] * n),
        "second_eigenvalue_share": float(shares[1]),
    }


def order_by_category(
    symbols: Sequence[str], categories: Mapping[str, str], *, first: str = "BTCUSDT"
) -> list[str]:
    """`first` el primero; despues por categoria (alfabetica) y por simbolo dentro de cada una."""
    names = list(dict.fromkeys(symbols))
    missing = [s for s in names if s not in categories]
    if missing:
        raise ValueError(f"simbolos sin categoria: {missing}")
    rest = sorted((s for s in names if s != first), key=lambda s: (categories[s], s))
    return [first, *rest] if first in names else rest


def _groups(
    order: Sequence[str], categories: Mapping[str, str], factor: str
) -> list[tuple[str, int, int]]:
    """Tramos consecutivos `(etiqueta, inicio, fin)` del orden; el factor va en el suyo."""
    labels = ["BTC" if s == factor else categories[s] for s in order]
    groups: list[tuple[str, int, int]] = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            groups.append((labels[start], start, i))
            start = i
    return groups


def plot_raw_correlation(
    corr: pd.DataFrame,
    categories: Mapping[str, str],
    summary: Mapping[str, float | int | str],
    eigenvalues: np.ndarray,
    *,
    title: str,
    factor: str = "BTCUSDT",
    without_data: Sequence[str] = (),
) -> Figure:
    """Figura de tres paneles: mapa de calor, correlacion con el factor y varianza explicada.

    `without_data` son las monedas apartadas del calculo: se anotan en el titulo de (c).
    """
    order = list(corr.index)
    n = len(order)
    short = [s.removesuffix("USDT") for s in order]
    groups = _groups(order, categories, factor)
    palette = [
        "#4c78a8",
        "#f58518",
        "#54a24b",
        "#b279a2",
        "#e45756",
        "#72b7b2",
        "#9d755d",
        "#bab0ac",
    ]
    labels = sorted({g[0] for g in groups})
    colors = {label: palette[i % len(palette)] for i, label in enumerate(labels)}

    fig = Figure(figsize=(17, 9.5))
    FigureCanvasAgg(fig)
    grid = fig.add_gridspec(
        2, 2, width_ratios=[1.35, 1], left=0.06, right=0.97, top=0.86, bottom=0.12, hspace=0.45
    )
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.985)

    # (a) Mapa de calor con escala divergente fija en [-1, 1].
    ax = fig.add_subplot(grid[:, 0])
    # Copia del colormap: en RdBu_r el blanco es correlacion 0, y una casilla sin dato no puede
    # parecer un 0 (D-02). No se toca el colormap registrado.
    cmap = colormaps["RdBu_r"].with_extremes(bad=_MISSING_COLOR)
    image = ax.imshow(
        corr.to_numpy(dtype="float64"), cmap=cmap, vmin=-1.0, vmax=1.0, interpolation="nearest"
    )
    ax.set_xticks(range(n), short, rotation=90, fontsize=6.5)
    ax.set_yticks(range(n), short, fontsize=6.5)
    for _, _, end in groups[:-1]:
        ax.axhline(end - 0.5, color="black", linewidth=0.8)
        ax.axvline(end - 0.5, color="black", linewidth=0.8)
    top = ax.secondary_xaxis("top")
    top.set_xticks(
        [(a + b - 1) / 2 for _, a, b in groups], [g[0] for g in groups], fontsize=8, rotation=90
    )
    top.tick_params(length=0)
    ax.set_title("(a) Correlacion de Pearson cruda, 1 minuto", fontsize=11, pad=48)
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02, label="correlacion")

    # (b) Correlacion de cada activo con el factor, en el mismo orden que el mapa.
    ax_b = fig.add_subplot(grid[0, 1])
    others = [s for s in order if s != factor]
    values = corr.loc[others, factor].to_numpy(dtype="float64")
    # Un par sin valor se pasa como NaN (sin barra): una barra de 0 mentiria (D-02).
    ax_b.bar(
        range(len(others)),
        values,
        color=[colors[categories[s]] for s in others],
    )
    ax_b.set_xticks(range(len(others)), [s.removesuffix("USDT") for s in others])
    ax_b.tick_params(axis="x", labelrotation=90, labelsize=6.5)
    ax_b.set_ylim(-1.0, 1.0)
    ax_b.axhline(0, color="black", linewidth=0.6)
    ax_b.set_ylabel(f"correlacion con {factor.removesuffix('USDT')}")
    name = factor.removesuffix("USDT")
    title_b = f"(b) Correlacion con {name} (mediana {summary['factor_median']:.3f})"
    absent = [s.removesuffix("USDT") for s, v in zip(others, values, strict=True) if np.isnan(v)]
    if absent:
        shown = ", ".join(absent[:6]) + (", ..." if len(absent) > 6 else "")
        title_b += f"\nsin dato ({len(absent)}): {shown}"
    ax_b.set_title(title_b, fontsize=11)
    ax_b.legend(
        handles=[Patch(color=colors[lab], label=lab) for lab in labels if lab != "BTC"],
        fontsize=7,
        ncol=4,
        loc="lower center",
    )

    # (c) Varianza que explica cada componente principal de la matriz de correlacion.
    ax_c = fig.add_subplot(grid[1, 1])
    ax_c.bar(range(1, len(eigenvalues) + 1), eigenvalues * 100, color="#4c78a8")
    ax_c.set_xlabel("componente principal")
    ax_c.set_ylabel("% de la varianza total")
    title_c = f"(c) Varianza explicada: 1.o {eigenvalues[0]:.1%}, 2.o {eigenvalues[1]:.1%}"
    if without_data:
        dropped = [s.removesuffix("USDT") for s in without_data]
        shown = ", ".join(dropped[:6]) + (", ..." if len(dropped) > 6 else "")
        title_c += f"\nsin datos, fuera del calculo ({len(dropped)}): {shown}"
    ax_c.set_title(title_c, fontsize=11)
    ax_c.set_xlim(0.3, len(eigenvalues) + 0.7)

    note = _METHOD_NOTE + (_MISSING_NOTE if corr.isna().to_numpy().any() else "")
    fig.text(0.06, 0.02, note, fontsize=8, wrap=True)
    return fig


def save_figure(
    figure: Figure, stem: str, *, out_dir: str | Path = FIGURES_DIR
) -> tuple[Path, Path]:
    """Guarda la figura como PNG (200 dpi) y PDF con el mismo nombre; devuelve ambas rutas."""
    directory = resolve_repo_path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    png = directory / f"{stem}.png"
    pdf = directory / f"{stem}.pdf"
    figure.savefig(png, dpi=DPI)
    figure.savefig(pdf)
    return png, pdf


def _format(value: float | str) -> str:
    """Texto estable de una cifra: mismo valor, mismos bytes."""
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    return _NUMBER_FORMAT % value


def _write_summary_csv(summary: Mapping[str, float | int | str], path: Path) -> None:
    """Escribe la tabla de cifras escalares (`metric,value`) en un orden y formato fijos."""
    table = pd.DataFrame({"metric": list(summary), "value": [_format(v) for v in summary.values()]})
    table.to_csv(path, index=False, lineterminator="\n")


def _eda_table(
    returns: pd.DataFrame,
    valid: pd.DataFrame,
    corr: pd.DataFrame,
    categories: Mapping[str, str],
    *,
    factor: str,
) -> pd.DataFrame:
    """Una fila por simbolo: categoria, % valido, % de `NaN`, std, curtosis y corr. con BTC."""
    tails = tail_report(returns)
    table = pd.DataFrame(
        {
            "category": [categories[s] for s in returns.columns],
            "valid_pct": valid[list(returns.columns)].mean().to_numpy() * 100,
            "return_nan_pct": returns.isna().mean().to_numpy() * 100,
            "std": tails.loc[list(returns.columns), "std"].to_numpy(),
            "kurtosis": tails.loc[list(returns.columns), "kurtosis"].to_numpy(),
            f"corr_{factor}": corr.loc[list(returns.columns), factor].to_numpy(),
        },
        index=pd.Index(list(returns.columns), name="symbol"),
    )
    return table


def _write_eda_csv(table: pd.DataFrame, path: Path) -> None:
    """Escribe la tabla por simbolo con formato numerico fijo."""
    table.to_csv(path, float_format=_NUMBER_FORMAT, lineterminator="\n")


def build_raw_correlation_report(
    period_start: pd.Timestamp | str,
    *,
    interval: str = "1m",
    universe: pd.DataFrame | None = None,
    factor: str = "BTCUSDT",
    min_periods: int = 1000,
    max_nan_share: float = 0.05,
    high: float = 0.5,
    rebuild: bool = False,
    base_dir: str | Path = DEFAULT_BASE_DIR,
    cache_dir: str | Path = PANEL_DIR,
    out_dir: str | Path = FIGURES_DIR,
    tables_dir: str | Path = TABLES_DIR,
) -> ReportPaths:
    """Genera la figura y las tablas de correlacion cruda de un periodo del universo.

    Es descriptivo del periodo completo: no debe usarse como filtro ni como entrada de un modelo.
    """
    start = (
        pd.Timestamp(period_start, tz="UTC")
        if isinstance(period_start, str)
        else pd.Timestamp(period_start)
    )
    if start.tzinfo is None:
        raise ValueError(f"period_start debe estar en UTC: {period_start}")
    frame = load_universe() if universe is None else universe
    starts = frame["period_start"].map(lambda moment: to_ms(pd.Timestamp(moment)))
    rows = frame[starts == to_ms(start)]
    if rows.empty:
        known = sorted({pd.Timestamp(m).strftime("%Y-%m-%d") for m in frame["period_start"]})
        raise ValueError(f"el periodo {start:%Y-%m-%d} no esta en el universo; hay {known}")
    if rows["period_end"].nunique() != 1:
        raise ValueError(f"el periodo {start:%Y-%m-%d} tiene varios period_end distintos")
    symbols = list(dict.fromkeys(rows["symbol"]))
    if factor not in symbols:
        raise ValueError(f"{factor} no esta en el universo del periodo {start:%Y-%m-%d}")
    categories = dict(zip(rows["symbol"], rows["category"], strict=True))
    order = order_by_category(symbols, categories, first=factor)

    end_ms = to_ms(pd.Timestamp(rows["period_end"].iloc[0]))
    # Mismo periodo y mismos simbolos que el universo: el panel viene de cache si ya existe.
    panel = cached_panel(
        symbols,
        interval,
        to_ms(start),
        end_ms,
        base_dir=base_dir,
        cache_dir=cache_dir,
        rebuild=rebuild,
    )
    returns = panel.features["return"][order]
    corr = raw_correlation(returns, min_periods=min_periods)
    without = symbols_without_data(corr)
    if factor in without:
        raise ValueError(f"{factor} no tiene datos en el periodo {start:%Y-%m-%d}")
    if len(order) - len(without) < 2:
        raise ValueError(
            f"menos de 2 simbolos con datos en el periodo {start:%Y-%m-%d}; sin datos: {without}"
        )
    if without:
        logger.warning(
            "Periodo %s: %d simbolos sin datos, fuera del resumen y de los componentes: %s",
            f"{start:%Y-%m-%d}",
            len(without),
            ", ".join(without),
        )
    # Un par deslistado dentro del periodo es legitimo (D-14): el 5 % solo cuenta pares que faltan
    # entre monedas con datos. El mapa y la tabla por simbolo usan `corr` completo.
    kept = corr.drop(index=without, columns=without)
    summary = correlation_summary(kept, factor=factor, high=high, max_nan_share=max_nan_share)
    eigenvalues = explained_variance(kept, max_nan_share=max_nan_share)

    label = f"{start:%Y-%m-%d}_{interval}"
    full_summary: dict[str, float | int | str] = {
        "period_start": f"{start:%Y-%m-%d}",
        "interval": interval,
        "n_rows": len(returns),
        "min_periods": min_periods,
        "n_symbols_in_universe": len(order),
        "n_symbols_without_data": len(without),
        "symbols_without_data": ";".join(without),
        **summary,
    }
    if without:
        title = (
            f"Correlacion cruda entre {len(order)} criptoactivos "
            f"({len(order) - len(without)} con datos), "
            f"trimestre desde {start:%Y-%m-%d} ({interval})"
        )
    else:
        title = (
            f"Correlacion cruda entre {len(order)} criptoactivos, "
            f"trimestre desde {start:%Y-%m-%d} ({interval})"
        )
    figure = plot_raw_correlation(
        corr,
        categories,
        summary,
        eigenvalues,
        title=title,
        factor=factor,
        without_data=without,
    )
    png, pdf = save_figure(figure, f"raw_correlation_{label}", out_dir=out_dir)
    tables = resolve_repo_path(tables_dir)
    tables.mkdir(parents=True, exist_ok=True)
    summary_csv = tables / f"raw_correlation_summary_{label}.csv"
    eda_csv = tables / f"panel_eda_{label}.csv"
    _write_summary_csv(full_summary, summary_csv)
    _write_eda_csv(_eda_table(returns, panel.valid, corr, categories, factor=factor), eda_csv)
    logger.info("Informe de correlacion cruda %s: %s, %s, %s", label, png, summary_csv, eda_csv)
    return ReportPaths(png=png, pdf=pdf, summary_csv=summary_csv, eda_csv=eda_csv)
