"""Tests de MIAX-198: universos trimestral, semestral y anual sobre los ficheros reales."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from miax.ingest.universe import (
    DEFAULT_EXCLUDED_CATEGORIES,
    UniverseParams,
    load_categories,
    load_universe,
    load_universe_params,
    universe_periods,
)
from miax.utils.config import CONFIGS_DIR

# nombre -> (fichero de parametros, fichero de universo, period_months)
VARIANTS: dict[str, tuple[str, str, int]] = {
    "quarterly": ("universe_params.yaml", "universe.yaml", 3),
    "semiannual": ("universe_params_semiannual.yaml", "universe_semiannual.yaml", 6),
    "annual": ("universe_params_annual.yaml", "universe_annual.yaml", 12),
}
LAST_END = pd.Timestamp("2026-10-01", tz="UTC")
NEW_ASSET_CATEGORIES = {
    "ASTER": "defi",
    "ONDO": "defi",
    "PUMP": "meme",
    "GIGGLE": "meme",
    "币安人生": "meme",
    "XPL": "l1",
    "KITE": "ai",
    "USD1": "stablecoin",
    "RLUSD": "stablecoin",
    "U": "stablecoin",
}
QUOTE = "USDT"


def _raw(universe_file: str) -> dict[str, Any]:
    """Lee el YAML del universo tal cual, sin pasar por load_universe."""
    text = (CONFIGS_DIR / universe_file).read_text(encoding="utf-8")
    return yaml.safe_load(text)


def _pairs_by_start(frame: pd.DataFrame) -> dict[pd.Timestamp, list[tuple[str, float]]]:
    """Pares (simbolo, volumen mediano) en orden de fichero para cada inicio de periodo."""
    return {
        start: list(zip(group["symbol"], group["median_quote_volume"], strict=True))
        for start, group in frame.groupby("period_start", sort=True)
    }


def _inconsistencies(
    smaller: pd.DataFrame, quarterly: pd.DataFrame
) -> list[tuple[pd.Timestamp, str]]:
    """Periodos de `smaller` cuyos pares o volumenes difieren del trimestre que empieza igual."""
    reference = _pairs_by_start(quarterly)
    problems: list[tuple[pd.Timestamp, str]] = []
    for start, pairs in _pairs_by_start(smaller).items():
        if start not in reference:
            problems.append((start, "no hay trimestre que empiece en el mismo instante"))
        elif pairs != reference[start]:
            problems.append((start, "pares o volumenes distintos"))
    return problems


@pytest.fixture(scope="module")
def params() -> dict[str, UniverseParams]:
    """Parametros reales de las tres variantes."""
    return {name: load_universe_params(files[0]) for name, files in VARIANTS.items()}


@pytest.fixture(scope="module")
def frames() -> dict[str, pd.DataFrame]:
    """Universos reales de las tres variantes."""
    return {name: load_universe(files[1]) for name, files in VARIANTS.items()}


@pytest.fixture(scope="module")
def raws() -> dict[str, dict[str, Any]]:
    """YAML crudos de las tres variantes."""
    return {name: _raw(files[1]) for name, files in VARIANTS.items()}


@pytest.fixture(scope="module")
def categories() -> dict[str, str]:
    """Mapa activo base -> categoria del categories.yaml real."""
    return load_categories()


# --- parametros ---


@pytest.mark.parametrize("name", list(VARIANTS))
def test_params_period_months(params: Mapping[str, UniverseParams], name: str) -> None:
    """Cada fichero de parametros tiene el period_months de su variante (3, 6 o 12)."""
    assert params[name].period_months == VARIANTS[name][2]


def test_params_differ_only_in_period_months(params: Mapping[str, UniverseParams]) -> None:
    """Igualando period_months, los tres ficheros de parametros son identicos."""
    base = params["quarterly"]
    for name in ("semiannual", "annual"):
        aligned = dataclasses.replace(params[name], period_months=base.period_months)
        assert aligned == base, name
    assert len({p.period_months for p in params.values()}) == 3


# --- universos congelados ---


@pytest.mark.parametrize("name", list(VARIANTS))
def test_universe_header_params_match_params_file(
    raws: Mapping[str, dict[str, Any]], params: Mapping[str, UniverseParams], name: str
) -> None:
    """La cabecera `params` del universo es exactamente el fichero de parametros que lo genero."""
    header = raws[name]["params"]
    assert header["period_months"] == VARIANTS[name][2]
    assert header == dataclasses.asdict(params[name])


@pytest.mark.parametrize("name", list(VARIANTS))
def test_universe_periods_contiguous_and_correct_length(
    raws: Mapping[str, dict[str, Any]], name: str
) -> None:
    """Periodos contiguos sin solape, de period_months meses salvo el ultimo, recortado."""
    months = VARIANTS[name][2]
    periods = raws[name]["periods"]
    assert periods
    for previous, current in pairwise(periods):
        assert current["period_start"] == previous["period_end"]
    for period in periods:
        start = pd.Timestamp(period["period_start"], tz="UTC")
        end = pd.Timestamp(period["period_end"], tz="UTC")
        assert start < end
    assert pd.Timestamp(periods[0]["period_start"], tz="UTC") == pd.Timestamp(
        "2018-01-01", tz="UTC"
    )
    for period in periods[:-1]:
        start = pd.Timestamp(period["period_start"], tz="UTC")
        end = pd.Timestamp(period["period_end"], tz="UTC")
        assert end == start + pd.DateOffset(months=months)
    last = periods[-1]
    last_start = pd.Timestamp(last["period_start"], tz="UTC")
    assert pd.Timestamp(last["period_end"], tz="UTC") == LAST_END
    assert last_start + pd.DateOffset(months=months) >= LAST_END


@pytest.mark.parametrize("name", list(VARIANTS))
def test_universe_periods_match_universe_periods_function(
    raws: Mapping[str, dict[str, Any]],
    params: Mapping[str, UniverseParams],
    frames: Mapping[str, pd.DataFrame],
    name: str,
) -> None:
    """Los periodos del fichero (y los no vacios del DataFrame) salen de universe_periods."""
    expected = [
        (
            pd.Timestamp(p.period_start_ms, unit="ms", tz="UTC").strftime("%Y-%m-%d"),
            pd.Timestamp(p.period_end_ms, unit="ms", tz="UTC").strftime("%Y-%m-%d"),
        )
        for p in universe_periods(params[name])
    ]
    in_file = [(p["period_start"], p["period_end"]) for p in raws[name]["periods"]]
    assert in_file == expected
    frame = frames[name]
    non_empty = set(zip(frame["period_start"], frame["period_end"], strict=True))
    as_dates = {(s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")) for s, e in non_empty}
    assert as_dates <= set(expected)


@pytest.mark.parametrize("name", list(VARIANTS))
def test_universe_volume_window_ends_day_before_period(
    raws: Mapping[str, dict[str, Any]], name: str
) -> None:
    """La ventana de volumen termina el dia anterior al inicio del periodo (sin fuga)."""
    for period in raws[name]["periods"]:
        start = pd.Timestamp(period["period_start"], tz="UTC")
        assert pd.Timestamp(period["volume_window"][1], tz="UTC") == start - pd.Timedelta(days=1)


@pytest.mark.parametrize("name", list(VARIANTS))
def test_universe_dates_are_utc(frames: Mapping[str, pd.DataFrame], name: str) -> None:
    """Todas las columnas de fecha del universo cargado son datetime64 con zona UTC."""
    frame = frames[name]
    for column in ("period_start", "period_end", "first_candle"):
        dtype = frame[column].dtype
        assert isinstance(dtype, pd.DatetimeTZDtype), column
        assert str(dtype.tz) == "UTC", column
    assert not frame.empty


@pytest.mark.parametrize("name", list(VARIANTS))
def test_universe_respects_caps(
    frames: Mapping[str, pd.DataFrame], params: Mapping[str, UniverseParams], name: str
) -> None:
    """Como mucho max_pairs por periodo y max_per_category por categoria y periodo."""
    frame = frames[name]
    per_period = frame.groupby("period_start").size()
    assert per_period.max() <= params[name].max_pairs
    per_category = frame.groupby(["period_start", "category"]).size()
    assert per_category.max() <= params[name].max_per_category


@pytest.mark.parametrize("name", list(VARIANTS))
def test_universe_header_counts_match_pairs(raws: Mapping[str, dict[str, Any]], name: str) -> None:
    """n_pairs y categories de cada periodo coinciden con la lista de pares."""
    for period in raws[name]["periods"]:
        assert period["n_pairs"] == len(period["pairs"])
        counted: dict[str, int] = {}
        for pair in period["pairs"]:
            counted[pair["category"]] = counted.get(pair["category"], 0) + 1
        assert period["categories"] == counted


@pytest.mark.parametrize("name", list(VARIANTS))
def test_universe_no_duplicate_symbols_in_period(
    frames: Mapping[str, pd.DataFrame], name: str
) -> None:
    """Un simbolo aparece una sola vez por periodo y todos son pares contra USDT."""
    frame = frames[name]
    assert not frame.duplicated(["period_start", "symbol"]).any()
    assert frame["symbol"].str.endswith(QUOTE).all()


@pytest.mark.parametrize("name", list(VARIANTS))
def test_universe_has_no_excluded_or_uncategorized_pairs(
    frames: Mapping[str, pd.DataFrame], categories: Mapping[str, str], name: str
) -> None:
    """Ningun par es stablecoin o pegged ni carece de categoria; la categoria es la del fichero."""
    frame = frames[name]
    assert not frame["category"].isin(list(DEFAULT_EXCLUDED_CATEGORIES)).any()
    bases = frame["symbol"].str.removesuffix(QUOTE)
    mapped = bases.str.upper().map(categories)
    assert mapped.notna().all(), sorted(set(bases[mapped.isna()]))
    assert (mapped == frame["category"]).all()
    assert not mapped.isin(list(DEFAULT_EXCLUDED_CATEGORIES)).any()


@pytest.mark.parametrize("name", list(VARIANTS))
def test_universe_volume_above_floor(
    frames: Mapping[str, pd.DataFrame], params: Mapping[str, UniverseParams], name: str
) -> None:
    """Todo par supera el suelo de volumen de los parametros."""
    assert (frames[name]["median_quote_volume"] >= params[name].min_quote_volume).all()


# --- consistencia entre universos ---


@pytest.mark.parametrize("name", ["semiannual", "annual"])
def test_periods_match_quarter_starting_at_same_instant(
    frames: Mapping[str, pd.DataFrame], name: str
) -> None:
    """Cada periodo semestral/anual tiene los mismos pares y volumenes que el trimestre gemelo."""
    assert _inconsistencies(frames[name], frames["quarterly"]) == []


def test_consistency_check_compares_every_semiannual_and_annual_period(
    frames: Mapping[str, pd.DataFrame],
) -> None:
    """La comprobacion de consistencia no es vacia: hay periodos que comparar."""
    assert len(_pairs_by_start(frames["semiannual"])) >= 10
    assert len(_pairs_by_start(frames["annual"])) >= 5


def test_consistency_check_fails_when_a_semiannual_pair_is_changed(
    tmp_path: Path, frames: Mapping[str, pd.DataFrame]
) -> None:
    """Mutacion: cambiar un par de un periodo en una copia del semestral rompe la consistencia."""
    raw = _raw(VARIANTS["semiannual"][1])
    target = raw["periods"][-3]
    assert target["pairs"], "el periodo elegido para mutar esta vacio"
    target["pairs"][0]["symbol"] = "ZZZZUSDT"
    copy = tmp_path / "universe_semiannual_mutated.yaml"
    copy.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
    problems = _inconsistencies(load_universe(copy), frames["quarterly"])
    assert [start for start, _ in problems] == [pd.Timestamp(target["period_start"], tz="UTC")]


def test_consistency_check_fails_when_a_volume_is_changed(
    tmp_path: Path, frames: Mapping[str, pd.DataFrame]
) -> None:
    """Mutacion: cambiar el volumen de un par en una copia del anual tambien se detecta."""
    raw = _raw(VARIANTS["annual"][1])
    target = raw["periods"][-2]
    target["pairs"][0]["median_quote_volume"] += 1.0
    copy = tmp_path / "universe_annual_mutated.yaml"
    copy.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
    problems = _inconsistencies(load_universe(copy), frames["quarterly"])
    assert [start for start, _ in problems] == [pd.Timestamp(target["period_start"], tz="UTC")]


def test_coin_sets_are_nested_annual_semiannual_quarterly(
    frames: Mapping[str, pd.DataFrame],
) -> None:
    """Monedas del anual incluidas en las del semestral, y estas en las del trimestral."""
    annual = set(frames["annual"]["symbol"])
    semiannual = set(frames["semiannual"]["symbol"])
    quarterly = set(frames["quarterly"]["symbol"])
    assert annual <= semiannual, sorted(annual - semiannual)
    assert semiannual <= quarterly, sorted(semiannual - quarterly)


def test_default_universe_is_the_quarterly_one(frames: Mapping[str, pd.DataFrame]) -> None:
    """load_universe() sin argumentos es universe.yaml y tiene periodos trimestrales."""
    default = load_universe()
    pd.testing.assert_frame_equal(default, load_universe("universe.yaml"))
    assert default.equals(frames["quarterly"])
    starts = sorted(default["period_start"].unique())
    quarter_starts = {
        pd.Timestamp(p["period_start"], tz="UTC") for p in _raw("universe.yaml")["periods"]
    }
    assert set(starts) <= quarter_starts
    assert all(s.month in (1, 4, 7, 10) and s.day == 1 for s in starts)
    spans = default.drop_duplicates("period_start")
    for start, end in zip(spans["period_start"], spans["period_end"], strict=True):
        assert end == min(start + pd.DateOffset(months=3), LAST_END)


# --- categories.yaml ---


def test_categories_load_and_have_no_asset_in_two_categories() -> None:
    """categories.yaml carga (load_categories rechaza duplicados) y cada activo cuenta una vez."""
    mapping = load_categories()
    raw = yaml.safe_load((CONFIGS_DIR / "categories.yaml").read_text(encoding="utf-8"))
    listed = [str(asset).strip().upper() for assets in raw.values() for asset in assets]
    assert len(listed) == len(set(listed))
    assert len(mapping) == len(listed)
    assert set(DEFAULT_EXCLUDED_CATEGORIES) <= set(raw)


@pytest.mark.parametrize(("asset", "category"), sorted(NEW_ASSET_CATEGORIES.items()))
def test_new_miax_198_assets_have_agreed_category(
    categories: Mapping[str, str], asset: str, category: str
) -> None:
    """Los activos clasificados a mano en MIAX-198 tienen la categoria acordada."""
    assert categories[asset.upper()] == category


def test_new_stablecoins_never_enter_any_universe(frames: Mapping[str, pd.DataFrame]) -> None:
    """USD1, RLUSD y U, clasificados como stablecoin, no aparecen en ningun universo."""
    stable = {f"{a}{QUOTE}" for a, c in NEW_ASSET_CATEGORIES.items() if c == "stablecoin"}
    for name, frame in frames.items():
        assert not (set(frame["symbol"]) & stable), name
