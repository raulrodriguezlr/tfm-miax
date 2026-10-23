"""Tests de la cache del panel en Parquet: ida y vuelta, invalidacion, atomicidad y CLI."""

from __future__ import annotations

import inspect
import json
import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import miax.features.__main__ as cli
import miax.features.panel_store as store_module
from miax.features import (
    FEATURE_NAMES,
    FEATURES_VERSION,
    Panel,
    build_panel,
    build_time_grid,
    period_panels,
)
from miax.features.panel_store import (
    META_FILE,
    cache_fingerprint,
    cached_panel,
    cached_period_panels,
    load_panel,
    panel_path,
    save_panel,
)
from miax.ingest import save_klines
from miax.ingest import store as ingest_store
from miax.ingest.klines import KLINE_COLUMNS, KLINE_DTYPES

MIN_MS = 60_000
T0 = 1_704_067_200_000  # 2024-01-01 00:00 UTC
SMALL = {"vol_window": 3, "activity_window": 5}
START, END = 20, 60  # minutos del rango de los tests


def _ms(minute: int) -> int:
    """Instante en ms de `minute` minutos despues de T0."""
    return T0 + minute * MIN_MS


def _data(seed: int, n: int) -> dict[str, np.ndarray]:
    """Cierre, volumen y operaciones sinteticos, positivos y con semilla fija."""
    rng = np.random.default_rng(seed)
    return {
        "close": 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.001, n))),
        "quote": np.exp(rng.normal(5.0, 1.0, n)),
        "trades": rng.integers(1, 100, n),
    }


def _cache(
    base_dir: Path, symbol: str, data: dict[str, np.ndarray], minutes: Iterable[int]
) -> None:
    """Guarda en `base_dir` las velas de `symbol` solo en los minutos dados."""
    rows = list(minutes)
    open_times = [_ms(k) for k in rows]
    n = len(rows)
    frame = pd.DataFrame(
        {
            "open_time": open_times,
            "open": data["close"][rows],
            "high": data["close"][rows],
            "low": data["close"][rows],
            "close": data["close"][rows],
            "volume": [1.0] * n,
            "close_time": [t + MIN_MS - 1 for t in open_times],
            "quote_asset_volume": data["quote"][rows],
            "num_trades": data["trades"][rows],
            "taker_buy_base": [0.5] * n,
            "taker_buy_quote": [5.0] * n,
            "ignore": [0.0] * n,
        },
        columns=KLINE_COLUMNS,
    ).astype(KLINE_DTYPES)
    save_klines(frame, symbol, "1m", base_dir=base_dir)


def _world(base_dir: Path, n: int = 120) -> None:
    """Tres simbolos: A con hueco (30..35), B con un minuto sin operaciones, C completo."""
    a, b, c = _data(1, n), _data(2, n), _data(3, n)
    _cache(base_dir, "A", a, [k for k in range(n) if not 30 <= k < 36])
    b["trades"] = b["trades"].copy()
    b["trades"][40] = 0
    _cache(base_dir, "B", b, range(n))
    _cache(base_dir, "C", c, range(n))


def _args(symbols: list[str] | None = None) -> tuple[list[str], str, int, int]:
    """Argumentos posicionales de `cached_panel` para el rango de los tests."""
    return (symbols or ["A", "B", "C"]), "1m", _ms(START), _ms(END)


class Counter:
    """Envuelve `build_panel` y cuenta las llamadas."""

    def __init__(self) -> None:
        self.calls = 0
        self._real: Callable[..., Panel] = store_module.build_panel

    def __call__(self, *args: object, **kwargs: object) -> Panel:
        """Cuenta y delega en el `build_panel` real."""
        self.calls += 1
        return self._real(*args, **kwargs)


@pytest.fixture
def counter(monkeypatch: pytest.MonkeyPatch) -> Counter:
    """Cuenta las llamadas a `build_panel` que hace `panel_store`."""
    spy = Counter()
    monkeypatch.setattr(store_module, "build_panel", spy)
    return spy


@pytest.fixture
def world(tmp_path: Path) -> tuple[Path, Path]:
    """Cache de velas sintetica y carpeta (vacia) de la cache del panel."""
    base = tmp_path / "klines"
    _world(base)
    return base, tmp_path / "panels"


def _built(base: Path) -> Panel:
    """Panel de A, B y C construido directamente, con ventanas pequenas."""
    return build_panel(*_args(), base_dir=base, **SMALL)


def _assert_same(first: Panel, second: Panel) -> None:
    """Dos paneles identicos bit a bit (la `freq` del indice no se guarda en Parquet)."""
    assert first.symbols == second.symbols
    assert first.interval == second.interval
    assert tuple(first.features) == tuple(second.features)
    for name in first.features:
        assert_frame_equal(
            first.features[name], second.features[name], check_exact=True, check_freq=False
        )
    assert_frame_equal(first.valid, second.valid, check_exact=True, check_freq=False)


def _saved(world: tuple[Path, Path]) -> tuple[Panel, Path]:
    """Construye, guarda y devuelve el panel y su directorio."""
    base, cache = world
    panel = _built(base)
    fingerprint = cache_fingerprint(panel.symbols, "1m", base_dir=base)
    directory = save_panel(
        panel,
        _ms(START),
        _ms(END),
        params=SMALL,
        fingerprint=fingerprint,
        cache_dir=cache,
    )
    return panel, directory


# --- ida y vuelta ------------------------------------------------------------------------------


def test_round_trip_is_bit_exact(world: tuple[Path, Path]) -> None:
    """Guardar y cargar devuelve lo mismo: dtypes, indice UTC, orden y huecos en NaN."""
    panel, directory = _saved(world)
    loaded = load_panel(directory)
    _assert_same(panel, loaded)
    grid = build_time_grid(_ms(START), _ms(END), "1m")
    assert isinstance(loaded.index, pd.DatetimeIndex)
    assert str(loaded.index.tz) == "UTC"
    assert loaded.index.equals(grid)
    assert loaded.symbols == ("A", "B", "C")
    assert tuple(loaded.features) == FEATURE_NAMES
    for frame in loaded.features.values():
        assert all(dtype == "float64" for dtype in frame.dtypes)
        assert list(frame.columns) == ["A", "B", "C"]
    assert all(dtype == bool for dtype in loaded.valid.dtypes)


def test_gaps_stay_nan_after_round_trip(world: tuple[Path, Path]) -> None:
    """Un hueco sigue siendo NaN tras guardar y cargar: nunca se convierte en 0."""
    panel, directory = _saved(world)
    loaded = load_panel(directory)
    invalid = ~panel.valid.to_numpy()
    assert invalid.any()
    for name in ("return", "rel_volume", "rel_trades"):
        before = panel.features[name].to_numpy()
        after = loaded.features[name].to_numpy()
        assert np.isnan(after[invalid]).all()
        np.testing.assert_array_equal(np.isnan(before), np.isnan(after))
        assert int((after == 0).sum()) == int((before == 0).sum())


def test_partial_load_respects_requested_order(world: tuple[Path, Path]) -> None:
    """Se cargan solo las features y simbolos pedidos, en el orden pedido, y `valid` acompana."""
    panel, directory = _saved(world)
    loaded = load_panel(directory, features=["rel_volume", "return"], symbols=["C", "A"])
    assert loaded.symbols == ("C", "A")
    assert tuple(loaded.features) == ("rel_volume", "return")
    for name in ("rel_volume", "return"):
        assert_frame_equal(
            loaded.features[name],
            panel.features[name][["C", "A"]],
            check_exact=True,
            check_freq=False,
        )
    assert_frame_equal(loaded.valid, panel.valid[["C", "A"]], check_exact=True, check_freq=False)


@pytest.mark.parametrize("kwargs", [{"features": ["nope"]}, {"symbols": ["ZZZ"]}])
def test_load_unknown_name_raises(world: tuple[Path, Path], kwargs: dict[str, list[str]]) -> None:
    """Una feature o simbolo desconocido lanza `ValueError`."""
    _, directory = _saved(world)
    with pytest.raises(ValueError, match="desconocidos"):
        load_panel(directory, **kwargs)


# --- caches rotas ------------------------------------------------------------------------------


def _remove_meta(directory: Path) -> None:
    (directory / META_FILE).unlink()


def _remove_parquet(directory: Path) -> None:
    (directory / "rel_volume.parquet").unlink()


def _corrupt_parquet(directory: Path) -> None:
    (directory / "return.parquet").write_bytes(b"esto no es un parquet" * 10)


def _drop_column(directory: Path) -> None:
    path = directory / "realized_vol.parquet"
    frame = pd.read_parquet(path)
    frame.drop(columns=["B"]).to_parquet(path)


def _corrupt_valid(directory: Path) -> None:
    (directory / "valid.parquet").write_bytes(b"\x00\x01basura")


BREAKERS = [_remove_meta, _remove_parquet, _corrupt_parquet, _drop_column, _corrupt_valid]


@pytest.mark.parametrize("breaker", BREAKERS, ids=lambda f: f.__name__)
def test_broken_cache_raises_value_error_on_load(
    world: tuple[Path, Path], breaker: Callable[[Path], None]
) -> None:
    """Sin meta, con un Parquet borrado, corrupto o sin columna, `load_panel` da `ValueError`."""
    _, directory = _saved(world)
    breaker(directory)
    with pytest.raises(ValueError):
        load_panel(directory)


@pytest.mark.parametrize("breaker", BREAKERS, ids=lambda f: f.__name__)
def test_cached_panel_rebuilds_a_broken_cache(
    world: tuple[Path, Path], counter: Counter, breaker: Callable[[Path], None]
) -> None:
    """Con la cache rota, `cached_panel` reconstruye sin traceback y deja la cache sana."""
    base, cache = world
    first = cached_panel(*_args(), base_dir=base, cache_dir=cache, **SMALL)
    assert counter.calls == 1
    directory = panel_path("1m", _ms(START), _ms(END), cache_dir=cache)
    breaker(directory)
    second = cached_panel(*_args(), base_dir=base, cache_dir=cache, **SMALL)
    assert counter.calls == 2
    _assert_same(first, second)
    _assert_same(first, load_panel(directory))
    cached_panel(*_args(), base_dir=base, cache_dir=cache, **SMALL)
    assert counter.calls == 2


# --- version del calculo de features -----------------------------------------------------------


def _set_features_version(directory: Path, value: object) -> None:
    """Reescribe `features_version` del meta; si `value` es `...`, quita la clave."""
    path = directory / META_FILE
    meta = json.loads(path.read_text(encoding="utf-8"))
    if value is ...:
        meta.pop("features_version")
    else:
        meta["features_version"] = value
    path.write_text(json.dumps(meta), encoding="utf-8")


def test_meta_stores_features_version(world: tuple[Path, Path]) -> None:
    """`save_panel` guarda `features_version` con el valor actual en el meta."""
    _, directory = _saved(world)
    meta = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
    assert meta["features_version"] == FEATURES_VERSION


@pytest.mark.parametrize("stale", [FEATURES_VERSION + 1, FEATURES_VERSION - 1, "x", None])
def test_load_rejects_other_features_version(world: tuple[Path, Path], stale: object) -> None:
    """Una cache con otra `features_version` da `ValueError`."""
    _, directory = _saved(world)
    _set_features_version(directory, stale)
    with pytest.raises(ValueError, match="features"):
        load_panel(directory)


def test_load_rejects_meta_without_features_version(world: tuple[Path, Path]) -> None:
    """Un meta antiguo sin la clave `features_version` da `ValueError`."""
    _, directory = _saved(world)
    _set_features_version(directory, ...)
    with pytest.raises(ValueError, match="features"):
        load_panel(directory)


@pytest.mark.parametrize("stale", [FEATURES_VERSION + 1, ...], ids=["other", "missing"])
def test_cached_panel_rebuilds_on_other_features_version(
    world: tuple[Path, Path], counter: Counter, stale: object
) -> None:
    """Con otra version (o sin ella), `cached_panel` reconstruye y deja la cache sana."""
    base, cache = world
    first = cached_panel(*_args(), base_dir=base, cache_dir=cache, **SMALL)
    assert counter.calls == 1
    directory = panel_path("1m", _ms(START), _ms(END), cache_dir=cache)
    _set_features_version(directory, stale)
    second = cached_panel(*_args(), base_dir=base, cache_dir=cache, **SMALL)
    assert counter.calls == 2
    _assert_same(first, second)
    meta = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
    assert meta["features_version"] == FEATURES_VERSION
    _assert_same(first, load_panel(directory))
    cached_panel(*_args(), base_dir=base, cache_dir=cache, **SMALL)
    assert counter.calls == 2


def test_cached_panel_rebuilds_when_code_version_changes(
    world: tuple[Path, Path], counter: Counter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Subir `FEATURES_VERSION` en el codigo invalida la cache ya guardada."""
    base, cache = world
    cached_panel(*_args(), base_dir=base, cache_dir=cache, **SMALL)
    monkeypatch.setattr(store_module, "FEATURES_VERSION", FEATURES_VERSION + 1)
    cached_panel(*_args(), base_dir=base, cache_dir=cache, **SMALL)
    assert counter.calls == 2
    directory = panel_path("1m", _ms(START), _ms(END), cache_dir=cache)
    meta = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
    assert meta["features_version"] == FEATURES_VERSION + 1
    cached_panel(*_args(), base_dir=base, cache_dir=cache, **SMALL)
    assert counter.calls == 2


def test_cached_period_panels_rebuild_on_other_features_version(
    tmp_path: Path, counter: Counter
) -> None:
    """Con otra version en el disco, una pasada de `cached_period_panels` reconstruye todo."""
    base, cache = tmp_path / "klines", tmp_path / "panels"
    for symbol, seed in (("A", 1), ("B", 2), ("C", 3)):
        _cache(base, symbol, _data(seed, 3000), range(3000))
    kwargs = {"base_dir": base, "cache_dir": cache, **SMALL}
    first = list(cached_period_panels(_universe(), "1m", **kwargs))
    assert counter.calls == 2
    for folder in (cache / "1m").iterdir():
        _set_features_version(folder, FEATURES_VERSION + 1)
    second = list(cached_period_panels(_universe(), "1m", **kwargs))
    assert counter.calls == 4
    for (_, want), (_, got) in zip(first, second, strict=True):
        _assert_same(want, got)
    for folder in (cache / "1m").iterdir():
        meta = json.loads((folder / META_FILE).read_text(encoding="utf-8"))
        assert meta["features_version"] == FEATURES_VERSION
    list(cached_period_panels(_universe(), "1m", **kwargs))
    assert counter.calls == 4


# --- cached_panel ------------------------------------------------------------------------------


def test_second_call_loads_from_cache(world: tuple[Path, Path], counter: Counter) -> None:
    """La primera vez construye; la segunda carga sin llamar a `build_panel`."""
    base, cache = world
    first = cached_panel(*_args(), base_dir=base, cache_dir=cache, **SMALL)
    assert counter.calls == 1
    second = cached_panel(*_args(), base_dir=base, cache_dir=cache, **SMALL)
    assert counter.calls == 1
    _assert_same(first, second)
    _assert_same(first, _built(base))


@pytest.mark.parametrize(
    "change",
    [
        {"vol_window": 4},
        {"activity_window": 6},
        {"vol_min_valid_share": 0.7},
        {"activity_min_valid_share": 0.6},
    ],
)
def test_changing_a_param_rebuilds(
    world: tuple[Path, Path], counter: Counter, change: dict[str, float]
) -> None:
    """Cambiar cualquier parametro de ventana reconstruye."""
    base, cache = world
    cached_panel(*_args(), base_dir=base, cache_dir=cache, **SMALL)
    cached_panel(*_args(), base_dir=base, cache_dir=cache, **{**SMALL, **change})
    assert counter.calls == 2


def test_changing_symbols_rebuilds(world: tuple[Path, Path], counter: Counter) -> None:
    """Cambiar los simbolos (o su orden) reconstruye."""
    base, cache = world
    cached_panel(*_args(["A", "B"]), base_dir=base, cache_dir=cache, **SMALL)
    cached_panel(*_args(["A"]), base_dir=base, cache_dir=cache, **SMALL)
    assert counter.calls == 2
    cached_panel(*_args(["B", "A"]), base_dir=base, cache_dir=cache, **SMALL)
    assert counter.calls == 3


def test_candle_added_outside_the_range_rebuilds(
    world: tuple[Path, Path], counter: Counter
) -> None:
    """Una vela nueva en la cache de un simbolo, fuera del rango, invalida el panel."""
    base, cache = world
    cached_panel(*_args(), base_dir=base, cache_dir=cache, **SMALL)
    # Minuto 500 (fuera de [20, 60)); la vela del mismo mes se fusiona y cambia el tamano.
    extra = _data(1, 600)
    _cache(base, "A", extra, [500])
    cached_panel(*_args(), base_dir=base, cache_dir=cache, **SMALL)
    assert counter.calls == 2


def test_rebuild_flag_always_rebuilds(world: tuple[Path, Path], counter: Counter) -> None:
    """Con `rebuild=True` se construye aunque la cache valga."""
    base, cache = world
    cached_panel(*_args(), base_dir=base, cache_dir=cache, **SMALL)
    cached_panel(*_args(), base_dir=base, cache_dir=cache, rebuild=True, **SMALL)
    cached_panel(*_args(), base_dir=base, cache_dir=cache, rebuild=True, **SMALL)
    assert counter.calls == 3


def test_touching_mtime_does_not_rebuild(world: tuple[Path, Path], counter: Counter) -> None:
    """Cambiar solo el mtime de los ficheros de velas no invalida la cache."""
    base, cache = world
    cached_panel(*_args(), base_dir=base, cache_dir=cache, **SMALL)
    for path in base.rglob("*.parquet"):
        stat = path.stat()
        os.utime(path, (stat.st_atime + 10_000, stat.st_mtime + 10_000))
    cached_panel(*_args(), base_dir=base, cache_dir=cache, **SMALL)
    assert counter.calls == 1


def test_cached_panel_collapses_duplicate_symbols(
    world: tuple[Path, Path], counter: Counter
) -> None:
    """Un simbolo repetido no duplica columnas ni rompe la huella."""
    base, cache = world
    panel = cached_panel(*_args(["B", "A", "B"]), base_dir=base, cache_dir=cache, **SMALL)
    assert panel.symbols == ("B", "A")
    cached_panel(*_args(["B", "A", "B"]), base_dir=base, cache_dir=cache, **SMALL)
    assert counter.calls == 1


# --- save_panel --------------------------------------------------------------------------------


def test_save_rejects_index_that_is_not_the_grid(world: tuple[Path, Path]) -> None:
    """Un indice que no es la rejilla del rango lanza `ValueError` y no escribe nada."""
    base, cache = world
    panel = _built(base)
    fingerprint = cache_fingerprint(panel.symbols, "1m", base_dir=base)
    with pytest.raises(ValueError, match="rejilla"):
        save_panel(
            panel, _ms(START), _ms(END + 1), params=SMALL, fingerprint=fingerprint, cache_dir=cache
        )
    assert not cache.exists() or list(cache.rglob("*")) == []


def test_save_rejects_fingerprint_with_other_symbols(world: tuple[Path, Path]) -> None:
    """Una huella con simbolos distintos lanza `ValueError` y no deja Parquets escritos."""
    base, cache = world
    panel = _built(base)
    wrong = cache_fingerprint(["A", "B"], "1m", base_dir=base)
    with pytest.raises(ValueError, match="huella"):
        save_panel(panel, _ms(START), _ms(END), params=SMALL, fingerprint=wrong, cache_dir=cache)
    assert list(cache.rglob("*.parquet")) == [] if cache.exists() else True
    assert not list(cache.rglob(META_FILE)) if cache.exists() else True


def test_numpy_scalars_in_params_are_serialized(world: tuple[Path, Path]) -> None:
    """Los escalares de numpy en `params` se guardan como numeros nativos de JSON."""
    base, cache = world
    panel = _built(base)
    params = {
        "vol_window": np.int64(3),
        "vol_min_valid_share": np.float64(0.8),
        "flag": np.bool_(True),
    }
    fingerprint = cache_fingerprint(panel.symbols, "1m", base_dir=base)
    directory = save_panel(
        panel, _ms(START), _ms(END), params=params, fingerprint=fingerprint, cache_dir=cache
    )
    meta = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
    assert meta["params"] == {"vol_window": 3, "vol_min_valid_share": 0.8, "flag": True}


def test_non_serializable_params_leave_nothing_written(world: tuple[Path, Path]) -> None:
    """Un param no serializable falla antes de escribir ningun Parquet."""
    base, cache = world
    panel = _built(base)
    fingerprint = cache_fingerprint(panel.symbols, "1m", base_dir=base)
    with pytest.raises(TypeError):
        save_panel(
            panel,
            _ms(START),
            _ms(END),
            params={"x": object()},
            fingerprint=fingerprint,
            cache_dir=cache,
        )
    assert not cache.exists() or list(cache.rglob("*.parquet")) == []


def test_meta_is_written_last(world: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    """Si falla un `write_parquet`, no queda `meta.json`: la carpeta no es cache valida."""
    base, cache = world
    panel = _built(base)
    fingerprint = cache_fingerprint(panel.symbols, "1m", base_dir=base)
    real = store_module.write_parquet
    calls = {"n": 0}

    def flaky(frame: pd.DataFrame, path: str | Path, *a: object, **k: object) -> Path:
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError("disco lleno")
        return real(frame, path, *a, **k)

    monkeypatch.setattr(store_module, "write_parquet", flaky)
    with pytest.raises(OSError, match="disco lleno"):
        save_panel(
            panel, _ms(START), _ms(END), params=SMALL, fingerprint=fingerprint, cache_dir=cache
        )
    directory = panel_path("1m", _ms(START), _ms(END), cache_dir=cache)
    assert not (directory / META_FILE).exists()
    with pytest.raises(ValueError):
        load_panel(directory)


def test_failed_resave_invalidates_the_previous_cache(
    world: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Si falla un re-guardado, el meta anterior ya no esta: no se mezclan versiones."""
    panel, directory = _saved(world)
    base, cache = world
    fingerprint = cache_fingerprint(panel.symbols, "1m", base_dir=base)

    def boom(*a: object, **k: object) -> Path:
        raise OSError("fallo")

    monkeypatch.setattr(store_module, "write_parquet", boom)
    with pytest.raises(OSError):
        save_panel(
            panel, _ms(START), _ms(END), params=SMALL, fingerprint=fingerprint, cache_dir=cache
        )
    assert not (directory / META_FILE).exists()
    with pytest.raises(ValueError):
        load_panel(directory)


# --- _stamp / panel_path -----------------------------------------------------------------------


def test_panel_path_distinguishes_minute_second_and_ms_bounds(tmp_path: Path) -> None:
    """Limites en minuto, segundo y ms exactos dan carpetas distintas."""
    start = _ms(0)
    ends = [_ms(10), _ms(10) + 1_000, _ms(10) + 1, _ms(10) + 999]
    paths = [panel_path("1m", start, end, cache_dir=tmp_path) for end in ends]
    assert len({p.name for p in paths}) == len(ends)
    # Un limite a medianoche usa solo la fecha.
    assert panel_path("1m", T0, T0 + 86_400_000, cache_dir=tmp_path).name == (
        "2024-01-01_2024-01-02"
    )
    assert panel_path("1m", _ms(1), _ms(10), cache_dir=tmp_path).parent.name == "1m"


# --- cached_period_panels ----------------------------------------------------------------------


def _universe() -> pd.DataFrame:
    """Universo sintetico de dos periodos de un dia con simbolos distintos."""
    rows = [
        ("2024-01-01", "2024-01-02", "A"),
        ("2024-01-01", "2024-01-02", "B"),
        ("2024-01-02", "2024-01-03", "B"),
        ("2024-01-02", "2024-01-03", "C"),
    ]
    frame = pd.DataFrame(rows, columns=["period_start", "period_end", "symbol"])
    for column in ("period_start", "period_end"):
        frame[column] = pd.to_datetime(frame[column], utc=True).astype("datetime64[us, UTC]")
    return frame


def test_cached_period_panels_match_period_panels(tmp_path: Path, counter: Counter) -> None:
    """Dan los mismos paneles que `period_panels`, y la segunda pasada carga de cache."""
    base, cache = tmp_path / "klines", tmp_path / "panels"
    for symbol, seed in (("A", 1), ("B", 2), ("C", 3)):
        _cache(base, symbol, _data(seed, 3000), range(3000))
    expected = list(period_panels(_universe(), "1m", base_dir=base, **SMALL))
    first = list(cached_period_panels(_universe(), "1m", base_dir=base, cache_dir=cache, **SMALL))
    assert counter.calls == 2
    second = list(cached_period_panels(_universe(), "1m", base_dir=base, cache_dir=cache, **SMALL))
    assert counter.calls == 2
    for run in (first, second):
        assert [s for s, _ in run] == [s for s, _ in expected]
        for (_, got), (_, want) in zip(run, expected, strict=True):
            _assert_same(want, got)


def test_cached_period_panels_validates_columns() -> None:
    """Faltan columnas del universo: `ValueError` que las nombra."""
    with pytest.raises(ValueError, match="period_end"):
        list(cached_period_panels(_universe().drop(columns=["period_end"]), "1m"))


# --- defaults ----------------------------------------------------------------------------------


@pytest.mark.parametrize("function", [cached_panel, cached_period_panels])
def test_defaults_match_build_panel(function: Callable[..., Any]) -> None:
    """Los defaults de ventana de la cache son los de `build_panel`."""
    own = inspect.signature(function).parameters
    reference = inspect.signature(build_panel).parameters
    for name in (
        "vol_window",
        "vol_min_valid_share",
        "activity_window",
        "activity_min_valid_share",
    ):
        assert own[name].default == reference[name].default


# --- CLI ---------------------------------------------------------------------------------------


def _write_universe(path: Path) -> Path:
    """YAML de universo con dos periodos de un dia."""
    lines = ["periods:"]
    for start, end, symbols in (
        ("2024-01-01", "2024-01-02", ["A", "B"]),
        ("2024-01-02", "2024-01-03", ["B", "C"]),
    ):
        lines += [f"  - period_start: '{start}'", f"    period_end: '{end}'", "    pairs:"]
        for symbol in symbols:
            lines += [
                f"      - symbol: {symbol}",
                "        category: test",
                "        median_quote_volume: 1.0",
                "        first_candle: '2023-01-01'",
            ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def cli_env(tmp_path: Path) -> dict[str, Path]:
    """Cache de velas, carpeta de paneles y universo YAML, todo en `tmp_path`."""
    base = tmp_path / "klines"
    for symbol, seed in (("A", 1), ("B", 2), ("C", 3)):
        _cache(base, symbol, _data(seed, 2900), range(2900))
    return {
        "base": base,
        "cache": tmp_path / "panels",
        "universe": _write_universe(tmp_path / "universe.yaml"),
    }


def _argv(env: dict[str, Path], *extra: str) -> list[str]:
    return [
        "panel",
        "--base-dir",
        str(env["base"]),
        "--cache-dir",
        str(env["cache"]),
        "--universe-file",
        str(env["universe"]),
        *extra,
    ]


def test_cli_builds_all_periods(
    cli_env: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """Sin `--period` crea el panel de cada periodo; el stdout queda vacio (sin print)."""
    assert cli.main(_argv(cli_env)) == 0
    assert capsys.readouterr().out == ""
    folders = sorted(p.name for p in (cli_env["cache"] / "1m").iterdir())
    assert folders == ["2024-01-01_2024-01-02", "2024-01-02_2024-01-03"]
    for folder in folders:
        assert (cli_env["cache"] / "1m" / folder / META_FILE).is_file()
        assert (cli_env["cache"] / "1m" / folder / "return.parquet").is_file()


def test_cli_single_period(cli_env: dict[str, Path]) -> None:
    """Con `--period` solo se crea ese periodo."""
    assert cli.main(_argv(cli_env, "--period", "2024-01-02")) == 0
    folders = [p.name for p in (cli_env["cache"] / "1m").iterdir()]
    assert folders == ["2024-01-02_2024-01-03"]


@pytest.mark.parametrize("period", ["2030-01-01", "no-es-fecha", "2024-1-1x"])
def test_cli_bad_period_exits_2(cli_env: dict[str, Path], period: str) -> None:
    """Un periodo inexistente o mal escrito da codigo 2 y no escribe nada."""
    assert cli.main(_argv(cli_env, "--period", period)) == 2
    assert not cli_env["cache"].exists()


@pytest.mark.parametrize("interval", ["5m", "1M"])
def test_cli_bad_interval_exits_2(cli_env: dict[str, Path], interval: str) -> None:
    """Un intervalo distinto de 1m o sin duracion fija da codigo 2."""
    assert cli.main(_argv(cli_env, "--interval", interval)) == 2


def test_cli_missing_universe_exits_2(cli_env: dict[str, Path]) -> None:
    """Un fichero de universo inexistente da codigo 2."""
    cli_env["universe"].unlink()
    assert cli.main(_argv(cli_env)) == 2


def test_cli_build_failure_exits_1(
    cli_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Un fallo al construir da codigo 1, sin traceback."""

    def boom(*args: object, **kwargs: object) -> NoReturn:
        raise RuntimeError("fallo simulado")

    monkeypatch.setattr(cli, "cached_period_panels", boom)
    assert cli.main(_argv(cli_env)) == 1


def test_cli_rebuild_flag(cli_env: dict[str, Path], counter: Counter) -> None:
    """Una segunda ejecucion carga de cache; con `--rebuild` vuelve a construir."""
    assert cli.main(_argv(cli_env)) == 0
    assert counter.calls == 2
    assert cli.main(_argv(cli_env)) == 0
    assert counter.calls == 2
    assert cli.main(_argv(cli_env, "--rebuild")) == 0
    assert counter.calls == 4


# --- refactor de store -------------------------------------------------------------------------


def test_month_files_alias() -> None:
    """`_month_files` sigue siendo la funcion publica `month_files`."""
    assert ingest_store._month_files is ingest_store.month_files
