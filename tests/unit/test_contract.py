"""Tests del contrato A -> B: esquema de features, `validate_panel` y su enganche en el panel."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import miax.features.panel as panel_module
import miax.features.panel_store as store_module
from miax.features import (
    FEATURE_NAMES,
    FEATURE_SPECS,
    FeatureSpec,
    Panel,
    build_panel,
    build_time_grid,
    cached_panel,
    load_panel,
    panel_path,
    validate_panel,
)
from miax.ingest import save_klines
from miax.ingest.klines import KLINE_COLUMNS, KLINE_DTYPES
from miax.utils.io import read_parquet, write_parquet

MIN_MS = 60_000
T0 = 1_704_067_200_000  # 2024-01-01 00:00 UTC
SMALL = {"vol_window": 3, "activity_window": 5}
START, END = 20, 60  # minutos del rango de los tests con cache sintetica
ROWS = 6  # filas de los paneles hechos a mano
SYMBOLS = ("A", "B")
REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_DOC = REPO_ROOT / "docs" / "datos" / "contrato-a-b.md"


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
    """Argumentos posicionales de `build_panel` y `cached_panel` para el rango de los tests."""
    return (symbols or ["A", "B", "C"]), "1m", _ms(START), _ms(END)


@pytest.fixture
def world(tmp_path: Path) -> tuple[Path, Path]:
    """Cache de velas sintetica y carpeta (vacia) de la cache del panel."""
    base = tmp_path / "klines"
    _world(base)
    return base, tmp_path / "panels"


class Counter:
    """Envuelve `build_panel` de `panel_store` y cuenta las llamadas."""

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


# --- panel hecho a mano ------------------------------------------------------------------------


def _index(rows: int = ROWS) -> pd.DatetimeIndex:
    """Rejilla de 1 minuto en ms y UTC, empezando en T0 + 1 minuto."""
    return build_time_grid(_ms(1), _ms(1 + rows), "1m")


def _frame(index: pd.DatetimeIndex, values: np.ndarray) -> pd.DataFrame:
    """DataFrame `float64` con las columnas `SYMBOLS` y el indice dado."""
    return pd.DataFrame(values.astype("float64"), index=index, columns=list(SYMBOLS))


def _arrays(rows: int = ROWS) -> dict[str, np.ndarray]:
    """Arrays que cumplen el contrato: todo valido, finito, y `realized_vol` positivo."""
    rng = np.random.default_rng(7)
    shape = (rows, len(SYMBOLS))
    return {
        "return": rng.normal(0.0, 0.001, shape),
        "realized_vol": rng.uniform(0.001, 0.01, shape),
        "rel_volume": rng.normal(0.0, 1.0, shape),
        "rel_trades": rng.normal(0.0, 1.0, shape),
        "valid": np.ones(shape, dtype=bool),
    }


def _panel(
    arrays: dict[str, np.ndarray] | None = None,
    *,
    index: pd.DatetimeIndex | None = None,
    names: tuple[str, ...] = FEATURE_NAMES,
    extra: tuple[str, ...] = (),
) -> Panel:
    """Panel a mano con los arrays dados (por defecto, uno que cumple el contrato)."""
    data = _arrays() if arrays is None else arrays
    idx = _index(len(data["valid"])) if index is None else index
    features = {name: _frame(idx, data[name]) for name in names}
    for name in extra:
        features[name] = _frame(idx, data["return"])
    valid = pd.DataFrame(data["valid"].astype(bool), index=idx, columns=list(SYMBOLS))
    return Panel(features=features, valid=valid, interval="1m", symbols=SYMBOLS)


def _assert_same(first: Panel, second: Panel) -> None:
    """Dos paneles identicos bit a bit (la `freq` del indice no se guarda en Parquet)."""
    assert first.symbols == second.symbols
    assert tuple(first.features) == tuple(second.features)
    for name in first.features:
        assert_frame_equal(
            first.features[name], second.features[name], check_exact=True, check_freq=False
        )
    assert_frame_equal(first.valid, second.valid, check_exact=True, check_freq=False)


# --- 1. esquema --------------------------------------------------------------------------------


def test_specs_have_same_keys_and_order_as_feature_names() -> None:
    """`FEATURE_SPECS` tiene las claves de `FEATURE_NAMES`, en el mismo orden, y `name` coincide."""
    assert tuple(FEATURE_SPECS) == FEATURE_NAMES
    for key, spec in FEATURE_SPECS.items():
        assert isinstance(spec, FeatureSpec)
        assert spec.name == key
        assert spec.unit
        assert spec.description


def test_spec_flags() -> None:
    """`realized_vol` admite valor con `valid=False` y es no negativa; las otras son NaN ahi."""
    vol = FEATURE_SPECS["realized_vol"]
    assert not vol.nan_if_invalid
    assert vol.non_negative
    for name in ("return", "rel_volume", "rel_trades"):
        assert FEATURE_SPECS[name].nan_if_invalid


def test_specs_are_read_only() -> None:
    """`FEATURE_SPECS` no se puede modificar desde fuera."""
    with pytest.raises(TypeError):
        FEATURE_SPECS["otra"] = FEATURE_SPECS["return"]  # type: ignore[index]


# --- 2. paneles reales pasan -------------------------------------------------------------------


def test_built_panel_passes_with_gaps_and_zero_trades(world: tuple[Path, Path]) -> None:
    """Un panel construido sobre cache con hueco y velas sin operaciones cumple el contrato."""
    base, _ = world
    panel = build_panel(*_args(), base_dir=base, **SMALL)
    assert not panel.valid.to_numpy().all()  # el hueco y el minuto sin operaciones existen
    validate_panel(panel)


def test_built_panel_with_symbol_without_cache_passes(world: tuple[Path, Path]) -> None:
    """Un simbolo sin cache (columna NaN, `valid` False) no viola el contrato."""
    base, _ = world
    panel = build_panel(*_args(["A", "ZZZ", "C"]), base_dir=base, **SMALL)
    assert panel.features["return"]["ZZZ"].isna().all()
    assert not panel.valid["ZZZ"].any()
    validate_panel(panel)


def test_built_panel_of_one_row_passes(world: tuple[Path, Path]) -> None:
    """Un panel de una sola fila cumple el contrato."""
    base, _ = world
    panel = build_panel(["A", "B", "C"], "1m", _ms(START), _ms(START + 1), base_dir=base, **SMALL)
    assert len(panel.index) == 1
    validate_panel(panel)


def test_loaded_panel_with_feature_and_symbol_subset_passes(world: tuple[Path, Path]) -> None:
    """`load_panel` con un subconjunto de features y de simbolos cumple el contrato."""
    base, cache = world
    cached_panel(*_args(), base_dir=base, cache_dir=cache, **SMALL)
    directory = panel_path("1m", _ms(START), _ms(END), cache_dir=cache)
    loaded = load_panel(directory, features=["return"], symbols=["C", "A"])
    assert tuple(loaded.features) == ("return",)
    assert loaded.symbols == ("C", "A")
    validate_panel(loaded)
    only_vol = load_panel(directory, features=["realized_vol", "rel_trades"], symbols=["B"])
    validate_panel(only_vol)


def test_handmade_valid_panel_passes() -> None:
    """El panel de referencia hecho a mano cumple el contrato."""
    validate_panel(_panel())


def test_subset_of_features_is_allowed() -> None:
    """Un panel con solo `return` es valido: el contrato admite subconjuntos."""
    validate_panel(_panel(names=("return",)))


def test_all_nan_and_invalid_panel_passes() -> None:
    """Todo NaN con `valid` False en todas partes cumple el contrato."""
    data = _arrays()
    for name in FEATURE_NAMES:
        data[name] = np.full_like(data[name], np.nan)
    data["valid"] = np.zeros_like(data["valid"], dtype=bool)
    validate_panel(_panel(data))


def test_one_row_panel_skips_step_check() -> None:
    """Con una sola fila no hay paso que comprobar."""
    data = {name: values[:1] for name, values in _arrays().items()}
    validate_panel(_panel(data))


def test_empty_panel_passes() -> None:
    """Un panel sin filas no viola nada."""
    data = {name: values[:0] for name, values in _arrays().items()}
    validate_panel(_panel(data, index=_index()[:0]))


# --- 3. realized_vol con valor donde valid=False -----------------------------------------------


def test_realized_vol_with_value_on_invalid_candle_is_not_a_violation() -> None:
    """`realized_vol` puede tener valor con `valid=False`: no es violacion."""
    data = _arrays()
    data["valid"][3, 0] = False
    for name in ("return", "rel_volume", "rel_trades"):
        data[name][3, 0] = np.nan
    data["return"][4, 0] = np.nan  # el retorno siguiente tampoco existe
    assert not np.isnan(data["realized_vol"][3, 0])
    validate_panel(_panel(data))


# --- 4. cada violacion por separado ------------------------------------------------------------


def _message(panel: Panel) -> str:
    """Mensaje del `ValueError` que lanza `validate_panel`."""
    with pytest.raises(ValueError, match="contrato A -> B") as info:
        validate_panel(panel)
    return str(info.value)


def test_violation_a_unknown_feature() -> None:
    """(a) Una feature fuera de `FEATURE_SPECS` se nombra en el mensaje."""
    message = _message(_panel(extra=("momentum",)))
    assert "fuera del contrato" in message
    assert "momentum" in message


@pytest.mark.parametrize("unit", ["ns", "us", "s"])
def test_violation_b_index_not_ms(unit: str) -> None:
    """(b) Un indice UTC que no esta en ms se rechaza y el mensaje dice el dtype."""
    index = _index().as_unit(unit)  # type: ignore[arg-type]
    message = _message(_panel(index=index))
    assert "indice" in message
    assert f"datetime64[{unit}, UTC]" in message
    assert "datetime64[ms, UTC]" in message


def test_violation_c_irregular_step() -> None:
    """(c) Un salto en el indice (una fila menos) se detecta y se cuenta."""
    data = _arrays()
    index = _index().delete(3)
    data = {name: np.delete(values, 3, axis=0) for name, values in data.items()}
    message = _message(_panel(data, index=index))
    assert "indice: 1 saltos" in message
    assert "1m" in message


def test_regular_but_not_clock_aligned_index_passes() -> None:
    """Un indice de paso fijo pero no alineado al reloj es valido (no se exige alineacion)."""
    index = pd.DatetimeIndex(
        [pd.Timestamp(_ms(1) + 17_000 + k * MIN_MS, unit="ms", tz="UTC") for k in range(ROWS)]
    ).as_unit("ms")
    validate_panel(_panel(index=index))


@pytest.mark.parametrize("name", FEATURE_NAMES)
@pytest.mark.parametrize("bad", [np.inf, -np.inf])
def test_violation_d_infinite_values(name: str, bad: float) -> None:
    """(d) Un +-inf en cualquier feature se detecta; el mensaje da feature, instante y simbolo."""
    data = _arrays()
    data[name][3, 1] = bad
    panel = _panel(data)
    message = _message(panel)
    assert f"{name}: valores infinitos: 1 celdas" in message
    assert str(panel.index[3]) in message
    assert "(B)" in message


@pytest.mark.parametrize("name", ["return", "rel_volume", "rel_trades"])
def test_violation_e_value_on_invalid_candle(name: str) -> None:
    """(e) Valor no NaN donde `valid=False` en una feature `nan_if_invalid` se detecta."""
    data = _arrays()
    data["valid"][2, 0] = False
    for other in ("return", "rel_volume", "rel_trades"):
        if other != name:
            data[other][2, 0] = np.nan
    data["return"][3, 0] = np.nan  # el retorno siguiente tampoco existe
    panel = _panel(data)
    message = _message(panel)
    assert f"{name}: valor no NaN en velas no validas: 1 celdas" in message
    assert str(panel.index[2]) in message
    assert "(A)" in message


def test_violation_f_return_bridges_a_gap() -> None:
    """(f) Un retorno con valor cuya vela anterior no es valida puentea el hueco."""
    data = _arrays()
    data["valid"][2, 1] = False
    for name in ("return", "rel_volume", "rel_trades"):
        data[name][2, 1] = np.nan
    # return[3, 1] conserva valor: la vela 3 es valida pero la 2 no.
    panel = _panel(data)
    message = _message(panel)
    assert "return: puentea un hueco" in message
    assert "1 celdas" in message
    assert str(panel.index[3]) in message
    assert "(B)" in message
    assert "valor no NaN en velas no validas" not in message


def test_violation_g_negative_realized_vol() -> None:
    """(g) Una volatilidad negativa se detecta."""
    data = _arrays()
    data["realized_vol"][4, 0] = -0.001
    panel = _panel(data)
    message = _message(panel)
    assert "realized_vol: valores negativos: 1 celdas" in message
    assert str(panel.index[4]) in message


def test_zero_realized_vol_is_not_negative() -> None:
    """Una volatilidad exactamente 0 es valida (por ejemplo, precio plano)."""
    data = _arrays()
    data["realized_vol"][:, :] = 0.0
    validate_panel(_panel(data))


# --- 5. puente de huecos: antifuga -------------------------------------------------------------


def test_bridge_check_skips_first_row_even_with_a_return() -> None:
    """La primera fila no se marca como puente aunque `return` tenga valor (vela previa fuera)."""
    data = _arrays()
    data["valid"][0, 0] = False  # el panel empieza con una vela invalida
    # return[0, 0] conserva valor: solo viola (e), nunca (f), porque su vela anterior no existe.
    assert not np.isnan(data["return"][0, 0])
    data["return"][1, 0] = np.nan  # el retorno de la fila 2 correcto tras un hueco
    message = _message(_panel(data))
    assert "puentea" not in message
    assert "return: valor no NaN en velas no validas: 1 celdas" in message


def test_bridge_check_flags_second_row_after_invalid_first_row() -> None:
    """Con la fila 1 invalida, un retorno con valor en la fila 2 se marca como puente."""
    data = _arrays()
    data["valid"][0, 0] = False
    for name in ("return", "rel_volume", "rel_trades"):
        data[name][0, 0] = np.nan
    panel = _panel(data)  # return[1, 0] conserva valor
    message = _message(panel)
    assert "return: puentea un hueco" in message
    assert "1 celdas" in message
    assert str(panel.index[1]) in message
    assert "(A)" in message


def test_bridge_check_accepts_nan_return_after_invalid_candle() -> None:
    """El retorno NaN justo despues de un hueco es lo correcto y no viola nada."""
    data = _arrays()
    data["valid"][2, 0] = False
    for name in ("return", "rel_volume", "rel_trades"):
        data[name][2, 0] = np.nan
    data["return"][3, 0] = np.nan
    validate_panel(_panel(data))


def test_bridge_check_ignores_valid_state_of_future_candle() -> None:
    """Una vela invalida posterior no afecta a un retorno anterior (solo mira al pasado)."""
    data = _arrays()
    data["valid"][4, 0] = False
    for name in ("return", "rel_volume", "rel_trades"):
        data[name][4, 0] = np.nan
    data["return"][5, 0] = np.nan
    validate_panel(_panel(data))


# --- 6. varias violaciones a la vez ------------------------------------------------------------


def test_all_violations_are_reported_in_a_single_error() -> None:
    """Varias violaciones de distinto tipo salen todas en un unico `ValueError`."""
    data = _arrays()
    data["return"][1, 0] = np.inf  # (d)
    data["realized_vol"][2, 1] = -1.0  # (g)
    data["valid"][3, 0] = False  # (e) y (f): return[3, 0] y la fila siguiente conservan valor
    data["rel_volume"][3, 0] = np.nan
    data["rel_trades"][3, 0] = np.nan
    index = _index().delete(4)  # (c)
    data = {name: np.delete(values, 4, axis=0) for name, values in data.items()}
    panel = _panel(data, index=index, extra=("momentum",))  # (a)
    message = _message(panel)
    for fragment in (
        "fuera del contrato",
        "indice: 1 saltos",
        "return: valores infinitos",
        "return: valor no NaN en velas no validas",
        "return: puentea un hueco",
        "realized_vol: valores negativos",
    ):
        assert fragment in message
    assert message.count("\n- ") >= 6


# --- 7. enganche -------------------------------------------------------------------------------


def _saved_directory(world: tuple[Path, Path]) -> tuple[Panel, Path]:
    """Construye y guarda el panel con `cached_panel`; devuelve el panel y su directorio."""
    base, cache = world
    panel = cached_panel(*_args(), base_dir=base, cache_dir=cache, **SMALL)
    return panel, panel_path("1m", _ms(START), _ms(END), cache_dir=cache)


def _put_return_on_invalid(directory: Path) -> None:
    """Pone un retorno donde `valid=False` en el Parquet de `return`."""
    valid = read_parquet(directory / "valid.parquet")
    returns = read_parquet(directory / "return.parquet")
    row, col = np.argwhere(~valid.to_numpy(dtype=bool))[0]
    returns.iloc[row, col] = 0.01
    write_parquet(returns, directory / "return.parquet")


def _put_inf(directory: Path) -> None:
    """Pone un `inf` en el Parquet de `rel_volume`."""
    frame = read_parquet(directory / "rel_volume.parquet")
    frame.iloc[5, 1] = np.inf
    write_parquet(frame, directory / "rel_volume.parquet")


def _put_negative_vol(directory: Path) -> None:
    """Pone una volatilidad negativa en el Parquet de `realized_vol`."""
    frame = read_parquet(directory / "realized_vol.parquet")
    frame.iloc[5, 2] = -0.5
    write_parquet(frame, directory / "realized_vol.parquet")


TAMPERS = [_put_return_on_invalid, _put_inf, _put_negative_vol]


@pytest.mark.parametrize("tamper", TAMPERS, ids=lambda f: f.__name__)
def test_load_panel_rejects_a_tampered_parquet(
    world: tuple[Path, Path], tamper: Callable[[Path], None]
) -> None:
    """Un Parquet de la cache que viola el contrato hace que `load_panel` lance `ValueError`."""
    _, directory = _saved_directory(world)
    tamper(directory)
    with pytest.raises(ValueError, match="contrato A -> B"):
        load_panel(directory)


@pytest.mark.parametrize("tamper", TAMPERS, ids=lambda f: f.__name__)
def test_cached_panel_rebuilds_a_tampered_cache(
    world: tuple[Path, Path], counter: Counter, tamper: Callable[[Path], None]
) -> None:
    """Con la cache fuera de contrato, `cached_panel` reconstruye y devuelve un panel valido."""
    base, cache = world
    first = cached_panel(*_args(), base_dir=base, cache_dir=cache, **SMALL)
    assert counter.calls == 1
    directory = panel_path("1m", _ms(START), _ms(END), cache_dir=cache)
    tamper(directory)
    second = cached_panel(*_args(), base_dir=base, cache_dir=cache, **SMALL)
    assert counter.calls == 2
    validate_panel(second)
    _assert_same(first, second)
    _assert_same(first, load_panel(directory))  # la cache queda sana
    cached_panel(*_args(), base_dir=base, cache_dir=cache, **SMALL)
    assert counter.calls == 2


def test_untouched_cache_is_not_rebuilt(world: tuple[Path, Path], counter: Counter) -> None:
    """Una cache que cumple el contrato se carga sin reconstruir."""
    base, cache = world
    cached_panel(*_args(), base_dir=base, cache_dir=cache, **SMALL)
    cached_panel(*_args(), base_dir=base, cache_dir=cache, **SMALL)
    assert counter.calls == 1


def test_build_panel_raises_if_it_would_produce_a_violation(
    world: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Si el calculo produjera un `inf`, `build_panel` lanza `ValueError` en vez de devolverlo."""
    base, _ = world
    real = panel_module.log_returns

    def with_inf(*args: object, **kwargs: object) -> pd.DataFrame:
        result = real(*args, **kwargs)  # type: ignore[arg-type]
        result = result.copy()
        result.iloc[-1, 0] = np.inf
        return result

    monkeypatch.setattr(panel_module, "log_returns", with_inf)
    with pytest.raises(ValueError, match="valores infinitos"):
        build_panel(*_args(), base_dir=base, **SMALL)


# --- 8. documento ------------------------------------------------------------------------------


def test_contract_doc_exists_and_names_every_feature_and_validator() -> None:
    """El documento del contrato existe y nombra cada feature y `validate_panel` entre backticks."""
    assert CONTRACT_DOC.is_file()
    text = CONTRACT_DOC.read_text(encoding="utf-8")
    for key in FEATURE_SPECS:
        assert f"`{key}`" in text
    assert "`validate_panel`" in text
