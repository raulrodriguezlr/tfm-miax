"""Punta a punta del preprocesado con huecos y alineacion, contra un oraculo independiente."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import miax.features.panel as panel_module
import miax.features.returns as returns_module
from miax.features import (
    GapCause,
    Panel,
    build_panel,
    build_time_grid,
    classify_klines,
    returns_from_cache,
)
from miax.ingest import save_klines
from miax.ingest.klines import KLINE_COLUMNS, KLINE_DTYPES
from miax.ingest.store import month_files

MIN_MS = 60_000
# 2024-01-31 21:00 UTC: 180 min despues llega la medianoche y el cambio de mes (un fichero por mes).
T0 = int(pd.Timestamp("2024-01-31 21:00", tz="UTC").timestamp() * 1000)
N_MINUTES = 360
MIDNIGHT = 180  # minuto en que abre la vela de 2024-02-01 00:00

# Ventanas pequenas para que el calentamiento (21 min) quepa antes del rango.
VOL_W = 5
ACT_W = 20
SMALL = {"vol_window": VOL_W, "activity_window": ACT_W}
START_MIN = 60
END_MIN = 340

SYMBOLS = ("A", "B", "C", "D")
SEEDS = {"A": 1, "B": 2, "C": 3, "D": 4}

# Escenario A: una vela suelta que falta y cinco seguidas a caballo de la medianoche.
A_SINGLE_HOLE = 100
A_MIDNIGHT_HOLE = range(178, 183)  # abren 23:58, 23:59, 00:00, 00:01 y 00:02
# Escenario B: un tramo sin operaciones y un retorno 0 autentico (con operaciones).
B_NO_TRADES = range(120, 126)
B_TRUE_ZERO = 140  # cierra igual que la vela 139 pero con operaciones
# Escenario C: listado tardio (primera vela) y deslistado (ultima vela).
C_FIRST = 90
C_LAST = 300
# Caida comun: ningun simbolo tiene vela, con C vivo, asi que es OUTAGE.
OUTAGE = range(220, 223)

TOLERANCE = 1e-12


def _ms(minute: int) -> int:
    """Instante en ms de `minute` minutos despues de T0."""
    return T0 + minute * MIN_MS


@dataclass(frozen=True)
class _World:
    """Mundo en memoria: por simbolo y minuto, si hay vela, su cierre, volumen y operaciones."""

    present: dict[str, np.ndarray]
    close: dict[str, np.ndarray]
    quote: dict[str, np.ndarray]
    trades: dict[str, np.ndarray]

    def is_valid(self, symbol: str, minute: int) -> bool:
        """Hay vela abierta en `minute` y con operaciones (D-18); fuera del mundo, no."""
        if not 0 <= minute < N_MINUTES:
            return False
        return bool(self.present[symbol][minute] and self.trades[symbol][minute] > 0)


@dataclass(frozen=True)
class _Truth:
    """Verdad del oraculo en la rejilla de cierres: instantes, validez y retornos (T, N)."""

    tau_ms: np.ndarray
    valid: np.ndarray
    returns: np.ndarray


def _random_series(seed: int) -> dict[str, np.ndarray]:
    """Cierre, volumen en quote y operaciones sinteticos, positivos y con semilla fija."""
    rng = np.random.default_rng(seed)
    return {
        "close": 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.001, N_MINUTES))),
        "quote": np.exp(rng.normal(5.0, 1.0, N_MINUTES)),
        "trades": rng.integers(1, 100, N_MINUTES),
    }


def _build_world() -> _World:
    """Construye los cuatro escenarios de huecos descritos en las constantes del modulo."""
    data = {symbol: _random_series(SEEDS[symbol]) for symbol in SYMBOLS}
    present = {symbol: np.ones(N_MINUTES, dtype=bool) for symbol in SYMBOLS}

    present["A"][A_SINGLE_HOLE] = False
    present["A"][list(A_MIDNIGHT_HOLE)] = False

    b = data["B"]
    first_no_trades = B_NO_TRADES.start
    # OHLC = cierre anterior repetido
    b["close"][list(B_NO_TRADES)] = b["close"][first_no_trades - 1]
    b["quote"][list(B_NO_TRADES)] = 0.0
    b["trades"][list(B_NO_TRADES)] = 0
    # retorno 0 real: mismo cierre, con operaciones
    b["close"][B_TRUE_ZERO] = b["close"][B_TRUE_ZERO - 1]

    present["C"][:C_FIRST] = False
    present["C"][C_LAST + 1 :] = False

    for symbol in SYMBOLS:
        present[symbol][list(OUTAGE)] = False
    return _World(
        present=present,
        close={s: data[s]["close"] for s in SYMBOLS},
        quote={s: data[s]["quote"] for s in SYMBOLS},
        trades={s: data[s]["trades"] for s in SYMBOLS},
    )


def _write_world(base_dir: Path, world: _World) -> None:
    """Guarda en la cache de `base_dir` solo las velas presentes de cada simbolo."""
    for symbol in SYMBOLS:
        minutes = np.flatnonzero(world.present[symbol])
        open_times = [_ms(int(k)) for k in minutes]
        close = world.close[symbol][minutes]
        frame = pd.DataFrame(
            {
                "open_time": open_times,
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": [1.0] * len(minutes),
                "close_time": [t + MIN_MS - 1 for t in open_times],
                "quote_asset_volume": world.quote[symbol][minutes],
                "num_trades": world.trades[symbol][minutes],
                "taker_buy_base": [0.5] * len(minutes),
                "taker_buy_quote": [5.0] * len(minutes),
                "ignore": [0.0] * len(minutes),
            },
            columns=KLINE_COLUMNS,
        ).astype(KLINE_DTYPES)
        save_klines(frame, symbol, "1m", base_dir=base_dir)


def _truth(world: _World, start_min: int, end_min: int) -> _Truth:
    """Oraculo con bucles simples: `valid` y retorno log de cada cierre `tau` en `[start, end)`."""
    taus = list(range(start_min, end_min))
    valid = np.zeros((len(taus), len(SYMBOLS)), dtype=bool)
    returns = np.full((len(taus), len(SYMBOLS)), np.nan)
    for j, symbol in enumerate(SYMBOLS):
        for i, tau in enumerate(taus):
            # La vela que cierra en tau abre en tau - 1 y la anterior en tau - 2: solo pasado.
            current, previous = tau - 1, tau - 2
            valid[i, j] = world.is_valid(symbol, current)
            if world.is_valid(symbol, current) and world.is_valid(symbol, previous):
                returns[i, j] = math.log(
                    world.close[symbol][current] / world.close[symbol][previous]
                )
    return _Truth(np.array([_ms(t) for t in taus], dtype="int64"), valid, returns)


def _return_mismatches(returns: pd.DataFrame, truth: _Truth) -> list[str]:
    """Diferencias de un DataFrame de retornos con el oraculo: indice, posiciones NaN y valores."""
    problems: list[str] = []
    if not np.array_equal(returns.index.as_unit("ms").asi8, truth.tau_ms):
        problems.append("return: el indice no es la rejilla de cierres del oraculo")
        return problems
    values = returns.to_numpy(dtype="float64")
    if values.shape != truth.returns.shape:
        return [*problems, f"return: forma {values.shape} distinta de {truth.returns.shape}"]
    nan_diff = np.isnan(values) != np.isnan(truth.returns)
    if nan_diff.any():
        problems.append(f"return: {int(nan_diff.sum())} celdas con NaN distinto del oraculo")
    injected = (values == 0) & np.isnan(truth.returns)
    if injected.any():
        problems.append(f"return: {int(injected.sum())} ceros inyectados donde el oraculo da NaN")
    both = ~np.isnan(values) & ~np.isnan(truth.returns)
    if (np.abs(values[both] - truth.returns[both]) > TOLERANCE).any():
        problems.append("return: valores distintos del oraculo (tolerancia 1e-12)")
    return problems


def _mismatches(panel: Panel, truth: _Truth) -> list[str]:
    """Lista las diferencias del panel con el oraculo (vacia si coincide todo)."""
    problems = _return_mismatches(panel.features["return"], truth)
    if not np.array_equal(panel.valid.index.as_unit("ms").asi8, truth.tau_ms):
        problems.append("valid: el indice no es la rejilla de cierres del oraculo")
    elif (panel.valid.to_numpy() != truth.valid).any():
        problems.append("valid: distinto del oraculo")
    invalid = ~panel.valid.to_numpy(dtype=bool)
    for name in ("rel_volume", "rel_trades"):
        leaked = ~np.isnan(panel.features[name].to_numpy(dtype="float64")) & invalid
        if leaked.any():
            problems.append(f"{name}: {int(leaked.sum())} celdas no NaN donde valid es False")
    return problems


@pytest.fixture(scope="module")
def world() -> _World:
    """Mundo sintetico en memoria, construido una sola vez; ningun test lo muta."""
    return _build_world()


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory: pytest.TempPathFactory, world: _World) -> Path:
    """Cache temporal con el mundo escrito una sola vez; ningun test la muta."""
    base_dir = tmp_path_factory.mktemp("gaps_cache")
    _write_world(base_dir, world)
    return base_dir


def _panel(base_dir: Path) -> Panel:
    """Panel de los cuatro simbolos en `[START_MIN, END_MIN)` con ventanas pequenas."""
    return build_panel(
        list(SYMBOLS), "1m", _ms(START_MIN), _ms(END_MIN), base_dir=base_dir, **SMALL
    )


# --- a. control --------------------------------------------------------------------------------


def test_control_panel_matches_the_oracle(cache_dir: Path, world: _World) -> None:
    """Sin mutar, `valid` y retornos coinciden con el oraculo y rel_* es NaN si valid es False."""
    truth = _truth(world, START_MIN, END_MIN)
    assert _mismatches(_panel(cache_dir), truth) == []


def test_control_returns_from_cache_matches_the_oracle(cache_dir: Path, world: _World) -> None:
    """`returns_from_cache` da los mismos retornos que el oraculo."""
    returns = returns_from_cache(
        list(SYMBOLS), "1m", _ms(START_MIN), _ms(END_MIN), base_dir=cache_dir
    )
    assert _return_mismatches(returns, _truth(world, START_MIN, END_MIN)) == []


def test_oracle_sees_each_gap_scenario() -> None:
    """El oraculo no es trivial: hay validos, invalidos, retornos NaN y exactamente un cero real."""
    truth = _truth(_build_world(), START_MIN, END_MIN)
    assert truth.valid.any()
    assert (~truth.valid).any()
    assert np.isnan(truth.returns).any()
    assert int((truth.returns == 0).sum()) == 1


# --- b. rejilla y cambio de mes ----------------------------------------------------------------


def test_every_frame_shares_the_grid_across_the_month_change(cache_dir: Path) -> None:
    """Todas las columnas comparten `build_time_grid` y la cache tiene un fichero por mes."""
    _, files = month_files("A", "1m", cache_dir)
    assert len(files) == 2
    panel = _panel(cache_dir)
    grid = build_time_grid(_ms(START_MIN), _ms(END_MIN), "1m")
    assert grid[0].month == 1
    assert grid[-1].month == 2
    assert panel.index.equals(grid)
    assert panel.valid.index.equals(grid)
    for frame in panel.features.values():
        assert frame.index.equals(grid)
    assert panel.shape == (END_MIN - START_MIN, len(SYMBOLS), 4)


def test_gap_across_midnight_leaves_nan_and_is_not_bridged(cache_dir: Path, world: _World) -> None:
    """El hueco a caballo de la medianoche deja NaN su retorno y el siguiente, sin puentear."""
    panel = _panel(cache_dir)
    returns = panel.features["return"]["A"]
    gap_first, gap_last = A_MIDNIGHT_HOLE.start, A_MIDNIGHT_HOLE[-1]
    # Velas ausentes gap_first..gap_last cierran en gap_first + 1..gap_last + 1; el siguiente
    # retorno (gap_last + 2) necesita la vela ausente gap_last como t - 1.
    nan_taus = range(gap_first + 1, gap_last + 3)
    for tau in nan_taus:
        stamp = pd.Timestamp(_ms(tau), unit="ms", tz="UTC")
        assert np.isnan(returns.loc[stamp]), tau
    assert pd.Timestamp(_ms(MIDNIGHT), unit="ms", tz="UTC") in returns.index
    assert pd.Timestamp(_ms(MIDNIGHT), unit="ms", tz="UTC") in panel.valid.index
    before = pd.Timestamp(_ms(nan_taus[0] - 1), unit="ms", tz="UTC")
    after = pd.Timestamp(_ms(nan_taus[-1] + 1), unit="ms", tz="UTC")
    assert not np.isnan(returns.loc[before])
    # La primera vela tras el hueco ya es valida pero su retorno aun no: el siguiente si lo es.
    assert not np.isnan(returns.loc[after])
    expected = math.log(world.close["A"][gap_last + 2] / world.close["A"][gap_last + 1])
    assert returns.loc[after] == pytest.approx(expected, abs=TOLERANCE)


# --- c. ningun cero inyectado ------------------------------------------------------------------


def test_no_zero_is_injected_in_any_gap_and_the_true_zero_survives(
    cache_dir: Path, world: _World
) -> None:
    """Ni retornos ni actividad relativa valen 0 en un hueco; el 0 real de B se conserva exacto."""
    panel = _panel(cache_dir)
    truth = _truth(world, START_MIN, END_MIN)
    returns = panel.features["return"].to_numpy(dtype="float64")
    assert np.isnan(returns[np.isnan(truth.returns)]).all()
    invalid = ~panel.valid.to_numpy(dtype=bool)
    for name in ("return", "rel_volume", "rel_trades"):
        assert np.isnan(panel.features[name].to_numpy(dtype="float64")[invalid]).all(), name
    true_zero_tau = pd.Timestamp(_ms(B_TRUE_ZERO + 1), unit="ms", tz="UTC")
    assert panel.features["return"].loc[true_zero_tau, "B"] == 0.0
    assert int((returns == 0).sum()) == int((truth.returns == 0).sum()) == 1


# --- d. causas de cada hueco -------------------------------------------------------------------


def _expected_causes() -> pd.DataFrame:
    """Causa esperada por minuto y simbolo, escrita a mano a partir de los escenarios."""
    causes = np.full((N_MINUTES, len(SYMBOLS)), GapCause.VALID, dtype="int8")
    col = {symbol: j for j, symbol in enumerate(SYMBOLS)}
    causes[A_SINGLE_HOLE, col["A"]] = GapCause.HALTED
    causes[list(A_MIDNIGHT_HOLE), col["A"]] = GapCause.HALTED
    causes[list(B_NO_TRADES), col["B"]] = GapCause.NO_TRADES
    causes[:C_FIRST, col["C"]] = GapCause.NO_DATA
    causes[C_LAST + 1 :, col["C"]] = GapCause.DELISTED
    causes[list(OUTAGE), :] = GapCause.OUTAGE
    return pd.DataFrame(causes, columns=list(SYMBOLS))


def test_classify_klines_labels_each_scenario(cache_dir: Path) -> None:
    """Cada escenario cae en su causa: HALTED, NO_TRADES, NO_DATA, DELISTED y OUTAGE."""
    causes = classify_klines(list(SYMBOLS), "1m", _ms(0), _ms(N_MINUTES), base_dir=cache_dir)
    np.testing.assert_array_equal(causes.to_numpy(), _expected_causes().to_numpy())


def test_classify_klines_gives_not_listed_when_the_listing_is_known(cache_dir: Path) -> None:
    """Con el listado conocido, los slots de C anteriores a su primera vela pasan a NOT_LISTED."""
    causes = classify_klines(
        list(SYMBOLS),
        "1m",
        _ms(0),
        _ms(N_MINUTES),
        base_dir=cache_dir,
        listed_ms={"C": _ms(C_FIRST)},
    )
    assert (causes["C"].to_numpy()[:C_FIRST] == GapCause.NOT_LISTED).all()


# --- e. mutaciones: un tratamiento de huecos incorrecto tiene que hacer fallar estos tests ------

_REAL_LOG_RETURNS = panel_module.log_returns
_REAL_VALIDITY_MASK = panel_module.validity_mask
_REAL_ALIGN_KLINES = panel_module.align_klines
_REAL_CLASSIFY_KLINES = panel_module.classify_klines
_REAL_SHIFT_TO_CLOSE = panel_module.shift_to_close
_REAL_LRP = panel_module.log_relative_to_past


def _fillna_zero(close: pd.DataFrame, valid: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Mutacion: los retornos que faltan se rellenan con 0."""
    return _REAL_LOG_RETURNS(close, valid, interval).fillna(0.0)


def _bridge_gaps(close: pd.DataFrame, valid: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Mutacion: el retorno se calcula desde el ultimo cierre valido, saltando el hueco."""
    bridged = np.log(close.where(valid)).ffill().diff().where(valid)
    return bridged.set_axis(_REAL_SHIFT_TO_CLOSE(bridged.index, interval), axis=0)


def _no_trades_valid(causes: pd.DataFrame) -> pd.DataFrame:
    """Mutacion: los minutos NO_TRADES se dan por validos."""
    return _REAL_VALIDITY_MASK(causes) | (causes == GapCause.NO_TRADES)


def _clamped_log_relative(
    values: pd.DataFrame, valid: pd.DataFrame, *args: object, **kwargs: object
) -> pd.DataFrame:
    """Mutacion: los valores <= 0 pasan a 1.0, para esquivar la guarda de `log_relative_to_past`."""
    return _REAL_LRP(values.where(values > 0, 1.0), valid, *args, **kwargs)


def _shifted_alignment(*args: object, **kwargs: object) -> dict[str, pd.DataFrame]:
    """Mutacion: todos los campos alineados se desplazan una fila (un minuto)."""
    return {name: frame.shift(1) for name, frame in _REAL_ALIGN_KLINES(*args, **kwargs).items()}


def _shifted_causes(*args: object, **kwargs: object) -> pd.DataFrame:
    """Mutacion: las causas se desplazan una fila, igual que los campos alineados."""
    causes = _REAL_CLASSIFY_KLINES(*args, **kwargs)
    return causes.shift(1, fill_value=int(GapCause.NO_DATA)).astype(causes.dtypes)


def _index_by_open(index: pd.DatetimeIndex, interval: str) -> pd.DatetimeIndex:
    """Mutacion: no se desplaza al cierre, se indexa por la apertura."""
    return index


@dataclass(frozen=True)
class _Mutation:
    """Parches `nombre -> reemplazo` y lo que se espera: una excepcion o textos del oraculo."""

    patches: Mapping[str, Callable[..., Any]]
    raises: tuple[type[Exception], str] | None = None
    oracle_texts: tuple[str, ...] = ()


PANEL_MUTATIONS: dict[str, _Mutation] = {
    "fillna_zero": _Mutation({"log_returns": _fillna_zero}, oracle_texts=("ceros inyectados",)),
    # Los minutos NO_TRADES pasan a validos y la guarda de log(0) del volumen lo para antes.
    "no_trades_valid": _Mutation(
        {"validity_mask": _no_trades_valid},
        raises=(ValueError, r"valor no finito o <= 0 en slots validos"),
    ),
    # Sin esa guarda la mutacion llega al oraculo.
    "no_trades_valid_unguarded": _Mutation(
        {"validity_mask": _no_trades_valid, "log_relative_to_past": _clamped_log_relative},
        oracle_texts=("ceros inyectados", "valid: distinto del oraculo"),
    ),
    "bridge_gaps": _Mutation(
        {"log_returns": _bridge_gaps}, oracle_texts=("celdas con NaN distinto del oraculo",)
    ),
    "shift_alignment": _Mutation(
        {"align_klines": _shifted_alignment, "classify_klines": _shifted_causes},
        oracle_texts=("valores distintos del oraculo", "valid: distinto del oraculo"),
    ),
    "index_by_open": _Mutation(
        {"shift_to_close": _index_by_open},
        raises=(RuntimeError, r"el indice de valid no coincide con la rejilla"),
    ),
}

RETURNS_MUTATIONS: dict[str, _Mutation] = {
    "fillna_zero": PANEL_MUTATIONS["fillna_zero"],
    # Aqui no hay guarda de volumen: la mutacion llega directa al oraculo.
    "no_trades_valid": _Mutation(
        {"validity_mask": _no_trades_valid}, oracle_texts=("ceros inyectados",)
    ),
    "bridge_gaps": PANEL_MUTATIONS["bridge_gaps"],
}


def _assert_panel_mutation_detected(
    spec: _Mutation, cache_dir: Path, truth: _Truth, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Aplica la mutacion y exige que se detecte exactamente por donde `spec` dice."""
    for target, replacement in spec.patches.items():
        monkeypatch.setattr(panel_module, target, replacement)
    if spec.raises is not None:
        exc_type, pattern = spec.raises
        with pytest.raises(exc_type, match=pattern):
            _panel(cache_dir)
        return
    problems = _mismatches(_panel(cache_dir), truth)  # una excepcion aqui hace fallar el test
    for text in spec.oracle_texts:
        assert any(text in problem for problem in problems), (text, problems)


@pytest.mark.parametrize("mutation", sorted(PANEL_MUTATIONS))
def test_wrong_gap_treatment_is_detected(
    mutation: str, cache_dir: Path, world: _World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cada tratamiento de huecos erroneo hace fallar `build_panel` o el oraculo, sin contrato."""
    truth = _truth(world, START_MIN, END_MIN)
    # Sin el contrato (`validate_panel`) para que lo detecten estos tests y no el MIAX-044.
    monkeypatch.setattr(panel_module, "validate_panel", lambda panel: None)
    # Con el contrato apagado el control sigue limpio.
    assert _mismatches(_panel(cache_dir), truth) == []
    _assert_panel_mutation_detected(PANEL_MUTATIONS[mutation], cache_dir, truth, monkeypatch)


@pytest.mark.parametrize("mutation", sorted(RETURNS_MUTATIONS))
def test_wrong_gap_treatment_is_detected_in_returns_from_cache(
    mutation: str, cache_dir: Path, world: _World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Los mismos tratamientos erroneos hacen que `returns_from_cache` difiera del oraculo."""
    truth = _truth(world, START_MIN, END_MIN)

    def run() -> pd.DataFrame:
        return returns_from_cache(
            list(SYMBOLS), "1m", _ms(START_MIN), _ms(END_MIN), base_dir=cache_dir
        )

    assert _return_mismatches(run(), truth) == []  # control sin mutar
    spec = RETURNS_MUTATIONS[mutation]
    for target, replacement in spec.patches.items():
        monkeypatch.setattr(returns_module, target, replacement)
    problems = _return_mismatches(run(), truth)
    for text in spec.oracle_texts:
        assert any(text in problem for problem in problems), (text, problems)
