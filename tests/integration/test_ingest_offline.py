"""Ingesta completa por la CLI real contra un Binance falso: sin red, con disco en `tmp_path`."""

from __future__ import annotations

import importlib.util
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from miax.ingest import __main__ as cli
from miax.ingest import load_klines, load_universe
from miax.ingest.klines import KLINE_DTYPES
from miax.utils.io import read_parquet

# El modo importlib de pytest no mete tests/ en sys.path: el falso se carga por su ruta.
_FAKE_PATH = Path(__file__).resolve().parents[1] / "fakes" / "binance.py"
_SPEC = importlib.util.spec_from_file_location("miax_tests_fake_binance", _FAKE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"No se puede cargar {_FAKE_PATH}")
_FAKE_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _FAKE_MODULE  # los dataclasses del modulo lo buscan por nombre
_SPEC.loader.exec_module(_FAKE_MODULE)
FakeBinance = _FAKE_MODULE.FakeBinance
FIXTURES_DIR: Path = _FAKE_MODULE.FIXTURES_DIR

NOW = datetime(2026, 10, 13, 12, 0, 0, tzinfo=UTC)
MINUTE_MS = 60_000
BTC_FIRST_OPEN_MS = 1_709_251_200_000  # 2024-03-01T00:00:00Z
BTC_LAST_OPEN_MS = BTC_FIRST_OPEN_MS + 1439 * MINUTE_MS  # 2024-03-01T23:59:00Z


def _raw(name: str) -> dict[str, Any] | list[Any]:
    """Lee un fixture JSON tal cual lo grabo el script."""
    return json.loads((FIXTURES_DIR / name).read_text(encoding="utf-8"))


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeBinance:
    """Engancha el Binance falso a la capa HTTP durante el test."""
    binance = FakeBinance()
    binance.install(monkeypatch)
    return binance


@pytest.fixture(autouse=True)
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Anota las esperas en vez de dormir y fija el reloj de la CLI."""
    recorded: list[float] = []
    monkeypatch.setattr(time, "sleep", recorded.append)
    monkeypatch.setattr(cli, "_utc_now", lambda: NOW)
    return recorded


def _download_btc(cache: Path) -> int:
    """Descarga por la CLI el dia completo de BTCUSDT 1m en la cache dada."""
    return cli.main(
        [
            "download",
            "--symbol",
            "BTCUSDT",
            "--interval",
            "1m",
            "--start",
            "2024-03-01",
            "--end",
            "2024-03-01",
            "--base-dir",
            str(cache),
        ]
    )


def test_download_full_day_saves_every_candle(fake: FakeBinance, tmp_path: Path) -> None:
    """Descargar un dia de BTCUSDT 1m guarda 1.440 velas con el formato y los tipos de Binance."""
    assert _download_btc(tmp_path) == 0

    candles = load_klines("BTCUSDT", "1m", base_dir=tmp_path)
    assert len(candles) == 1440
    assert candles.dtypes.astype(str).to_dict() == KLINE_DTYPES
    assert int(candles["open_time"].iloc[0]) == BTC_FIRST_OPEN_MS
    assert int(candles["open_time"].iloc[-1]) == BTC_LAST_OPEN_MS
    assert candles["open_time"].diff().dropna().eq(MINUTE_MS).all()
    first_raw = _raw("klines_BTCUSDT_1m.json")[0]
    assert candles["open"].iloc[0] == float(first_raw[1])
    assert int(candles["num_trades"].iloc[0]) == first_raw[8]


def test_rerun_does_not_download_candles_again(fake: FakeBinance, tmp_path: Path) -> None:
    """Relanzar el mismo rango no baja velas nuevas ni pide nada anterior a lo ya cacheado."""
    assert _download_btc(tmp_path) == 0
    before = len(fake.kline_requests)

    assert _download_btc(tmp_path) == 0

    new_requests = fake.kline_requests[before:]
    # Como mucho la peticion vacia de final de rango.
    assert len(new_requests) <= 1
    assert all(item.n_rows == 0 for item in new_requests)
    # La reanudacion solo mira hacia delante de la ultima vela cacheada (MIAX-021).
    assert all(item.params["startTime"] > BTC_LAST_OPEN_MS for item in new_requests)
    assert len(load_klines("BTCUSDT", "1m", base_dir=tmp_path)) == 1440


def test_validate_clean_cache_has_no_issues(fake: FakeBinance, tmp_path: Path) -> None:
    """Validar la cache de un dia completo de BTCUSDT sale con 0 y sin incidencias."""
    cache = tmp_path / "cache"
    assert _download_btc(cache) == 0
    report = tmp_path / "issues.parquet"

    code = cli.main(
        ["validate", "--symbol", "BTCUSDT", "--base-dir", str(cache), "--output", str(report)]
    )

    assert code == 0
    assert read_parquet(report).empty


def test_validate_luna_reports_gap_as_warning(fake: FakeBinance, tmp_path: Path) -> None:
    """El hueco de mayo de 2022 de LUNAUSDT sale como aviso `gaps` del tamano exacto, con 0."""
    rows = _raw("klines_LUNAUSDT_1m.json")
    cache, report = tmp_path / "cache", tmp_path / "issues.parquet"
    download = [
        "download",
        "--symbol",
        "LUNAUSDT",
        "--interval",
        "1m",
        "--start",
        "2022-05-12",
        "--end",
        "2022-05-31",
        "--base-dir",
        str(cache),
    ]
    assert cli.main(download) == 0
    assert len(load_klines("LUNAUSDT", "1m", base_dir=cache)) == len(rows)

    code = cli.main(
        ["validate", "--symbol", "LUNAUSDT", "--base-dir", str(cache), "--output", str(report)]
    )

    assert code == 0
    issues = read_parquet(report)
    assert issues["severity"].tolist() == ["warning"]
    assert issues["check"].tolist() == ["gaps"]
    # Las velas que faltan entre las dos filas que rodean el hueco (borde del fixture).
    expected_missing = (int(rows[100][0]) - int(rows[99][0])) // MINUTE_MS - 1
    assert expected_missing > 1000
    assert issues["count"].tolist() == [expected_missing]


def test_pairs_filters_usdt_break_from_catalog(fake: FakeBinance, tmp_path: Path) -> None:
    """`pairs --quote USDT --status BREAK` guarda justo los pares USDT en BREAK del fixture."""
    symbols = _raw("exchange_info.json")["symbols"]
    expected = sorted(
        item["symbol"]
        for item in symbols
        if item["quoteAsset"] == "USDT" and item["status"] == "BREAK"
    )
    # El filtro solo se prueba si hay pares que debe dejar fuera, por estado y por cotizado.
    assert expected
    assert len(expected) < sum(1 for item in symbols if item["status"] == "BREAK")
    assert len(expected) < sum(1 for item in symbols if item["quoteAsset"] == "USDT")
    output = tmp_path / "pairs.parquet"

    code = cli.main(["pairs", "--quote", "USDT", "--status", "BREAK", "--output", str(output)])

    assert code == 0
    saved = read_parquet(output)
    assert saved["symbol"].tolist() == expected
    assert set(saved["status"]) == {"BREAK"}
    assert set(saved["quote_asset"]) == {"USDT"}


def test_rate_limit_429_waits_retry_after_and_retries(
    fake: FakeBinance, tmp_path: Path, sleeps: list[float]
) -> None:
    """Un 429 con Retry-After en la primera peticion espera lo indicado y reintenta la misma."""
    fake.schedule_error(429, retry_after=1)

    assert _download_btc(tmp_path) == 0

    assert sleeps == [1.0]
    first, second = fake.kline_requests[:2]
    assert (first.status, second.status) == (429, 200)
    assert first.params == second.params
    assert len(load_klines("BTCUSDT", "1m", base_dir=tmp_path)) == 1440


def _build_universe(tmp_path: Path, name: str, min_volume: float) -> pd.DataFrame:
    """Ejecuta `universe` por la CLI con ese umbral de volumen y devuelve el universo guardado."""
    categories = {
        "l1": ["BTC", "ETH", "EOS"],
        "defi": ["UNI"],
        "infra": ["LINK"],
        "meme": ["DOGE"],
        "stablecoin": ["USDC"],
    }
    categories_file = tmp_path / f"categories_{name}.yaml"
    categories_file.write_text(yaml.safe_dump(categories), encoding="utf-8")
    params_file = tmp_path / f"universe_params_{name}.yaml"
    params_file.write_text(
        yaml.safe_dump(
            {
                "quote_asset": "USDT",
                "first_period_start": "2024-01-01",
                "last_period_end": "2025-01-01",
                "period_months": 12,
                "volume_window_months": 3,
                "min_listing_days": 30,
                "min_quote_volume": min_volume,
                "max_per_category": 10,
                "max_pairs": 40,
                "min_categories": 4,
                "categories_file": str(categories_file),
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / f"universe_{name}.yaml"

    assert cli.main(["universe", "--params", str(params_file), "--out", str(out)]) == 0
    return load_universe(out)


def test_universe_from_fixtures_evaluates_break_pairs_like_the_rest(
    fake: FakeBinance, tmp_path: Path
) -> None:
    """El universo de 2024 filtra por volumen y evalua igual el par en BREAK (D-13)."""
    exchange = {item["symbol"]: item["status"] for item in _raw("exchange_info.json")["symbols"]}
    assert exchange["EOSUSDT"] == "BREAK"  # sin esto el test no probaria el caso de D-13

    # Mediana esperada en la ventana oct-dic de 2023, calculada a mano desde el fixture crudo.
    window_start = int(datetime(2023, 10, 1, tzinfo=UTC).timestamp() * 1000)
    window_end = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1000) - 1
    medians = {
        symbol: statistics.median(
            float(row[7]) for row in rows if window_start <= int(row[0]) <= window_end
        )
        for symbol, rows in _raw("klines_1d.json").items()
    }
    # Las medianas fijan el orden que necesitan los dos umbrales: EOS (BREAK) < UNI < resto.
    assert (
        medians["EOSUSDT"]
        < medians["UNIUSDT"]
        < min(m for s, m in medians.items() if s not in ("EOSUSDT", "UNIUSDT"))
    )

    # Umbral bajo: pasan todos menos la stablecoin. Cubre D-13 (EOS en BREAK entra con su
    # volumen real, no se descarta por su estado), la exclusion de la stablecoin y el periodo.
    low = _build_universe(tmp_path, "low", min_volume=1_000_000.0)
    assert set(low["symbol"]) == {s for s in medians if s != "USDCUSDT"}
    assert set(low["period_start"]) == {pd.Timestamp("2024-01-01", tz="UTC")}
    eos = low[low["symbol"] == "EOSUSDT"].iloc[0]
    assert eos["category"] == "l1"
    assert eos["median_quote_volume"] == pytest.approx(medians["EOSUSDT"], abs=0.01)

    # Umbral entre EOS y UNI: cubre el filtro de volumen (EOS cae por volumen, UNI se queda).
    threshold = (medians["EOSUSDT"] + medians["UNIUSDT"]) / 2
    mid = _build_universe(tmp_path, "mid", min_volume=threshold)
    assert set(mid["symbol"]) == {
        s for s, m in medians.items() if s != "USDCUSDT" and m >= threshold
    }
    assert "EOSUSDT" not in set(mid["symbol"])
    assert "UNIUSDT" in set(mid["symbol"])
