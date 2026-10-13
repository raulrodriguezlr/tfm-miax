# Estado del equipo

**Bitácora compartida entre sesiones.** Somos tres personas con tres Claude distintos que no se ven entre sí. Este fichero es el único sitio donde se enteran de lo que hacen los demás.

> **Si eres un agente:** lee esto antes de tocar nada, y añade tu entrada antes de que tu humano haga push. No borres entradas de otros. Si hay conflicto de merge aquí, se quedan las dos versiones.

---

## 1. Ahora mismo

| Persona | Ticket en curso | Rama | Desde | Estado |
|---|---|---|---|---|
| Raúl | — (reglas de equipo) | `docs/reglas-equipo` | 2026-10-12 | en review |
| Piettro | — | — | — | libre |
| Alonso | MIAX-030 | `feature/MIAX-030` | 2026-10-13 | en review |

**Versión actual:** `v0.1.0` (2026-10-11, en `main` con tag) · **Snapshot de `develop`:** `v0.2.0-snapshot.5` (este PR será `.6`)

---

## 2. Decisiones vigentes

Decisiones tomadas que afectan a todos. Si vas a contradecir una, hablas con el equipo antes.

**Numeración:** en tu rama apunta la decisión nueva como `D-??`. El número definitivo se pone al mergear, cogiendo el siguiente libre en `develop` en ese momento. Así dos ramas en paralelo no cogen el mismo número.

| # | Decisión | Por qué | Ticket |
|---|---|---|---|
| D-01 | El plan se hace en Opus; la ejecución, con subagentes Sonnet. | Planificar es donde se decide bien o mal. Ejecutar es mecánico. | MIAX-003 |
| D-02 | Los huecos de velas **no** se rellenan con `fillna(0)`. | Un retorno cero equivale a rellenar el precio hacia delante e inyecta autocorrelación falsa. Se usa máscara de validez. | MIAX-035 |
| D-03 | La residualización guarda betas y cargas, y las **proyecta** sobre el test. | Recalcular el factor en test es fuga de datos. | MIAX-051 |
| D-04 | Se evalúan varios horizontes (1m, 5m, 15m, 60m), no solo 1 minuto. | Rebalancear cada minuto condena el test económico por construcción, no por evidencia. | MIAX-102 |
| D-05 | El grafo se recalcula **dentro de cada fold** de entrenamiento. | Construirlo con la muestra completa contamina todos los resultados. | MIAX-137 |
| D-06 | Python 3.12 para todo el equipo. | Todo el stack tiene ruedas para 3.12 en Windows, Linux y macOS, y es la versión que deja más margen para bajar torch si PyG Temporal falla. | MIAX-005 |
| D-07 | Toda dependencia entra en `requirements.txt` con versión exacta (`==`), nunca con rango. | Un rango instala cosas distintas según el día. Lo vigila `tests/unit/test_requirements.py`. | MIAX-005 |
| D-08 | Las dependencias solo se declaran en `requirements.txt`; `pyproject.toml` no lista ninguna. | Una sola fuente de verdad: dos listas acaban desincronizadas, y leerlas desde el pyproject rompería con las opciones de índice que traerá torch (MIAX-115). | MIAX-006 |
| D-09 | El estilo lo decide ruff: sus reglas por defecto más `ANN`, `D`, `NPY` y `E501` (100); `T20` se añadió después (ver D-11). `ruff check .` y `ruff format --check .` deben pasar en limpio antes de cada push. | Nadie discute estilo en los PR, y las normas del repo (tipos, docstrings, semillas, UTC) las vigila la máquina. | MIAX-007 |
| D-10 | Todo Parquet se lee y escribe con `miax.utils.io` (`read_parquet` / `write_parquet`); la escritura es atómica. | Una sola forma de hacerlo; una descarga cortada no deja ficheros corruptos que la reanudación daría por buenos. | MIAX-011 |
| D-11 | En `src/` no se usa `print`: se loguea con `miax.utils.get_logger`. Ruff lo hace cumplir con `T20`, eximido en `notebooks/`, `scripts/` y `tests/`. | Un formato y un nivel comunes en producción; `print` no se filtra ni se puede silenciar. | MIAX-012 |
| D-12 | La prohibición de tocar la red en los tests la vigila una guardia autouse en `tests/conftest.py`, no la revisión humana. | Un `monkeypatch` sobre `socket.socket.connect`/`connect_ex` falla siempre y se restaura solo, aunque el test falle. | MIAX-008 |
| D-13 | El catálogo de pares se consulta completo, incluidos los pares en `BREAK` (parados o deslistados); filtrar por estado es decisión del universo (MIAX-025 a 028), no de la consulta. | Sin los pares en `BREAK` no se puede medir el sesgo de supervivencia (MIAX-031). | MIAX-024 |
| D-14 | El universo se elige con el volumen de una ventana anterior al periodo de estudio, nunca con el del propio periodo: así la selección no usa información del futuro, y los pares que entonces eran líquidos entran aunque hoy estén deslistados. | Elegir por el volumen de hoy o del test es sesgo de selección y de supervivencia. | MIAX-025 |

---

## 3. Interfaces entre áreas

El contrato que evita que unos bloqueen a otros. **Si cambias una, avisas antes, no después.**

| Interfaz | Qué se entrega | Formato | Estado |
|---|---|---|---|
| `A → B` | Panel limpio de retornos alineados | Parquet particionado por símbolo | sin definir |
| `B → C` | Matriz de adyacencia por *fold* | `.npz` + metadatos de retardo | sin definir |

---

## 4. Bloqueos abiertos

| Quién | Qué bloquea | Desde | Qué se necesita |
|---|---|---|---|
| Alonso | Universo por periodos desde 2017 (opción B, MIAX-028): provisional. Si se confirma, el grafo, la GNN y la validación (E3+) deben admitir un conjunto de nodos que cambia por periodo | 2026-10-12 | Validarlo con el tutor en la tutoría del 2026-10-13 y, si sigue, abrir tickets para nodos variables |

---

## 5. Bitácora

Lo más reciente arriba. Una entrada por sesión de trabajo.

### Plantilla

```markdown
### YYYY-MM-DD · Nombre · MIAX-XXX
- **Hecho:** qué he dejado funcionando.
- **Decisión:** qué he decidido que no era obvio, y por qué.
- **Ojo:** qué tiene que saber el siguiente que toque esto.
- **Pendiente:** qué queda sin hacer, y en qué ticket.
```

---

### 2026-10-13 · Alonso · MIAX-030
- **Hecho:** `miax.ingest.coverage` con `coverage_from_frame(candles, interval, start_ms=None, end_ms=None) -> dict` (pura), `coverage_report(symbols, interval, *, base_dir, start_ms=None, end_ms=None) -> pd.DataFrame` (una fila por símbolo, también si no hay caché) y `universe_coverage(universe, interval, *, base_dir, now_ms=None) -> pd.DataFrame` (cada par en cada periodo de `load_universe()`). Columnas (`COVERAGE_COLUMNS`): `symbol`, `first_open`, `last_open`, `expected`, `present`, `coverage_pct`, `traded_pct` (% de minutos con `num_trades > 0` sobre los esperados), `n_gaps`, `max_gap_candles`, `missing_before`, `missing_after`. Exportadas desde `miax.ingest`. Subcomando `python -m miax.ingest coverage (--symbol A,B | --universe) [--universe-file] [--interval] [--base-dir] [--start --end] [--output]`: sale con 2 si los argumentos no valen, 1 si algún símbolo no tiene velas en el rango (o falla `--output`), 0 si no.
- **Decisión:** con rango, la cobertura se mide sobre la rejilla del intervalo en `[start, end)` (esperadas = ranuras de la rejilla), y `missing_before`/`missing_after` cuentan las ranuras vacías antes de la primera y después de la última vela: así una serie cortada (BCC 2019, BCHABC 2020, LUNA 2022) sale con `missing_after` alto aunque no tenga huecos internos. Sin rango, el rango es de la primera a la última vela y los extremos valen 0. Los huecos internos reutilizan `find_gaps` de `integrity.py`. `1M` y otros intervalos sin paso fijo dan `ValueError` (código 2 en la CLI).
- **Ojo:** `universe_coverage` es un informe **descriptivo**: mide el propio periodo, así que no debe usarse para elegir los pares de ese mismo periodo (sería mirar al futuro, D-14). `now_ms` recorta el periodo en curso y omite los futuros.
- **Ojo (suelo de 5 M):** `traded_pct` es la medida pedida en la nota de MIAX-028 para validar el suelo de liquidez (efecto Epps). Aún no hay velas descargadas, así que el informe del universo real no se ha podido ejecutar; hace falta la descarga del universo (fuera de este ticket).
- **Prueba de humo real (2026-10-13):** velas de 1m descargadas de Binance a una carpeta temporal y `python -m miax.ingest coverage` sobre ellas. `LUNAUSDT` abr–jun 2022: 104.800 de 131.040 velas (80 %), **un hueco de 26.240 velas (~18 días) a mediados de mayo**, tras el que la serie vuelve ya como Terra 2.0 con el mismo símbolo: el informe lo detecta. `BTCUSDT` (una semana de 2024): 100 % presente y 100 % con operaciones. `ARUSDT` ene-2023 (el par más débil del universo de 2023, 5,1 M): 100 % presente pero **solo 95,07 % de minutos con operaciones**, justo en el límite del criterio del 95 % propuesto para el suelo de 5 M. `--universe` sobre esa caché parcial: 241 de 244 pares-periodo sin velas, código 1 (correcto).
- **Pendiente:** ejecutar `coverage --universe` cuando haya caché y decidir con `traded_pct` si el suelo de 5 M se mantiene o se cambia el criterio.

### 2026-10-13 · Alonso · MIAX-028
- **Hecho:** universo **por periodos** congelado en un fichero. `configs/universe_params.yaml` (parámetros) y, en `miax.ingest.universe`: `UniverseParams`, `UniversePeriod`, `load_universe_params(path="universe_params.yaml")` (valida y lanza `ConfigError`), `universe_periods(params) -> list[UniversePeriod]`, `select_period_universe(pairs, first_candles, volumes, categories, period, params) -> pd.DataFrame` (pura: antigüedad, volumen, diversificación), `write_universe(universes, params, path="universe.yaml", *, generated_at=None) -> Path` (YAML atómico, tipos nativos) y `load_universe(path="universe.yaml") -> pd.DataFrame` (una fila por par y periodo). `build_universe(params, categories, *, client, now_ms, base_url=...) -> list[tuple[UniversePeriod, pd.DataFrame]]` orquesta la descarga (catálogo, primeras velas, volúmenes por ventana) sin escribir. Todo exportado desde `miax.ingest`. El subcomando `python -m miax.ingest universe [--params] [--out] [--base-url]` (con red) llama a `build_universe` y escribe `configs/universe.yaml`.
- **Decisión (provisional hasta la tutoría del 2026-10-13):** opción B, el universo **no es fijo**: cada periodo anual (2018 a 2026-10-01, 9 periodos) tiene el suyo, elegido solo con datos anteriores. La ventana de volumen son los 3 meses previos al periodo y acaba 1 ms antes de que empiece; la antigüedad mínima (30 días) se mide contra el **inicio de la ventana**, no del periodo, para que el volumen no incluya la euforia de lanzamiento. Si el tutor prefiere un universo fijo basta con configurar un único periodo. `min_categories=4` solo **avisa** (los primeros años son casi todo L1) y un periodo sin ningún par útil queda vacío con aviso, sin lanzar.
- **Ojo:** el prefiltro de `build_universe` descarta los pares sin categoría útil antes de consultar la red, así que `configs/categories.yaml` debe cubrir los activos que fueron líquidos en cada periodo, incluidos los hoy deslistados; si falta uno, no entra. Los pares en `BREAK` entran (D-13). Los periodos vacíos no aparecen en `load_universe`; para saber qué periodos hay, usar `universe_periods(load_universe_params())`.
- **Ojo:** `configs/universe.yaml` lo genera el subcomando `universe` y no se edita a mano. El subcomando no se prueba con red; su lógica vive en funciones de `src/` con tests sin red. El subcomando hace toda la red antes de escribir: si `write_universe` falla, hay que relanzarlo entero.
- **Refactor:** `universe.py` ya no define `DAY_MS` ni el peso de `/klines`: importa `DAY_MS` y `KLINES_WEIGHT` de `klines.py` (mismos valores, mismo comportamiento).
- **CLI:** el subcomando `universe` sigue el patrón de CLAUDE.md §6.6 (entrada en `_COMMANDS`, lógica en `src/`, códigos de salida 0/1/2: `ConfigError` da 2, fallos de red o disco dan 1, sin `print`).
- **Categorías ampliadas:** `configs/categories.yaml` pasa de 90 a 210 activos con un barrido real (2026-10-13) de los pares USDT líquidos (≥ 10 M) en alguna ventana anual desde 2017, incluidos los deslistados (EOS, LUNA, FTT, XMR, OMG, SRM, BCC/BCHABC…), y una categoría nueva `gaming` (SAND, MANA, AXS, CHZ…), que en 2022 tenía mucho peso. Sin ellos, el universo de 2019-2022 estaría sesgado hacia los supervivientes. Clasificación a mano: revisarla en el PR.
- **Decisión (con Alonso):** suelo de liquidez de **5 M** USDT/día y **40 pares** como tamaño objetivo (`max_pairs`), en vez de 10 M / 50. Con un umbral absoluto alto el tamaño seguía al ciclo de mercado (25 pares en 2023 tras FTX, 50 en 2022 y 2025); así el universo es estable (40 pares, 7-8 categorías) desde 2022 y el grafo apenas cambia de nodos.
- **Ojo (5 M es provisional):** 5 M no sale de un criterio formal: es el suelo más alto con el que se llega a 40 pares todos los años desde 2022. Solo decide cuando no hay 40 pares líquidos: en 2022, 2024 y 2025 manda el tope de 40 (el par más débil tiene 11-32 M), pero en **2023** 15 de los 40 pares están entre 5 y 10 M (y 6 en 2026). A 1 minuto lo que contamina es que el precio no se actualice (efecto Epps), y eso no lo mide el volumen diario. Criterio defendible pendiente: exigir que el 95-99 % de las velas de 1 minuto de la ventana tengan operaciones (`num_trades > 0`), empezando por medir los 15 pares débiles de 2023. Encaja en MIAX-030 o en la Épica 2; si sale mal, se sube el suelo o se cambia el criterio.
- **Prueba de humo real (2026-10-13):** `python -m miax.ingest universe` contra Binance, código 0. Pares por periodo (categorías): 2018: 2 (BTC, ETH) · 2019: 11 (2) · 2020: 10 (3) · 2021: 21 (5) · 2022 a 2026: 40 (7-8).
- **Ojo (para la tutoría):** **2018-2020 no dan un grafo útil** (2 a 11 nodos, casi todo L1). No es un fallo: Binance abrió en julio de 2017 y sus pares USDT empezaron el 17-08-2017 solo con BTC y ETH; en 2017 las altcoins cotizaban contra BTC y ETH (23 pares cada uno antes de octubre). Ni contra BTC hay más: para 2018 solo ETH, BCC y NEO superan 10 M. Sintetizar precios USDT desde pares BTC metería el movimiento de BTC en cada serie, justo lo que contamina el *lead-lag*. Conclusión: el grafo útil empieza en 2021-2022.
- **Ojo (MIAX-030):** hay pares que dejan de cotizar dentro de su periodo y su serie se corta: BCC (renombrado a BCHABC en nov-2018) en 2019, BCHABC (a BCH en nov-2020) en 2020, LUNA (colapso en mayo de 2022) en 2022. Además `LUNAUSDT` es Terra clásica hasta mayo de 2022 y Terra 2.0 después, en la misma serie. La cobertura por periodo debe detectarlo.
- **Ojo (descarga, siguiente paso):** aún no hay velas descargadas (`data/` vacío); `configs/universe.yaml` es la lista. El universo implica 104 símbolos distintos y ≈ 123 M velas de 1 minuto (cota superior: algunos pares dejan de cotizar a mitad), el 80 % en 2022-2026. Bajando cada símbolo de forma continua de su primer a su último periodo: ≈ 129 M velas, ≈ 129.000 peticiones, unas 3,5-4 h con el limitador actual (reanudable) y unos 5-8 GB en Parquet (estimación, sin medir). Los volcados mensuales de `data.binance.vision` lo acelerarían (ticket aparte).
- **Ojo (5 minutos):** las velas de 5m (y 15m, 1h) **no se descargan aparte**: salen de remuestrear las de 1m (MIAX-042, D-04), porque una vela de 5m de Binance es la agregación exacta de sus cinco de 1m y un único origen evita incoherencias entre horizontes. La reducción de ruido viene del horizonte, no de la descarga. Bajar 5m directamente (≈ 26 M velas, ≈ 45 min) solo valdría como comprobación del remuestreo.
- **Pendiente:** si se confirma la opción B, el grafo, la GNN y la validación (E3+) han de admitir un conjunto de nodos que cambia por periodo (ver Bloqueos abiertos). Descarga de velas 1m del universo, fuera de este ticket.

### 2026-10-12 · Raúl · Reglas de equipo
- **Hecho:** `CLAUDE.md`, `AGENTS.md` y `ONBOARDING.md` recogen las prácticas acordadas durante MIAX-024 a 029. Son reglas nuevas que **leen ya los Claude de los tres**:
  - **Refactorizar sí, a lo loco no** (§1 y §6.5): el código de tickets cerrados se refactoriza antes que duplicarlo o hacer una chapuza, con el mismo comportamiento y diciéndolo en el PR; nunca el de un ticket en curso; nada de mil líneas por ticket.
  - **Sin atribución a Claude** en commits ni en PR (§4.4 y §4.5).
  - **Prueba de humo real antes del PR** si el código toca Binance o datos reales (§6.4).
  - **Al mergear** (§4.7): squash, con un mensaje que reúne todos los commits de la rama (el repo ya lo hace por defecto), la tarjeta a Done y la rama borrada tras comprobar con `gh pr view` que está mergeada.
  - **Decisiones `D-??`** también en §5.
  - **Patrón de CLI con subcomandos** (§6.6).
- **Arreglo:** las tres líneas del CHANGELOG de MIAX-029 habían caído dentro de la sección ya publicada `v0.1.0`. Ahora están en `[Sin publicar]`, y `v0.1.0` queda idéntica a la de `main`. **Ojo al añadir líneas:** van siempre bajo `[Sin publicar]`.
- **Decisión:** cambio de proceso pedido por Raúl. No es un ticket del backlog, así que va en una rama `docs/` y no `feature/MIAX-XXX`.
- **Ojo (Piettro y Alonso):** la regla nº 1 queda matizada. "No aprovechar para refactorizar" significa no refactorizar a lo loco, no prohibir el refactor que el ticket necesita.

### 2026-10-12 · Raúl · MIAX-029
- **Hecho:** `miax.ingest.integrity` con `validate_klines(symbol, interval, *, base_dir) -> pd.DataFrame` (revisa la caché fichero a fichero y devuelve una tabla de incidencias `file`, `check`, `severity`, `count`, `detail`; vacía si todo está bien), más las funciones puras `check_klines_frame` (un fichero) y `find_gaps` (huecos de un conjunto de `open_time`). Comprobaciones: `unreadable`, `schema`, `duplicates` (dentro de un fichero y entre ficheros), `unordered`, `misaligned`, `close_time`, `wrong_month`, `ohlc` (errores) y `gaps` (aviso). Exportadas desde `miax.ingest`.
- **Decisión:** error frente a aviso: un fichero corrupto es un error; un hueco es un aviso, porque Binance tiene huecos reales (mantenimientos). Solo lectura: el validador nunca arregla ni borra nada. Revisión fichero a fichero con `read_parquet`, no con `load_klines`, para atribuir cada problema a su fichero y que uno ilegible no tumbe el resto. Con un esquema incorrecto no se hacen las demás comprobaciones de ese fichero. Para intervalos sin duración fija (`1M`) no se comprueban alineación, `close_time` ni huecos.
- **Refactor:** la regla de intervalo a ms vive ahora en `klines.py` (`interval_ms`, `candle_offset_ms` y las constantes `INTERVAL_PATTERN`, `UNIT_MS`, `WEEK_OFFSET_MS`, `DAY_MS`), exportadas desde `miax.ingest`, y la CLI (`_last_closed_ms`) la usa desde ahí con el mismo comportamiento.
- **Ojo (MIAX-034):** los huecos se reportan, pero no se clasifica su causa.
- **CLI:** ahora tiene subcomandos (`download`, `validate`, `pairs`) con un registro dict `{nombre: manejador}` en `__main__.py`; un comando nuevo es una entrada más. Sin subcomando se asume `download`, así que `python -m miax.ingest --symbol X` sigue funcionando. `validate` sale con 1 si hay errores o un símbolo sin caché; `pairs` sin flags lista el catálogo completo (incluido BREAK; `--quote`/`--status` filtran); `validate`/`pairs` aceptan `--output` (Parquet).
- **Arreglo:** `download_klines` pasa `KLINES_WEIGHT = 2` al limitador (el peso real de `/klines`); resuelve el "Ojo (MIAX-018)" anotado en MIAX-025.
- **Pendiente:** el informe de cobertura es MIAX-030.

### 2026-10-12 · Alonso · MIAX-027
- **Hecho:** `configs/categories.yaml` (mapa categoría → activos base, a mano: `l1`, `l2`, `defi`, `meme`, `exchange`, `infra`, `ai`, `stablecoin`, `pegged`) y, en `miax.ingest.universe`, `load_categories(path="categories.yaml") -> dict[str, str]` (mapa activo → categoría; ruta relativa contra `configs/`; `ConfigError` si falta el fichero, no es un mapeo de listas o un activo está en dos categorías) y `diversify_universe(pairs, categories, *, max_per_category=10, max_pairs=50, min_categories=4, excluded_categories=("stablecoin", "pegged")) -> pd.DataFrame` (función pura que añade la columna `category`, descarta activos sin categoría y de categorías excluidas, aplica el tope por categoría y luego el total respetando el orden de entrada, y lanza `ValueError` si quedan menos de `min_categories` categorías). Ambas exportadas desde `miax.ingest`.
- **Decisión:** un activo sin categoría se **descarta** (con log que lista sus `base_asset` para añadirlos al YAML), no se agrupa en "otros". `diversify_universe` no reordena por volumen: se queda con los primeros de cada categoría en el orden que recibe, así que hay que pasarle la salida de `filter_by_volume`. `stablecoin` y `pegged` (envueltos como WBTC/WBETH, oro como PAXG y fiat como EUR) se excluyen por defecto: su precio es plano o replica otro activo y darían *lead-lag* trivial. El YAML incluye ticker antiguo y nuevo de los renombrados (MATIC/POL, FTM/S, RNDR/RENDER).
- **Ojo (uso previsto, MIAX-028):** `list_pairs(quote_asset="USDT")` → `filter_by_listing_age` → `filter_by_volume` (ordena por volumen) → `diversify_universe(pairs, load_categories())`. El resultado lo congela MIAX-028 en `configs/universe.yaml`. No hay CLI.
- **Ojo:** las categorías están hechas a mano con el conocimiento de hoy (sesgo menor: un activo se clasifica como lo vemos ahora, no como cuando empezó el periodo). Conviene revisar el YAML en el PR.
- **Decisión (con Alonso):** YAML a mano y **no** CoinGecko en el pipeline. CoinGecko da varias etiquetas por moneda (habría que elegir una con otra regla subjetiva), sus tickers chocan con los de Binance (`S`, `MEME`, `NEIRO`…), sus etiquetas cambian sin histórico (no reproducible) y añade API, clave y fallos de red; para 30–50 activos su ventaja de escala no compensa. El sesgo de clasificar con la visión de hoy es el mismo en ambos.
- **Pendiente (MIAX-028):** para poder citar una fuente externa en la memoria, consultar CoinGecko **una sola vez** a mano, contrastar y justificar el YAML con esa consulta (anotando la fecha) y seguir usando el YAML congelado. Sin meter la API en el código.
- **Pendiente:** los valores por defecto (10 por categoría, 50 en total, mínimo 4 categorías) son provisionales; los definitivos se fijan en MIAX-028.

### 2026-10-12 · Alonso · MIAX-026
- **Hecho:** en `miax.ingest.universe`, `first_candle_time(symbols, reference_ms, *, client, base_url) -> pd.Series` (una llamada por símbolo a `/klines` con `interval=1d`, `startTime=0`, `endTime=reference_ms`, `limit=1` y peso 2; devuelve el `open_time` de la primera vela, `NA` si no hay ninguna no posterior a la referencia, dtype `Int64`) y `filter_by_listing_age(pairs, first_candles, reference_ms, min_listing_days=90) -> pd.DataFrame` (función pura que añade la columna `first_candle_ms` y conserva los pares con primera vela `<=` `reference_ms - min_listing_days` días, límite inclusivo). Ambas exportadas desde `miax.ingest`. Tests en `tests/unit/test_universe.py` (19 nuevos, sin red), con los límites de fuga comprobados a ±1 ms.
- **Decisión:** `first_candle_time` no usa `download_klines`, que pagina hasta `end_ms` y con `limit=1` pediría día a día. Los pares sin velas hasta la referencia (`endTime` es inclusivo) se descartan y el log cuenta cuántos por recientes y cuántos por no tener velas. `filter_by_listing_age` lanza `ValueError` si `first_candles` trae símbolos repetidos. A diferencia de `filter_by_volume`, `filter_by_listing_age` no reordena: conserva el orden de entrada, así que se puede combinar con el filtro de volumen en cualquier orden.
- **Ojo (uso previsto, MIAX-028):** `reference_ms` debe ser el inicio de la ventana de volumen de MIAX-025. Así la mediana de volumen nunca incluye la euforia de lanzamiento: el caso ARBUSDT que anotó Raúl queda fuera por construcción.
- **Ojo:** el valor de 90 días es provisional; el definitivo irá a `configs/universe.yaml` (MIAX-028). No hay CLI.
- **Ojo (MIAX-027/028, acordado con Alonso):** el universo es solo pares **contra USDT** (README), sin cruces entre criptos (ETHBTC ≈ ETHUSDT/BTCUSDT da *lead-lag* mecánico, y el volumen de MIAX-025 solo está en USDT si el par cotiza contra USDT). Y **no se descargan stablecoins contra USDT** (USDCUSDT, FDUSDUSDT, TUSDUSDT…): pasan el filtro de volumen de sobra, pero su precio es plano y solo meten ruido. Ninguno de los dos filtros está forzado aún en el código (`first_candle_time`/`filter_by_listing_age` aceptan cualquier par; `list_pairs` tiene `quote_asset` opcional): hay que aplicarlos al construir y congelar el universo.
- **Pendiente:** exigir días con velas dentro de la ventana para pares parados o deslistados a mitad (MIAX-030).

### 2026-10-12 · Raúl · MIAX-025
- **Hecho:** `miax.ingest.universe` con `median_daily_quote_volume` (mediana del volumen diario en USDT, `quote_asset_volume` de las velas `1d`, de cada símbolo en una ventana `start_ms`–`end_ms`) y `filter_by_volume` (función pura que une el volumen a `pairs` en la columna `median_quote_volume`, deja los de volumen `>=` umbral y los ordena de mayor a menor). Ambas exportadas desde `miax.ingest`.
- **Decisión:** D-14: la ventana es obligatoria y debe ser anterior al periodo de estudio. Se usa la mediana, que no se deja arrastrar por un día de *pump*. Umbral por defecto de 10 M USDT/día (el valor definitivo irá a `universe.yaml`). Un par sin velas en la ventana da `NaN` y se descarta, y el log cuenta cuántos. Los pares en `BREAK` se evalúan igual que los demás (D-13). Sin cliente inyectado, se crea uno con `RateLimiter()` y se cierra al terminar.
- **Ojo (MIAX-031, importante):** los pares deslistados SÍ conservan su histórico de velas. Comprobado en vivo con FTTUSDT, SRMUSDT y 1INCHUPUSDT: devuelven velas de cuando cotizaban (FTT, junio de 2022: mediana diaria de 11,6 M) y 0 velas fuera de su periodo de vida. El riesgo 8 del README ("la API no sirve histórico de pares deslistados") es falso para los pares que siguen en `exchangeInfo` como `BREAK`; el sesgo de supervivencia se puede corregir en buena parte, y MIAX-031 debe corregir el README.
- **Ojo (MIAX-018):** `download_klines` llama a `get` con el peso por defecto (1), pero en Binance `/klines` pesa 2. El limitador se resincroniza con la cabecera `X-MBX-USED-WEIGHT-1M`, así que no es grave. Queda anotado sin tocar.
- **Ojo (MIAX-026):** un par listado dentro de la ventana se mide con pocos días, normalmente de euforia de lanzamiento (ARBUSDT en el primer trimestre de 2023: solo 9 días y una mediana de 328 M). La antigüedad mínima debe exigir días suficientes dentro de la ventana.
- **Pendiente:** ventana y umbral reales en `configs/universe.yaml` (MIAX-028). El código no puede impedir que alguien pase una ventana que solape el estudio; eso lo vigila MIAX-028, donde se fijan las fechas.

### 2026-10-12 · Raúl · MIAX-024
- **Hecho:** `miax.ingest.exchange_info` con `fetch_exchange_info` (GET crudo, peso 20), `pairs_from_exchange_info` (pares a DataFrame) y `list_pairs` (catálogo completo con filtros opcionales por `quote_asset` y `status`), exportadas desde `miax.ingest`.
- **Decisión:** D-13: los pares en `BREAK` se conservan; filtrar es cosa del universo. El peso 20 se pasa al limitador. Sin modo listado en la CLI: se usa como función.
- **Ojo (MIAX-031):** `exchangeInfo` trae 2.349 pares en `BREAK` (254 contra USDT) de 3.724: el catálogo sí deja ver pares parados o deslistados, aunque el README diga que la API no sirve su histórico. Hay que comprobar si sus klines siguen disponibles.
- **Ojo:** la respuesta pesa unos 17 MB; no conviene pedirla en bucle.
- **Pendiente:** modo listado en la CLI, si se quiere, en un ticket aparte.

### 2026-10-11 · Raúl · Release v0.1.0
- **Hecho:** release `v0.1.0`, que cierra la Épica 0: CHANGELOG versionado, merge `develop` → `main` y tag. La regla de ramas exige ahora el check `Lint y tests` del CI para mergear en `main` y `develop`, que era lo que le faltaba a MIAX-015. Orden: tarjetas 019–023 a Done, ramas `feature/*` ya mergeadas borradas y tabla "Ahora mismo" al día.
- **Decisión:** la release se llama `v0.1.0` aunque el CHANGELOG ya tenía una entrada `v0.1.0` del 18-09, porque esa nunca llegó a `main` ni tuvo tag; se fusiona en esta. Incluye MIAX-016 a MIAX-023 de la Épica 1 porque ya estaban en `develop`. La siguiente MINOR, `v0.2.0`, sale al cerrar la Épica 1.
- **Ojo:** desde hoy no se puede mergear con el CI en rojo. Los PR mergeados con *squash* (#10, #15, #18) dejan una rama que git no ve contenida en `develop`; se puede borrar igual tras comprobar que su PR está mergeado.
- **Pendiente:** un test que falle si en ESTADO hay dos decisiones con el mismo número o queda un `D-??` (propuesto, sin ticket).

### 2026-10-09 · Alonso · MIAX-023
- **Hecho:** `--symbol` acepta una lista separada por comas (`BTCUSDT,ETHUSDT,SOLUSDT`); un solo símbolo sigue valiendo. `download_symbols(symbols, interval, start_ms, end_ms, *, client, base_dir, base_url) -> dict[str, int | None]` en `src/miax/ingest/__main__.py` recorre los símbolos en orden con `update_symbol`, loguea `[i/N] SÍMBOLO: K velas nuevas` y un resumen final (OK, fallidos y cuáles). Tests en `tests/unit/test_batch.py` con `update_symbol` parcheado: sin red ni disco.
- **Decisión:** `main` crea **un único** `HttpClient(rate_limiter=RateLimiter())` para toda la tanda y lo pasa a `download_symbols`: el presupuesto de peso es por IP, así que un limitador compartido evita el baneo colectivo. Es secuencial, sin concurrencia. **Aislamiento de errores por símbolo:** `HttpError`/`ValueError`/`OSError` de uno se loguean y la tanda sigue; el resultado de ese símbolo es `None`. `main` devuelve 0 si todos OK, 1 si alguno falló (tras procesarlos todos) y 2 para fechas/rango inválidos o lista de símbolos vacía. Se descartan espacios, vacíos y repetidos manteniendo el orden.
- **Ojo:** `download_symbols` vive en `__main__.py` (los tests de la CLI parchean `cli.update_symbol`). El último mensaje del log ya no es el resumen por símbolo de MIAX-022, sino `Resumen: N simbolos OK, M fallidos`; el test de resumen de `test_cli.py` se adaptó. Un 429/418 o una red caída persistente (`RetriesExhaustedError`) puede repetirse en todos los símbolos siguientes, y con un 418 (IP baneada) seguir golpeando alarga el baneo. Por eso hay un **cortacircuitos**: `download_symbols(..., max_consecutive_failures=3)` cuenta los `RetriesExhaustedError` consecutivos (un éxito o un fallo propio del símbolo —`ClientError`, `ValueError`, `OSError`, cualquier `requests.exceptions.RequestException`— lo resetea) y al llegar al límite corta la tanda: los símbolos restantes quedan con el valor `NOT_PROCESSED` (-1, distinto de `None` = fallido), se loguea el motivo y `main` devuelve 1. `_parse_symbols` normaliza a **mayúsculas** y deduplica después (`btcusdt` y `BTCUSDT` son el mismo).
- **Pendiente:** universo y `exchangeInfo` (MIAX-024 y siguientes), huecos e integridad (MIAX-029).

### 2026-10-09 · Alonso · MIAX-022
- **Hecho:** `src/miax/ingest/__main__.py`: `python -m miax.ingest --symbol BTCUSDT [--interval] [--start] [--end] [--base-url] [--base-dir]`. `main(argv) -> int` construye `HttpClient(rate_limiter=RateLimiter())` (se cierra con `with`), llama a `update_symbol` y loguea un resumen con las velas nuevas. Tests en `tests/unit/test_cli.py` con `update_symbol` y el reloj parcheados: sin red ni disco.
- **Decisión:** defaults `--interval 1m`, `--start` = hace 30 días y `--end` = ahora (UTC). `--start`/`--end` son `YYYY-MM-DD` en UTC: `--start` es 00:00:00 de ese día y `--end` es **inclusivo** (hasta 23:59:59.999 de ese día). Fecha inválida o `start > end` devuelve 2 y `HttpError`/`ValueError` de la descarga devuelven 1, siempre con mensaje por el logger y sin traceback (`OSError` incluido). Con esto el `RateLimiter` queda enganchado en la CLI.
- **Ojo:** el limitador usa peso 1 por petición (`get` por defecto) aunque klines con `limit=1000` pesa 2: el presupuesto de 1200/min es conservador y la cabecera `X-MBX-USED-WEIGHT-1M` lo resincroniza. **La CLI nunca descarga la vela en curso**: recorta `end` a la última vela cerrada (`close_time <= ahora`) para intervalos de duración fija (`s/m/h/d/w`; las semanales alinean a lunes 00:00 UTC); para `1M` u otros no mapeables no recorta y avisa con un `warning`. `update_symbol`/`download_klines` no filtran la vela en curso por sí mismos. `--symbol` se pasa tal cual: Binance exige mayúsculas.
- **Pendiente:** descarga masiva de N símbolos y progreso (MIAX-023), universo y `exchangeInfo` (MIAX-024 y siguientes), huecos e integridad (MIAX-029).

### 2026-10-09 · Alonso · MIAX-021
- **Hecho:** `src/miax/ingest/resume.py` con `cached_last_open_time(symbol, interval, *, base_dir) -> int | None` (el `open_time` máximo del mes cacheado más reciente, `None` sin cache) y `update_symbol(symbol, interval, start_ms, end_ms, *, client, base_dir, base_url, limit) -> int` (nº de velas nuevas), ambas exportadas desde `miax.ingest`. Relanzar con el rango ya cacheado no hace ninguna llamada al cliente. Tests en `tests/unit/test_resume.py` con un cliente espía, sin red y sobre `tmp_path`.
- **Decisión:** punto de reanudación `max(última_cacheada + 1, start_ms)`; si supera `end_ms` se devuelve 0 sin descargar. Como `save_klines` sobrescribe el mes entero, antes de guardar se fusiona lo nuevo con lo cacheado desde el inicio del mes de reanudación (`load_klines`), se deduplica por `open_time` (gana lo nuevo) y se ordena; los meses anteriores no se reescriben. No se ha tocado `download_klines`, `save_klines` ni `load_klines`.
- **Ojo:** es reanudación **hacia delante**: solo mira la última vela cacheada, no rellena huecos internos ni detecta ficheros mensuales ausentes en medio (MIAX-029). Si la última vela cacheada estaba sin cerrar, no se vuelve a pedir (`+1`); la validación de la vela abierta es de MIAX-029.
- **Ojo:** tampoco se cubre el hueco **previo** a la primera vela cacheada (p. ej. cache desde marzo y se pide desde enero: enero y febrero no se piden); queda para MIAX-029, igual que los huecos internos.
- **Pendiente:** CLI (MIAX-022), descarga masiva de N símbolos (MIAX-023), huecos e integridad (MIAX-029).

### 2026-10-09 · Alonso · MIAX-020
- **Hecho:** `src/miax/ingest/store.py` con `save_klines(df, symbol, interval, *, base_dir="data/klines") -> list[Path]` y `load_klines(symbol, interval, *, base_dir="data/klines", start_ms=None, end_ms=None) -> DataFrame`, exportadas desde `miax.ingest`. Guarda un Parquet por mes con `miax.utils.io`; releer devuelve lo guardado con los mismos valores y dtypes (`assert_frame_equal`). Tests en `tests/unit/test_store.py` con `tmp_path`, sin red y sin tocar `data/`.
- **Decisión:** layout `{base_dir}/{symbol}/{interval}/{YYYY-MM}.parquet`; el mes se deriva de `open_time` en UTC. Se ordena por `open_time` (orden estable) antes de escribir y el índice se resetea, de modo que guardar es determinista. En lectura el rango `[start_ms, end_ms]` es inclusivo: primero se saltan los ficheros de meses que no lo solapan y luego se filtran las filas. Sin datos, `load_klines` devuelve un DataFrame vacío con las columnas y dtypes de `KLINE_DTYPES`.
- **Validación:** `symbol` e `interval` solo admiten `[A-Za-z0-9_.-]` (y no `.`/`..`), si no `ValueError`; `save_klines` descarta las columnas extra (persiste solo `KLINE_COLUMNS`) y falla con `ValueError` si faltan; `load_klines` ignora con aviso los `.parquet` cuyo nombre no sea `YYYY-MM`.
- **Ojo:** guardar un mes que ya existe lo **sobrescribe** entero con las velas del DataFrame que se pasa; no fusiona con lo cacheado (eso es de la reanudación, MIAX-021). No se valida ni limpia nada: duplicados, huecos, desorden y la última vela posiblemente sin cerrar se guardan tal cual (MIAX-029). Un mes con velas de varias descargas debe guardarse con todas ellas a la vez.
- **Pendiente:** reanudación sin redescargar lo cacheado (MIAX-021), CLI (MIAX-022), descarga masiva (MIAX-023), validación de integridad (MIAX-029).
### 2026-10-09 · Alonso · MIAX-019
- **Hecho:** `src/miax/ingest/ratelimit.py` con `RateLimiter(max_weight=1200, window_seconds=60.0, clock=time.monotonic)`, exportado desde `miax.ingest`. `acquire(weight=1)` reserva peso y, si no cabe en la ventana deslizante, duerme (`time.sleep`) lo justo hasta que salgan los consumos necesarios; `sync_used_weight(used)` alinea la cuenta con `X-MBX-USED-WEIGHT-1M`. `HttpClient` acepta `rate_limiter` (por defecto `None`, sin cambios) y `get(url, params, weight=1)`: llama a `acquire(weight)` antes de cada intento y a `sync_used_weight` con la cabecera de cada respuesta. Tests en `tests/unit/test_ratelimit.py` y `tests/unit/test_http.py`, sin red y sin esperas reales.
- **Decisión:** presupuesto por defecto 1200 de peso por 60 s, deliberadamente conservador (Binance spot da hoy 6000/min y antes 1200/min); ajustable por parametro y pendiente de leerse desde `configs/`. `sync_used_weight` solo sube la cuenta (el exceso se fecha ahora), nunca la baja, porque la cuenta propia incluye peticiones en vuelo que el servidor aun no ha contado. `acquire` se llama en cada intento, reintentos incluidos, porque cada uno consume peso. Un `weight` mayor que `max_weight` lanza `ValueError` en vez de esperar para siempre.
- **Ojo:** `download_klines` NO crea ni usa un limitador por defecto; quien quiera frenar pasa `HttpClient(rate_limiter=RateLimiter())` como `client`. Cada klines con `limit=1000` pesa 2 en spot: habra que pasar `weight` si se quiere precision (MIAX-022/023). El limitador no es thread-safe.
- **Pendiente:** Parquet (MIAX-020), reanudación (MIAX-021), CLI (MIAX-022) y descarga masiva (MIAX-023), donde se decidirá cablear el limitador; leer `max_weight` de `configs/`.

### 2026-10-02 · Alonso · MIAX-018
- **Hecho:** `miax.ingest.download_klines(symbol, interval, start_ms, end_ms, *, client, base_url, limit)` descarga un rango de `/api/v3/klines` paginando hacia delante: tras cada bloque el cursor pasa a `open_time` de la última vela + 1 ms. Devuelve un DataFrame con las columnas crudas de Binance (`open_time`, `open`, `high`, `low`, `close`, `volume`, `close_time`, `quote_asset_volume`, `num_trades`, `taker_buy_base`, `taker_buy_quote`, `ignore`) y sus dtypes; vacío con esas columnas si no hay datos. Usa `HttpClient`. Tests en `tests/unit/test_klines.py` con un Binance simulado, sin red.
- **Decisión:** la paginación corta si el bloque viene vacío, si el cursor no avanza (ese bloque no se añade) o si el cursor supera `end_ms`; ya NO corta por bloque corto, a costa de una llamada extra (vacía) por símbolo. `endTime` es inclusivo en Binance, así que `end_ms` también. Si no se pasa cliente se crea un `HttpClient` y se cierra al terminar; uno inyectado no se cierra. `limit` fuera de 1..1000 lanza `ValueError`.
- **Ojo:** devuelve datos crudos, sin ordenar ni deduplicar más allá de lo que da el cursor; sin retornos, alineación ni huecos (Épica 2). Los huecos de mercado se saltan y el bloque se rellena hasta `limit`, así que un bloque corto no indica fin de datos (por eso el corte ya no se basa en él); la validación de cobertura es de MIAX-029/030. Con `end_ms` futuro la última vela puede venir sin cerrar: a tratar en MIAX-020/029.
- **Pendiente:** guardar en Parquet (MIAX-020), reanudación (MIAX-021), CLI (MIAX-022), descarga masiva de N símbolos (MIAX-023).

### 2026-10-02 · Alonso · MIAX-017
- **Hecho:** `HttpClient.get` trata 429 y 418 como reintentables. Lee `Retry-After` (segundos enteros o fecha HTTP RFC 1123, vía el helper privado `_retry_after_seconds`) y espera ese tiempo; si falta o no se entiende, usa el backoff exponencial. Nuevo parámetro `max_retry_after` (300 s por defecto, debe ser > 0) que recorta la espera por `Retry-After`. Cada reintento se loguea con el código y los segundos. Tests en `tests/unit/test_http.py`, sin red y con `time.sleep` parcheado.
- **Decisión:** 429 y 418 cuentan contra `max_retries`; al agotarse lanzan `RetriesExhaustedError`, igual que los transitorios. El tope solo recorta el valor de `Retry-After`, no el backoff de fallback. Una fecha HTTP ya pasada espera 0 s.
- **Ojo:** el contrato de 418/429 cambió respecto a MIAX-016: ya no lanzan `ClientError`, se reintentan. El resto de 4xx (400, 404, 499...) sigue lanzando `ClientError` sin reintento. `test_4xx_is_not_retried` ya no incluye 418/429.
- **Pendiente:** limitador proactivo por peso de IP para no llegar al 429 (MIAX-019); klines, paginación y exchangeInfo (MIAX-018 y siguientes).

### 2026-10-02 · Alonso · MIAX-016
- **Hecho:** `src/miax/ingest/http.py` con `HttpClient.get(url, params)` sobre `requests.Session`: timeout configurable (10 s por defecto) y reintentos con backoff exponencial (`backoff_base * 2**(reintento-1)`, base 0.5 s, sin jitter) ante `ConnectionError`, `Timeout` y 5xx. Cada reintento se loguea con `get_logger`. Excepciones `HttpError`, `ClientError` (4xx, con `status_code`) y `RetriesExhaustedError`, exportadas desde `miax.ingest`. Tests en `tests/unit/test_http.py`, sin red.
- **Decisión:** `max_retries=3` son reintentos tras el primer intento (4 intentos en total). Un 4xx lanza `ClientError` sin reintentar, 429 y 418 incluidos por ahora. Al agotar reintentos se lanza `RetriesExhaustedError` encadenada con la causa.
- **Ojo (MIAX-017):** el 429/418 con `Retry-After` se engancha en la rama 4xx de `HttpClient.get`, hoy un `ClientError`. Los tests parchean `requests.Session.get` y `miax.ingest.http.time.sleep`.
- **Pendiente:** limitador por peso de IP (MIAX-019); klines, paginación y exchangeInfo (MIAX-018 y siguientes).

### 2026-10-01 · Alonso · MIAX-014
- **Hecho:** `.github/pull_request_template.md` (qué hace, cómo se ha comprobado, qué queda fuera, riesgo de fuga de datos y checklist), plantillas de issue `incidencia.md` y `tarea.md` en `.github/ISSUE_TEMPLATE/`, y `config.yml` con los enlaces al tablero de Trello y al backlog.
- **Decisión:** `blank_issues_enabled: false`: los issues salen siempre de una plantilla. La plantilla de tarea recuerda que el backlog canónico es `docs/project/BACKLOG.md` (CLAUDE.md §7.5) y que lo normal es añadir el ticket ahí. La sección de fuga de datos tiene preguntas guía para obligar a razonarla, no a escribir "ninguno".
- **Ojo:** GitHub solo precarga la plantilla de PR cuando ya está en la rama base (`develop`), así que no se ve hasta el merge de este ticket. Las etiquetas de las plantillas de issue (`bug` en incidencia, `enhancement` en tarea) existen por defecto en GitHub, no hay que crear ninguna.
- **Pendiente:** el CI (MIAX-013) y la protección de ramas (MIAX-015) quedan en sus tickets.
### 2026-10-01 · Alonso · MIAX-013
- **Hecho:** `.github/workflows/ci.yml`: en cada PR a `develop` o `main` instala `requirements.txt` y luego `pip install -e .` sobre Python 3.12 (ubuntu-latest, caché de pip), y ejecuta `ruff check .`, `ruff format --check .` y `pytest`. El job se llama `Lint y tests`. Añadido `.gitattributes` con `eol=lf`.
- **Decisión:** `.gitattributes` fuerza LF para que `ruff format --check` dé el mismo resultado en el runner Linux aunque alguien suba CRLF desde Windows. No se ha renormalizado ningún fichero existente.
- **Ojo:** en el CI `tests/unit/test_seeds.py` salta el test de torch (no está instalado hasta MIAX-115), es lo esperado. Cuando entren torch y tigramite (MIAX-115/077) irán en `requirements.txt` y el CI los instalará solo.
- **Pendiente:** marcar el check `Lint y tests` como *required* en la protección de `develop` y `main` es MIAX-015. Hasta entonces el check aparece en el PR pero no bloquea el merge.

### 2026-10-01 · Piettro · MIAX-008
- **Hecho:** `[tool.pytest.ini_options]` en `pyproject.toml` (`testpaths = ["tests"]`, `--strict-markers`, `--strict-config`, `--import-mode=importlib`, marcador `integration`); guardia de red `autouse` en `tests/conftest.py`; tests que comprueban ambas cosas. `pytest` recoge 22 tests de `tests/unit` y 8 de `tests/integration`, y pasa en verde con `ruff check .` y `ruff format --check .` limpios.
- **Decisión:** D-12 (era D-11; se renumeró al integrar con `develop`, porque MIAX-012 ya había tomado D-11). La guardia bloquea solo `socket.socket.connect`/`connect_ex`, no la creación del socket entero, porque `test_package.py` y `test_ruff_config.py` ya lanzan subprocesos y eso tiene que seguir funcionando.
- **Ojo:** la guardia no protege dentro de subprocesos propios (otro proceso no hereda el `monkeypatch`); MIAX-032 tendrá que cuidarlo con sus propios fixtures al escribir los tests de ingesta.
- **Ojo (entorno):** `pip install -e .` falla con Python 3.14; el `requires-python` exige 3.12 (D-06). Si tenéis otra versión por defecto, el `.venv` del repo hay que crearlo con `py -3.12 -m venv .venv`.
- **Pendiente:** el workflow de CI que ejecuta `pytest` en cada PR es MIAX-013, que ya no está bloqueado por este ticket.

### 2026-09-29 · Alonso · MIAX-012
- **Hecho:** `src/miax/utils/logging.py` con `get_logger(name, level="INFO")`, exportada desde `miax.utils`. Formato común `timestamp UTC | NIVEL | nombre | mensaje`, nivel configurable (texto o entero) e idempotente: llamarla otra vez sobre el mismo logger solo actualiza el nivel, no añade handlers. Tests en `tests/unit/test_logging.py`.
- **Decisión:** D-11. `T20` activado en ruff y exigido solo en `src/` (eximido en `notebooks/**`, `scripts/**` y `tests/**` vía `per-file-ignores`). El número es D-11 y no D-10 porque D-10 ya lo tomó MIAX-011.
- **Ojo:** el logger tiene `propagate = False` y escribe a stderr; `caplog` de pytest no lo ve, hay que leer `capsys` (ver los tests). El fichero se llama `logging.py` dentro de `miax.utils`: dentro del paquete se importa como `miax.utils.logging`, nunca con `import logging` desde un script ejecutado dentro de `src/miax/utils/`.
- **Pendiente:** enganchar `get_logger` en el resto del pipeline; cada módulo lo usará cuando exista. Cierra el pendiente sobre `T20` de MIAX-007.

---

### 2026-09-28 · Raúl · MIAX-011
- **Hecho:** `miax.utils.io` con `resolve_repo_path`, `write_parquet` y `read_parquet`, exportadas desde `miax.utils`. Las rutas relativas se resuelven desde la raíz del repo y las absolutas se respetan. Releer devuelve el mismo DataFrame: tipos, índice UTC, categóricos, `Int64` con nulos y `MultiIndex` en filas o columnas.
- **Decisión:** D-10. La escritura es atómica (`.tmp` + `os.replace`), para que una descarga cortada no deje un Parquet corrupto que la reanudación (MIAX-021) daría por bueno. Se reutiliza `REPO_ROOT` de `utils/config.py` en lugar de duplicarlo.
- **Ojo:** `DatetimeIndex.freq` no se guarda en Parquet y vuelve como `None`. Si alguien la necesita, que la reasigne a mano; nunca inferida, porque con huecos (D-02) no hay frecuencia regular.
- **Ojo (MIAX-020/043):** el particionado por mes o símbolo se añade en esos tickets, sobre estas funciones.
- **Pendiente:** `config.py` e `io.py` resuelven rutas cada uno a su manera; unificarlo en un `paths.py` sería un ticket de limpieza.

### 2026-09-27 · Alonso · MIAX-010
- **Hecho:** `src/miax/utils/seeds.py` con `set_global_seed(seed: int = 42)` (fija `random`, el estado global de `numpy` y, si está instalado, `torch` con CUDA) y `make_rng(seed: int = 42)` (devuelve `numpy.random.default_rng(seed)` para no depender del estado global). Exportadas desde `miax.utils`. Tests en `tests/unit/test_seeds.py`.
- **Decisión:** `set_global_seed` fija además el estado global legacy de numpy (`np.random.seed`, con `noqa: NPY002` justificado en el propio comentario) porque librerías externas (sklearn, etc.) llaman a `np.random.*` directamente y no basta con el `Generator` de `make_rng`.
- **Ojo:** torch no está en `requirements.txt` todavía (llega en MIAX-115); `set_global_seed` hace `import torch` dentro de un `try/except ImportError` y no falla si no está. El test de torch usa `pytest.importorskip("torch")` y se salta limpio en este entorno.
- **Pendiente:** enganchar la semilla en el resto del pipeline es MIAX-174. `PYTHONHASHSEED` y el determinismo profundo de cuDNN quedan fuera de este ticket (MIAX-176 y posteriores).
### 2026-09-27 · Alonso · MIAX-009
- **Hecho:** `miax.utils.config.load_config` lee un YAML tipado contra un dataclass, con `ConfigError` claro (nombra el campo) si falta un campo obligatorio, si un tipo no coincide o si hay un campo desconocido. YAML de ejemplo en `configs/example.yaml` con su `ExampleConfig`, y tests en `tests/unit/test_config.py`.
- **Decisión:** validación con dataclasses de la librería estándar + pyyaml, sin pydantic ni otras deps nuevas (ya decidido por el humano). `load_config` acepta rutas absolutas (útil para tests con `tmp_path`), o si no relativas a la raíz del repo, o si no relativas a `configs/`, probando en ese orden.
- **Ojo:** `_check_field_type` solo valida tipos simples (`str`, `int`, `float`, `bool`); con genéricos (`list[str]`, `dict[...]`) lanza `ConfigError` claro en vez de validarlos, sin profundizar en su contenido. Quien defina un esquema con esos tipos deberá ampliar la función.
- **Pendiente:** los configs reales del pipeline (universo, horizontes, etc.) no se han tocado; quedan para sus tickets correspondientes.

### 2026-09-21 · Raúl · MIAX-007
- **Hecho:** configuración de ruff en `pyproject.toml` y repo en limpio (`ruff check .` y `ruff format --check .`). Arreglados los 4 avisos que había, sin cambiar comportamiento.
- **Decisión:** D-09. Ruff 0.16.8 ya trae unas 430 reglas por defecto (F, UP, B, SIM, DTZ, PT, RUF, PL*…), así que se amplían con `ANN`, `D`, `NPY` y `E501` en lugar de sustituirlas. Los notebooks no exigen tipos, docstrings ni longitud de línea.
- **Ojo:** las reglas por defecto de ruff cambian entre versiones; al subir ruff en `requirements.txt` hay que volver a pasar `ruff check .`.
- **Ojo (MIAX-013/179):** el CI tiene que ejecutar `ruff check .` y `ruff format --check .`.
- **Pendiente:** prohibir `print` en `src/` (regla `T20`) se decide en MIAX-012, junto con el logging.

### 2026-09-21 · Raúl · MIAX-006
- **Hecho:** `pyproject.toml` con setuptools; `miax` y sus 7 subpaquetes son paquetes Python, y con `pip install -e .` se importa `miax` desde cualquier directorio.
- **Decisión:** las dependencias solo viven en `requirements.txt` (D-08). `requires-python = ">=3.12,<3.13"` hace cumplir D-06. La versión es estática, `0.1.0`.
- **Ojo (MIAX-007/008):** la configuración de ruff (`[tool.ruff]`) y de pytest (`[tool.pytest.ini_options]`) va en este `pyproject.toml`.
- **Ojo (MIAX-013):** el CI tiene que instalar con `pip install -r requirements.txt` y luego `pip install -e .`; sin lo segundo falla `tests/integration/test_package.py`.
- **Pendiente:** la `version` de `pyproject.toml` se sube a mano en cada release (PR a `main`).

### 2026-09-21 · Raúl · MIAX-005
- **Hecho:** `requirements.txt` con las 13 dependencias directas del stack base fijadas con `==` (datos, cálculo, grafos, configuración YAML, visualización y desarrollo), un test que falla si alguna entra con rango, y los pasos de instalación en `ONBOARDING.md`.
- **Decisión:** Python 3.12 (D-06) y versión exacta siempre (D-07). Cada pieza del stack se fija en su ticket: torch y PyG Temporal en MIAX-115, tigramite en MIAX-077, el congelado completo con transitivas en MIAX-173. Se ha creado `develop` desde `main` para que los PR de la Épica 0 tengan base; protegerla sigue siendo MIAX-015.
- **Trello:** la asignación de tarjetas pasa a ser una etiqueta por persona (🟢 `raul`, 🔵 `piettro`, 🟣 `alonso`, ya creadas en el tablero) y la URL del PR va al final de la descripción, porque el conector no puede asignar miembros ni comentar. Reglas en `CLAUDE.md` §7.4 y `AGENTS.md`. Entra en este PR por decisión del humano, aunque no es de este ticket.
- **GitHub CLI:** los PR se abren con `gh`. Instalación, reinicio de Claude y `gh auth login` en `CLAUDE.md` §9.1 y `ONBOARDING.md` §2.4. También entra por decisión del humano.
- **Ojo (MIAX-115):** PyG Temporal 0.56.2 (la última, de julio de 2025) exige `torch-scatter` y `torch-sparse`, y su `import` los necesita (`evolvegcno`). Solo hay ruedas precompiladas en data.pyg.org hasta torch 2.12.x, así que torch ≤ 2.12.1. El fallo de `to_dense_adj` con PyG ≥ 2.5 ya está parcheado en 0.56.x. Fija `decorator==4.4.2`. Resolución comprobada con `uv pip compile --no-build` (Python 3.12): esta base + tigramite 5.2.10.1 + torch 2.12.1 + PyG 2.8.0.post1 + PyG Temporal 0.56.2 resuelve en Windows, Linux y macOS 14+. El import real no está probado.
- **Ojo (MIAX-006):** el `requires-python` del `pyproject.toml` tiene que casar con D-06.
- **Pendiente:** torch en CPU o en CUDA se decide en MIAX-115.

### 2026-09-18 · Raúl · MIAX-001 a MIAX-004
- **Hecho:** repositorio arrancado. README como prototipo de la memoria, estructura de carpetas, `CLAUDE.md` con las reglas del equipo, cuatro subagentes, comprobación de entorno al arrancar sesión, backlog completo y este fichero.
- **Decisión:** los diagramas se hacen en SVG a mano, no con modelos de imagen. Un modelo de difusión no dibuja una flecha etiquetada `SOL → LINK con 2 min de retardo`. El banner de portada sí es generado, porque es decorativo.
- **Decisión:** los cuatro subagentes van versionados en `.claude/agents/`, no en la configuración personal de cada uno. Así los tres ejecutan el mismo flujo sin tener que configurarlo.
- **Ojo:** `check_setup.py` falla a propósito si estás en `main`. No es un bug.
- **Pendiente:** crear la rama `develop` y proteger `main` (MIAX-015). Nadie ha escrito código de producción todavía.
