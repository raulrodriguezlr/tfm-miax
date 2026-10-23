# Changelog

Qué hace cada versión del proyecto. No qué commits tiene.

Formato basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/). Versionado según [CLAUDE.md §4.2](CLAUDE.md).

- **Snapshot** (`v0.X.0-snapshot.N`) — cada merge de `feature/*` a `develop`.
- **Release** (`v0.X.0`) — cada merge de `develop` a `main`. Lleva tag de git.
- **MINOR** sube al cerrar una épica. **PATCH**, al arreglar algo ya publicado.
- `v1.0.0` es la versión que se entrega al tribunal.

Cada línea explica **qué hace**, en castellano, y acaba con su ticket.

---

## [Sin publicar]

### Añadido
- Catálogo de pares de Binance: `list_pairs` consulta `/exchangeInfo` y devuelve cada par con su estado (`TRADING` o `BREAK`), activos base y cotizado y si admite spot y margen, con filtros opcionales por activo cotizado y estado (MIAX-024)
- Filtro de universo por volumen: `median_daily_quote_volume` mide la mediana del volumen diario en USDT en una ventana dada y `filter_by_volume` descarta los pares por debajo de un umbral explícito (10 M USDT/día por defecto) o sin velas en la ventana (MIAX-025)
- Filtro de universo por antigüedad: `first_candle_time` consulta la primera vela diaria de cada símbolo hasta una fecha de referencia y `filter_by_listing_age` descarta los pares con menos de 90 días de cotización (por defecto) o sin velas, para dejar fuera tokens recién listados con datos ruidosos (MIAX-026)
- Diversificación del universo por categoría: `configs/categories.yaml` clasifica los activos (L1, L2, DeFi, memecoins, infraestructura, IA, exchange, stablecoins y activos replicados), `load_categories` lo lee y `diversify_universe` limita pares por categoría (10) y en total (50), descarta stablecoins, activos replicados y activos sin categoría, y falla si quedan menos de 4 categorías (MIAX-027)
- Validador de integridad de la caché de klines: `validate_klines` revisa cada Parquet por separado y reporta ficheros ilegibles, esquema incorrecto, duplicados, desorden, velas desalineadas o en el mes equivocado, `close_time` u OHLC incoherentes (errores) y huecos (avisos), sin modificar nada (MIAX-029)
- Subcomandos de la CLI de ingesta: `python -m miax.ingest validate` revisa la integridad de la caché (sale con código 1 si hay errores o si un símbolo no tiene caché) y `python -m miax.ingest pairs` lista el catálogo de pares completo, con `--quote` y `--status` opcionales para filtrar, ambos con `--output` para guardar el resultado en Parquet; `download` sigue siendo el comando por defecto (MIAX-029)
- Las descargas de `/klines` cuentan con su peso real de 2 en el limitador, en lugar de 1 (MIAX-029)
- Universo congelado por periodos: `configs/universe_params.yaml` fija los parámetros y `python -m miax.ingest universe` elige, para cada año desde 2018, hasta 40 pares con al menos 5 M USDT/día de volumen mediano en los tres meses anteriores (sin usar datos del propio periodo) y los guarda en `configs/universe.yaml`, que `load_universe` lee de vuelta; `configs/categories.yaml` se amplía con los activos históricos y deslistados que fueron líquidos desde 2017 y una categoría `gaming` (MIAX-028)
- Tests de ingesta sin red: fixtures grabados de Binance (`tests/fixtures/binance/`, regenerables con `scripts/dev/record_fixtures.py`) y un Binance falso que atiende `/klines` y `/exchangeInfo` desde ellos; la CLI real (`download`, `validate`, `pairs`, `universe`) se prueba de punta a punta, incluidos la reanudación, el hueco de LUNAUSDT, un 429 con `Retry-After` y el par EOS en `BREAK`. La guardia de red apunta además los proxies a una dirección inalcanzable para que un subproceso con `requests` no salga a internet (MIAX-032)
- Documento del sesgo de supervivencia (`docs/datos/sesgo-supervivencia.md`): mide que la API de Binance sí sirve el histórico de los pares deslistados, explica cómo el universo por periodos los incluye (12 de 104 símbolos hoy en `BREAK`) y lista qué pares siguen faltando y por qué; se corrige el riesgo 8 del README, que afirmaba lo contrario (MIAX-031)
- Informe de cobertura temporal de la caché de klines: `coverage_report` da, por símbolo, la fecha inicial y final, el porcentaje de velas presentes, el de minutos con operaciones (`num_trades > 0`), los huecos y las velas que faltan al principio o al final del rango, y `universe_coverage` lo mide para cada par en su periodo del universo para detectar series cortadas (BCC, BCHABC, LUNA); `python -m miax.ingest coverage` lo ejecuta con `--symbol` o `--universe`, y sale con código 1 si algún símbolo no tiene velas (MIAX-030)
- Alineación a una rejilla temporal común en UTC: `align_klines` devuelve, por campo, una tabla `tiempo × símbolo` con el mismo índice para todos los símbolos en un rango dado; lo que falta queda como `NaN`, nunca como 0 ni relleno, y una vela duplicada o desalineada falla con el símbolo en el error (MIAX-033)
- Clasificación de los huecos por causa: `classify_slots` y `classify_klines` etiquetan cada slot de la rejilla común como válido, vela sin operaciones, par aún no listado, parada del símbolo, caída de Binance (falta en al menos dos símbolos y en la mitad o más de los vivos), par deslistado o sin datos (`download_ends_from_universe` da el fin de descarga de cada par para no confundir el fin de la caché con un deslistado); `gap_runs` lista los tramos y `cause_counts` los cuenta por símbolo y causa. Es una clasificación ex post, para limpiar y para la memoria, nunca una feature del modelo; `cache_bounds` da la primera y la última vela de la caché de un símbolo (MIAX-034)
- Máscara de validez de huecos sin rellenar con ceros: `validity_mask` marca válido solo el slot con operaciones, `return_validity` exige que lo sean `t` y `t-1`, y `mask_invalid` deja `NaN` (nunca 0) en lo inválido, de modo que un hueco no se convierte en un retorno cero (MIAX-035)
- Retornos logarítmicos a 1 paso sobre los precios alineados, indexados por el instante en que se conocen (`open_time + intervalo`: el de la vela de 10:00 a 10:01 va en la fila 10:01): `log_returns` deja `NaN` (nunca 0) donde no son válidos `t` y `t-1`, y `returns_from_cache` los da desde la caché usando la vela anterior al rango para que el primer slot tenga retorno (MIAX-036)
- Decisión y herramientas sobre los valores extremos de los retornos: por defecto no se recorta nada (en 2025 el 0,1 % de los retornos aporta del 12 % al 40 % de la varianza y no parecen errores de datos), `tail_report` describe las colas de cada símbolo y `fit_clip_bounds` y `clip_returns` permiten winsorizar solo como análisis de sensibilidad, con límites ajustados en train y sin convertir nada en 0; documentado en `docs/datos/colas-retornos.md` (MIAX-037)
- Volatilidad realizada por ventana: `realized_volatility` calcula la raíz de la suma de retornos al cuadrado de las últimas `W` filas (60 por defecto), mirando solo hacia atrás, y la reescala por `W / n` para no infravalorar los pares con huecos; da `NaN` si menos del 80 % de la ventana tiene retorno válido, y `realized_volatility_from_cache` la entrega desde la caché con la ventana del primer slot ya completa (MIAX-038)
- Volumen relativo al propio pasado de cada activo: `log_relative_to_past` da `ln(V_τ)` menos la media del log de las 10080 velas válidas anteriores (7 días a 1m, sin incluir la vela actual ni mirar al futuro), de modo que BTC y un token pequeño quedan en la misma escala y un cambio de régimen se absorbe en una semana; es `NaN` (nunca 0) si el slot no es válido o menos de la mitad de la ventana lo es, y `relative_volume_from_cache` lo entrega desde la caché con la base del primer slot completa (MIAX-039)
- Número de operaciones relativo al propio pasado de cada activo: `relative_trades_from_cache` aplica a `num_trades` la misma base móvil de 7 días hacia atrás que el volumen relativo (sin la vela actual ni mirar al futuro), de modo que mide cuántas manos operan y no cuánto dinero, y `relative_activity_from_cache` hace lo mismo para cualquiera de `quote_asset_volume`, `volume` o `num_trades` (MIAX-040)
- Remuestreo de las velas de 1m a barras de 5m, 15m y 1h (`resample_from_cache`) y retornos a esos horizontes (`horizon_returns_from_cache`) desde el mismo origen: una barra es válida si su último minuto es válido y al menos el 80 % de sus minutos lo son, `open` es el del primer minuto válido, `high` y `low` el máximo y el mínimo de los minutos válidos, `close` el último y los volúmenes y operaciones se suman sin reescalar (MIAX-042)
- Panel maestro `tiempo × activo × feature`: `build_panel` entrega en un objeto `Panel` (estructura que no se reasigna ni admite features nuevas; los DataFrames internos no están protegidos, copia antes de modificar) el retorno, la volatilidad realizada, el volumen relativo y el número de operaciones relativo a 1m (cada uno idéntico a su `*_from_cache`) con la máscara de validez y `to_array()` de forma `(T, N, F)`, leyendo la caché dos veces en lugar de ocho; `period_panels` rinde un panel por periodo del universo, de uno en uno, y `NaN` (nunca 0) marca lo no válido (MIAX-041)

### Cambiado
- `load_klines` acepta `columns` para leer solo las columnas necesarias (más `open_time`), y la aritmética de slots de la cobertura se reutiliza como `slot_bounds`, sin cambiar el comportamiento (MIAX-033)
- La lista de ficheros de mes de un símbolo (`_month_files`) pasa de `integrity.py` a `store.py` para reutilizarla en `cache_bounds`, y `_grid_ms` de `alignment.py` pasa a ser pública (`grid_ms`) para reutilizarla en la clasificación, sin cambiar el comportamiento (MIAX-034)
- El desplazamiento del índice al instante de cierre de la vela pasa a la función pública `shift_to_close` y la comprobación de los parámetros de ventana a `check_window_params`, para reutilizarlas en el volumen relativo sin duplicarlas; `log_returns` y `realized_volatility` no cambian su comportamiento (MIAX-039)
- La cache de klines se guarda por defecto en `data/raw/klines` en lugar de `data/klines`: la descarga, la validación y la cobertura leen y escriben ahí, de modo que la carpeta `klines` del Drive se copia tal cual dentro de `data/raw/` (MIAX-197)
- Reglas de equipo ampliadas en `CLAUDE.md` y `AGENTS.md`: se refactoriza código de tickets cerrados antes que duplicarlo (sin cambios masivos), sin atribución a Claude en commits y PR, prueba de humo real antes del PR, cierre al mergear con squash (con el mensaje de todos los commits) y CLI con subcomandos
- `relative_volume_from_cache` delega en la nueva `relative_activity_from_cache` para no duplicar la lectura de la caché con el número de operaciones, sin cambiar el comportamiento (MIAX-040)

---

## [v0.1.0] — 2026-10-11

Primera release. Cierra la Épica 0: entorno fijado, paquete instalable, lint, tests, CI, utilidades comunes y ramas protegidas. Incluye además el arranque de la Épica 1 (ingesta de Binance, MIAX-016 a MIAX-023), que ya estaba en `develop`.

### Añadido · Épica 0 (fundación)
- README que explica el TFM completo a alguien sin contexto previo: problema, espejismo de factor, pregunta de investigación, metodología, evaluación y riesgos (MIAX-001)
- Tres diagramas vectoriales de la memoria: concepto de lead-lag, espejismo de factor y pipeline metodológico (MIAX-001)
- Banner de portada del repositorio (MIAX-001)
- Estructura de carpetas del proyecto con marcadores de posición (MIAX-001)
- `CLAUDE.md` con las reglas que cargan automáticamente todos los Claude del equipo: flujo de trabajo, modelo de ramas, convenciones de código y qué no se hace (MIAX-002)
- `AGENTS.md` para herramientas que no leen `CLAUDE.md` (MIAX-002)
- Cuatro subagentes versionados que ejecutan el flujo `coder → auditor → fixer → tester` en Sonnet (MIAX-003)
- Comprobación automática de entorno al arrancar sesión: avisa de paquetes, skills y subagentes que falten, y de estar en `main` (MIAX-004)
- Backlog completo del proyecto, numerado y ordenado por dependencias, desde la fase semilla hasta la defensa (MIAX-002)
- Proceso de equipo documentado: tablero, límite de trabajo en curso, definición de hecho y reparto por épicas (MIAX-002)
- Bitácora compartida entre sesiones para que los tres Claude sepan qué están haciendo los demás (MIAX-002)
- Entorno fijado sobre Python 3.12: `pip install -r requirements.txt` instala las mismas versiones exactas de datos, cálculo, grafos, configuración, visualización y desarrollo en cualquier máquina (MIAX-005)
- Test que falla si una dependencia entra en `requirements.txt` con rango en lugar de versión exacta (MIAX-005)
- Guía para instalar GitHub CLI e iniciar sesión, necesaria para abrir PR desde Claude (MIAX-005)
- Paquete `miax` instalable en modo editable: con `pip install -e .`, `import miax` funciona desde cualquier directorio (MIAX-006)
- Lint y formato con ruff en todo el repo: exige type hints y docstrings en funciones públicas, semillas con `default_rng` y fechas con zona horaria (MIAX-007)
- Configuración de pytest en `pyproject.toml`: recorre `tests/unit` y `tests/integration`, y falla ante marcadores o configuración desconocidos (MIAX-008)
- Guardia de red `autouse` que hace fallar cualquier test que intente conectar por socket, con mensaje explicando qué regla se ha roto (MIAX-008)
- Carga tipada de configuración desde YAML: `load_config` valida un fichero de `configs/` contra un dataclass y falla con un mensaje claro si falta un campo obligatorio, si un tipo no coincide o si hay un campo desconocido (MIAX-009)
- `utils.seeds`: fija la misma semilla en `random`, `numpy` y `torch` (si está instalado), y crea generadores numpy independientes con `default_rng` (MIAX-010)
- Lectura y escritura de Parquet con rutas relativas a la raíz del repo: la escritura es atómica y releer devuelve el mismo DataFrame, tipos e índice incluidos (MIAX-011)
- `get_logger`: todos los módulos loguean con el mismo formato (hora UTC, nivel, nombre, mensaje) y con nivel configurable; ruff prohíbe `print` en `src/` (MIAX-012)
- Plantillas de GitHub: abrir un PR precarga qué hace, cómo se ha comprobado, qué queda fuera, riesgo de fuga de datos y checklist; los issues salen de plantillas de incidencia o tarea, no en blanco (MIAX-014)
- CI en GitHub Actions: cada PR a `develop` o `main` ejecuta `ruff check`, `ruff format --check` y `pytest` sobre Python 3.12, y fuerza finales de línea LF para que el formato no dependa del sistema operativo (MIAX-013)
- Ramas `main` y `develop` protegidas: todo entra por PR, sin force-push ni borrado, y el check `Lint y tests` del CI tiene que estar en verde para mergear (MIAX-015)

### Añadido · Épica 1 (ingesta, en curso)
- Cliente HTTP base de la ingesta (`HttpClient`): GET con timeout y reintentos con backoff exponencial ante timeouts, fallos de conexión y 5xx; un 4xx falla al instante sin reintentar (MIAX-016)
- El cliente HTTP trata 429 y 418 como reintentables: espera lo que indique `Retry-After` (segundos o fecha HTTP, con tope configurable) o, si no viene, el backoff exponencial, de modo que un rate limit no rompe la descarga (MIAX-017)
- Descarga de klines con paginación hacia delante: un rango de más de 1000 velas se pide en bloques avanzando el cursor a la última `open_time` + 1 ms y se devuelve entero, con las columnas crudas de Binance tipadas (MIAX-018)
- Cacheo local de klines en Parquet, un fichero por mes UTC (`data/klines/{símbolo}/{intervalo}/YYYY-MM.parquet`): `save_klines` guarda ordenado por `open_time` y `load_klines` relee, con rango opcional, exactamente lo guardado (MIAX-020)
- Limitador de peso por ventana deslizante (`RateLimiter`): el cliente HTTP puede frenar antes de superar el presupuesto de peso de Binance y resincronizarse con la cabecera `X-MBX-USED-WEIGHT-1M`, para que una descarga larga no provoque 429 ni baneo de IP (MIAX-019)
- Reanudación de la descarga de klines: `update_symbol` baja solo las velas posteriores a la última cacheada conservando las velas previas del mes parcial, y relanzar con el rango ya cubierto no hace ninguna llamada a Binance (MIAX-021)
- CLI de ingesta: `python -m miax.ingest --symbol BTCUSDT` descarga y cachea klines de un rango en UTC (por defecto, 1m y los últimos 30 días), reanudando lo ya bajado y frenando con el limitador de peso (MIAX-022)
- Descarga masiva desde la CLI: `--symbol BTCUSDT,ETHUSDT,SOLUSDT` baja N símbolos en orden con un único limitador de peso compartido, muestra el progreso `[i/N]` y un resumen final, y un símbolo que falla no aborta el resto (MIAX-023)

### Cambiado
- Las tarjetas de Trello se asignan con una etiqueta por persona y la URL del PR se añade a la descripción, porque el conector de Trello no puede asignar miembros ni comentar (MIAX-005)

[Sin publicar]: https://github.com/raulrodriguezlr/tfm-miax/compare/v0.1.0...develop
[v0.1.0]: https://github.com/raulrodriguezlr/tfm-miax/releases/tag/v0.1.0
