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

### Cambiado
- Reglas de equipo ampliadas en `CLAUDE.md` y `AGENTS.md`: se refactoriza código de tickets cerrados antes que duplicarlo (sin cambios masivos), sin atribución a Claude en commits y PR, prueba de humo real antes del PR, cierre al mergear con merge commit y CLI con subcomandos

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
