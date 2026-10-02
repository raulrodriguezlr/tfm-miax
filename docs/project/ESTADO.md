# Estado del equipo

**Bitácora compartida entre sesiones.** Somos tres personas con tres Claude distintos que no se ven entre sí. Este fichero es el único sitio donde se enteran de lo que hacen los demás.

> **Si eres un agente:** lee esto antes de tocar nada, y añade tu entrada antes de que tu humano haga push. No borres entradas de otros. Si hay conflicto de merge aquí, se quedan las dos versiones.

---

## 1. Ahora mismo

| Persona | Ticket en curso | Rama | Desde | Estado |
|---|---|---|---|---|
| Raúl | — | — | — | libre |
| Piettro | MIAX-008 | `feature/MIAX-008` | 2026-09-22 | en review |
| Alonso | — | — | — | libre |

**Versión actual:** `v0.1.0` · **Snapshot de `develop`:** `v0.1.0-snapshot.7`

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
| — | — | — | — |

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
