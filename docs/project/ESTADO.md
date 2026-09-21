# Estado del equipo

**Bitácora compartida entre sesiones.** Somos tres personas con tres Claude distintos que no se ven entre sí. Este fichero es el único sitio donde se enteran de lo que hacen los demás.

> **Si eres un agente:** lee esto antes de tocar nada, y añade tu entrada antes de que tu humano haga push. No borres entradas de otros. Si hay conflicto de merge aquí, se quedan las dos versiones.

---

## 1. Ahora mismo

| Persona | Ticket en curso | Rama | Desde | Estado |
|---|---|---|---|---|
| Raúl | MIAX-007 | `feature/MIAX-007` | 2026-09-21 | en review |
| — | — | — | — | — |
| — | — | — | — | — |

**Versión actual:** `v0.1.0` · **Snapshot de `develop`:** `v0.1.0-snapshot.2`

---

## 2. Decisiones vigentes

Decisiones tomadas que afectan a todos. Si vas a contradecir una, hablas con el equipo antes.

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
| D-09 | El estilo lo decide ruff: sus reglas por defecto más `ANN`, `D`, `NPY` y `E501` (100). `ruff check .` y `ruff format --check .` deben pasar en limpio antes de cada push. | Nadie discute estilo en los PR, y las normas del repo (tipos, docstrings, semillas, UTC) las vigila la máquina. | MIAX-007 |

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
