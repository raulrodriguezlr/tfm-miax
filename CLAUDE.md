# CLAUDE.md — Reglas del proyecto

Este fichero lo carga Claude Code automáticamente al abrir el repositorio. **No hay que pedir que se lea.** Aplica a los tres integrantes del equipo y a cualquier agente que trabaje aquí.

Si trabajas con otra herramienta (Codex, Cursor, Gemini CLI), lee [AGENTS.md](AGENTS.md) — apunta a este mismo documento.

---

## 0. Qué es este proyecto

TFM sobre relaciones de *lead-lag* entre criptoactivos. Contexto completo en [README.md](README.md). Léelo antes de tocar código.

**Vamos a estar meses con esto.** No tiene que funcionar el primer día.

---

## 1. Regla número uno

> **Una IA no hace el proyecto. Una IA hace *una tarea*.**

Si a un agente se le da una tarea acotada, la coge, la hace y se acaba ahí. No continúa con la siguiente. No "aprovecha para" refactorizar otra cosa. No inventa alcance.

**Prohibido:**
- Pedir "implementa el pipeline entero".
- Aceptar un PR que toca más de un ticket.
- Dejar que un agente decida qué hacer después.

**Obligatorio:**
- Un ticket = una rama = un PR.
- El humano decide qué ticket entra. Siempre.

---

## 2. Flujo de trabajo obligatorio

Cada ticket recorre este camino. Sin atajos.

```
   Backlog  →  Sprint actual  →  In Progress  →  Review  →  Done
```

### 2.1. Fases

| Fase | Quién | Qué pasa |
|---|---|---|
| **1. Plan** | **Opus**, con el humano | Se lee el ticket, se acota, se decide el diseño. Sale un plan escrito. |
| **2. Código** | Subagente `miax-coder` (Sonnet) | Implementa el plan. Nada más que el plan. |
| **3. Auditoría** | Subagente `miax-auditor` (Sonnet) | Revisa el código contra el plan y contra estas reglas. Solo señala; no toca nada. |
| **4. Corrección** | Subagente `miax-fixer` (Sonnet) | Aplica los hallazgos del auditor. Solo esos. |
| **5. Test** | Subagente `miax-tester` (Sonnet) | Escribe y ejecuta los tests. Reporta lo que falla tal cual. |
| **6. PR** | El humano | Abre el PR, lo mueve a Review, pide revisión a un compañero. |

**El plan se hace en Opus. La ejecución va en Sonnet.** Planificar es donde se decide bien o mal; ejecutar es mecánico.

### 2.2. Cómo se lanza

```
1. Lee el ticket MIAX-XXX en docs/project/BACKLOG.md
2. Pide un plan (modo plan, Opus)
3. Apruebas el plan
4. Lanzas la cadena de subagentes: coder → auditor → fixer → tester
5. Revisas el diff tú mismo
6. Abres el PR
```

Los cuatro subagentes están definidos en `.claude/agents/`. Se invocan por nombre.

---

## 3. Skills obligatorias

| Trabajo | Skill |
|---|---|
| Cualquier código Python (pipeline, modelos, tests) | `senior-engineer` |
| Cualquier cosa del dashboard (`app/`) | `frontend-design` |

Si no están instaladas, el arranque de sesión te avisa (ver §9). No empieces a programar sin ellas: el objetivo es que el código de los tres se parezca.

---

## 4. Git

### 4.1. Modelo de ramas

Tres niveles. Nada se salta un nivel.

```
   feature/MIAX-XXX  ──merge──►  develop  ──merge──►  main
                                    │                   │
                                 SNAPSHOT            RELEASE
```

| Rama | Qué es | Quién escribe |
|---|---|---|
| `main` | Solo releases. Estado presentable en cualquier momento. | Nadie directamente. Solo merges desde `develop`. |
| `develop` | Integración. Aquí viven los tickets ya revisados. | Nadie directamente. Solo merges desde `feature/*`. |
| `feature/MIAX-XXX` | Un ticket. Nace de `develop` y vuelve a `develop`. | Tú. |

**Nunca se hace push directo a `main` ni a `develop`.** Siempre vía PR.

Una rama por ticket. El nombre **es** el ticket, sin descripción — el ticket ya la tiene:

```
feature/MIAX-018
feature/MIAX-087
fix/MIAX-092          # arreglo sobre algo ya integrado
```

### 4.2. Snapshots y releases

| Evento | Qué se genera | Versión |
|---|---|---|
| Merge de `feature/*` → `develop` | **Snapshot** | `v0.5.0-snapshot.N` (N sube en cada merge) |
| Merge de `develop` → `main` | **Release** | `v0.5.0` con tag de git |

Numeración:

- **MINOR** (`0.4.0` → `0.5.0`): se cierra una épica completa.
- **PATCH** (`0.5.0` → `0.5.1`): arreglos sobre una release ya publicada.
- **`v1.0.0`**: la versión que se entrega al tribunal.

Toda release lleva tag:

```bash
git tag -a v0.5.0 -m "v0.5.0: grafo de lead-lag completo"
git push origin v0.5.0
```

### 4.3. Changelog

[CHANGELOG.md](CHANGELOG.md) se actualiza **siempre**, y dice qué hace cada versión, no qué commits tiene.

- En el PR a `develop`: añades tu línea bajo `## [Sin publicar]`.
- En el PR a `main`: `[Sin publicar]` se convierte en la versión con fecha, y se abre un `[Sin publicar]` vacío.

Una línea por cambio, en castellano, explicando **qué hace** — no el nombre del fichero:

```
- Descarga de klines con paginación y backoff ante 429 (MIAX-018)
```

No vale `- Cambios en ingest.py`.

### 4.4. Commits

Formato: `MIAX-XXX: qué hace, en imperativo`

```
MIAX-018: paginar la descarga de klines hacia delante
MIAX-092: medir similitud de adyacencia entre ventanas
```

Un commit debe poder explicarse en una línea. Si no puedes, son dos commits.

### 4.5. Pull requests

- Base del PR: **`develop`**, nunca `main`.
- Título: `MIAX-XXX: título del ticket`.
- El cuerpo dice: qué hace, cómo se ha comprobado, qué queda fuera, y si hay riesgo de fuga de datos.
- **Lo revisa una persona del equipo, no un agente.** El auditor ya pasó antes; el PR es control humano.
- No se mergea con tests en rojo.

### 4.6. Antes de cada push

Tres cosas, siempre, sin excepción:

1. Actualizar [docs/project/ESTADO.md](docs/project/ESTADO.md) con lo que has hecho (§5).
2. Añadir tu línea al [CHANGELOG.md](CHANGELOG.md) bajo `[Sin publicar]`.
3. Pasar los tests.

---

## 5. Bitácora compartida entre sesiones

Somos tres personas con tres Claudes distintos que no se ven entre sí. [docs/project/ESTADO.md](docs/project/ESTADO.md) es el único sitio donde se enteran de lo que hacen los demás.

**Si eres un agente trabajando en este repo:**

- **Lee `ESTADO.md` al empezar.** Antes de tocar nada. Te dice qué tickets están en curso, qué decisiones se han tomado y qué interfaces han cambiado.
- **Actualízalo antes de que tu humano haga push.** Tu entrada dice: qué ticket, qué has tocado, qué decisión no obvia has tomado, y qué queda pendiente.
- **No borres entradas de otros.** El fichero crece; ya se podará al cerrar cada épica.
- **Si hay conflicto de merge en `ESTADO.md`, se quedan las dos versiones.** Nunca se descarta la entrada de otro.

Sin esto, tres personas resuelven el mismo problema tres veces y de tres maneras distintas.

---

## 6. Código

### 6.1. Docstrings

**Una frase. Dos como mucho. Directas y al pie.**

```python
def residualize(returns: pd.DataFrame, factor: pd.Series) -> pd.DataFrame:
    """Elimina el factor comun de cada activo por OLS. Devuelve los residuos."""
```

No queremos esto:

```python
def residualize(returns, factor):
    """
    This function performs residualization of the returns.

    Parameters
    ----------
    returns : pd.DataFrame
        A DataFrame containing the returns...
    """
```

Nada de secciones `Parameters` / `Returns` / `Examples`. Los tipos van en la firma, no en el texto.

### 6.2. Reglas duras

- **Type hints en todas las funciones públicas.** Sin excepción.
- **Nada de rutas absolutas.** Todo relativo a la raíz del repo o desde `configs/`.
- **Nada de magia oculta**: si una función usa un umbral, va como parámetro con valor por defecto explícito.
- **Ninguna función mira al futuro.** Cualquier `shift`, ventana o split se comenta con una línea diciendo por qué no hay fuga.
- **Semillas fijas** en todo lo que tenga aleatoriedad.
- **Los datos no se commitean.** `data/` está ignorado salvo los `.gitkeep`.

### 6.3. Estructura

```
src/miax/
├── ingest/     # descarga y cacheo desde Binance
├── features/   # retornos, residualizacion, features por nodo
├── graph/      # lead-lag, PCMCI, estabilidad
├── models/     # GNN temporal y baselines
├── eval/       # purga/embargo, metricas, test economico
├── viz/        # figuras para la memoria
└── utils/      # io, config, semillas, logging
```

Los notebooks exploran. **La lógica vive en `src/`.** Si un notebook tiene una función que sirve para algo, se mueve a `src/` en el mismo PR.

### 6.4. Tests

- Todo lo de `src/` que no sea I/O puro lleva test.
- Los que tocan fugas de datos (purga, embargo, `shift`) llevan test **sí o sí**. Es el riesgo número uno del proyecto.
- `pytest` desde la raíz. Nada de tests que necesiten red.

---

## 7. Tablero (Trello)

**Tablero:** https://trello.com/b/PBzf0gtT/tfm-miax-lead-lag-cripto

Cada integrante conecta su cuenta de Trello a Claude (ajustes de conectores en claude.ai → Trello). Así su Claude lee el tablero, coge tickets y los mueve de columna sin que nadie copie nada a mano.

### 7.1. Listas

| Lista | Qué contiene |
|---|---|
| 📚 **Épicas (roadmap)** | Una tarjeta por épica. Contexto y riesgos. No se mueven. |
| 📥 **Backlog** | Tickets definidos, sin empezar. |
| 🎯 **Sprint actual** | Lo que el equipo se ha comprometido a hacer ahora. |
| 🔨 **In Progress** | Alguien está trabajando en ello ahora mismo. |
| 👀 **Review (PR abierto)** | PR abierto, esperando revisión humana. |
| ✅ **Done** | Mergeado en `develop`. |

### 7.2. Cómo se mueven las tarjetas

El *Backlog* tiene los 196 tickets, ordenados por número. **Cada uno se sirve de ahí**: eliges el ticket más bajo disponible cuyas dependencias estén en *Done*, y lo arrastras a *Sprint actual*.

```
📥 Backlog  ──tú eliges──►  🎯 Sprint actual  ──►  🔨 In Progress  ──PR──►  👀 Review  ──merge──►  ✅ Done
```

- **Un ticket por persona en *In Progress*.** Si te bloqueas, lo devuelves a *Sprint actual* con una nota. No abres otro.
- De *In Progress* a *Review* se pasa **abriendo el PR**.
- De *Review* a *Done* se pasa cuando **otra persona** aprueba y se mergea.
- Las tarjetas de *Épicas* nunca se mueven: son el mapa, no trabajo.
- Cada tarjeta enlaza a su épica en la descripción, y lleva el prefijo `E0`–`E13` en el título. Buscando `E4` en Trello salen todos los tickets de esa épica.

### 7.3. Reglas del sprint

- El sprint se cierra y se repuebla **entre los tres**, no unilateralmente.
- No se traen del Backlog tickets con dependencias abiertas, por muy apetecibles que parezcan.
- Si un ticket se queda a medias al cerrar el sprint, se queda en *Sprint actual*; no vuelve al Backlog.

### 7.4. Qué hace el agente con Trello

Si tienes el conector activo:

1. **Al empezar:** lee *Sprint actual*, comprueba dependencias y propone el ticket más bajo disponible.
2. **Al arrancar el ticket:** mueve la tarjeta a *In Progress* y se asigna a tu usuario.
3. **Al abrir el PR:** mueve la tarjeta a *Review* y pega la URL del PR como comentario en la tarjeta.
4. **Nunca mueve a *Done* por su cuenta.** Eso lo hace una persona tras aprobar y mergear.

### 7.5. Fuente de verdad

**El backlog canónico es [docs/project/BACKLOG.md](docs/project/BACKLOG.md)**, con los 196 tickets, sus dependencias y su criterio de "hecho". Trello es la vista de estado, no el contenido.

Si los dos discrepan, manda el fichero del repo. Si un ticket cambia de alcance, se cambia en el markdown y luego en la tarjeta.

Reglas de proceso completas en [docs/project/WORKFLOW.md](docs/project/WORKFLOW.md).

---

## 8. Qué NO se hace

- No se mergea nada sin revisión humana.
- No se acepta un resultado bueno sin haber comprobado que no hay fuga de datos.
- No se cambia el alcance de un ticket a mitad. Se abre otro.
- No se instala una dependencia sin decirlo en el PR.
- No se optimiza nada antes de que funcione y esté medido.
- No se escribe en `main`.

---

## 9. Primera vez en el repo

Al abrir sesión se lanza `scripts/dev/check_setup.py`, que comprueba entorno y skills y dice qué falta. Si algo no está, lo arreglas antes de empezar.

Manualmente:

```bash
python scripts/dev/check_setup.py
```

---

## 10. Idioma

- **Código, nombres de variables, funciones y ramas:** inglés.
- **Commits, docstrings, comentarios, documentación, tickets y PRs:** castellano.

El tribunal lee castellano y el equipo también. Lo único en inglés es lo que se lee dentro del código.
