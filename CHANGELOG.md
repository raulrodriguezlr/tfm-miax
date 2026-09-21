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
- Entorno fijado sobre Python 3.12: `pip install -r requirements.txt` instala las mismas versiones exactas de datos, cálculo, grafos, configuración, visualización y desarrollo en cualquier máquina (MIAX-005)
- Test que falla si una dependencia entra en `requirements.txt` con rango en lugar de versión exacta (MIAX-005)
- Guía para instalar GitHub CLI e iniciar sesión, necesaria para abrir PR desde Claude (MIAX-005)

### Cambiado
- Las tarjetas de Trello se asignan con una etiqueta por persona y la URL del PR se añade a la descripción, porque el conector de Trello no puede asignar miembros ni comentar (MIAX-005)

---

## [v0.1.0] — 2026-09-18

Primera versión. El proyecto queda definido y el equipo tiene un método de trabajo común.

### Añadido
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

[Sin publicar]: https://github.com/raulrodriguezlr/tfm-miax/compare/v0.1.0...develop
[v0.1.0]: https://github.com/raulrodriguezlr/tfm-miax/releases/tag/v0.1.0
