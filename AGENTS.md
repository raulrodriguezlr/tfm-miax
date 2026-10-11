# AGENTS.md

Las reglas de este repositorio están en **[CLAUDE.md](CLAUDE.md)**. Léelo entero antes de hacer nada.

Resumen para agentes que no cargan `CLAUDE.md` automáticamente:

- **Una tarea por sesión.** Haces el ticket que se te da y paras. No continúas con el siguiente.
- **Un ticket = una rama = un PR.** Rama `feature/MIAX-XXX`, commits con prefijo `MIAX-XXX:`.
- **Nunca se escribe en `main`.**
- **Docstrings de una o dos frases.** Sin secciones `Parameters` ni `Returns`.
- **Type hints obligatorios** en funciones públicas.
- **Ninguna función mira al futuro.** Todo `shift`, ventana o split lleva un comentario explicando por qué no hay fuga de datos.
- **Los datos no se commitean.**
- Código y commits en inglés; docstrings, comentarios y documentación en castellano.

## Tablero

**Trello:** https://trello.com/b/PBzf0gtT/tfm-miax-lead-lag-cripto

Conecta tu cuenta de Trello al agente (en Claude: ajustes de conectores → Trello) para poder leer el tablero y mover tarjetas sin copiar nada a mano.

Listas: `📚 Épicas` → `📥 Backlog` → `🎯 Sprint actual` → `🔨 In Progress` → `👀 Review` → `✅ Done`

- Al coger un ticket, muévelo a *In Progress* y ponle **la etiqueta con el nombre de tu humano** (🟢 `raul`, 🔵 `piettro`, 🟣 `alonso`). El conector no puede asignar miembros: la etiqueta es la asignación.
- Al abrir el PR, muévelo a *Review* y añade `**PR:** <url>` al final de la descripción, leyéndola antes para no pisarla. El conector no puede comentar.
- **Nunca muevas a *Done*.** Eso lo hace una persona tras aprobar y mergear.
- Un ticket por persona en *In Progress*.

**El backlog canónico es el fichero del repo, no Trello.** Si discrepan, manda el fichero.

---

Backlog: [docs/project/BACKLOG.md](docs/project/BACKLOG.md) · Proceso: [docs/project/WORKFLOW.md](docs/project/WORKFLOW.md) · Estado: [docs/project/ESTADO.md](docs/project/ESTADO.md)
