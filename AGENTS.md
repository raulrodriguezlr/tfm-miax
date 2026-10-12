# AGENTS.md

Las reglas de este repositorio están en **[CLAUDE.md](CLAUDE.md)**. Léelo entero antes de hacer nada.

Resumen para agentes que no cargan `CLAUDE.md` automáticamente:

- **Una tarea por sesión.** Haces el ticket que se te da y paras. No continúas con el siguiente.
- **Un ticket = una rama = un PR.** Rama `feature/MIAX-XXX`, commits con prefijo `MIAX-XXX:`.
- **Refactorizar sí, a lo loco no.** Si el ticket lo necesita, se refactoriza código de tickets ya cerrados en lugar de duplicarlo, con el mismo comportamiento y diciéndolo en el PR. Nunca se toca código de un ticket en curso de otro, ni se cambian mil líneas por ticket.
- **Sin atribución a Claude:** ni `Co-Authored-By` en los commits ni el pie "Generated with Claude Code" en los PR.
- **Prueba de humo real antes del PR** si el código habla con Binance o con datos reales, con el resultado en el cuerpo del PR. Los tests nunca tocan la red.
- **Al mergear:** con squash, con un mensaje que recoja todos los commits de la rama (el repo ya lo hace por defecto); la tarjeta a Done y la rama borrada tras comprobar con `gh pr view` que está mergeada.
- **Decisiones nuevas en ESTADO como `D-??`.** El número se pone al integrar en `develop`.
- **CLI:** una capa fina de subcomandos sobre funciones de `src/`, con códigos de salida 0, 1 y 2 y sin `print`.
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
