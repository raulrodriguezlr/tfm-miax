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

Backlog: [docs/project/BACKLOG.md](docs/project/BACKLOG.md) · Proceso: [docs/project/WORKFLOW.md](docs/project/WORKFLOW.md)
