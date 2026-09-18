# Estado del equipo

**Bitácora compartida entre sesiones.** Somos tres personas con tres Claude distintos que no se ven entre sí. Este fichero es el único sitio donde se enteran de lo que hacen los demás.

> **Si eres un agente:** lee esto antes de tocar nada, y añade tu entrada antes de que tu humano haga push. No borres entradas de otros. Si hay conflicto de merge aquí, se quedan las dos versiones.

---

## 1. Ahora mismo

| Persona | Ticket en curso | Rama | Desde | Estado |
|---|---|---|---|---|
| Raúl | — | — | — | libre |
| — | — | — | — | — |
| — | — | — | — | — |

**Versión actual:** `v0.1.0` · **Snapshot de `develop`:** —

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

### 2026-09-18 · Raúl · MIAX-001 a MIAX-004
- **Hecho:** repositorio arrancado. README como prototipo de la memoria, estructura de carpetas, `CLAUDE.md` con las reglas del equipo, cuatro subagentes, comprobación de entorno al arrancar sesión, backlog completo y este fichero.
- **Decisión:** los diagramas se hacen en SVG a mano, no con modelos de imagen. Un modelo de difusión no dibuja una flecha etiquetada `SOL → LINK con 2 min de retardo`. El banner de portada sí es generado, porque es decorativo.
- **Decisión:** los cuatro subagentes van versionados en `.claude/agents/`, no en la configuración personal de cada uno. Así los tres ejecutan el mismo flujo sin tener que configurarlo.
- **Ojo:** `check_setup.py` falla a propósito si estás en `main`. No es un bug.
- **Pendiente:** crear la rama `develop` y proteger `main` (MIAX-015). Nadie ha escrito código de producción todavía.
