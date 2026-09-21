# Bienvenido al TFM

Léete esto entero antes de escribir una línea de código. Son diez minutos y te ahorra semanas.

---

## 1. De qué va el TFM

Queremos saber si **hay criptomonedas que se mueven sistemáticamente antes que otras**, y si esa ventaja —una vez descontado el arrastre de Bitcoin— sirve para predecir algo que sobreviva a las comisiones.

Parece fácil hasta que ves el problema: **en cripto casi todo se mueve con BTC**. Si mides a pelo, "descubres" que todo está conectado con todo, que es no decir nada. El trabajo real es quitar primero la parte que explica BTC y *entonces* preguntar quién adelanta a quién.

**Los tres finales posibles aprueban:**

| | Qué pasa |
|---|---|
| **A** | El grafo aporta valor y sobrevive a costes → hallazgo fuerte |
| **B** | Aporta estadísticamente pero los costes se lo comen → resultado realista, muy defendible |
| **C** | Tras quitar BTC no queda nada explotable → refutación rigurosa, también vale |

**El único fracaso real es un resultado no reproducible o contaminado por fuga de datos.**

Contexto completo en el [README](../../README.md).

---

## 2. Qué tienes que hacer hoy

### 2.1. Clonar, instalar y comprobar el entorno

El entorno es **Python 3.12** (D-06). Todas las dependencias van en `requirements.txt` con versión exacta (D-07): no instales nada suelto, y si necesitas una nueva, entra con `==` en tu PR.

```bash
git clone https://github.com/raulrodriguezlr/tfm-miax.git
cd tfm-miax
py -3.12 -m venv .venv          # Linux/macOS: python3.12 -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python scripts/dev/check_setup.py
```

Te va a decir qué te falta. Arréglalo antes de seguir.

### 2.2. Instalar las skills obligatorias

```bash
claude plugin install anthropic-skills
```

| Trabajo | Skill |
|---|---|
| Código Python | `senior-engineer` |
| Dashboard (`app/`) | `frontend-design` |

No son opcionales: son lo que hace que el código de los tres se parezca.

### 2.3. Crear cuenta de Trello y conectarla a Claude

1. Cuenta en [trello.com](https://trello.com) (gratis).
2. Pídele a Raúl que te invite al tablero.
3. **En Claude → ajustes de conectores → conecta Trello.**

El paso 3 es el importante. Con el conector activo, tu Claude lee el tablero, coge tickets y mueve tarjetas solo. Sin él, tendrás que ir copiando a mano.

**Tablero:** https://trello.com/b/PBzf0gtT/tfm-miax-lead-lag-cripto

### 2.4. Instalar GitHub CLI y conectarla

Sin ella no puedes abrir PR desde Claude. Tres pasos: `winget install --id GitHub.cli`, cerrar Claude del todo y volver a abrirlo, y `gh auth login`. Detalle en [CLAUDE.md §9.1](../../CLAUDE.md#91-github-cli-para-abrir-pr).

---

## 3. La regla que manda sobre todas

> **Una IA no hace el proyecto. Una IA hace *una tarea*.**

Si le das una tarea acotada, la coge, la hace y para. No sigue con la siguiente. No "aprovecha para" refactorizar otra cosa.

**Prohibido:**
- Pedir "implementa el pipeline entero".
- Aceptar un PR que toca más de un ticket.
- Dejar que un agente decida qué hacer después.

Vamos a estar meses con esto. No tiene que funcionar el primer día.

---

## 4. El flujo de trabajo

Cada ticket recorre esto. Sin atajos.

```
 1. Lees ESTADO.md            qué están haciendo los otros dos
 2. Coges un ticket           el más bajo disponible sin dependencias abiertas
 3. Creas la rama             git checkout develop && git pull
                              git checkout -b feature/MIAX-XXX
 4. Planificas EN OPUS        modo plan, con el ticket delante
 5. Ejecutas CON SONNET       coder → auditor → fixer → tester
 6. Revisas el diff TÚ        no delegues esto
 7. Actualizas ESTADO.md      y CHANGELOG.md, antes del push
 8. Abres el PR               base develop
 9. Un compañero aprueba      merge → snapshot
```

### Los cuatro subagentes

Están en `.claude/agents/`, versionados en el repo. No hay que configurar nada.

| Agente | Qué hace | Qué NO hace |
|---|---|---|
| `miax-coder` | Implementa el plan aprobado | No amplía alcance |
| `miax-auditor` | Señala problemas | **No edita nada** |
| `miax-fixer` | Aplica los hallazgos del auditor | Solo esos |
| `miax-tester` | Escribe y ejecuta tests | **Nunca reporta verde algo en rojo** |

**El plan se hace en Opus. La ejecución va en Sonnet.** Planificar es donde se decide bien o mal; ejecutar es mecánico.

---

## 5. Git

```
 feature/MIAX-XXX  ──PR──►  develop  ──PR──►  main
                               │                │
                            SNAPSHOT         RELEASE
```

- **Nunca se escribe directo en `main` ni en `develop`.** Todo por PR.
- Rama = el ticket: `feature/MIAX-014`. Sin descripción, el ticket ya la tiene.
- Commit: `MIAX-014: qué hace, en imperativo`.
- Cada merge a `develop` = snapshot. Cada merge a `main` = release con tag.
- `MINOR` sube al cerrar una épica. `v1.0.0` = entrega al tribunal.

### Antes de cada push, tres cosas

1. Actualizar `docs/project/ESTADO.md`.
2. Añadir tu línea al `CHANGELOG.md` bajo `[Sin publicar]`.
3. Pasar los tests.

---

## 6. La bitácora

Somos tres personas con tres Claude que no se ven entre sí. **[ESTADO.md](ESTADO.md) es el único sitio donde se enteran de lo que hacen los demás.**

- Lo lees al empezar.
- Lo actualizas antes de hacer push.
- **No borras entradas de otros.** Si hay conflicto de merge, se quedan las dos.

Sin esto, los tres resolvemos el mismo problema de tres maneras distintas.

---

## 7. Cómo escribimos código

### Docstrings: una frase, dos como mucho

```python
def residualize(returns: pd.DataFrame, factor: pd.Series) -> pd.DataFrame:
    """Elimina el factor comun de cada activo por OLS. Devuelve los residuos."""
```

Nada de secciones `Parameters` / `Returns` / `Examples`. Los tipos van en la firma.

### Reglas duras

- Type hints en todas las funciones públicas.
- Nada de rutas absolutas. Todo relativo a la raíz o desde `configs/`.
- Si una función usa un umbral, va como parámetro con valor por defecto explícito.
- **Ninguna función mira al futuro.** Todo `shift`, ventana o split lleva un comentario de una línea diciendo por qué no hay fuga.
- Semillas fijas en todo lo aleatorio.
- **Los datos no se commitean.**
- Los notebooks exploran. **La lógica vive en `src/`.**

### Idioma

- **Inglés:** nombres de variables, funciones, clases, ramas.
- **Castellano:** commits, docstrings, comentarios, documentación, tickets y PRs.

---

## 8. El riesgo número uno: fuga de datos

Es lo que hunde la mayoría de TFM de este tipo. Produce backtests espectaculares e irreproducibles.

> **Regla de oro.** Toda transformación que use estadísticas —medias, betas, escalados, **y el propio grafo**— se ajusta *solo* con datos de entrenamiento y se aplica al test. La residualización y la construcción del grafo se recalculan **dentro de cada fold**, nunca sobre la muestra completa.

**Si un resultado sale demasiado bien, se para y se busca la fuga antes de celebrarlo.**

Todo PR que toque datos, features, splits o evaluación lleva un apartado obligatorio de riesgo de fuga.

---

## 9. Decisiones ya tomadas

Si vas a contradecir una, se habla antes.

| # | Decisión | Por qué |
|---|---|---|
| D-01 | Plan en Opus, ejecución con subagentes Sonnet | Planificar es donde se decide bien o mal |
| D-02 | Los huecos **no** se rellenan con `fillna(0)` | Un retorno cero equivale a rellenar el precio hacia delante: autocorrelación falsa |
| D-03 | La residualización guarda betas y las **proyecta** sobre test | Recalcular el factor en test es fuga |
| D-04 | Horizontes 1m, 5m, 15m y 60m, no solo 1m | Rebalancear cada minuto condena el test económico por construcción |
| D-05 | El grafo se recalcula **dentro de cada fold** | Construirlo con la muestra completa contamina todo |
| D-06 | Python 3.12 para todo el equipo | Todo el stack tiene ruedas para 3.12 y deja margen para bajar torch si PyG Temporal falla |
| D-07 | Dependencias con versión exacta (`==`), nunca rangos | Un rango instala cosas distintas según el día |
| D-08 | Dependencias solo en requirements.txt, no en pyproject.toml | Una sola fuente de verdad |

---

## 10. Reparto

| Persona | Área | Épicas |
|---|---|---|
| **Raúl** | Datos e infraestructura | 0, 1, 2, 12 |
| **Piettro** | Estructura del grafo | 3, 4, 5, 6 |
| **Alonso** | Modelado y evaluación | 7, 8, 9, 10 |
| Los tres | Dashboard y memoria | 11, 13 |

*Reparto provisional: se acuerda entre los tres antes de arrancar.*

El área no es propiedad exclusiva: si estás bloqueado, coges un ticket de otra épica cuyas dependencias estén cerradas. Pero **el que conoce el área revisa el PR**.

### Interfaces entre áreas

El contrato que evita bloqueos. Si cambias una, **avisas antes, no después**:

- **Datos → Grafo:** panel limpio de retornos alineados en Parquet.
- **Grafo → Modelado:** matriz de adyacencia por *fold*.

---

## 11. Qué NO se hace

- No se mergea nada sin revisión humana.
- No se acepta un resultado bueno sin comprobar que no hay fuga.
- No se cambia el alcance de un ticket a mitad. Se abre otro.
- No se instala una dependencia sin decirlo en el PR.
- No se optimiza nada antes de que funcione y esté medido.
- No se escribe en `main`.

---

## 12. Dónde está cada cosa

| Necesitas | Fichero |
|---|---|
| Entender el TFM | [README.md](../../README.md) |
| Las reglas (las carga Claude solo) | [CLAUDE.md](../../CLAUDE.md) |
| Los 196 tickets | [BACKLOG.md](BACKLOG.md) |
| El proceso de equipo | [WORKFLOW.md](WORKFLOW.md) |
| Qué hace cada uno ahora | [ESTADO.md](ESTADO.md) |
| Qué hace cada versión | [CHANGELOG.md](../../CHANGELOG.md) |
| El tablero | https://trello.com/b/PBzf0gtT/tfm-miax-lead-lag-cripto |

---

## 13. Primer ticket

Mira **🎯 Sprint actual** en Trello. Coge el más bajo que no tenga dependencias abiertas y que nadie haya cogido.

Ahora mismo: **MIAX-005** (`requirements.txt`) y **MIAX-006** (paquete instalable) son los que desbloquean todo lo demás.
