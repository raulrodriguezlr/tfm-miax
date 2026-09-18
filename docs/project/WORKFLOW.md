# Cómo trabajamos

Proceso de equipo para el TFM. Tres personas, meses de trabajo, un repositorio. Las reglas técnicas están en [CLAUDE.md](../../CLAUDE.md); aquí está el proceso.

---

## 1. Tablero

Cuatro columnas. Un ticket solo puede estar en una.

| Columna | Qué significa | Cómo se sale |
|---|---|---|
| **Backlog** | Definido, sin empezar. | Alguien lo coge y crea la rama. |
| **In Progress** | Alguien está trabajando en él ahora. | Se abre el PR. |
| **Review** | PR abierto, esperando revisión humana. | Otra persona aprueba y se mergea. |
| **Done** | Mergeado en `main`. | No se sale. |

### Límite de trabajo en curso

**Un ticket por persona en *In Progress*.** Sin excepciones.

Si te bloqueas, no abres otro: lo dices en el grupo y lo mueves de vuelta a *Backlog* con una nota de por qué. Tener tres cosas a medias es la forma más rápida de no terminar ninguna.

### Dónde vive el tablero

El backlog canónico, numerado y ordenado, está en [BACKLOG.md](BACKLOG.md).

Para el tablero visual, GitHub Projects sobre este repo. Los tickets se crean como *issues* con el título `MIAX-XXX: título` y se enlazan al PR.

---

## 2. Vida de un ticket

```
  1. Lees ESTADO.md         →  qué están haciendo los otros dos
  2. Coges el ticket        →  el más bajo disponible que no esté bloqueado
  3. Lees sus dependencias  →  si no están en Done, no empieces
  4. Creas la rama          →  git checkout develop && git pull
                               git checkout -b feature/MIAX-XXX
  5. Planificas             →  modo plan, en Opus, con el ticket delante
  6. Ejecutas               →  coder → auditor → fixer → tester (Sonnet)
  7. Revisas el diff TÚ     →  no delegues esto
  8. Actualizas ESTADO.md   →  y CHANGELOG.md, antes del push
  9. Abres el PR            →  base develop, pasa a Review
 10. Un compañero aprueba   →  merge a develop = snapshot, pasa a Done
```

### Definición de *listo para empezar*

Un ticket se puede coger si:
- Sus dependencias están en *Done*.
- Se entiende qué hay que hacer sin preguntar.
- Tiene un criterio de "hecho" comprobable.

Si no cumple las tres, primero se arregla el ticket.

### Definición de *hecho*

Un ticket está hecho cuando:
- El código está en `src/` (no en un notebook).
- Tiene tests y pasan.
- Un humano ha leído el diff entero.
- Un compañero ha aprobado el PR.
- Está mergeado en `main`.

---

## 3. Ramas, snapshots y releases

```
   feature/MIAX-XXX  ──PR──►  develop  ──PR──►  main
                                 │                │
                              SNAPSHOT         RELEASE
                       v0.5.0-snapshot.N       v0.5.0 + tag
```

| Cosa | Formato |
|---|---|
| Rama de ticket | `feature/MIAX-XXX` (nace de `develop`, vuelve a `develop`) |
| Rama de arreglo | `fix/MIAX-XXX` |
| Commit | `MIAX-XXX: qué hace, en imperativo` |
| PR | `MIAX-XXX: título del ticket`, con base `develop` |

**Nunca se escribe directamente en `main` ni en `develop`.** Ambas están protegidas.

- **Cada merge a `develop` genera un snapshot.** Es el estado integrado más reciente: puede estar incompleto, pero pasa los tests.
- **Cada merge a `main` genera una release** con tag y entrada en el [CHANGELOG](../../CHANGELOG.md). `main` tiene que estar presentable en cualquier momento — si mañana hay que enseñar el proyecto, se enseña `main`.
- `MINOR` sube al cerrar una épica. `v1.0.0` es lo que se entrega al tribunal.

### Cuerpo del PR

```markdown
## Qué hace
Dos o tres frases.

## Cómo se ha comprobado
Tests que se han escrito, comando ejecutado, resultado.

## Qué queda fuera
Lo que he visto y NO he tocado, con su ticket nuevo si procede.

## Riesgo de fuga de datos
Ninguno / o explicación de por qué no la hay.

## Checklist
- [ ] ESTADO.md actualizado
- [ ] CHANGELOG.md actualizado bajo [Sin publicar]
- [ ] Tests en verde
```

El apartado de fuga de datos es obligatorio en todo PR que toque datos, features, splits o evaluación. Es el riesgo número uno del proyecto.

---

## 4. Revisión

- **Revisa una persona, no un agente.** El auditor automático ya pasó en el paso 5; el PR es control humano.
- No se revisa por encima. Si no entiendes un trozo, preguntas antes de aprobar.
- No se aprueba con tests en rojo.
- Si el PR toca más de un ticket, se rechaza y se parte.

Comentarios de revisión: una línea, ubicación, problema, arreglo. Sin párrafos.

---

## 5. Reparto

| Persona | Área principal | Épicas |
|---|---|---|
| **A** | Datos e infraestructura | 0, 1, 2, 12 |
| **B** | Estructura del grafo | 3, 4, 5, 6 |
| **C** | Modelado y evaluación | 7, 8, 9, 10 |
| **Los tres** | Dashboard y memoria | 11, 13 |

El área principal no es propiedad exclusiva: si estás bloqueado, coges un ticket de otra épica cuyas dependencias estén cerradas. Pero el que conoce el área revisa el PR.

### Interfaces entre áreas

Son el contrato que evita que unos bloqueen a otros:

- **A → B:** panel limpio de retornos alineados en Parquet.
- **B → C:** matriz de adyacencia por *fold*.

Si la interfaz cambia, se avisa antes, no después.

---

## 6. Ritmo

- **Semanal:** repaso del tablero. Qué se movió, qué se atascó, qué entra la semana que viene.
- **Cuando algo se atasca más de tres días:** se habla. No se sufre en silencio.
- **Al cerrar cada épica:** una figura o tabla que vaya directa a la memoria. Si una épica no produce nada publicable, estaba mal planteada.

No escribimos la memoria al final. Cada épica deja su material escrito mientras está fresco.

---

## 7. Qué hacemos cuando un resultado sale mal

Nada. Se documenta y se sigue.

El TFM está diseñado para que un resultado negativo sea una conclusión válida (ver README §9). El único fracaso real es un resultado no reproducible o contaminado por fuga de datos.

Si un resultado sale **demasiado bien**, se para y se busca la fuga antes de celebrarlo.
