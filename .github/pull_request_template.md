<!--
Título del PR: `MIAX-XXX: título del ticket`.
Base del PR: `develop`, nunca `main`.
Un PR = un ticket.
-->

## Qué hace

<!-- Qué cambia este PR, en 2-3 frases. Ticket: MIAX-XXX. -->

## Cómo se ha comprobado

<!-- Qué has ejecutado y qué ha salido. Pega el resumen, no solo "funciona". -->

## Qué queda fuera

<!-- Lo que has visto y no has tocado, o lo que el ticket no cubre. Si algo merece su propio ticket, di cuál. -->

## Riesgo de fuga de datos

<!--
Es el riesgo número uno del proyecto. No lo dejes en blanco ni escribas solo "ninguno":
razónalo. Si el PR no toca nada de esto, di por qué.

- ¿Hay algún `shift`, ventana, split, purga o embargo? ¿Dónde está su comentario de por qué no mira al futuro?
- ¿Algo se calcula con la muestra completa y luego se usa dentro de un fold (grafo, betas, factor, escalado)?
- ¿Hay test que lo cubra?
-->

## Checklist

- [ ] Los tests están en verde.
- [ ] `ruff check .` y `ruff format --check .` salen limpios.
- [ ] `docs/project/ESTADO.md` actualizado con mi entrada.
- [ ] Añadida mi línea a `CHANGELOG.md` bajo `[Sin publicar]`.
- [ ] El PR toca un solo ticket.
- [ ] La base del PR es `develop`.
- [ ] Si hay dependencias nuevas, están declaradas en el PR.
- [ ] Si toca purga, embargo o `shift`, hay test.
