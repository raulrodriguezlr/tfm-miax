# Correlación cruda entre activos

La foto del «antes» (MIAX-046): cuánto se parecen entre sí los retornos de 1 minuto del universo **sin quitar el factor común**. Es la prueba numérica del espejismo de factor del README §2.3, y se compara después con la misma figura sobre el panel residualizado (MIAX-060). Todo es **descriptivo**: no es una feature ni un filtro, y no debe entrar en ningún modelo.

---

## 1. Qué muestra la figura

Una figura de tres paneles (PNG a 200 dpi y PDF, mismo nombre) más dos tablas CSV:

- **(a) Mapa de calor N×N** (40×40 en 2025T1; N son todas las monedas del periodo) de la correlación de Pearson cruda entre los retornos log de 1 minuto. Escala divergente fija en `[-1, 1]`: si todo sale en el mismo extremo del rojo, ese es el mensaje. Los activos van ordenados **por categoría** (BTC primero, luego las categorías del universo en orden alfabético y, dentro de cada una, los símbolos en orden alfabético), con una línea negra entre categorías. Las casillas sin dato (monedas sin datos en el periodo) se pintan en **gris claro**, no en blanco: en la escala `RdBu_r` el blanco es correlación 0 y parecería un valor (D-02); la nota del método añade «Gris = sin dato».
- **(b) Correlación de cada activo con BTC**, en el mismo orden, coloreada por categoría.
- **(c) Varianza que explica cada componente principal** de la matriz de correlación. Si hay monedas sin datos, el título añade una segunda línea con ellas (sin `USDT`, abreviadas si son más de 6).

El orden es por categoría y no por agrupación jerárquica a propósito: la categoría es una etiqueta exógena y no fabrica bloques; ordenar por la propia correlación los fabricaría incluso en un mercado de un solo factor.

Tablas (en `reports/tables/`):

- `raw_correlation_summary_{periodo}_{intervalo}.csv`: cifras escalares (`metric,value`). Tras `min_periods` lleva `n_symbols_in_universe`, `n_symbols_without_data` y `symbols_without_data` (nombres unidos por `;`, vacío si no hay); `n_symbols` es el número de símbolos **con datos**.
- `panel_eda_{periodo}_{intervalo}.csv`: una fila por símbolo con la categoría, el % de datos válidos, el % de `NaN` de `return`, la desviación típica y la curtosis (`tail_report`, MIAX-037) y la correlación con BTC.

Las salidas se escriben con orden y formato numérico fijos: misma entrada, mismo CSV byte a byte (comprobado en la misma máquina: los autovalores de LAPACK pueden diferir en los últimos dígitos entre máquinas).

---

## 2. Cómo se calcula

- Retornos `return` del panel (`cached_panel`), logarítmicos de 1 minuto, **válidos y sin recortar** (D-22). Lo no válido es `NaN`, nunca 0 (D-02).
- Pearson **por pares con observaciones comunes** (`DataFrame.corr` con `min_periods=1000`: un par con menos observaciones comunes queda `NaN`). Sin residualizar y sin retardo.
- Autovalores de la matriz de correlación, de mayor a menor, como fracción de `N` (N = monedas con datos; en 2026T3, 31 y no 32). Si faltan pares, se sustituyen por la media de los pares con valor (no por 0) y falla con `ValueError` si faltan más del 5 % o si, tras el relleno, la matriz no es semidefinida positiva (el relleno no lo garantiza).
- Una serie constante, toda `NaN` o con menos de `min_periods` datos no tiene diagonal 1: su fila, columna y diagonal son `NaN`. Esa moneda se aparta del resumen y de los componentes (ver la viñeta de MIAX-213), sigue en el mapa en gris y en el panel (b) aparece sin barra, no como 0, con los símbolos anotados. `n_missing_pairs` (tabla de resumen) cuenta solo los pares que faltan entre monedas con datos; en 2025T1 no falta ninguno (`n_missing_pairs` = 0).
- **Monedas sin datos en el periodo (MIAX-213):** las de diagonal `NaN` se apartan del resumen y de los componentes principales, se anotan en la figura (título de (c)) y en la tabla de resumen, y siguen en el mapa de calor (en gris) y en la tabla por símbolo. El 5 % de `max_nan_share` cuenta solo pares que faltan **entre monedas con datos**. Motivo: un par deslistado dentro del periodo, como TONUSDT en 2026T3 (Binance lo deslistó el 30-jun-2026), es legítimo en el universo (D-14); contar sus 31 pares como ausentes hacía fallar la figura con 32 monedas (6,2 % > 5 %). Si la moneda sin datos es BTC (el factor), o si quedan menos de 2 monedas con datos, falla con `ValueError`.
- **Caso borde:** solo se aparta una moneda por diagonal `NaN`. Una moneda con datos propios suficientes pero sin ningún par válido (no se solapa al menos `min_periods` observaciones con ninguna otra) sigue contando sus pares como ausentes y puede hacer fallar el límite del 5 % a propósito: no es un deslistado sino una anomalía que conviene ver.
- Se calcula con toda la muestra del periodo. No hay fuga porque no alimenta nada; precisamente por eso no debe usarse como filtro ni como entrada de un modelo.

---

## 3. Lo medido

2025T1 (2025-01-01 a 2025-04-01), los 40 símbolos de `load_universe()`, 129 600 filas, retornos log válidos sin recortar. Los 40 símbolos tienen más del 90 % de datos. El panel tiene 129 600 filas por símbolo pero solo 129 598 retornos posibles (la primera fila no tiene retorno y el de la última vela queda fuera por indexarse al cierre, D-19), así que `valid_pct` y `return_nan_pct` no suman 100. Categorías del periodo: l1 10, meme 10, defi 7, ai 5, infra 3, l2 2, gaming 2, exchange 1.

| Horizonte | Correlación media entre pares | mediana | mín. | máx. | % pares > 0,5 | 1.er autovalor (de 40) | 2.º | Correlación con BTC: mediana | mín. | máx. |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 min | 0,644 | 0,657 | 0,315 | 0,868 | 88,6 % | 26,38 = 66,0 % | 2,3 % | 0,635 | 0,400 (1000SATSUSDT) | 0,757 (ETHUSDT) |
| 5 min | 0,682 | 0,700 | 0,344 | 0,890 | 93,3 % | 27,86 = 69,7 % | 2,6 % | 0,661 | 0,409 (1000SATSUSDT) | 0,773 (ETHUSDT) |
| 60 min | 0,705 | 0,725 | 0,296 | 0,911 | 92,2 % | 28,84 = 72,1 % | 2,9 % | 0,694 | 0,370 (1000SATSUSDT) | 0,783 (ETHUSDT) |

La figura es solo la de 1 minuto (el horizonte del panel). Las filas de 5 y 60 minutos son **contexto medido** a parte, con el mismo método sobre las barras de 5 y 60 minutos construidas desde las velas de 1 minuto (MIAX-042).

**Lección:** un solo factor común explica entre el 66 % y el 72 % de la varianza y el segundo componente apenas el 2-3 %. La correlación sube con el horizonte (menos ruido de microestructura).

### Otros trimestres (MIAX-213)

Mismo método a 1 min, medido en dos trimestres más:

| Periodo | Símbolos | Correlación media | mediana | mín. | % pares > 0,5 | 1.er componente | 2.º | Correlación con BTC: mediana | mín. | máx. |
|---|---|---|---|---|---|---|---|---|---|---|
| 2022T2 | 40 | 0,570 | 0,623 | 0,099 | 75,5 % | 59,7 % | 2,4 % | 0,676 | 0,214 (LUNAUSDT) | 0,888 (ETHUSDT) |
| 2026T3 | 32 (31 con datos: TONUSDT sin velas) | 0,409 | 0,437 | 0,072 | 31,4 % | 45,5 % | 3,6 % | 0,511 | 0,161 (TRXUSDT) | 0,835 (ETHUSDT) |

Las cifras de 2026T3 están medidas con `python -m miax.viz raw-correlation --period 2026-07-01` el 2026-10-30 (TONUSDT apartada). Lectura: en 2026T3 entre el 19 % y el 22 % de los minutos de WLFI, ASTER y CHZ no tienen operaciones (`NO_TRADES` sueltos, con rachas de 24 minutos como mucho, no huecos de la caché: WLFI 21,5 %, ASTER 21,8 %, CHZ 19,4 %). Como cada minuto sin operaciones invalida dos retornos (D-18), alrededor de un tercio de sus retornos de 1 min están vacíos (34,1 %, 34,0 %, 32,1 %), y esa iliquidez puede bajar la correlación a 1 min (efecto Epps, MIAX-070).

---

## 4. Qué no se ha verificado

- La tabla principal de §3 es **un solo trimestre** (2025T1) y **40 símbolos**; 2022T2 y 2026T3 están medidos solo a 1 min y no se han medido los otros universos.
- La figura es **solo de 1 minuto**; las cifras de 5 y 60 minutos no salen de la figura.
- El **efecto Epps** (precios rancios que parecen reaccionar tarde) y las **colas** (saltos conjuntos, `docs/datos/colas-retornos.md`) pueden inflar o deformar la correlación a 1 minuto. No se ha separado su contribución.
- La correlación de Pearson es sensible a las colas; no se ha medido con una medida robusta en esta figura.
- Que el primer componente sea «Bitcoin» o el mercado en general es una interpretación; aquí solo se mide cuánta varianza explica.

---

## 5. Cómo regenerarla

```bash
python -m miax.viz raw-correlation --period 2025-01-01 --interval 1m
```

Usa el panel de la caché (`data/processed/panel`, MIAX-043) y lo reconstruye si no está o si pasas `--rebuild`. Escribe en `reports/figures/` y `reports/tables/` (ignoradas por git: no se versionan). Códigos de salida: 0 bien, 1 si falla el trabajo (por ejemplo un periodo que no está en el universo), 2 si los argumentos no valen. Desde Python: `miax.viz.build_raw_correlation_report("2025-01-01")`.

---

## 6. Qué la hará comparable con MIAX-060

MIAX-060 repite la figura sobre el panel residualizado. Para que la comparación sea limpia debe mantener el mismo periodo, el mismo orden por categoría, la misma escala `[-1, 1]` y el mismo método de cálculo (Pearson por pares, `min_periods`, mismo cálculo de autovalores) y comparar las mismas cifras de la tabla `raw_correlation_summary_*`: la correlación media entre pares, el % de pares > 0,5, la varianza del 1.er componente y la correlación con BTC.
