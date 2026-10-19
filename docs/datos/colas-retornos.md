# Colas de los retornos

Decisión sobre los valores extremos de los retornos de 1 minuto (MIAX-037). Base del capítulo de datos de la memoria (MIAX-186). Las cifras están medidas sobre la caché del paquete del Drive, retornos de `returns_from_cache` (MIAX-036).

---

## 1. Qué se recorta

**Nada, por defecto.** `log_returns` y `returns_from_cache` devuelven los retornos tal cual, y ningún paso del pipeline aplica `clip_returns` por su cuenta.

La winsorización existe solo como **análisis de sensibilidad**, a petición explícita, con límites **ajustados únicamente en train** y proyectados sobre test (`fit_clip_bounds` y `clip_returns`, en `miax.features.tails`). Un recorte nunca produce 0: lo que no se recorta queda igual y lo inválido sigue siendo `NaN` (D-02, D-18).

---

## 2. Por qué no se recorta

Cripto tiene colas gordas de verdad, y recortar de más mata la señal de los eventos que más importan. Tres hechos medidos sostienen la decisión:

1. **Las colas pesan mucho en la suma de cuadrados.** Lo medido es que el 0,1 % de los retornos más grandes aporta entre el 12 % y el 40 % de la suma de cuadrados. Que por eso la correlación, el OLS y el PCA se muevan por ellas es conocimiento general de estadística, no medido aquí.
2. **No parecen errores de datos.** Solo entre el 14,5 % y el 20,8 % de los retornos con |z| > 8 se revierte al minuto siguiente (signo contrario y módulo mayor que la mitad del suyo, es decir, más de la mitad del salto); el resto persiste.
3. **Parecen eventos de mercado.** Los saltos son conjuntos: de 5.459 minutos con algún símbolo a |z| > 5, en 1.048 saltan 3 o más a la vez (y en 3.777 solo uno). Que sean eventos de mercado es una interpretación de estos conteos, no una demostración.

Recortar sin más eliminaría justo lo que une a los activos en un salto común, que es lo que el proyecto quiere estudiar.

---

## 3. Lo medido

Retornos logarítmicos de 1 minuto **válidos**, 2025 completo (2025-01-01 a 2026-01-01), 10 símbolos.

| Símbolo | n | std | curtosis | p0,1 | p99,9 | z máx | n > 5 sd | n > 10 sd | var top 0,1 % | reversión \|z\| > 8 |
|---|---|---|---|---|---|---|---|---|---|---|
| BTCUSDT | 525600 | 0.0006284 | 80.67 | -0.003631 | 0.003636 | 69.78 | 1699 | 183 | 0.1321 | 0.1846 |
| ETHUSDT | 525600 | 0.001029 | 276.1 | -0.005988 | 0.005908 | 98 | 1649 | 192 | 0.1679 | 0.1745 |
| SOLUSDT | 525600 | 0.00121 | 29.83 | -0.007062 | 0.007156 | 30.98 | 1671 | 197 | 0.1216 | 0.2015 |
| DOGEUSDT | 525600 | 0.001423 | 3222 | -0.007355 | 0.007382 | 189.2 | 1180 | 143 | 0.2733 | 0.1685 |
| PEPEUSDT | 525526 | 0.001951 | 2303 | -0.008452 | 0.008648 | 141 | 678 | 82 | 0.219 | 0.2083 |
| WIFUSDT | 509832 | 0.002353 | 2334 | -0.01013 | 0.01041 | 167.4 | 634 | 94 | 0.2326 | 0.1801 |
| BONKUSDT | 520827 | 0.002106 | 8493 | -0.009108 | 0.009723 | 243.7 | 662 | 74 | 0.3237 | 0.145 |
| 1000SATSUSDT | 482062 | 0.002352 | 3223 | -0.0115 | 0.01116 | 159.9 | 882 | 171 | 0.3832 | 0.1615 |
| LINKUSDT | 524646 | 0.00149 | 1894 | -0.007075 | 0.007228 | 139.7 | 917 | 110 | 0.2323 | 0.1574 |
| AAVEUSDT | 525132 | 0.001687 | 11780 | -0.007145 | 0.007619 | 231.9 | 668 | 86 | 0.3981 | 0.1986 |

Cómo se calcula cada columna, sobre los retornos válidos (sin `NaN`) de cada símbolo:

- `n`: número de retornos válidos. `std`: desviación típica de esa misma muestra.
- `curtosis`: en exceso (una normal vale 0), `Series.kurt()` de pandas.
- `p0,1` y `p99,9`: cuantiles 0,001 y 0,999.
- `z`: `|r| / std`, con la `std` de la propia muestra. `z máx` es su máximo; `n > 5 sd` y `n > 10 sd` cuentan los retornos con z mayor que 5 y 10.
- `var top 0,1 %`: fracción de la suma de cuadrados que aportan el 0,1 % de retornos más grandes en valor absoluto.
- `reversión |z| > 8`: fracción de los retornos con |z| > 8 (valor absoluto, no z > 8 con signo) cuyo retorno del minuto siguiente tiene signo contrario y módulo mayor que 0,5 veces el suyo.

### Eventos de mercado

Mismos 10 símbolos y año, con z calculado con la std de cada símbolo:

| Minutos con… | Número |
|---|---|
| al menos 1 símbolo a \|z\| > 5 | 5.459 |
| al menos 3 símbolos a la vez | 1.048 |
| un solo símbolo | 3.777 |

Hay **68 minutos con 10 de 10 símbolos a \|z\| > 5** en 2025. Los cinco primeros en orden cronológico son 2025-01-07 15:43, 2025-01-10 13:31, 2025-01-13 14:37, 2025-01-14 13:31 y 2025-01-15 13:31 UTC. Por hora:minuto, los 68 se reparten así: 12:31 → 5, 13:31 → 4, 14:37 → 2, 20:17 → 2, y el resto una vez cada uno (15:43, 22:49, 00:51, 06:57…). De los 1.048 minutos con 3 o más símbolos extremos, 9 caen a las 13:30-13:31.

Las etiquetas 12:31 y 13:31 son el instante en que se conoce el retorno de la vela que abre a las 12:30 o las 13:30 (convención D-19: `open_time + intervalo`).

**Hipótesis NO verificada:** en 9 de los 68 minutos (13 %) la vela es la de las 12:30 o las 13:30 UTC, que corresponden a las 8:30 de Nueva York en horario de verano y de invierno, hora habitual de publicaciones macro de EE. UU.; por azar serían un 0,14 % (730 / 525.600: 2 de los 1.440 minutos del día). No se ha contrastado con un calendario macro, y la interpretación del cambio de hora es mía.

### Crisis: el umbral global engaña

Una semana de crisis, retornos válidos de 1 minuto:

| Semana | Símbolo | n | std | curtosis | p0,1 | p99,9 | z máx | n > 5 sd | n > 10 sd | var top 0,1 % |
|---|---|---|---|---|---|---|---|---|---|---|
| 2022-05-07..2022-05-14 | LUNAUSDT | 8681 | 0.03831 | 63.36 | -0.3703 | 0.2456 | 21.04 | 54 | 13 | 0.1807 |
| 2022-11-06..2022-11-12 | FTTUSDT | 8640 | 0.0124 | 23 | -0.08154 | 0.08784 | 14.5 | 54 | 3 | 0.1002 |

Las ventanas son semiabiertas, **fin exclusivo**: terminan a las 00:00 UTC del último día sin incluirlo (con fin 2022-11-13, FTTUSDT daría otra cosa: n 10080, curtosis 24,27).

Referencia, BTCUSDT las mismas semanas: curtosis 13,1 y 92,75; z máx 14,02 y 24,41.

El z máx de LUNA es solo 21, aunque su cuantil 0,1 % es -0,37 en un minuto. En una crisis la propia caída infla la std de la ventana, así que **un umbral en sigmas globales subestima los extremos** justo cuando más importan. Es otra razón para no filtrar por z.

---

## 4. Estabilidad de las colas

Límites ajustados con `fit_clip_bounds` (por defecto 0,001 y 0,999) sobre el **primer semestre** de 2025 (2025-01-01 a 2025-07-01, fin exclusivo) y proyectados con `clip_returns` sobre el **segundo** (2025-07-01 a 2026-01-01), mismos 10 símbolos, retornos válidos de 1 minuto. Si las colas fueran estables, caería fuera de límites ~0,2 % (0,1 % por cada lado).

| Símbolo | n válidos 2S | fuera de límites | % fuera | ratio vs 0,2 % |
|---|---|---|---|---|
| BTCUSDT | 264960 | 203 | 0,0766 | 0,38 |
| ETHUSDT | 264960 | 242 | 0,0913 | 0,46 |
| SOLUSDT | 264960 | 173 | 0,0653 | 0,33 |
| DOGEUSDT | 264960 | 335 | 0,1264 | 0,63 |
| PEPEUSDT | 264886 | 298 | 0,1125 | 0,56 |
| WIFUSDT | 250541 | 283 | 0,1130 | 0,57 |
| BONKUSDT | 260552 | 412 | 0,1581 | 0,79 |
| 1000SATSUSDT | 236905 | 671 | 0,2832 | 1,42 |
| LINKUSDT | 264327 | 358 | 0,1354 | 0,68 |
| AAVEUSDT | 264545 | 312 | 0,1179 | 0,59 |

Nueve de diez símbolos quedan por debajo del 0,2 % esperado y 1000SATSUSDT por encima. Es un resultado descriptivo de 10 símbolos y un solo reparto 1S/2S.

Sensibilidad, correlación de Pearson en el segundo semestre, sin recorte → con recorte: BTCUSDT–ETHUSDT 0,7691 → 0,7840; BTCUSDT–DOGEUSDT 0,6105 → 0,7069. El recorte no introdujo ningún 0 nuevo (396.799 ceros antes y después) y respetó los `NaN`.

Esto **no es un fold de validación**: es un solo corte, sin purga ni embargo. El flujo real lo hará MIAX-135.

---

## 5. Qué no se ha verificado

- La hipótesis de las 12:30 y 13:30 UTC (publicaciones macro de EE. UU.): no se ha contrastado con un calendario de eventos.
- Todo es de **2025 y 10 símbolos** (más las dos semanas de crisis). No se ha medido en los 101 símbolos del universo ni en otros años.
- Todo es de **1 minuto**. Los retornos a 5, 15 y 60 minutos (MIAX-042) tienen colas distintas y habrá que medirlos aparte.
- La fracción de reversión depende de la definición usada (signo contrario y más de la mitad del salto —módulo estrictamente mayor que 0,5 veces el suyo—, a un minuto); con otra ventana saldría otra cifra.
- No se ha elegido un percentil de recorte: no hay evidencia para fijarlo, y depende del estimador.

---

## 6. Cómo reproducirlo

```python
from miax.features import returns_from_cache, tail_report

returns = returns_from_cache(symbols, "1m", start_ms, end_ms)
report = tail_report(returns)
```

`tail_report` devuelve una fila por símbolo con `n`, `std`, `kurtosis`, los cuantiles pedidos, `max_abs_z`, un conteo por nivel de sigma y `top_var_share`. Es **descriptivo de la muestra que se le pasa**: `std` y `max_abs_z` usan toda ella, así que no sirven como filtro ni como umbral (mirarían al futuro).

---

## 7. Cuándo se usaría la winsorización

Solo como sensibilidad, para ver cuánto depende un resultado de las colas:

- **MIAX-056** (análisis de sensibilidad del resultado al factor elegido; el recorte sería una variante más).
- **MIAX-079**: PCMCI con `ParCorr` (correlación parcial; que sea sensible a las colas es conocimiento general de estadística, no medido aquí).
- **MIAX-052 y MIAX-050**: residuos por OLS y factor como primera componente principal, ambos, por conocimiento general de estadística (no medido aquí), muy influidos por los extremos.

Siempre así: `fit_clip_bounds(train)` con solo las filas de train (por activo, nunca con la muestra completa) y `clip_returns(test, bounds)` para proyectar los mismos límites sobre test. Ajustar los límites con train y test juntos es fuga de datos.
