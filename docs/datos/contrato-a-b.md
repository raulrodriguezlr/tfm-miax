# Contrato A → B: el panel de features

Qué entrega el área A (datos y features) y qué puede dar por hecho el área B (residualización y lead-lag) sin preguntar a nadie (MIAX-044). Todo lo que sigue se comprueba por código con `validate_panel` (sección 6).

---

## 1. Qué es y para quién

El **panel** es el único objeto que pasa de A a B: cuatro features por minuto y por activo, más una máscara de validez. Lo consume B para residualizar los retornos y construir el grafo de lead-lag, y después C para entrenar la GNN.

Es un objeto `Panel` (`miax.features.Panel`), con caché en Parquet. Hay **un panel por periodo** del universo trimestral (D-25), no uno global.

---

## 2. Cómo obtener los paneles

```python
from miax.features import cached_period_panels
from miax.ingest.universe import load_universe

universe = load_universe()
for period_start, panel in cached_period_panels(universe, "1m"):
    returns = panel.features["return"]  # DataFrame tiempo x símbolo
    valid = panel.valid  # DataFrame bool, mismo índice y columnas
    array = panel.to_array()  # (T, N, F), con los NaN conservados
```

`cached_period_panels` rinde los paneles de uno en uno y en orden de periodo; si el panel está en caché lo carga y, si no, lo construye y lo guarda. Para leer solo los retornos de un panel guardado (lo que usará el grafo), sin cargar el resto de features:

```python
from miax.features import load_panel, panel_path

panel = load_panel(panel_path("1m", start_ms, end_ms), features=["return"])
```

`to_array()` apila las features en el orden de `panel.features` (o en el de `order=[...]`) y da un `float64` de forma `(T, N, F)`: tiempo, símbolo, feature.

---

## 3. Esquema

Todas las features son `float64`, con el mismo índice y las mismas columnas que `valid`. Están definidas en código en `miax.features.FEATURE_SPECS`.

| Feature | Unidad | Qué es | `NaN` si la vela no es válida | ≥ 0 |
|---|---|---|---|---|
| `return` | log-retorno, unidades naturales (no %) | `ln P_t − ln P_{t-1}` del cierre; solo existe si `t` y `t−1` son válidos | sí | no |
| `realized_vol` | desviación típica del retorno de 1 paso, sin anualizar | `sqrt((W/n) · Σ r²)` de los `W` retornos anteriores, incluido el actual, con `n` los no `NaN` (D-20); `NaN` si `n/W` queda por debajo del mínimo. Por defecto `W = 60` (`vol_window`) y mínimo 80 % (`vol_min_valid_share`) | **no** (puede tener valor con `valid=False`) | sí |
| `rel_volume` | diferencia de logaritmos, sin unidad | `ln(volumen en quote)` menos la media de `ln V` de las velas válidas de la ventana anterior, sin la actual (D-21); `NaN` si la parte válida de la ventana queda por debajo del mínimo. Por defecto la ventana es de 10080 velas (`activity_window`) y el mínimo 50 % (`activity_min_valid_share`) | sí | no |
| `rel_trades` | diferencia de logaritmos, sin unidad | lo mismo con el número de operaciones | sí | no |

Además del conjunto de features:

- `valid` (`bool`): `True` si la vela que cierra en esa fila tuvo operaciones (D-18).
- Los parámetros de ventana y de cobertura mínima (`vol_window`, `vol_min_valid_share`, `activity_window`, `activity_min_valid_share`) son argumentos de `build_panel` y se guardan en `meta.json`; los valores de arriba son los de por defecto, no constantes del contrato.
- **Índice**: `DatetimeIndex` `datetime64[ms, UTC]`, creciente, sin duplicados y de paso fijo igual al intervalo (1 minuto), con `freq = None` (D-16).
- **Columnas**: los símbolos del periodo (`panel.symbols`), en el mismo orden en todas las tablas.

---

## 4. Convención de tiempo

Cada fila está indexada por **el instante en que se conoce** (D-19): la fila `τ` es la vela `[τ − 1m, τ)`. La vela de 10:00 a 10:01 va en la fila **10:01**.

Consecuencias para B:

- Lo que tiene hora `τ` se sabía en `τ`: usar en `τ` las filas con hora `≤ τ` no es fuga. Lo que se quiera predecir va siempre en filas posteriores a las que se usan para predecirlo.
- Para leer el índice de tiempo: el retorno de `j` en la fila `τ + k` se conoce `k` minutos más tarde que el de `i` en `τ`. El estimador y la convención de lead-lag (quién lidera a quién, signo de `k`) son del área B y no forman parte de este contrato.
- La **primera fila de cada periodo** usa velas anteriores al periodo (calentamiento de las ventanas de volatilidad y de volumen). Solo es pasado: no hay fuga, pero el periodo no es independiente de lo anterior en esas filas.

---

## 5. Validez y huecos

- Lo que falta es `NaN`, **nunca 0** (D-02, D-18).
- Un retorno es válido solo si lo son la vela `t` y la vela `t−1`. No se puentea un hueco calculando el retorno desde el último precio válido. Un hueco invalida su retorno y el siguiente.
- Un retorno 0 con operaciones se conserva: es un dato real.
- `realized_vol` **puede tener valor con `valid=False`**: es una ventana de retornos (60 por defecto), y que la vela actual falle no la anula si hay cobertura suficiente (80 % por defecto). Usa `valid` para enmascarar si lo necesitas.
- La causa del hueco (`GapCause`) **no está en el panel** y no debe usarse como feature: se clasifica mirando al futuro (D-17).

---

## 6. Garantías que comprueba `validate_panel`

`validate_panel(panel)` reúne todas las violaciones y lanza **un único** `ValueError`. Las violaciones de feature (d–g) dicen la feature, el número de celdas, el primer instante afectado y el símbolo; las de índice y de features desconocidas (a–c) se describen sin celdas. Comprueba:

1. **a.** Toda feature del panel está en `FEATURE_SPECS`. Un subconjunto vale: `load_panel` puede leer solo `return`.
2. **b.** El índice es `datetime64[ms, UTC]` (D-16).
3. **c.** El paso entre filas consecutivas es siempre el del intervalo. No se exige alineación al reloj.
4. **d.** Ninguna feature contiene `±inf`.
5. **e.** `return`, `rel_volume` y `rel_trades` son `NaN` donde `valid` es `False`.
6. **f.** `return` no puentea huecos (D-18): un retorno no `NaN` exige que la vela anterior sea válida. La primera fila no se comprueba porque su vela anterior queda fuera del panel.
7. **g.** `realized_vol` no tiene valores negativos.

Se ejecuta **automáticamente** al final de `build_panel` y de `load_panel`. Un panel de la caché que no la cumpla hace que `cached_panel` lo reconstruya.

---

## 7. Lo que B NO debe hacer

- **`fillna(0)`** sobre las features: un 0 es un dato, y rellenar los huecos con ceros inyecta autocorrelación falsa (D-02).
- **Ajustar el factor común o las betas con el test** (D-03): se ajustan en train y se proyectan.
- **Construir el grafo con la muestra completa** (D-05): se recalcula dentro de cada fold.
- **Modificar los DataFrames del panel sin copiar**: la estructura no se reasigna, pero los DataFrames internos no están protegidos. Haz `.copy()` antes de tocar nada.
- Usar la causa del hueco como feature (D-17).

---

## 8. Periodos y símbolos

- Hay **un panel por periodo del universo trimestral** (D-25). Los semestral y anual son pruebas de robustez y se consumen igual.
- Las columnas son los símbolos de ese periodo, **en el orden del universo**, y **cambian entre periodos**: no se pueden apilar paneles de periodos distintos sin alinear columnas.
- Un par del universo sin ninguna vela en el periodo sale como columna con `valid=False` en todas las filas y `return`, `rel_volume` y `rel_trades` en `NaN`, no desaparece. `realized_vol` puede conservar valores en las primeras filas si el par tenía velas justo antes del periodo (ver sección 5). Caso conocido: `TONUSDT` en 2026T3, que sale entera `NaN`.
- `LUNAUSDT` no se parte (D-24): `valid=False` en 26 240 filas dentro de su periodo; `return`, `rel_volume` y `rel_trades` son `NaN` en todo el hueco (`realized_vol` conserva unos pocos valores al inicio).

---

## 9. Límites

De momento **solo a 1 minuto**. Las features a 5, 15 y 60 minutos son MIAX-102, y entonces el contrato se amplía.

---

## 10. Versiones y cambios

- `FEATURES_VERSION` (en `miax.features.panel`) cambia cuando cambia **cómo se calcula** una feature o su validez. Si sube, la caché se reconstruye.
- `PANEL_FORMAT_VERSION` (en `miax.features.panel_store`) cambia cuando cambia el **formato en disco**.
- Cualquier cambio de este contrato (features, unidades, convención de tiempo, validez) se avisa **antes** en [ESTADO.md](../project/ESTADO.md) §3 y con una decisión nueva.
