# Residuo sin el retraso respecto al mercado

Control de robustez D-39 (b), MIAX-216. Al residualizar, a cada moneda se le quita el mercado **del mismo minuto** (D-34, D-35, D-36). Si una moneda reacciona tarde, en su residuo queda el mercado de los minutos anteriores, y el lead-lag podría ser solo eso: «el mercado se movió y esta moneda llegó tarde» (riesgo R2). Aquí se quita también ese retraso y se repite el contraste de señal de MIAX-057.

> **Resultado:** **la señal sobrevive.** Con los residuos sin retraso, la señal cruzada sale en **12 de 12 folds** con la media y con el PCA (regla de D-44, fijada antes de ver los datos), y su tamaño apenas cambia (ratio frente al azar 6,5 → 6,4). El retraso respecto al mercado solo es sistemático con el factor **BTC** (γ ≈ 0,05 en los 12 folds); con la media y el PCA es prácticamente cero. D-34, D-35 y D-36 siguen siendo el residuo principal.

---

## 1. Qué se compara

`LaggedFactor` (`miax.features.factor_lagged`) envuelve cualquier factor (BTC, media, PCA) y, en dos pasos, todo con train:
1. Ajusta el factor base igual que siempre: **las betas del mismo minuto no cambian**.
2. Regresión de cada residuo base sobre el factor de mercado de los **5 minutos anteriores**, `e_i(t) ~ f(t−1), …, f(t−5)` (OLS con constante, que no se resta, como D-34). Los coeficientes `γ` se congelan.

Residuo sin retraso: `e_i(t) − Σ_ℓ γ_iℓ · f(t−ℓ)`.
- Los retardos se toman **por tiempo**: si falta el minuto `t−ℓ`, el residuo de `t` queda vacío; nunca se usa otro minuto.
- Solo usa el pasado, sin fuga (pasa el contrato de `FactorModel` y el detector de MIAX-058).
- Los primeros 5 minutos de cada muestra quedan vacíos, porque su pasado no está en ella (5 de ~130 000 por trimestre).
- El nodo mercado del grafo no cambia: `factors()` es el del base.
- 5 retardos, fijados de antemano: los mismos que mira el contraste de 057.

---

## 2. Método

- Los **12 folds de D-37** (prueba 2023T1–2025T4, 12 meses de train). Todo se mide en test.
- **(A) Diagnóstico** (`market_lag_diagnostics.csv`), mediana entre monedas: `corr(e_i(t), f(t−ℓ))` con signo y en valor absoluto para ℓ = 1..5, varianza quitada y la extra de la versión sin retraso, y la mediana de `γ` por retardo.
- **(B) Señal** (`market_lag_signal_by_fold.csv`, `market_lag_signal_decision.csv`): el contraste de 057 (`study_signal`, 1 min) con `btc`, `mean`, `pca`, `mean_lag` y `pca_lag`, en ese orden para que las semillas coincidan con 057.
  - **Comprobación:** las filas de `btc`, `mean` y `pca` reproducen **exactamente** las de MIAX-057 (`idio_signal_by_fold.csv`).
  - **Regla (D-44, fijada antes):** queda señal si `p_cross < 0,05` en ≥ 9 de 12 folds con `mean_lag` **y** `pca_lag`; pivote si ≤ 3 con alguno; si no, no concluyente.
- Reproducir: `python -m miax.eval market-lag` (unos 28 minutos; `--skip-signal` hace solo la parte A).

---

## 3. Resultados

### 3.1. ¿Hay retraso respecto al mercado? (mediana de los 12 folds)

| | `btc` | `btc_lag` | `mean` | `mean_lag` | `pca` | `pca_lag` |
|---|---|---|---|---|---|---|
| `corr(e(t), f(t−1))` con signo | **+0,027** | −0,005 | −0,004 | −0,004 | −0,004 | −0,005 |
| \|`corr(e(t), f(t−1))`\| | 0,027 | 0,014 | 0,021 | 0,019 | 0,022 | 0,022 |
| \|`corr(e(t), f(t−2))`\| | 0,006 | 0,006 | 0,007 | 0,010 | 0,007 | 0,009 |
| `γ` del retardo 1 (train) | — | **0,051** | — | −0,001 | — | 0,000 |
| Varianza quitada | 0,342 | 0,342 | 0,604 | 0,601 | 0,600 | 0,598 |

- **Con BTC el retraso es real:** la correlación con el BTC del minuto anterior es positiva en los 12 folds (de +0,007 a +0,109), con `γ` entre 0,035 y 0,081. La versión sin retraso la quita fuera de muestra (con signo, +0,027 → −0,005; en valor absoluto baja en 10 de 12 folds). Es lo que vio MIAX-053.
- **Con la media y el PCA no hay retraso sistemático:** la correlación con signo es ≈ 0 y las `γ` de train también. El 0,02 en valor absoluto son monedas con signos distintos que **no se repiten de train a test**: corregirlo apenas lo baja (6 y 7 de 12 folds), y en algún fold lo empeora (PCA en 2024T3–T4). Tiene sentido: la media y el PCA ya incluyen a las monedas lentas, así que el factor «llega tarde» con ellas.
- Retardos 2 a 5: ≈ 0 en todos los casos.
- La varianza que quita la corrección es despreciable (< 0,001 en la mediana).

### 3.2. ¿Sobrevive el lead-lag?

| Factor | Folds con `p_cross < 0,05` | Ratio cruzado frente al azar (mediana) | Ratio propio | Ratio de asimetría |
|---|---|---|---|---|
| `mean` (057) | 12 / 12 | 6,48 | 341 | 5,90 |
| **`mean_lag`** | **12 / 12** | **6,35** | 365 | 5,69 |
| `pca` (057) | 12 / 12 | 6,48 | 354 | 5,90 |
| **`pca_lag`** | **12 / 12** | **6,43** | 359 | 6,07 |
| `btc` (referencia) | 12 / 12 | 13,10 | 172 | 12,92 |

**Veredicto con la regla de D-44: queda señal** con los residuos sin retraso (y, como en 057, con los de siempre). El ratio cruzado cambia un 2 % como mucho. 2025T4 sigue dominado por el desplome del 10 de octubre, como en 057.

---

## 4. Qué significa para la Épica 4

- **R2 no se confirma:** el lead-lag entre residuos de media y PCA no es el mercado llegando tarde. Se puede construir el grafo sobre D-35/D-36.
- **Con el factor BTC sí hay que tenerlo en cuenta:** su residuo arrastra el BTC del minuto anterior, y en un grafo saldría como «BTC → todas, con un minuto de retraso». Es otro motivo, además de MIAX-055, para no usar BTC como factor principal.
- **Control obligatorio (D-39 b):** cada grafo se reporta también con la versión sin retraso (`LaggedFactor` sobre el mismo factor). Cuesta poco: unos segundos por fold.
- La corrección con media y PCA no aporta y puede meter ruido (sus `γ` no son estables): no se usa como residuo principal.

---

## 5. Limitaciones

- Solo 1 minuto y la beta por fold de 12 meses (D-43).
- `γ` fijas por fold: si el retraso cambia dentro del año, no se recoge (con BTC baja de 0,06–0,08 en 2023 a 0,035–0,05 en 2025).
- Una moneda sin 1000 minutos completos (residuo y los 5 retardos) en train queda fuera de la versión sin retraso; en los 12 folds no ha pasado con ninguna (solo BTC con el factor BTC, cuyo residuo ya es vacío por D-34).
- No se ha combinado con el factor sin la propia moneda (MIAX-215).

---

## Relacionado

- D-34, D-35, D-36: los residuos de siempre. D-39 (b): este control. D-43: MIAX-053, que midió el retardo 1. D-44: la regla de señal de MIAX-057.
- `docs/datos/senal-idiosincratica.md` (057) y `docs/datos/betas-moviles-y-horizonte.md` (053).
