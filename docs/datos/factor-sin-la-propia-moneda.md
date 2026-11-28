# Factor sin la propia moneda (leave-one-out)

Control de robustez D-39 (c), MIAX-215. Con el factor media (D-35) y el PCA (D-36), cada moneda entra en el factor que se le resta, y por eso sus residuos **suman cero** en cada minuto: son linealmente dependientes. Aquí se calcula el factor de cada moneda **con las demás** y se compara.

> **Resultado:** el leave-one-out (LOO) **quita la dependencia exacta**, pero **no quita la correlación negativa de fondo** (≈ −1/(N−1)), y apenas cambia las correlaciones por pares (0,008 de media frente a un |ρ| de 0,055). Sale de la dependencia porque sus betas quedan algo por debajo de 1, es decir, **dejando un poco de mercado en el residuo**. D-35 y D-36 siguen siendo el residuo principal.

---

## 1. Qué se compara

| Modelo | Factor de la moneda i en el minuto t | Beta de i |
|---|---|---|
| `mean` (D-35) | media de **todas** las monedas con dato | OLS contra ese factor, en train |
| `loo_mean` | media de **las demás** con dato (`NaN` si tienen dato menos de la mitad de las demás) | OLS contra su propio factor, en train |
| `pca` (D-36) | regresión transversal sobre las cargas del PCA de **todas** | la carga |
| `loo_pca` | regresión transversal sobre las cargas del PCA de la correlación **sin i** | OLS contra su propio factor, en train |

- Las clases son `LooMeanFactor` y `LooPcaFactor` (`miax.features.factor_loo`). Cumplen el contrato de `FactorModel` y pasan el detector de fugas de MIAX-058: todo se ajusta en train y se congela.
- **El nodo mercado sigue siendo uno solo:** `factors()` devuelve el mismo factor que `MeanFactor` o `PcaFactor`. Los factores por moneda se ven con `loo_factors()`.
- La moneda i no entra en su factor **ni en el último decimal**: cambiar su retorno no cambia su factor.

---

## 2. Por qué la correlación negativa no desaparece

La cuenta con el caso más sencillo: todas las monedas siguen al mercado con beta 1, más un ruido propio `u_i` de la misma varianza.
- **Con la media de todas:** `e_i = u_i − media(u)`. La correlación entre dos residuos es exactamente **−1/(N−1)** y la suma de los residuos es 0.
- **Con la media de las demás:** `e_i = u_i − media(u de las demás)`. La correlación vuelve a ser **−1/(N−1)**, y si la beta es 1, la suma también es 0.

Es decir, restar a cada moneda «lo que hacen las otras» crea esa correlación negativa se incluya o no a la propia moneda. Lo único que cambia con el LOO es la beta: al estimarla contra un factor que no contiene a la moneda, sale **por debajo de 1** (la parte de ruido del factor la atenúa). Con beta < 1 la suma deja de ser exactamente 0… porque en el residuo queda una fracción `1 − β` del mercado.

Por eso, antes de ver los datos, lo esperable era: la dependencia exacta desaparece en la medida en que las betas se alejan de 1, y la correlación media con signo apenas se mueve. Es lo que sale.

---

## 3. Método

- Los **12 folds de D-37** (prueba de 2023T1 a 2025T4, 12 meses de train; `load_fold` de MIAX-057). Cada modelo se ajusta en train y se mide **solo en test**.
- Métricas por fold y modelo (`miax.eval.loo_factor`):
  - `pair_mean` y `pair_abs_mean`: correlación media con signo y en valor absoluto entre pares de residuos (Pearson por pares, ≥ 1000 minutos); `floor` = −1/(N−1).
  - `correlation_drop`: cuánta correlación quita el factor (D-41).
  - `min_eigenvalue`: **autovalor mínimo** de la correlación de los residuos en los minutos en que todas las monedas tienen dato. 0 = dependencia exacta (la matriz no se puede invertir). `min_eigenvalue_pairwise` es la misma cuenta con la correlación por pares; con huecos no es una matriz válida y puede dar negativos, por eso no es la principal.
  - `sum_ratio`: desviación de la suma de los residuos frente a la que tendría si fueran independientes. 0 = suman cero.
  - `market_exposure`: mediana entre monedas de |corr(residuo, factor de mercado)|.
  - `beta_mean`: media de las betas.
  - `matrix_change`: media de |ρ_LOO − ρ_estándar| sobre los mismos pares (cuánto cambiaría un grafo construido par a par).
- Reproducir: `python -m miax.eval loo-factor` (unos 3,5 minutos). Tablas en `reports/tables/loo_factor_folds.csv` y `loo_factor_summary.csv`.

---

## 4. Resultados (mediana de los 12 folds, ~33 monedas por fold)

| | `mean` | `loo_mean` | `pca` | `loo_pca` |
|---|---|---|---|---|
| ρ medio con signo (suelo −0,032) | −0,028 | −0,026 | −0,022 | −0,016 |
| \|ρ\| medio | 0,056 | 0,054 | 0,059 | 0,054 |
| Caída de la correlación (D-41) | 90,2 % | 90,6 % | 90,1 % | 90,4 % |
| **Autovalor mínimo** | 0,002 | 0,024 | **0,000** | 0,040 |
| Suma de residuos (`sum_ratio`) | 0,07 | 0,21 | 0,24 | 0,32 |
| Mercado que queda en el residuo | 0,092 | 0,086 | 0,088 | 0,107 |
| Beta media | 1,010 | 0,970 | 1,000 | 0,960 |
| Cambio de la matriz frente al estándar | — | 0,008 | — | 0,009 |

Por fold:
- **El PCA estándar es singular en los 12 folds** (autovalor mínimo 0 exacto: sus residuos cumplen `Σ bᵢeᵢ = 0`, D-36). `loo_pca` lo arregla en los 12: entre 0,003 (2025T4) y 0,26 (2024T2).
- **La media estándar está casi en 0** (de 0,0005 a 0,012; no es exacto porque train tiene huecos, D-35). `loo_mean` la supera en 9 de 12 folds, pero **no en 2025T1, 2025T2 y 2025T4**, donde sus betas son ≈ 0,99: con betas casi 1 el LOO vuelve a la dependencia, como predice §2.
- El ρ medio con signo sigue cerca del suelo en todos los folds; `loo_pca` es el que más se aleja (hasta −0,005 en 2024T3), en parte porque deja más mercado en el residuo.
- La caída de la correlación es la misma con y sin LOO (diferencias < 1 punto).

---

## 5. Qué significa para la Épica 4

- **Grafo par a par (061–066):** el LOO **no cambia nada relevante**. Las correlaciones entre pares se mueven 0,008 de media, una séptima parte del |ρ| típico. Se puede seguir con D-35 o D-36.
- **Métodos que usan todas las monedas a la vez (VAR, condicionar sobre todas en PCMCI):** con D-35/D-36 la matriz es singular o casi. `loo_pca` es la variante que la hace invertible en los 12 folds, pero con un autovalor mínimo pequeño (mediana 0,04): sigue haciendo falta regularizar, o quitar una moneda del sistema. `loo_mean` no es fiable para esto (2025).
- **La correlación negativa −1/(N−1) no es un defecto del factor elegido**, sino de restar un mercado estimado con las mismas monedas. Al interpretar aristas negativas pequeñas hay que compararlas con ese suelo (D-35, D-36), con o sin LOO.
- **El coste del LOO** es dejar algo más de mercado en el residuo (beta media 0,96–0,97; con `loo_pca` la exposición sube de 0,088 a 0,107). Para lead-lag es justo lo que no se quiere: «todo sigue al mercado» (R2 de D-39).

---

## 6. Limitaciones

- Solo 1 minuto y solo la beta por fold de 12 meses (D-43).
- `min_eigenvalue` y `sum_ratio` usan los minutos con todas las monedas: de 21 000 (2023T3) a 121 000 (2024T4) por fold.
- No se ha repetido el contraste de señal de MIAX-057 con el LOO; `study_signal` admite otros factores si hiciera falta.
- El retraso respecto al mercado se trata en MIAX-216.

---

## Relacionado

- D-35 y D-36: los factores media y PCA, y su dependencia lineal.
- D-39 (c): este control. D-41: la métrica de caída. D-37: los folds.
- MIAX-216: residuo sin el retraso respecto al mercado.
