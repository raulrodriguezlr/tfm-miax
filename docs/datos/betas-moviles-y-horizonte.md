# Betas móviles y betas por horizonte

Decisión sobre dos dudas abiertas de la residualización (MIAX-053, control (d) de D-39 y riesgo R4): si la beta de cada moneda debe ser **fija por fold** o **móvil**, y si conviene estimarla con retornos de **1 minuto** o de **5 y 15 minutos**. Es un estudio **descriptivo**: no cambia `BtcFactor`, `MeanFactor` ni `PcaFactor`. Si la decisión fuera adoptar otro esquema, la implementación es un ticket nuevo.

> **Estado:** medido sobre los 12 folds (2023T1 a 2025T4) con datos reales de Binance. **Decisión: se mantienen las betas por fold a 1 min** (ninguna alternativa cumple la regla en los tres factores). Las cifras de §4 están copiadas de `beta_study_*.csv`.

---

## 1. La pregunta

- **A. ¿Beta fija por fold o beta móvil?** Las tres definiciones de factor (D-34 a D-36) ajustan las betas una vez en train y las congelan (D-03). Las betas cripto cambian entre regímenes, así que una beta fija de 12 meses puede ser rígida.
- **B. ¿Estimar la beta a 1 minuto o a 5 y 15 minutos?** Con negociación asíncrona (efecto Epps), la beta a 1 min de una moneda poco líquida sale sesgada a la baja. El residuo conserva entonces parte del mercado y el lead-lag que se vea después puede ser solo ese resto (D-39 R1 y R2).

---

## 2. Regla de decisión (fijada antes de ver los datos)

Se **mantienen las betas por fold a 1 min** (la opción más simple y la ya implementada) **salvo** que una alternativa mejore la métrica primaria en **al menos 9 de 12 folds para cada uno de los tres factores** (BTC, media y PCA).

| Pregunta | Esquemas alternativos | Métrica primaria (gana un fold si...) |
|---|---|---|
| A | beta móvil de 30 y de 90 días | quita **más** varianza (`var_removed`) **y** deja **menos** mercado instantáneo (`corr_lag0`), las dos a la vez |
| B | beta ajustada a 5 y a 15 min | `\|corr_lag1\|` **menor** que con la beta a 1 min |

Para B se reporta además si el cociente `β_5m/β_1m` crece con la iliquidez (Spearman positivo), pero **no es parte de la regla**. La **magnitud** de la mejora se reporta aparte y es de criterio estadístico (se comenta con el tutor). Un fold con la métrica en `NaN` no cuenta como victoria. El umbral es el parámetro `min_wins` (9).

- **El umbral es absoluto:** `min_wins=9` no se reescala si se piden menos de 12 folds (`--tests`). Con 4 folds la regla nunca pasa; es conservadora, no una inconsistencia.
- **La regla B puede dar victoria sin mejorar nada más:** gana un fold con solo bajar `|corr_lag1|`, aunque `var_removed` y `corr_lag0` no mejoren. Por eso la magnitud y el resto de métricas se miran aparte.

---

## 3. Método

- **Folds:** los 12 de D-37, test trimestral de 2023T1 a 2025T4, entrenando con los 12 meses anteriores (ventana deslizante) y las monedas de la lista del trimestre de prueba; una moneda con menos del 50 % de datos válidos en train queda fuera del fold. **2026 no se toca.**
- **`fold` (referencia, igual que `h1m`):** `fit(retornos de 1 min de train).transform(retornos de 1 min de test)`.
- **`roll30` y `roll90`:** cada 7 días, un modelo nuevo `fit(30 o 90 días anteriores).transform(bloque de 7 días)`. La ventana acaba justo antes del bloque (índice estrictamente menor que su primer minuto), así que lo que se proyecta nunca entra en el ajuste. Las primeras ventanas de cada fold toman datos del tramo de train, que es anterior al test. En cada ventana quedan fuera las monedas con menos del 50 % de datos válidos.
- **`h5m` y `h15m`:** `fit(retornos de 5 o 15 min de train).transform(retornos de 1 min de test)`. Los retornos de 5 y 15 min salen de `horizon_returns_from_cache` (barras de 1 min agregadas con la regla de D-23); cierran antes de que empiece el test (D-19). Los tres horizontes se proyectan sobre exactamente los mismos minutos de test.
- **Métricas** (solo en test, por moneda, resumidas con la mediana entre monedas): `var_removed = 1 − Σe²/Σr²`; `corr_lag0 = |corr(e_i(t), f(t))|`; `corr_lag1 = corr(e_i(t), f(t−1))` con signo y en valor absoluto. El residuo del propio factor BTC queda vacío (D-34). Con menos de `min_obs = 1000` observaciones comunes, la métrica de la moneda es `NaN`.
- **Sesgo por iliquidez** (solo betas OLS: BTC y media; las cargas del PCA van normalizadas a media 1 y su cociente no vale): `β_5m/β_1m` y `β_15m/β_1m` por moneda frente a su iliquidez en train (`1 − fracción de minutos válidos`), con la correlación de Spearman entre monedas y la mediana del cociente por tercil de liquidez. Solo cuentan las monedas con `β_1m > 0`.
- **Fallos de ajuste:** un ajuste que falla (PCA sin pares con solape, bloque sin datos previos) deja el residuo en `NaN`, nunca en 0 (D-02), y se cuenta en `n_failed` de la tabla; una beta `NaN` se ve porque baja `n_coins`.

---

## 4. Lo medido

Ejecución completa de los 12 folds (`python -m miax.eval beta-study`): sale con código 0 y sin ajustes fallidos (`n_failed = 0` en todas las filas: ningún bloque móvil, ningún PCA y ningún ajuste a 5 o 15 min falló). Los folds tienen 38, 40, 40, 35 y ocho veces 40 monedas en el universo; con la regla del 50 % de datos válidos en train se quedan 24, 23, 20, 24, 27, 23, 17, 22, 24, 24, 24 y 32 monedas por fold (de 2023T1 a 2025T4), y `n_coins` baja alguna más en varios folds (por ejemplo 2023T2 y 2024T2 con el factor media), sin que se haya desglosado la causa (el CSV solo cuenta monedas con métrica; el factor BTC además no cuenta a BTC). Tiempo: **1262 s (21 min) la primera vez**, construyendo los paneles y mientras se hacían otras pruebas en la misma máquina, y **621 s y 678 s (10 a 11 min) con los paneles en caché**; un fold con caché, 62 s. Memoria máxima del proceso, 3,3 GB. Los paneles de los 12 folds ocupan en `data/processed/panel/1m/` unos 5,4 GB más (de 0,7 a 6,1 GB de `data/processed`).

### 4.1. Diagnóstico por fold y esquema (`beta_study_diagnostics.csv`)

Mediana entre monedas, solo en test. `fold` es la referencia (beta de train a 1 min); `roll30` y `roll90` son betas móviles; `h5m` y `h15m` son betas de 5 y 15 min.
**Factor BTC**

| Fold | Esquema | n_coins | var_removed | corr_lag0 | corr_lag1 | \|corr_lag1\| | n_failed |
|---|---|---:|---:|---:|---:|---:|---:|
| 2023-01-01 | fold | 23 | 0.3798 | 0.1001 | 0.0378 | 0.0378 | 0 |
| 2023-01-01 | roll30 | 23 | 0.3844 | 0.0582 | 0.0391 | 0.0391 | 0 |
| 2023-01-01 | roll90 | 23 | 0.3662 | 0.0717 | 0.0379 | 0.0379 | 0 |
| 2023-01-01 | h5m | 23 | 0.3567 | 0.1383 | 0.0374 | 0.0374 | 0 |
| 2023-01-01 | h15m | 23 | 0.3553 | 0.1535 | 0.0367 | 0.0367 | 0 |
| 2023-04-01 | fold | 21 | 0.3031 | 0.0922 | 0.0444 | 0.0444 | 0 |
| 2023-04-01 | roll30 | 21 | 0.3259 | 0.0321 | 0.0493 | 0.0493 | 0 |
| 2023-04-01 | roll90 | 21 | 0.3268 | 0.0294 | 0.0482 | 0.0482 | 0 |
| 2023-04-01 | h5m | 21 | 0.2909 | 0.1450 | 0.0429 | 0.0429 | 0 |
| 2023-04-01 | h15m | 21 | 0.2812 | 0.1629 | 0.0423 | 0.0423 | 0 |
| 2023-07-01 | fold | 19 | 0.2983 | 0.0416 | 0.0265 | 0.0265 | 0 |
| 2023-07-01 | roll30 | 19 | 0.2803 | 0.0395 | 0.0189 | 0.0189 | 0 |
| 2023-07-01 | roll90 | 19 | 0.2921 | 0.0231 | 0.0251 | 0.0251 | 0 |
| 2023-07-01 | h5m | 19 | 0.2962 | 0.0520 | 0.0263 | 0.0263 | 0 |
| 2023-07-01 | h15m | 19 | 0.2961 | 0.0704 | 0.0259 | 0.0259 | 0 |
| 2023-10-01 | fold | 23 | 0.2726 | 0.0639 | 0.0072 | 0.0095 | 0 |
| 2023-10-01 | roll30 | 23 | 0.2653 | 0.0427 | 0.0098 | 0.0107 | 0 |
| 2023-10-01 | roll90 | 23 | 0.2711 | 0.0633 | 0.0101 | 0.0118 | 0 |
| 2023-10-01 | h5m | 23 | 0.2717 | 0.0864 | 0.0051 | 0.0090 | 0 |
| 2023-10-01 | h15m | 23 | 0.2691 | 0.0880 | 0.0046 | 0.0088 | 0 |
| 2024-01-01 | fold | 26 | 0.3531 | 0.0639 | 0.0574 | 0.0574 | 0 |
| 2024-01-01 | roll30 | 26 | 0.3508 | 0.0291 | 0.0561 | 0.0561 | 0 |
| 2024-01-01 | roll90 | 26 | 0.3509 | 0.0444 | 0.0560 | 0.0560 | 0 |
| 2024-01-01 | h5m | 26 | 0.3470 | 0.1025 | 0.0582 | 0.0582 | 0 |
| 2024-01-01 | h15m | 26 | 0.3406 | 0.1136 | 0.0584 | 0.0584 | 0 |
| 2024-04-01 | fold | 19 | 0.4212 | 0.0859 | 0.0266 | 0.0266 | 0 |
| 2024-04-01 | roll30 | 19 | 0.4368 | 0.0248 | 0.0264 | 0.0264 | 0 |
| 2024-04-01 | roll90 | 19 | 0.4371 | 0.0644 | 0.0260 | 0.0260 | 0 |
| 2024-04-01 | h5m | 19 | 0.4311 | 0.0679 | 0.0273 | 0.0273 | 0 |
| 2024-04-01 | h15m | 19 | 0.4338 | 0.0647 | 0.0275 | 0.0275 | 0 |
| 2024-07-01 | fold | 16 | 0.4680 | 0.1073 | 0.0307 | 0.0307 | 0 |
| 2024-07-01 | roll30 | 16 | 0.4759 | 0.0285 | 0.0295 | 0.0295 | 0 |
| 2024-07-01 | roll90 | 16 | 0.4750 | 0.0563 | 0.0302 | 0.0302 | 0 |
| 2024-07-01 | h5m | 16 | 0.4576 | 0.1626 | 0.0312 | 0.0312 | 0 |
| 2024-07-01 | h15m | 16 | 0.4451 | 0.1901 | 0.0316 | 0.0316 | 0 |
| 2024-10-01 | fold | 21 | 0.3588 | 0.1192 | 0.0250 | 0.0250 | 0 |
| 2024-10-01 | roll30 | 21 | 0.3604 | 0.0295 | 0.0253 | 0.0253 | 0 |
| 2024-10-01 | roll90 | 21 | 0.3587 | 0.0861 | 0.0258 | 0.0258 | 0 |
| 2024-10-01 | h5m | 21 | 0.3616 | 0.0965 | 0.0250 | 0.0250 | 0 |
| 2024-10-01 | h15m | 21 | 0.3620 | 0.0899 | 0.0250 | 0.0250 | 0 |
| 2025-01-01 | fold | 23 | 0.4093 | 0.0852 | 0.0277 | 0.0277 | 0 |
| 2025-01-01 | roll30 | 23 | 0.4101 | 0.0790 | 0.0260 | 0.0260 | 0 |
| 2025-01-01 | roll90 | 23 | 0.4262 | 0.0634 | 0.0280 | 0.0280 | 0 |
| 2025-01-01 | h5m | 23 | 0.4100 | 0.0573 | 0.0277 | 0.0277 | 0 |
| 2025-01-01 | h15m | 23 | 0.4099 | 0.0508 | 0.0277 | 0.0277 | 0 |
| 2025-04-01 | fold | 23 | 0.4377 | 0.1491 | 0.0208 | 0.0208 | 0 |
| 2025-04-01 | roll30 | 23 | 0.4503 | 0.0400 | 0.0216 | 0.0216 | 0 |
| 2025-04-01 | roll90 | 23 | 0.4361 | 0.0895 | 0.0214 | 0.0214 | 0 |
| 2025-04-01 | h5m | 23 | 0.4352 | 0.1164 | 0.0213 | 0.0213 | 0 |
| 2025-04-01 | h15m | 23 | 0.4354 | 0.1131 | 0.0215 | 0.0215 | 0 |
| 2025-07-01 | fold | 23 | 0.3578 | 0.1821 | 0.0075 | 0.0119 | 0 |
| 2025-07-01 | roll30 | 23 | 0.3726 | 0.0101 | 0.0056 | 0.0124 | 0 |
| 2025-07-01 | roll90 | 23 | 0.3751 | 0.0291 | 0.0064 | 0.0128 | 0 |
| 2025-07-01 | h5m | 23 | 0.3630 | 0.1639 | 0.0074 | 0.0120 | 0 |
| 2025-07-01 | h15m | 23 | 0.3651 | 0.1565 | 0.0074 | 0.0121 | 0 |
| 2025-10-01 | fold | 31 | 0.2371 | 0.0701 | 0.1088 | 0.1088 | 0 |
| 2025-10-01 | roll30 | 31 | 0.2198 | 0.0235 | 0.1088 | 0.1088 | 0 |
| 2025-10-01 | roll90 | 31 | 0.2252 | 0.0799 | 0.1090 | 0.1090 | 0 |
| 2025-10-01 | h5m | 31 | 0.2400 | 0.0631 | 0.1091 | 0.1091 | 0 |
| 2025-10-01 | h15m | 31 | 0.2425 | 0.0585 | 0.1091 | 0.1091 | 0 |

**Factor media**

| Fold | Esquema | n_coins | var_removed | corr_lag0 | corr_lag1 | \|corr_lag1\| | n_failed |
|---|---|---:|---:|---:|---:|---:|---:|
| 2023-01-01 | fold | 24 | 0.5821 | 0.0935 | -0.0050 | 0.0265 | 0 |
| 2023-01-01 | roll30 | 24 | 0.5810 | 0.0328 | -0.0123 | 0.0253 | 0 |
| 2023-01-01 | roll90 | 24 | 0.5804 | 0.0802 | -0.0062 | 0.0254 | 0 |
| 2023-01-01 | h5m | 24 | 0.5807 | 0.0872 | -0.0056 | 0.0264 | 0 |
| 2023-01-01 | h15m | 24 | 0.5799 | 0.0825 | -0.0062 | 0.0259 | 0 |
| 2023-04-01 | fold | 22 | 0.5818 | 0.0866 | 0.0014 | 0.0313 | 0 |
| 2023-04-01 | roll30 | 22 | 0.5767 | 0.0419 | -0.0068 | 0.0300 | 0 |
| 2023-04-01 | roll90 | 22 | 0.5784 | 0.0378 | -0.0090 | 0.0309 | 0 |
| 2023-04-01 | h5m | 22 | 0.5815 | 0.0878 | 0.0014 | 0.0319 | 0 |
| 2023-04-01 | h15m | 22 | 0.5809 | 0.0932 | 0.0014 | 0.0328 | 0 |
| 2023-07-01 | fold | 20 | 0.5190 | 0.0616 | 0.0023 | 0.0177 | 0 |
| 2023-07-01 | roll30 | 20 | 0.5127 | 0.0274 | -0.0006 | 0.0256 | 0 |
| 2023-07-01 | roll90 | 20 | 0.5176 | 0.0504 | 0.0036 | 0.0158 | 0 |
| 2023-07-01 | h5m | 20 | 0.5195 | 0.0720 | 0.0051 | 0.0178 | 0 |
| 2023-07-01 | h15m | 20 | 0.5195 | 0.0632 | 0.0062 | 0.0171 | 0 |
| 2023-10-01 | fold | 24 | 0.5325 | 0.0938 | -0.0015 | 0.0084 | 0 |
| 2023-10-01 | roll30 | 24 | 0.5339 | 0.0376 | -0.0007 | 0.0081 | 0 |
| 2023-10-01 | roll90 | 24 | 0.5354 | 0.0472 | -0.0017 | 0.0069 | 0 |
| 2023-10-01 | h5m | 24 | 0.5359 | 0.0944 | -0.0017 | 0.0085 | 0 |
| 2023-10-01 | h15m | 24 | 0.5370 | 0.1008 | -0.0015 | 0.0085 | 0 |
| 2024-01-01 | fold | 27 | 0.5735 | 0.0922 | 0.0026 | 0.0222 | 0 |
| 2024-01-01 | roll30 | 27 | 0.5790 | 0.0414 | 0.0049 | 0.0210 | 0 |
| 2024-01-01 | roll90 | 27 | 0.5784 | 0.0525 | 0.0041 | 0.0219 | 0 |
| 2024-01-01 | h5m | 27 | 0.5749 | 0.0990 | 0.0015 | 0.0222 | 0 |
| 2024-01-01 | h15m | 27 | 0.5742 | 0.1056 | 0.0014 | 0.0217 | 0 |
| 2024-04-01 | fold | 20 | 0.7063 | 0.1111 | -0.0038 | 0.0325 | 0 |
| 2024-04-01 | roll30 | 20 | 0.7075 | 0.0641 | -0.0052 | 0.0345 | 0 |
| 2024-04-01 | roll90 | 20 | 0.7105 | 0.0712 | -0.0057 | 0.0350 | 0 |
| 2024-04-01 | h5m | 20 | 0.7053 | 0.0908 | -0.0033 | 0.0320 | 0 |
| 2024-04-01 | h15m | 20 | 0.7060 | 0.1091 | -0.0038 | 0.0323 | 0 |
| 2024-07-01 | fold | 17 | 0.6949 | 0.0617 | 0.0002 | 0.0138 | 0 |
| 2024-07-01 | roll30 | 17 | 0.7033 | 0.0338 | 0.0009 | 0.0138 | 0 |
| 2024-07-01 | roll90 | 17 | 0.6998 | 0.0744 | 0.0008 | 0.0138 | 0 |
| 2024-07-01 | h5m | 17 | 0.6956 | 0.0943 | 0.0002 | 0.0139 | 0 |
| 2024-07-01 | h15m | 17 | 0.6918 | 0.0741 | 0.0001 | 0.0136 | 0 |
| 2024-10-01 | fold | 22 | 0.6242 | 0.1197 | -0.0006 | 0.0100 | 0 |
| 2024-10-01 | roll30 | 22 | 0.6442 | 0.0610 | 0.0006 | 0.0113 | 0 |
| 2024-10-01 | roll90 | 22 | 0.6380 | 0.0951 | 0.0003 | 0.0110 | 0 |
| 2024-10-01 | h5m | 22 | 0.6215 | 0.1282 | -0.0014 | 0.0104 | 0 |
| 2024-10-01 | h15m | 22 | 0.6217 | 0.1212 | -0.0017 | 0.0095 | 0 |
| 2025-01-01 | fold | 24 | 0.6966 | 0.1364 | -0.0038 | 0.0134 | 0 |
| 2025-01-01 | roll30 | 24 | 0.7083 | 0.0312 | -0.0016 | 0.0148 | 0 |
| 2025-01-01 | roll90 | 24 | 0.7064 | 0.0455 | -0.0027 | 0.0147 | 0 |
| 2025-01-01 | h5m | 24 | 0.6976 | 0.1185 | -0.0040 | 0.0131 | 0 |
| 2025-01-01 | h15m | 24 | 0.6982 | 0.1029 | -0.0037 | 0.0127 | 0 |
| 2025-04-01 | fold | 24 | 0.6751 | 0.0708 | -0.0078 | 0.0237 | 0 |
| 2025-04-01 | roll30 | 24 | 0.6816 | 0.0272 | -0.0081 | 0.0239 | 0 |
| 2025-04-01 | roll90 | 24 | 0.6707 | 0.0745 | -0.0083 | 0.0250 | 0 |
| 2025-04-01 | h5m | 24 | 0.6752 | 0.0640 | -0.0077 | 0.0235 | 0 |
| 2025-04-01 | h15m | 24 | 0.6761 | 0.0625 | -0.0078 | 0.0235 | 0 |
| 2025-07-01 | fold | 24 | 0.6481 | 0.0929 | -0.0061 | 0.0267 | 0 |
| 2025-07-01 | roll30 | 24 | 0.6532 | 0.0269 | -0.0014 | 0.0260 | 0 |
| 2025-07-01 | roll90 | 24 | 0.6528 | 0.0569 | -0.0023 | 0.0262 | 0 |
| 2025-07-01 | h5m | 24 | 0.6487 | 0.0900 | -0.0056 | 0.0266 | 0 |
| 2025-07-01 | h15m | 24 | 0.6487 | 0.0888 | -0.0056 | 0.0268 | 0 |
| 2025-10-01 | fold | 32 | 0.5840 | 0.1896 | 0.0134 | 0.1503 | 0 |
| 2025-10-01 | roll30 | 32 | 0.5926 | 0.1649 | 0.0024 | 0.1451 | 0 |
| 2025-10-01 | roll90 | 32 | 0.5898 | 0.1022 | -0.0057 | 0.1259 | 0 |
| 2025-10-01 | h5m | 32 | 0.5855 | 0.1801 | 0.0096 | 0.1452 | 0 |
| 2025-10-01 | h15m | 32 | 0.5879 | 0.1778 | 0.0100 | 0.1444 | 0 |

**Factor PCA**

| Fold | Esquema | n_coins | var_removed | corr_lag0 | corr_lag1 | \|corr_lag1\| | n_failed |
|---|---|---:|---:|---:|---:|---:|---:|
| 2023-01-01 | fold | 24 | 0.5825 | 0.1034 | -0.0063 | 0.0263 | 0 |
| 2023-01-01 | roll30 | 24 | 0.5862 | 0.0293 | -0.0104 | 0.0239 | 0 |
| 2023-01-01 | roll90 | 24 | 0.5881 | 0.0773 | -0.0044 | 0.0237 | 0 |
| 2023-01-01 | h5m | 24 | 0.5796 | 0.0927 | -0.0067 | 0.0243 | 0 |
| 2023-01-01 | h15m | 24 | 0.5777 | 0.0826 | -0.0073 | 0.0238 | 0 |
| 2023-04-01 | fold | 22 | 0.5702 | 0.0939 | -0.0028 | 0.0328 | 0 |
| 2023-04-01 | roll30 | 22 | 0.5656 | 0.0595 | -0.0088 | 0.0344 | 0 |
| 2023-04-01 | roll90 | 22 | 0.5671 | 0.0496 | -0.0103 | 0.0323 | 0 |
| 2023-04-01 | h5m | 22 | 0.5741 | 0.0826 | -0.0033 | 0.0328 | 0 |
| 2023-04-01 | h15m | 22 | 0.5734 | 0.0912 | -0.0037 | 0.0326 | 0 |
| 2023-07-01 | fold | 20 | 0.5059 | 0.0572 | 0.0041 | 0.0208 | 0 |
| 2023-07-01 | roll30 | 20 | 0.5133 | 0.0389 | 0.0046 | 0.0203 | 0 |
| 2023-07-01 | roll90 | 20 | 0.5180 | 0.0526 | 0.0041 | 0.0181 | 0 |
| 2023-07-01 | h5m | 20 | 0.5222 | 0.0635 | 0.0083 | 0.0198 | 0 |
| 2023-07-01 | h15m | 20 | 0.5205 | 0.0502 | 0.0080 | 0.0203 | 0 |
| 2023-10-01 | fold | 24 | 0.5308 | 0.0927 | -0.0010 | 0.0101 | 0 |
| 2023-10-01 | roll30 | 24 | 0.5324 | 0.0348 | 0.0009 | 0.0089 | 0 |
| 2023-10-01 | roll90 | 24 | 0.5361 | 0.0550 | -0.0012 | 0.0070 | 0 |
| 2023-10-01 | h5m | 24 | 0.5343 | 0.1135 | -0.0016 | 0.0103 | 0 |
| 2023-10-01 | h15m | 24 | 0.5340 | 0.1044 | -0.0018 | 0.0100 | 0 |
| 2024-01-01 | fold | 27 | 0.5776 | 0.0820 | 0.0031 | 0.0214 | 0 |
| 2024-01-01 | roll30 | 27 | 0.5813 | 0.0288 | 0.0061 | 0.0201 | 0 |
| 2024-01-01 | roll90 | 27 | 0.5847 | 0.0492 | 0.0045 | 0.0210 | 0 |
| 2024-01-01 | h5m | 27 | 0.5782 | 0.1053 | 0.0046 | 0.0212 | 0 |
| 2024-01-01 | h15m | 27 | 0.5777 | 0.0964 | 0.0046 | 0.0206 | 0 |
| 2024-04-01 | fold | 20 | 0.7077 | 0.0872 | -0.0020 | 0.0344 | 0 |
| 2024-04-01 | roll30 | 20 | 0.7100 | 0.0637 | -0.0041 | 0.0357 | 0 |
| 2024-04-01 | roll90 | 20 | 0.7093 | 0.0791 | -0.0041 | 0.0365 | 0 |
| 2024-04-01 | h5m | 20 | 0.7066 | 0.1011 | -0.0018 | 0.0337 | 0 |
| 2024-04-01 | h15m | 20 | 0.7078 | 0.1041 | -0.0027 | 0.0335 | 0 |
| 2024-07-01 | fold | 17 | 0.6941 | 0.0444 | -0.0004 | 0.0129 | 0 |
| 2024-07-01 | roll30 | 17 | 0.6944 | 0.0281 | 0.0012 | 0.0133 | 0 |
| 2024-07-01 | roll90 | 17 | 0.6944 | 0.0502 | 0.0003 | 0.0129 | 0 |
| 2024-07-01 | h5m | 17 | 0.6929 | 0.0600 | -0.0002 | 0.0127 | 0 |
| 2024-07-01 | h15m | 17 | 0.6930 | 0.0371 | -0.0006 | 0.0125 | 0 |
| 2024-10-01 | fold | 22 | 0.6228 | 0.1150 | 0.0007 | 0.0101 | 0 |
| 2024-10-01 | roll30 | 22 | 0.6479 | 0.0531 | 0.0018 | 0.0130 | 0 |
| 2024-10-01 | roll90 | 22 | 0.6412 | 0.0855 | 0.0011 | 0.0106 | 0 |
| 2024-10-01 | h5m | 22 | 0.6220 | 0.1194 | 0.0003 | 0.0101 | 0 |
| 2024-10-01 | h15m | 22 | 0.6219 | 0.1050 | -0.0003 | 0.0098 | 0 |
| 2025-01-01 | fold | 24 | 0.7035 | 0.1151 | -0.0043 | 0.0124 | 0 |
| 2025-01-01 | roll30 | 24 | 0.7156 | 0.0296 | -0.0021 | 0.0148 | 0 |
| 2025-01-01 | roll90 | 24 | 0.7134 | 0.0512 | -0.0036 | 0.0144 | 0 |
| 2025-01-01 | h5m | 24 | 0.7058 | 0.0958 | -0.0043 | 0.0122 | 0 |
| 2025-01-01 | h15m | 24 | 0.7067 | 0.0873 | -0.0042 | 0.0117 | 0 |
| 2025-04-01 | fold | 24 | 0.6699 | 0.0827 | -0.0097 | 0.0226 | 0 |
| 2025-04-01 | roll30 | 24 | 0.6791 | 0.0464 | -0.0087 | 0.0232 | 0 |
| 2025-04-01 | roll90 | 24 | 0.6642 | 0.0802 | -0.0086 | 0.0245 | 0 |
| 2025-04-01 | h5m | 24 | 0.6695 | 0.0750 | -0.0098 | 0.0223 | 0 |
| 2025-04-01 | h15m | 24 | 0.6714 | 0.0732 | -0.0100 | 0.0224 | 0 |
| 2025-07-01 | fold | 24 | 0.6494 | 0.0953 | -0.0072 | 0.0268 | 0 |
| 2025-07-01 | roll30 | 24 | 0.6475 | 0.0265 | -0.0010 | 0.0285 | 0 |
| 2025-07-01 | roll90 | 24 | 0.6461 | 0.0593 | -0.0020 | 0.0286 | 0 |
| 2025-07-01 | h5m | 24 | 0.6496 | 0.0901 | -0.0070 | 0.0266 | 0 |
| 2025-07-01 | h15m | 24 | 0.6499 | 0.0889 | -0.0071 | 0.0267 | 0 |
| 2025-10-01 | fold | 32 | 0.5867 | 0.2020 | -0.0061 | 0.1438 | 0 |
| 2025-10-01 | roll30 | 32 | 0.6069 | 0.1586 | -0.0179 | 0.1380 | 0 |
| 2025-10-01 | roll90 | 32 | 0.5946 | 0.1270 | -0.0177 | 0.1267 | 0 |
| 2025-10-01 | h5m | 32 | 0.5851 | 0.1814 | -0.0070 | 0.1356 | 0 |
| 2025-10-01 | h15m | 32 | 0.5861 | 0.1781 | -0.0092 | 0.1356 | 0 |

### 4.2. Sesgo por iliquidez (`beta_study_bias.csv`)

Cociente `β_h/β_1m` por moneda; «Spearman» es la correlación entre el cociente y la iliquidez en train; las columnas de tercil son la mediana del cociente en cada tercil de liquidez. Solo BTC y media. Con 5m y 15m el cociente de las monedas más líquidas es ya distinto de 1 con el factor BTC.

| Fold | Factor | Horizonte | n_coins | Spearman | Mediana líquido | Mediana medio | Mediana ilíquido | Mediana total |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 2023-01-01 | BTC | 5m | 23 | 0.6548 | 1.0484 | 1.0449 | 1.0630 | 1.0521 |
| 2023-01-01 | BTC | 15m | 23 | 0.7319 | 1.0568 | 1.0652 | 1.0914 | 1.0688 |
| 2023-01-01 | media | 5m | 24 | 0.3481 | 1.0018 | 0.9967 | 1.0041 | 1.0016 |
| 2023-01-01 | media | 15m | 24 | 0.6719 | 0.9885 | 0.9958 | 1.0109 | 0.9988 |
| 2023-04-01 | BTC | 5m | 22 | 0.6591 | 1.0504 | 1.0542 | 1.0775 | 1.0563 |
| 2023-04-01 | BTC | 15m | 22 | 0.7433 | 1.0664 | 1.0832 | 1.1282 | 1.0834 |
| 2023-04-01 | media | 5m | 23 | 0.4857 | 0.9962 | 1.0003 | 1.0060 | 1.0023 |
| 2023-04-01 | media | 15m | 23 | 0.7240 | 0.9819 | 0.9955 | 1.0171 | 0.9997 |
| 2023-07-01 | BTC | 5m | 19 | 0.7986 | 1.0616 | 1.0760 | 1.1175 | 1.0799 |
| 2023-07-01 | BTC | 15m | 19 | 0.7047 | 1.0738 | 1.0939 | 1.1561 | 1.0905 |
| 2023-07-01 | media | 5m | 20 | 0.5437 | 0.9880 | 0.9988 | 1.0115 | 0.9988 |
| 2023-07-01 | media | 15m | 20 | 0.7244 | 0.9867 | 1.0007 | 1.0145 | 0.9974 |
| 2023-10-01 | BTC | 5m | 23 | 0.4438 | 1.0819 | 1.1083 | 1.1161 | 1.0919 |
| 2023-10-01 | BTC | 15m | 23 | 0.4013 | 1.0921 | 1.1307 | 1.1462 | 1.1189 |
| 2023-10-01 | media | 5m | 24 | 0.3412 | 0.9912 | 1.0020 | 0.9989 | 0.9989 |
| 2023-10-01 | media | 15m | 24 | 0.4108 | 0.9824 | 1.0058 | 1.0013 | 0.9988 |
| 2024-01-01 | BTC | 5m | 26 | 0.6018 | 1.0511 | 1.0633 | 1.1156 | 1.0674 |
| 2024-01-01 | BTC | 15m | 26 | 0.5943 | 1.0745 | 1.0814 | 1.1493 | 1.0960 |
| 2024-01-01 | media | 5m | 27 | 0.3965 | 0.9905 | 0.9918 | 1.0242 | 0.9976 |
| 2024-01-01 | media | 15m | 27 | 0.5271 | 0.9880 | 0.9905 | 1.0232 | 0.9983 |
| 2024-04-01 | BTC | 5m | 22 | 0.5082 | 1.0464 | 1.0466 | 1.1076 | 1.0515 |
| 2024-04-01 | BTC | 15m | 22 | 0.5953 | 1.0793 | 1.0829 | 1.1263 | 1.0910 |
| 2024-04-01 | media | 5m | 23 | 0.4225 | 0.9937 | 0.9980 | 1.0210 | 0.9985 |
| 2024-04-01 | media | 15m | 23 | 0.4581 | 0.9846 | 1.0001 | 1.0252 | 1.0001 |
| 2024-07-01 | BTC | 5m | 16 | -0.1711 | 1.0571 | 1.0457 | 1.0380 | 1.0441 |
| 2024-07-01 | BTC | 15m | 16 | -0.1239 | 1.0829 | 1.0748 | 1.0735 | 1.0773 |
| 2024-07-01 | media | 5m | 17 | -0.4772 | 1.0084 | 1.0092 | 0.9918 | 1.0069 |
| 2024-07-01 | media | 15m | 17 | -0.1418 | 1.0171 | 1.0073 | 0.9918 | 1.0049 |
| 2024-10-01 | BTC | 5m | 21 | 0.3375 | 1.0434 | 1.0379 | 1.0552 | 1.0400 |
| 2024-10-01 | BTC | 15m | 21 | 0.0365 | 1.0656 | 1.0731 | 1.0671 | 1.0673 |
| 2024-10-01 | media | 5m | 22 | -0.0051 | 1.0080 | 1.0012 | 1.0141 | 1.0080 |
| 2024-10-01 | media | 15m | 22 | 0.0085 | 0.9963 | 1.0250 | 1.0039 | 1.0110 |
| 2025-01-01 | BTC | 5m | 23 | 0.2701 | 1.0423 | 1.0492 | 1.0534 | 1.0492 |
| 2025-01-01 | BTC | 15m | 23 | 0.1390 | 1.0486 | 1.0639 | 1.0652 | 1.0630 |
| 2025-01-01 | media | 5m | 24 | 0.2283 | 1.0000 | 1.0041 | 1.0136 | 1.0058 |
| 2025-01-01 | media | 15m | 24 | 0.2730 | 0.9915 | 1.0090 | 1.0219 | 1.0080 |
| 2025-04-01 | BTC | 5m | 23 | 0.4083 | 1.0337 | 1.0440 | 1.0452 | 1.0412 |
| 2025-04-01 | BTC | 15m | 23 | 0.4663 | 1.0395 | 1.0554 | 1.0625 | 1.0516 |
| 2025-04-01 | media | 5m | 24 | 0.4439 | 0.9917 | 1.0037 | 1.0042 | 0.9997 |
| 2025-04-01 | media | 15m | 24 | 0.6208 | 0.9853 | 0.9996 | 1.0089 | 0.9984 |
| 2025-07-01 | BTC | 5m | 23 | 0.4907 | 1.0282 | 1.0404 | 1.0459 | 1.0374 |
| 2025-07-01 | BTC | 15m | 23 | 0.4286 | 1.0408 | 1.0572 | 1.0590 | 1.0497 |
| 2025-07-01 | media | 5m | 24 | 0.5954 | 0.9888 | 0.9981 | 1.0063 | 0.9992 |
| 2025-07-01 | media | 15m | 24 | 0.6513 | 0.9871 | 0.9988 | 1.0066 | 0.9988 |
| 2025-10-01 | BTC | 5m | 31 | 0.3599 | 1.0306 | 1.0426 | 1.0397 | 1.0330 |
| 2025-10-01 | BTC | 15m | 31 | 0.6088 | 1.0316 | 1.0424 | 1.0619 | 1.0452 |
| 2025-10-01 | media | 5m | 32 | 0.5034 | 0.9880 | 1.0052 | 1.0047 | 0.9988 |
| 2025-10-01 | media | 15m | 32 | 0.4122 | 0.9859 | 1.0037 | 1.0053 | 0.9981 |

**Resumen de las 12 tablas** (mediana de las medianas por tercil entre folds): con el factor BTC, `β_5m/β_1m` = 1,0474 (líquido), 1,0461 (medio) y 1,0591 (ilíquido), y `β_15m/β_1m` = 1,0660, 1,0740 y 1,0824; con el factor media, `β_5m/β_1m` = 0,9927, 1,0008 y 1,0061, y `β_15m/β_1m` = 0,9869, 1,0004 y 1,0099. Spearman medio entre folds: BTC 0,4468 (5m) y 0,4438 (15m); media 0,3189 (5m) y 0,4450 (15m). El Spearman es positivo (el cociente crece con la iliquidez) en **11 de 12 folds** con BTC en 5m y 15m y con media en 15m, y en **10 de 12** con media en 5m. La mediana del cociente del tercil ilíquido es mayor que la del líquido en 11 de 12 folds en las cuatro combinaciones. Con el factor BTC el cociente es mayor que 1 en los tres terciles (la beta a 1 min queda por debajo de la de 5 y 15 min incluso en monedas líquidas); con el factor media es ≈ 1 y solo se separa de 1 en los extremos, en torno a ±1 %.

### 4.3. Folds ganados y regla (`beta_study_decision.csv`)

| Esquema | Factor | Pregunta | Folds | Folds ganados | Mínimo | ¿Pasa? | Folds con Spearman > 0 |
|---|---|---|---:|---:|---:|---|---:|
| roll30 | BTC | A | 12 | 8 | 9 | no |  |
| roll30 | media | A | 12 | 9 | 9 | sí |  |
| roll30 | PCA | A | 12 | 10 | 9 | sí |  |
| roll30 | todos | A | 12 | 8 | 9 | no |  |
| roll90 | BTC | A | 12 | 5 | 9 | no |  |
| roll90 | media | A | 12 | 7 | 9 | no |  |
| roll90 | PCA | A | 12 | 8 | 9 | no |  |
| roll90 | todos | A | 12 | 5 | 9 | no |  |
| h5m | BTC | B | 12 | 5 | 9 | no | 11 |
| h5m | media | B | 12 | 6 | 9 | no | 10 |
| h5m | PCA | B | 12 | 9 | 9 | sí |  |
| h5m | todos | B | 12 | 5 | 9 | no |  |
| h15m | BTC | B | 12 | 5 | 9 | no | 11 |
| h15m | media | B | 12 | 9 | 9 | sí | 11 |
| h15m | PCA | B | 12 | 12 | 9 | sí |  |
| h15m | todos | B | 12 | 5 | 9 | no |  |

### 4.4. Contraste independiente y reproducibilidad

- **Contraste con numpy puro del fold 2025T1** (cov/var y `np.linalg.lstsq`, sin pasar por los modelos ni por `beta_study`; retornos de 1 min de la caché del panel y de 5 y 15 min de `horizon_returns_from_cache`): `var_removed`, `corr_lag0`, `corr_lag1` y `|corr_lag1|` (mediana entre monedas) de `fold`, `h5m` y `h15m` para BTC y media coinciden con el CSV en las 10 cifras significativas con que se escribe; la beta de ETHUSDT por `lstsq` y por cov/var difiere en 7e-14 (BTC) y 3e-15 (media). La mediana del cociente (total y por tercil) coincide. ETHUSDT (BTC): `β_1m` = 0,98094237, `β_5m` = 0,98516893, `β_15m` = 0,98552571. 1000SATSUSDT no entra en ese fold (0 % de datos válidos en train); como moneda poco líquida se tomó APEUSDT (2,27 % de minutos no válidos, la más ilíquida del fold): `β_1m` = 1,2206682, `β_5m` = 1,313335 (cociente 1,0759149) y `β_15m` = 1,2976124 con el factor BTC.
- **Spearman con empates:** se calcula con rangos promediados (el Pearson de los rangos de iliquidez y de cociente, con los empates promediados), así que no depende del orden de las monedas; los terciles desempatan por iliquidez y, a igualdad, por símbolo alfabético. Contrastado con numpy y `scipy.stats.spearmanr` en 2025T1: BTC 5m 0,2701 y 15m 0,1390; media 5m 0,2283 y 15m 0,2730, iguales al CSV.
- **Regeneración:** con el Spearman de rangos promediados se volvió a ejecutar el estudio completo dos veces con los paneles en caché (621 s y 678 s) y los tres CSV salieron idénticos byte a byte (`md5`: `beta_study_bias.csv` 7005bec3…, `beta_study_decision.csv` a173b0d2…, `beta_study_diagnostics.csv` b5e73218…). `beta_study_diagnostics.csv` no cambió respecto a la ejecución con Spearman por rango `first`; en `beta_study_decision.csv` solo cambió el recuento de folds con Spearman > 0 de `h5m` con media (de 11 a 10). Ejecutar solo el fold 2025T1 da las mismas filas que ese fold dentro de la ejecución completa.
- **Magnitud.** Mediana entre folds de la diferencia con `fold`: `roll30` quita 0,0031 (BTC), 0,0053 (media) y 0,0037 (PCA) más de `var_removed` y baja `corr_lag0` en 0,053, 0,049 y 0,048; `roll90`, −0,0008, 0,0045 y 0,0054 de `var_removed` y −0,025, −0,038 y −0,031 de `corr_lag0`; `h5m` y `h15m` cambian `var_removed` y `|corr_lag1|` en 10⁻³ a 10⁻⁴ y `corr_lag0` en menos de 0,008. `corr_lag0` baja con `roll30` en los 12 folds para los tres factores, pero `var_removed` solo sube en 8, 9 y 10 folds; la regla exige las dos a la vez.

---

## 5. Decisión

Regla de la §2 aplicada tal cual a la tabla 4.3: una alternativa se adopta solo si gana en **≥ 9 de 12 folds para cada uno de los tres factores**.

- **A, beta móvil:** `roll30` gana en 8 folds con BTC, 9 con media y 10 con PCA; falla con BTC por un fold. `roll90`, 5, 7 y 8. **Ninguna pasa.**
- **B, beta a 5 y 15 min:** `h5m` gana en 5 (BTC), 6 (media) y 9 (PCA); `h15m`, en 5, 9 y 12. Con BTC no llega a 9 en ninguna. **Ninguna pasa.**

**Decisión: se mantienen las betas por fold a 1 min** (la opción más simple y la ya implementada). Es una decisión de la regla, no de la magnitud, y **`roll30` es un caso cercano que conviene comentar con el tutor**: con las betas móviles de 30 días `corr_lag0` baja en los 12 folds y en los tres factores (la mediana entre folds de la bajada es 0,053, 0,049 y 0,048 con BTC, media y PCA, frente a valores de `fold` de 0,04 a 0,20 según el fold), pero `var_removed` casi no cambia (+0,003 a +0,005) y solo sube en 8, 9 y 10 folds, de modo que con BTC se queda a un fold de la regla. La regla exige las dos mejoras a la vez, y `var_removed` sube en cantidades muy pequeñas que pueden ser ruido; el resultado de A no es «la beta móvil no ayuda», sino «no mejora lo bastante en las dos métricas a la vez». Las mejoras de `h5m`, `h15m` y `roll90` sí son pequeñas (§4.4). La parte secundaria de B sí sale: la beta a 1 min queda por debajo de la de 5 y 15 min cuando se usa el factor BTC y el cociente crece con la iliquidez en 10 a 11 de 12 folds según factor y horizonte (§4.2); eso no cambia la decisión, pero es el dato para el control de Epps (MIAX-070) y para comentarlo con el tutor.

---

## 6. Qué no se ha verificado

- Es **un solo periodo** (2023T1 a 2025T4), con 35 a 40 monedas en el universo de cada trimestre y entre 17 y 32 que pasan el filtro del 50 % de datos válidos en train (D-37). Ese filtro deja fuera entre el 20 % (8 de 40 monedas en 2025T4) y el 57,5 % (23 de 40 en 2024T3) del universo de cada fold, y en 8 de los 12 folds el 40 % o más; importa porque las monedas más ilíquidas, justo las que más sesgo de Epps deberían mostrar, son las que quedan fuera, y el estudio de sesgo solo ve las que se negocian lo bastante.
- Solo se han probado ventanas de **30 y 90 días** con reajuste cada 7 días; el PCA móvil es un reajuste semanal, no un PCA recursivo.
- **No se mide aún el efecto sobre el grafo** de lead-lag: solo el residuo.
- La regla de ≥ 9 de 12 folds y las ventanas son decisiones de criterio, pendientes de comentar con el tutor.
- El retardo 1 (`corr_lag1`) solo se **mide**, no se corrige (betas con retardos: MIAX-216).
- **La composición de monedas no se aparea entre esquemas:** la mediana de cada esquema se calcula sobre conjuntos de monedas y de minutos algo distintos (cambia `n_coins`; en la media y el PCA el factor de cada esquema también cambia). Una victoria puede venir en parte de la composición y no solo de la beta.
- **No se aplicó `check_factor_model` al esquema móvil:** queda cubierto por los tests de no-fuga propios y porque los tres modelos ya pasan su contrato.

---

## 7. Cómo regenerarlo

```bash
python -m miax.eval beta-study --windows 30 90 --horizons 1m 5m 15m
```

Por defecto recorre los 12 trimestres de D-37 (`--tests` acepta otros inicios de trimestre, `YYYY-MM-DD` en UTC). Lee la caché de velas y guarda el panel de cada fold (12 + 3 meses con las monedas del trimestre de prueba) en `data/processed/panel/1m/`; `--rebuild` lo reconstruye. **Coste de disco:** un panel de 15 meses por fold en `data/processed/panel/1m/`, solapados un 75 % entre folds contiguos; en total varios GB, ignorados por git. Escribe `beta_study_diagnostics.csv`, `beta_study_bias.csv` y `beta_study_decision.csv` en `reports/tables/` (ignorada por git), con orden y formato numérico fijos: misma entrada, mismo CSV byte a byte. Códigos de salida: 0 bien, 1 si falla el trabajo (trimestre fuera del universo, sin caché), 2 si los argumentos no valen. Desde Python: `miax.eval.compare_schemes` y `miax.eval.write_tables`.

---

## 8. Relacionado

- MIAX-054 (métrica oficial de caída de correlación) y MIAX-055 (comparación de los tres factores): usan el esquema que salga de aquí.
- MIAX-070 (control de Epps): estudia el mismo fenómeno desde la correlación.
- MIAX-216 (betas con retardos): trata el retraso respecto al mercado que aquí solo se mide.
