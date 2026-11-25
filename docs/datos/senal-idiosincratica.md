# Señal idiosincrática tras residualizar

Comprobación del punto de decisión de la Épica 3 (MIAX-057): después de quitar el factor común, **¿queda algo que modelar?** Si no queda, el TFM pivota al escenario C del README (§9): demostrar que el lead-lag idiosincrático en cripto es ruido.

> **Estado:** la regla se fijó el 2026-11-23, **antes de ejecutar el estudio con datos reales**, y después se midieron los 12 folds. **Veredicto: queda señal (12 de 12 folds con la media y con el PCA); no se activa el pivote.** La señal es pequeña y aguanta sin los minutos extremos (§4.3).

---

## 1. La pregunta

«Algo que modelar» quiere decir que el residuo de una moneda en `t` se puede predecir, más allá del azar, con los residuos **pasados**:
- de **otras** monedas: es el lead-lag, el objeto del TFM;
- o con **el suyo**: es lo que aprovecha el baseline univariante.

Es un contraste **global** por fold: un solo número para todas las monedas a la vez. No construye aristas ni da p-valores por par; eso es la Épica 4 (061 a 066).

Que pase este filtro es **necesario, no suficiente**. No dice si la señal es grande ni si cubre comisiones: eso lo dicen el modelo y el test económico.

---

## 2. Regla de decisión (fijada antes de ver los datos)

- **Horizonte que decide:** 1 minuto, con retardos de 1 a 5 barras.
- **Factores que deciden:** la **media** y el **PCA**, los dos que puede elegir la Épica 4 según MIAX-055. BTC deja mercado en el residuo (055), así que su señal no sería idiosincrática: se reporta solo como referencia.
- **Estadístico que decide:** el cruzado, `S_cross` (§3).

| Veredicto | Condición |
|---|---|
| **Queda señal** | `p_cross < 0,05` en **al menos 9 de 12 folds**, con la media **y** con el PCA |
| **Pivote** (escenario C) | `p_cross < 0,05` en **3 folds o menos**, con la media **o** con el PCA. Por azar se esperan 0,6 de 12 |
| **No concluyente** | cualquier otro caso; se lleva al tutor antes de seguir |

- **Hacen falta los 12 folds:** un factor decisor con menos de 12 folds con p-valor válido (por pedir menos trimestres o porque falló algún ajuste) no puede dar ni señal ni pivote. Su veredicto es **no concluyente**, para que la falta de datos nunca active el pivote.
- 5 y 15 minutos se reportan con la misma tabla, pero **no deciden**.
- La magnitud (`ratio_cross`), la asimetría y la señal propia se reportan aparte; su lectura es de criterio.
- Un fold con el p-valor en `NaN` (ajuste fallido) no cuenta como fold con señal.

---

## 3. Método

- **Folds:** los 12 de D-37, igual que MIAX-053.
  - El test es cada trimestre de 2023T1 a 2025T4.
  - Se entrena con los 12 meses anteriores, usando las monedas del trimestre de prueba.
  - Queda fuera la moneda con menos del 50 % de datos válidos en train.
  - **2026 no se toca.**
- **Residuos:** cada factor se ajusta en train y se proyecta sobre el test con las betas congeladas (D-03). A 5 y 15 minutos se ajusta y se proyecta con los retornos de ese horizonte (D-23); los de train cierran antes de que empiece el test (D-19).
- **Correlación con retardo:**
  - Para cada retardo `ℓ = 1..5`, se calcula la matriz `C_ℓ[i, j] = corr(e_i(t − ℓ), e_j(t))`.
  - Se calcula por pares, con las observaciones comunes y solo dentro del test.
  - Una casilla con menos de 1000 observaciones comunes queda en `NaN`.
- **Estadísticos:** agrupan los cinco retardos. Cada uno es la suma de los cuadrados de todas las casillas finitas de los cinco retardos, dividida entre el número de esas casillas.
  - **cruzado** `S_cross`: media de `C_ℓ[i, j]²` para `i ≠ j`;
  - **propio** `S_own`: media de `C_ℓ[i, i]²`;
  - **asimetría** `S_asym`: media de `(C_ℓ[i, j] − C_ℓ[j, i])²` para `i < j`, porque el lead-lag exige una dirección.
- **El azar (nulo):**
  - Se desplaza circularmente el bloque del pasado, todas las monedas a la vez, un número aleatorio de filas: entre un día y la longitud del test menos un día.
  - Se hace 99 veces, con semilla fija.
  - El desplazamiento conserva la dinámica de cada serie (colas, rachas de volatilidad) y la correlación contemporánea entre monedas, y rompe cualquier relación entre el pasado y el presente.
  - El p-valor es `(1 + nº de nulos ≥ S) / (nº de nulos finitos + 1)`. Normalmente los 99 son finitos y el mínimo es 0,01.
  - El efecto es `S / mediana de los nulos`.
- **Lo que queda en el mismo minuto**, solo descriptivo (retoma los puntos (c) y (d) de MIAX-055):
  - el primer autovalor de la correlación de los residuos, como fracción de N;
  - cuántos autovalores superan el borde de Marchenko-Pastur `(1 + √(N/T))²`.

---

## 4. Lo medido

Ejecución completa el 2026-11-23 (`python -m miax.eval idio-signal`), que sale con código 0.
- **Duración:** 49 minutos, construyendo los 12 paneles de fold, que no estaban en caché. Una fila a 1 minuto tarda unos 40 s.
- **Monedas y filas:** entran de 28 a 35 monedas por fold y 131 760 minutos de test (26 352 a 5 min y 8 784 a 15 min).
- **Errores:** ninguna fila tiene error.

### 4.1. Resultado por horizonte y factor (`idio_signal_by_fold.csv`, mediana de los 12 folds)

- `rms` es la raíz del estadístico: la correlación con retardo «típica» entre pares.
- `azar` es la misma cifra en la mediana de los 99 desplazamientos.
- Los cocientes son `S / mediana del azar` (cuadrados).

| Horizonte | Factor | Folds con p < 0,05 | rms cruzado | rms azar | cociente cruzado | cociente asimetría | rms propio | 1.er autovalor / N | autovalores sobre MP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1m | BTC | 12 | 0,0101 | 0,0028 | 13,1 | 12,9 | 0,037 | 0,38 | 2 |
| 1m | media | **12** | 0,0071 | 0,0028 | 6,5 | 5,9 | 0,051 | 0,077 | 9,5 |
| 1m | PCA | **12** | 0,0071 | 0,0028 | 6,5 | 5,9 | 0,052 | 0,079 | 10 |
| 5m | media | 12 | 0,0102 | 0,0063 | 2,7 | 2,3 | 0,030 | 0,087 | 8 |
| 5m | PCA | 12 | 0,0101 | 0,0063 | 2,6 | 2,3 | 0,030 | 0,089 | 8 |
| 15m | media | 12 | 0,0151 | 0,0109 | 1,9 | 1,8 | 0,032 | 0,092 | 7 |
| 15m | PCA | 12 | 0,0151 | 0,0109 | 1,9 | 1,7 | 0,032 | 0,095 | 7 |

Los 108 p-valores (cruzado, propio y asimetría) son 0,01, el mínimo posible con 99 desplazamientos. BTC a 5 y 15 min también sale significativo en los 12 folds.

### 4.2. Un evento extremo puede dominar el fold: 2025T4

En 2025T4 el cociente cruzado es **4262** con la media y **4704** con el PCA. En el resto de folds va de 3 a 26.
- **Causa:** el crash del **10 de octubre de 2025**. Los peores minutos (21:19 a 21:29 UTC) tienen residuos de hasta **239 desviaciones típicas**.
- **Por qué pesa tanto:** la correlación de Pearson sobre retornos sin recortar (D-22) se la llevan esos pocos minutos.
- **Lo que queda sin él:** quitando el 10 y el 11 de octubre, el rms cruzado de ese fold a 1 retardo baja de 0,187 a 0,014.

### 4.3. Sensibilidad a los minutos extremos (prueba de humo, fuera del pipeline)

Para ver si la señal la fabrican unos pocos minutos, se repitió el cálculo a 1 minuto con la media y el PCA en los 12 folds:
- quitando los minutos en que alguna moneda tiene un residuo de más de 10 desviaciones (≈ 1050 minutos por fold, el 0,8 %);
- quitando los de más de 5 (≈ 8000 minutos, el 6 %).

Se hizo con 39 desplazamientos, así que el p-valor mínimo es 0,025.

| Factor | Minutos quitados | Folds con p < 0,05 (cruzado / asimetría) | cociente cruzado, mediana (mín–máx) | cociente asimetría, mediana | rms propio |
|---|---|---:|---|---:|---:|
| media | ninguno | 12 / 12 | 6,6 (3,2 – 4156) | 6,0 | 0,051 |
| media | \|z\| > 10 | 12 / 12 | 3,7 (2,4 – 6,0) | 2,5 | 0,055 |
| media | \|z\| > 5 | 12 / 12 | 3,8 (2,2 – 6,0) | 2,2 | 0,063 |
| PCA | ninguno | 12 / 12 | 6,6 (3,2 – 4233) | 5,9 | 0,052 |
| PCA | \|z\| > 10 | 12 / 12 | 3,7 (2,6 – 7,1) | 2,5 | 0,055 |
| PCA | \|z\| > 5 | 12 / 12 | 3,9 (2,2 – 7,2) | 2,2 | 0,063 |

**La señal no depende de los extremos:** sin ellos sigue en los 12 folds, aunque con la mitad de tamaño (el cociente baja de 6,5 a 3,7 y el rms a 1 retardo de 0,007 a 0,006). Este cálculo no forma parte del estudio ni de la regla; el script está en la descripción del PR.

---

## 5. Decisión

Regla de la §2 aplicada tal cual a `idio_signal_decision.csv`: con la media y con el PCA, a 1 minuto, `p_cross < 0,05` en **12 de 12 folds**.

**Veredicto: queda señal. No se activa el pivote:** el TFM sigue con la Épica 4.

Cómo leerlo:
- **Es real pero pequeña.**
  - La correlación típica entre el pasado de una moneda y el presente de otra es de 0,006–0,007. El azar da 0,003.
  - Una correlación de 0,007 explica un 0,005 % de la varianza de un par. El grafo tendrá que agregar muchas relaciones pequeñas, y el test económico lo tendrá difícil a 1 minuto (D-39).
- **Tiene dirección:** la asimetría también sale en los 12 folds, así que no es solo el eco de que las monedas se mueven juntas y cada una se parece a su pasado.
- **El pasado propio pesa más que el cruzado:** rms de 0,05 frente a 0,007. El baseline univariante (§5.5 del README) parte con ventaja, y el grafo tiene que superarlo.
- **Se debilita con el horizonte** (cociente 6,5 a 1 min, 2,7 a 5 min y 1,9 a 15 min), pero sigue en los 12 folds.
- **Queda estructura en el mismo minuto:** con la media o el PCA, el primer autovalor de los residuos es el 8 % de N (1/N ≈ 3 %), y entre 7 y 10 autovalores superan el borde de Marchenko-Pastur. Son sectores o factores secundarios que el factor único no quita (punto (c) de MIAX-055).
- **Con BTC la señal es mayor, pero no es idiosincrática:** su residuo conserva el mercado (primer autovalor del 38 %, MIAX-055).
- **Para la Épica 4:** un solo evento como el del 10-oct-2025 puede multiplicar el estadístico por mil. Los p-valores por par (064) y la sensibilidad de las colas (D-22) tienen que contar con ello.

---

## 6. Qué no se ha verificado

- **Predecible no es lo mismo que lead-lag causal:**
  - Una correlación contemporánea entre residuos, sumada a la autocorrelación propia, también da `C_ℓ[i, j] ≠ 0` sin que `i` adelante a `j`.
  - Eso sigue siendo algo que modelar, pero la dirección la miden la asimetría y, después, el grafo.
- **Sesgo por iliquidez:** un precio rancio de una moneda poco líquida también parece «ir detrás» (efecto Epps, D-39). Lo controla MIAX-217.
- **Predecible no es lo mismo que rentable:** la magnitud es pequeña (§5) y no se ha medido el valor económico.
- **El p-valor está en su suelo:** 99 desplazamientos no distinguen entre una señal fuerte y una enorme. Para comparar tamaños se usa el cociente.
- **Las monedas más ilíquidas no están:** el filtro del 50 % de datos válidos en train (D-37) las deja fuera, igual que en MIAX-053.

---

## 7. Cómo regenerarlo

```bash
python -m miax.eval idio-signal
```

Escribe `idio_signal_by_fold.csv` e `idio_signal_decision.csv` en `reports/tables/` (ignorada por git).

---

## 8. Relacionado

- MIAX-054 y MIAX-055: cuánta correlación quita cada factor. MIAX-053: qué beta usar.
- MIAX-061 a 066: el grafo de lead-lag, par a par, con p-valores corregidos.
- MIAX-215 (factor sin la propia moneda), MIAX-216 (betas con retardo) y MIAX-217 (control de liquidez).
