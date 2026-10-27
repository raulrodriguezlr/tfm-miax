# Sesgo de supervivencia

Limitación declarada del universo de datos (riesgo 8 del [README](../../README.md)). Base del capítulo de datos de la memoria (MIAX-186). Las cifras están medidas el **2026-10-13**.

---

## 1. Qué es el riesgo

Si el universo solo contiene activos que existen hoy, se ignoran los que se deslistaron tras desplomarse. Los resultados salen optimistas y las relaciones de *lead-lag* quedan sesgadas hacia los supervivientes.

---

## 2. La premisa original era falsa

El README decía que la API pública de Binance no sirve el histórico de pares deslistados. Medido el 2026-10-13, no es cierto:

- `exchangeInfo` devuelve 3.725 pares, 761 contra USDT. De ellos, **254 están en `BREAK`** (parados o deslistados) y **conservan su histórico de velas**: el 2026-10-13 se consultó la primera vela de los 254 y **los 254 devuelven velas** (0 sin velas). Antes, Raúl lo había comprobado en vivo en MIAX-025 con FTTUSDT, SRMUSDT y 1INCHUPUSDT: devuelven velas de cuando cotizaban y 0 velas fuera de su vida.
- Los volcados oficiales de `data.binance.vision` (bucket público, `data/spot/monthly/klines/`) listan 3.745 símbolos spot, 757 contra USDT. **Solo uno contra USDT, `NBTUSDT`, falta en `exchangeInfo`** (purgado del catálogo). El resto de los deslistados sigue accesible por la API.

Conclusión: el sesgo de supervivencia dentro de Binance se puede corregir en lo esencial. Lo que queda está en el apartado 4.

---

## 3. Qué está corregido y cómo

| Qué | Dónde | Efecto |
|---|---|---|
| El catálogo se consulta completo, incluidos los pares en `BREAK` | D-13 (MIAX-024) | Los deslistados son visibles para el universo |
| El universo se elige año a año con el volumen de la ventana anterior (oct-dic del año previo), nunca con el de hoy | D-14 (MIAX-025), MIAX-028 | Un par entra si era líquido entonces, aunque hoy esté deslistado |
| `configs/categories.yaml` incluye los activos que fueron líquidos en alguna ventana desde 2017, deslistados incluidos (EOS, LUNA, FTT, XMR, OMG, SRM, BCC/BCHABC…) | MIAX-028 | Sin ellos el prefiltro los descartaría por no tener categoría |

Resultado en `configs/universe.yaml`: de los 104 símbolos del universo, **12 están hoy en `BREAK`**: BCCUSDT, BCHABCUSDT, EOSUSDT, FTMUSDT, HNTUSDT, LRCUSDT, MATICUSDT, OMGUSDT, RNDRUSDT, STORJUSDT, SXPUSDT, YFIIUSDT. Unos están deslistados y otros solo renombrados (BCC/BCHABC → BCH, MATIC → POL, FTM → S, RNDR → RENDER).

Pares-periodo hoy en `BREAK`, por año:

| Año | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|
| En `BREAK` / total | 0/2 | 2/11 | 2/10 | 4/21 | 5/40 | 2/40 | 3/40 | 0/40 | 0/40 |

LUNA y FTT entran en el universo de 2022 (antes de sus colapsos), y FTT también en 2023 y 2024.

---

## 4. Qué pares faltan y por qué

Sesgos residuales, de más a menos relevante para el resultado:

| # | Qué falta | Por qué | Magnitud | Cómo se trata |
|---|---|---|---|---|
| 1 | **Activos sin par contra USDT en Binance** | El universo es solo contra USDT (decisión de MIAX-027/028). En 2017-2018 las altcoins cotizaban sobre todo contra BTC y ETH: antes de octubre de 2017 había 2 pares contra USDT, 23 contra BTC y 23 contra ETH | 2018-2020 tienen solo 2-11 pares | No se sintetizan precios USDT desde pares BTC: metería el movimiento de BTC en cada serie y contaminaría el *lead-lag*. Se acota el periodo: el grafo útil empieza en 2021-2022 |
| 2 | **Sesgo de exchange** | Solo Binance, que sobrevivió (FTX no). Los activos que nunca cotizaron en Binance, o solo en otros exchanges, quedan fuera | No medida | No se mezclan exchanges a propósito: generaría *lead-lag* espurio entre plataformas. Se declara |
| 3 | **Renombrados y símbolos reutilizados** | BCC → BCHABC → BCH, MATIC → POL, FTM → S y RNDR → RENDER son símbolos distintos (nodos distintos en años distintos). `LUNAUSDT` es Terra clásica hasta mayo de 2022 y Terra 2.0 después, con el mismo símbolo; la clásica sigue como `LUNCUSDT` | 5 renombrados (BCCUSDT, BCHABCUSDT, MATICUSDT, FTMUSDT, RNDRUSDT) y 1 símbolo reutilizado (LUNAUSDT) | Se detecta con `coverage` de MIAX-030, en review (hueco de unos 18 días en mayo de 2022). Lo debe tratar la Épica 2 (MIAX-034) |
| 4 | **Pares que dejan de cotizar dentro de su año** | Su serie se corta: BCC en 2019, BCHABC en 2020, LUNA en 2022. No faltan, pero están incompletos | 3 casos conocidos | Se mide con `python -m miax.ingest coverage --universe` (`coverage` es de MIAX-030, en review, aún no está en `develop`); lo trata la Épica 2 |
| 5 | **Categorías con la visión de hoy** | `categories.yaml` clasifica cada activo como se ve ahora, no como se veía entonces | Sesgo menor | Se declara |
| 6 | **Purgados del catálogo** | `NBTUSDT` no aparece en `exchangeInfo` | 1 de 757 pares USDT | Despreciable; se declara |

---

## 5. Cómo reproducir las cifras

- **Catálogo:** `python -m miax.ingest pairs --quote USDT` (y `--status BREAK` para los parados o deslistados).
- **Universo:** `configs/universe.yaml`, regenerable con `python -m miax.ingest universe`.
- **Símbolos de `data.binance.vision`:** listado del bucket público S3 `https://s3-ap-northeast-1.amazonaws.com/data.binance.vision?prefix=data/spot/monthly/klines/&delimiter=/` (paginado con `marker`), comparado con `exchangeInfo`.
- **Fecha de las cifras:** 2026-10-13. El catálogo cambia con el tiempo (nuevos listados, nuevos deslistados): al citar una cifra hay que citar su fecha.
