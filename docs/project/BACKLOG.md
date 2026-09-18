# Backlog

Todas las tareas del TFM, de la fase semilla a la defensa. **Numeradas en el orden en que hay que hacerlas.**

- Un ticket = una rama `feature/MIAX-XXX` = un PR contra `develop`.
- No cojas un ticket si sus dependencias no están en *Done*.
- Si un ticket resulta ser dos tickets, se parte. No se amplía.

Proceso en [WORKFLOW.md](WORKFLOW.md) · Reglas en [CLAUDE.md](../../CLAUDE.md) · Estado del equipo en [ESTADO.md](ESTADO.md)

**Leyenda:** `Dep.` = tickets que deben estar cerrados antes. `—` = sin dependencias.

---

## Índice

| Épica | Tickets | Área | Qué deja hecho |
|---|---|---|---|
| [0. Fundación](#épica-0--fundación-del-proyecto) | 001–015 | A | Repo, reglas, entorno, CI |
| [1. Ingesta](#épica-1--ingesta-de-datos) | 016–032 | A | Histórico de Binance en local |
| [2. Panel y preprocesado](#épica-2--panel-y-preprocesado) | 033–046 | A | Panel limpio de retornos |
| [3. Residualización](#épica-3--residualización-del-factor-común) | 047–060 | B | Retornos idiosincráticos |
| [4. Lead-lag](#épica-4--lead-lag-línea-base) | 061–076 | B | Grafo dirigido con FDR |
| [5. Descubrimiento causal](#épica-5--descubrimiento-causal) | 077–090 | B | Grafo causal con PCMCI |
| [6. Estabilidad](#épica-6--estabilidad-del-grafo) | 091–100 | B | ¿Aguanta el grafo en el tiempo? |
| [7. Baselines](#épica-7--baselines-predictivos) | 101–114 | C | Los rivales del GNN |
| [8. GNN temporal](#épica-8--gnn-temporal) | 115–130 | C | El modelo de grafos |
| [9. Validación](#épica-9--validación-sin-fuga-de-datos) | 131–142 | C | Purga, embargo, métricas |
| [10. Test económico](#épica-10--test-económico) | 143–158 | C | ¿Sobrevive a los costes? |
| [11. Dashboard](#épica-11--dashboard) | 159–172 | Todos | Interfaz de resultados |
| [12. Reproducibilidad](#épica-12--reproducibilidad) | 173–182 | A | Que se pueda replicar |
| [13. Memoria y defensa](#épica-13--memoria-y-defensa) | 183–196 | Todos | El documento y el pitch |

---

## Épica 0 — Fundación del proyecto

**Release objetivo:** `v0.1.0`

| ID | Tarea | Dep. | Hecho cuando |
|---|---|---|---|
| MIAX-001 | README, diagramas y estructura de carpetas | — | ✅ Hecho |
| MIAX-002 | `CLAUDE.md`, `AGENTS.md`, backlog y proceso de equipo | 001 | ✅ Hecho |
| MIAX-003 | Cuatro subagentes versionados en `.claude/agents` | 002 | ✅ Hecho |
| MIAX-004 | Comprobación de entorno al arrancar sesión | 002 | ✅ Hecho |
| MIAX-005 | `requirements.txt` con versiones exactas fijadas | — | `pip install -r` reproduce el entorno |
| MIAX-006 | `pyproject.toml` y paquete instalable con `pip install -e .` | 005 | `import miax` funciona desde cualquier sitio |
| MIAX-007 | Configurar `ruff` (lint y formato) | 006 | `ruff check .` pasa en limpio |
| MIAX-008 | Configurar `pytest` y estructura de `tests/` | 006 | `pytest` corre y encuentra los tests |
| MIAX-009 | `utils/config.py`: cargar configuración desde YAML | 006 | Un YAML de `configs/` se lee tipado |
| MIAX-010 | `utils/seeds.py`: semillas deterministas globales | 006 | Dos ejecuciones dan el mismo resultado |
| MIAX-011 | `utils/io.py`: lectura y escritura de Parquet con rutas del repo | 006 | Guardar y releer un DataFrame es idéntico |
| MIAX-012 | `utils/logging.py`: logging con formato común | 006 | Todos los módulos loguean igual |
| MIAX-013 | CI en GitHub Actions: `ruff` + `pytest` en cada PR | 007, 008 | El PR muestra el check en verde |
| MIAX-014 | Plantillas de PR e issue en `.github/` | 002 | Abrir un PR precarga la plantilla |
| MIAX-015 | Crear `develop` y proteger `main` y `develop` | 013 | No se puede hacer push directo a ninguna |

---

## Épica 1 — Ingesta de datos

**Release objetivo:** `v0.2.0` · **Área:** A

| ID | Tarea | Dep. | Hecho cuando |
|---|---|---|---|
| MIAX-016 | Cliente HTTP base con timeout y reintentos | 006, 012 | Un fallo transitorio se reintenta solo |
| MIAX-017 | Backoff ante 429 y 418 respetando `Retry-After` | 016 | Simular un 429 no rompe la descarga |
| MIAX-018 | Descarga de klines con paginación hacia delante | 016, 017 | Un rango mayor de 1000 velas se descarga entero |
| MIAX-019 | Limitador de peticiones por peso de IP | 017 | Una descarga larga no provoca baneo |
| MIAX-020 | Guardar un símbolo en Parquet particionado por mes | 011, 018 | El fichero se relee y coincide |
| MIAX-021 | Reanudación: no volver a descargar lo ya cacheado | 020 | Relanzar la descarga no repite llamadas |
| MIAX-022 | CLI de descarga: símbolo, intervalo y rango de fechas | 020 | `python -m miax.ingest --symbol BTCUSDT` funciona |
| MIAX-023 | Descarga masiva de N símbolos con progreso | 021, 022 | 30 símbolos se descargan sin intervención |
| MIAX-024 | Consultar `/exchangeInfo`: pares disponibles y su estado | 016 | Se lista el catálogo completo de pares |
| MIAX-025 | Filtro de universo por volumen mínimo | 024 | Se descartan pares ilíquidos con un umbral |
| MIAX-026 | Filtro de universo por antigüedad mínima de listado | 024 | Se descartan tokens recién listados |
| MIAX-027 | Diversificación del universo por categoría (L1, L2, DeFi, memecoins) | 025, 026 | El universo no es solo L1 |
| MIAX-028 | Congelar el universo elegido en `configs/universe.yaml` | 027 | El universo es un fichero versionado, no código |
| MIAX-029 | Validador de integridad: huecos, duplicados, timestamps desordenados | 020 | Un fichero corrupto se detecta y se reporta |
| MIAX-030 | Informe de cobertura temporal por símbolo | 029 | Tabla de fecha inicial, final y % de velas presentes |
| MIAX-031 | Documentar el sesgo de supervivencia como limitación declarada | 024 | Queda escrito qué pares faltan y por qué |
| MIAX-032 | Tests de ingesta con fixtures, sin tocar la red | 018, 021 | `pytest` pasa sin conexión |

---

## Épica 2 — Panel y preprocesado

**Release objetivo:** `v0.3.0` · **Área:** A · **Salida:** interfaz `A → B`

| ID | Tarea | Dep. | Hecho cuando |
|---|---|---|---|
| MIAX-033 | Alinear todas las series a una rejilla temporal común en UTC | 020, 028 | Todos los símbolos comparten índice |
| MIAX-034 | Detectar y clasificar huecos: sin trades, símbolo caído, listado tardío | 033 | Cada hueco tiene una causa etiquetada |
| MIAX-035 | Política de huecos con máscara de validez, **sin** `fillna(0)` | 034 | Un hueco no se convierte en un retorno cero |
| MIAX-036 | Retornos logarítmicos sobre precios alineados | 035 | Los retornos son aditivos en el tiempo |
| MIAX-037 | Tratamiento de valores extremos y decisión sobre winsorización | 036 | Queda documentado qué se recorta y por qué |
| MIAX-038 | Volatilidad realizada por ventana | 036 | Feature disponible por activo y periodo |
| MIAX-039 | Volumen normalizado por activo | 033 | Comparable entre activos de tamaño distinto |
| MIAX-040 | Número de operaciones normalizado como proxy de actividad | 033 | Feature disponible por activo y periodo |
| MIAX-041 | Constructor del panel maestro `[tiempo × activo × feature]` | 036–040 | Un solo objeto alimenta todo lo demás |
| MIAX-042 | Remuestreo a 5m, 15m y 1h desde las velas de 1m | 036 | Los cuatro horizontes salen del mismo origen |
| MIAX-043 | Cachear el panel en Parquet particionado | 011, 041 | Cargar el panel tarda segundos, no minutos |
| MIAX-044 | Contrato de la interfaz `A → B`: esquema y validación | 041 | B puede consumir el panel sin preguntar |
| MIAX-045 | Tests de preprocesado, con foco en huecos y alineación | 035, 036 | Un hueco mal tratado hace fallar un test |
| MIAX-046 | EDA del panel y figura de correlación cruda para la memoria | 041 | Figura en `reports/figures/` |

---

## Épica 3 — Residualización del factor común

**Release objetivo:** `v0.4.0` · **Área:** B · **Riesgo cubierto:** espejismo de factor

| ID | Tarea | Dep. | Hecho cuando |
|---|---|---|---|
| MIAX-047 | Interfaz común `FactorModel` con `fit` / `transform` | 044 | Las tres definiciones de factor son intercambiables |
| MIAX-048 | Factor = retorno de BTC | 047 | Devuelve residuos por activo |
| MIAX-049 | Factor = media transversal de retornos | 047 | Devuelve residuos por activo |
| MIAX-050 | Factor = primera componente principal | 047 | Devuelve residuos y cargas |
| MIAX-051 | `fit` solo en train: guardar betas y cargas, y **proyectarlas** sobre test | 048–050 | Test existe un test que falla si se recalcula en test |
| MIAX-052 | Residuos por OLS con betas congeladas | 051 | Los residuos de test usan betas de train |
| MIAX-053 | Betas móviles frente a betas por fold: comparar y decidir | 052 | Decisión documentada con datos |
| MIAX-054 | Métrica: caída de correlación media tras residualizar | 052 | Un número que dice cuánto factor se ha quitado |
| MIAX-055 | Comparativa de las tres definiciones de factor | 048–050, 054 | Tabla con las tres columnas |
| MIAX-056 | Análisis de sensibilidad del resultado al factor elegido | 055 | Se sabe si la conclusión depende de la elección |
| MIAX-057 | Comprobar que queda señal idiosincrática tras residualizar | 054 | Si no queda, se activa el pivote del README §7.1 |
| MIAX-058 | Test antifuga específico de la residualización | 051 | Inyectar futuro hace fallar el test |
| MIAX-059 | Tests unitarios de residualización | 052 | Cobertura de casos borde |
| MIAX-060 | Figura de memoria: correlación antes y después de residualizar | 054 | Figura en `reports/figures/` |

---

## Épica 4 — Lead-lag, línea base

**Release objetivo:** `v0.5.0` · **Área:** B

| ID | Tarea | Dep. | Hecho cuando |
|---|---|---|---|
| MIAX-061 | Correlación cruzada con retardo entre dos series | 052 | Función probada contra un caso sintético |
| MIAX-062 | Barrido de retardos de 1 a L para todos los pares | 061 | Matriz `[activo × activo × retardo]` |
| MIAX-063 | Selección del retardo de correlación máxima por par | 062 | Un retardo y un signo por par ordenado |
| MIAX-064 | p-valor por permutación de bloques, corrigiendo el sesgo de selección | 063 | El p-valor tiene en cuenta que se eligió el máximo |
| MIAX-065 | Corrección de tests múltiples por FDR de Benjamini-Hochberg | 064 | Miles de contrastes no generan cientos de falsos |
| MIAX-066 | Construir aristas dirigidas `i → j` a partir de lo significativo | 065 | Lista de aristas con retardo y peso |
| MIAX-067 | Matriz de adyacencia ponderada | 066 | Objeto que consume el modelo |
| MIAX-068 | Umbral de peso y poda del grafo | 067 | El grafo no es una malla completa |
| MIAX-069 | Estimador de Hayashi-Yoshida para datos asíncronos | 061 | Alternativa disponible si se baja a tick |
| MIAX-070 | Control del efecto Epps y de la microestructura | 069 | Queda acotado el lead-lag espurio por iliquidez |
| MIAX-071 | Grafo **sin** residualizar, como anclaje de contraste | 062 | Se ve el "todo apunta a BTC" |
| MIAX-072 | Comparativa grafo crudo frente a grafo residualizado | 067, 071 | Se cuantifica qué elimina la residualización |
| MIAX-073 | Métricas de red: grado, centralidad, reciprocidad | 067 | Tabla de nodos más influyentes |
| MIAX-074 | Contrato de la interfaz `B → C`: formato de adyacencia por fold | 067 | C puede consumirla sin preguntar |
| MIAX-075 | Tests del estimador de lead-lag con series de retardo conocido | 061, 063 | Detecta un retardo inyectado a propósito |
| MIAX-076 | Figura de memoria: grafo de lead-lag | 068, 073 | Figura en `reports/figures/` |

---

## Épica 5 — Descubrimiento causal

**Release objetivo:** `v0.6.0` · **Área:** B

| ID | Tarea | Dep. | Hecho cuando |
|---|---|---|---|
| MIAX-077 | Integrar `tigramite` y congelar su versión | 005 | Se importa y corre un ejemplo mínimo |
| MIAX-078 | Adaptador del panel al formato de `tigramite` | 077, 052 | Los residuos entran sin transformación manual |
| MIAX-079 | PCMCI con `ParCorr` y `tau_max` configurable | 078 | Devuelve matrices de valor y de p-valores |
| MIAX-080 | Corrección FDR sobre la matriz de p-valores | 079 | Se obtiene la matriz de q-valores |
| MIAX-081 | Extracción de enlaces significativos con su retardo | 080 | Lista de aristas causales |
| MIAX-082 | Reducir coste remuestreando a 5 minutos | 042, 079 | El tiempo de ejecución baja a algo viable |
| MIAX-083 | Reducir coste restringiendo al subgrafo candidato de la línea base | 066, 079 | PCMCI solo evalúa pares plausibles |
| MIAX-084 | Banco de tiempos de PCMCI por tamaño de universo | 082, 083 | Tabla que dice qué cabe en el calendario |
| MIAX-085 | Comparar PCMCI frente a correlación cruzada | 066, 081 | Cuántas aristas sobreviven y cuáles caen |
| MIAX-086 | Cuantificar las aristas indirectas eliminadas | 085 | Número concreto para la memoria |
| MIAX-087 | Grafo causal recalculado por fold | 081 | Un grafo por fold, nunca uno global |
| MIAX-088 | Tests con series sintéticas de causalidad conocida | 079 | Recupera la estructura inyectada |
| MIAX-089 | Figura: grafo causal frente a grafo de correlación | 085 | Figura en `reports/figures/` |
| MIAX-090 | Notebook de descubrimiento causal para la memoria | 086 | Narrativa reproducible |

---

## Épica 6 — Estabilidad del grafo

**Release objetivo:** `v0.7.0` · **Área:** B · **Riesgo cubierto:** no estacionariedad

| ID | Tarea | Dep. | Hecho cuando |
|---|---|---|---|
| MIAX-091 | Estimar el grafo por ventana deslizante | 067 | Una secuencia de grafos en el tiempo |
| MIAX-092 | Medir similitud entre matrices de adyacencia consecutivas | 091 | Serie temporal de similitud |
| MIAX-093 | Persistencia de aristas: cuántas sobreviven entre ventanas | 091 | Distribución de vida útil de las aristas |
| MIAX-094 | Detectar cambios de régimen en la estructura del grafo | 092 | Fechas marcadas de ruptura |
| MIAX-095 | Estabilidad segmentada por categoría de activo | 091, 027 | ¿Las L1 son más estables que las memecoins? |
| MIAX-096 | Comparar grafo en periodos de estrés frente a calma | 094 | Dos grafos contrastados |
| MIAX-097 | Métrica agregada de estabilidad para la memoria | 092, 093 | Un número defendible ante el tribunal |
| MIAX-098 | Tests de las métricas de estabilidad | 092 | Casos sintéticos estable e inestable |
| MIAX-099 | Figura: evolución temporal del grafo | 092, 094 | Figura en `reports/figures/` |
| MIAX-100 | Notebook de estabilidad | 097 | Narrativa reproducible |

---

## Épica 7 — Baselines predictivos

**Release objetivo:** `v0.8.0` · **Área:** C

| ID | Tarea | Dep. | Hecho cuando |
|---|---|---|---|
| MIAX-101 | Definir la tarea: objetivo, horizonte y unidad de predicción | 044 | Queda escrito qué se predice exactamente |
| MIAX-102 | Soportar horizontes múltiples: 1m, 5m, 15m y 60m | 042, 101 | El mismo código sirve para los cuatro |
| MIAX-103 | Baseline martingala: predicción constante cero | 101 | Referencia mínima disponible |
| MIAX-104 | Baseline AR(p) por activo | 101 | Un modelo por activo, entrenado |
| MIAX-105 | Selección de `p` por criterio de información | 104 | `p` elegido con datos, no a ojo |
| MIAX-106 | Baseline VAR multivariante | 101 | Captura dependencias lineales sin grafo |
| MIAX-107 | Control de dimensionalidad del VAR | 106 | No explota con 50 activos |
| MIAX-108 | **Baseline LSTM por activo, sin grafo** — el juez real | 101 | Red equivalente al GNN pero sin estructura |
| MIAX-109 | Dataset y cargador temporal comunes a todos los modelos | 101 | Ningún modelo define su propio split |
| MIAX-110 | Bucle de entrenamiento común | 109 | Todos los modelos se entrenan igual |
| MIAX-111 | Parada temprana sobre validación temporal | 110 | No se entrena sobre el futuro |
| MIAX-112 | Registro de experimentos: parámetros, semilla y métricas | 110 | Cada ejecución queda trazada |
| MIAX-113 | Tests de los baselines | 103, 104, 106, 108 | Cada uno predice algo razonable en sintético |
| MIAX-114 | Tabla comparativa de baselines | 112 | Tabla para la memoria |

---

## Épica 8 — GNN temporal

**Release objetivo:** `v0.9.0` · **Área:** C

| ID | Tarea | Dep. | Hecho cuando |
|---|---|---|---|
| MIAX-115 | Congelar versiones de PyTorch y PyTorch Geometric Temporal | 005 | Entorno reproducible que importa sin errores |
| MIAX-116 | Plan B: implementar GConvGRU propio si la librería rompe | 115 | Existe alternativa si el ecosistema falla |
| MIAX-117 | Convertir adyacencia a `edge_index` y `edge_weight` | 074 | El grafo entra en el modelo |
| MIAX-118 | Dataset de grafo temporal por ventana | 109, 117 | Lotes con nodos, features y aristas |
| MIAX-119 | Modelo A3TGCN base | 115, 118 | Entrena sin errores y baja la pérdida |
| MIAX-120 | Cabeza de regresión, una predicción por nodo | 119 | Salida con la forma esperada |
| MIAX-121 | Bucle de entrenamiento del GNN | 110, 120 | Reutiliza el bucle común |
| MIAX-122 | Regularización: L2, dropout y modelo pequeño primero | 121 | Se combate el sobreajuste desde el diseño |
| MIAX-123 | Búsqueda de hiperparámetros acotada y documentada | 122 | Rejilla pequeña, sin sobreajustar el test |
| MIAX-124 | Grafo estático frente a grafo recalculado por fold | 087, 121 | Comparativa con datos |
| MIAX-125 | **Ablación: grafo vacío** (sin aristas) | 121 | Aísla la aportación de las aristas |
| MIAX-126 | **Ablación: aristas aleatorias** | 121 | Descarta que cualquier grafo sirva |
| MIAX-127 | Ablación de features por nodo | 121 | Se sabe qué feature aporta |
| MIAX-128 | Determinismo: misma semilla, mismo resultado | 010, 121 | Dos ejecuciones idénticas |
| MIAX-129 | Tests del GNN | 119, 120 | Formas, gradientes y determinismo |
| MIAX-130 | Curvas de entrenamiento para la memoria | 121 | Figura en `reports/figures/` |

---

## Épica 9 — Validación sin fuga de datos

**Release objetivo:** `v0.10.0` · **Área:** C · **Riesgo cubierto:** fuga de datos

| ID | Tarea | Dep. | Hecho cuando |
|---|---|---|---|
| MIAX-131 | Walk-forward básico por bloques temporales | 109 | Genera pares train/test ordenados |
| MIAX-132 | Purga: eliminar del train la franja contigua al test | 131 | Sin solape de ventanas |
| MIAX-133 | Embargo tras el train | 132 | La autocorrelación no cruza el corte |
| MIAX-134 | Validación combinatoria purgada (CPCV) | 133 | Varias combinaciones train/test |
| MIAX-135 | Ajustar escalados y normalizaciones **solo** en train | 134 | Ninguna estadística ve el test |
| MIAX-136 | Ejecutar la residualización dentro de cada fold | 051, 134 | El factor se estima por fold |
| MIAX-137 | Construir el grafo dentro de cada fold | 087, 134 | El grafo se estima por fold |
| MIAX-138 | **Test antifuga:** inyectar futuro y verificar que se detecta | 135–137 | El test falla si alguien introduce fuga |
| MIAX-139 | Métricas MAE y RMSE | 134 | Comparables entre modelos |
| MIAX-140 | Information Coefficient (Spearman previsto vs realizado) | 134 | Métrica de sección cruzada |
| MIAX-141 | Diebold-Mariano con errores agrupados por activo | 139 | Significatividad sin inflar por panel |
| MIAX-142 | Tests de la validación | 132, 133 | Purga y embargo comprobados en sintético |

---

## Épica 10 — Test económico

**Release objetivo:** `v0.11.0` · **Área:** C · **Riesgo cubierto:** explotabilidad

| ID | Tarea | Dep. | Hecho cuando |
|---|---|---|---|
| MIAX-143 | Convertir predicciones en pesos de cartera | 140 | Señal → pesos, documentado |
| MIAX-144 | Cartera long-short dólar-neutral | 143 | Exposición neta cercana a cero |
| MIAX-145 | Normalización de exposición bruta | 144 | Apalancamiento controlado |
| MIAX-146 | Desplazar la posición un periodo y verificar que no mira al futuro | 145 | Test que falla sin el desplazamiento |
| MIAX-147 | Cálculo de rotación de cartera | 146 | Serie temporal de turnover |
| MIAX-148 | Modelo de comisiones con la tarifa de referencia documentada | 147 | Coste por operación explícito |
| MIAX-149 | Modelo de slippage | 147 | Coste de impacto estimado |
| MIAX-150 | Banda de no-negociación: operar solo si el cambio supera un umbral | 147 | La rotación baja de forma medible |
| MIAX-151 | PnL bruto y neto | 148, 149 | Dos series comparables |
| MIAX-152 | Sharpe anualizado por horizonte | 102, 151 | Una cifra por horizonte |
| MIAX-153 | Sensibilidad al coste: barrido de puntos básicos | 151 | Curva Sharpe frente a coste |
| MIAX-154 | Sensibilidad al horizonte de rebalanceo | 152, 150 | Se ve dónde muere la señal |
| MIAX-155 | Drawdown máximo y métricas de riesgo | 151 | Tabla de riesgo |
| MIAX-156 | Comparativa económica del GNN frente a todos los baselines | 114, 152 | Tabla final del TFM |
| MIAX-157 | Tests del backtest, con foco en look-ahead | 146 | Cualquier look-ahead hace fallar un test |
| MIAX-158 | Figura: curva de PnL neta por modelo | 151 | Figura en `reports/figures/` |

---

## Épica 11 — Dashboard

**Release objetivo:** `v0.12.0` · **Área:** todos · **Skill:** `frontend-design`

| ID | Tarea | Dep. | Hecho cuando |
|---|---|---|---|
| MIAX-159 | Decidir el stack del frontend y justificarlo | 015 | Decisión escrita en `ESTADO.md` |
| MIAX-160 | Esqueleto de la aplicación y navegación | 159 | Arranca en local con una página |
| MIAX-161 | Sistema de diseño: tokens de color, tipografía y espaciado | 160 | Ninguna vista usa colores sueltos |
| MIAX-162 | Vista: universo y cobertura de datos | 030, 161 | Se ve qué hay descargado y qué falta |
| MIAX-163 | Vista: grafo de lead-lag interactivo | 076, 161 | Se puede explorar nodo a nodo |
| MIAX-164 | Selector de ventana temporal sobre el grafo | 091, 163 | El grafo cambia al mover la ventana |
| MIAX-165 | Vista: estabilidad del grafo en el tiempo | 099, 161 | Serie de similitud navegable |
| MIAX-166 | Vista: comparativa de modelos | 114, 161 | Tabla y gráfico de métricas |
| MIAX-167 | Vista: backtest, costes y sensibilidad | 153, 158, 161 | Curvas de PnL con controles de coste |
| MIAX-168 | Estados de carga, vacío y error en todas las vistas | 162–167 | Ninguna vista se queda en blanco |
| MIAX-169 | Comportamiento responsive | 168 | Usable en pantalla estrecha |
| MIAX-170 | Accesibilidad: contraste, foco y navegación por teclado | 169 | Auditoría básica en verde |
| MIAX-171 | Tests del frontend | 168 | Las vistas críticas tienen test |
| MIAX-172 | Documentar cómo lanzar el dashboard | 171 | Un comando en el README |

---

## Épica 12 — Reproducibilidad

**Release objetivo:** `v0.13.0` · **Área:** A · **Riesgo cubierto:** irreproducibilidad

| ID | Tarea | Dep. | Hecho cuando |
|---|---|---|---|
| MIAX-173 | Congelar todas las versiones exactas del entorno | 005, 115 | `pip freeze` versionado y comprobado |
| MIAX-174 | Semilla global aplicada en todo el pipeline | 010, 128 | Una ejecución completa es determinista |
| MIAX-175 | Manifiesto de datos: rango, hash y recuento por símbolo | 029 | Se puede verificar que los datos son los mismos |
| MIAX-176 | Script de reproducción de principio a fin | 174, 175 | Un comando reproduce todos los resultados |
| MIAX-177 | Tareas automatizadas para los comandos habituales | 176 | `make test`, `make data`, `make report` |
| MIAX-178 | CI ejecutando la suite completa en cada PR | 013, 142 | Ningún PR entra con tests en rojo |
| MIAX-179 | CI ejecutando el lint | 007, 013 | Estilo homogéneo garantizado |
| MIAX-180 | Documentar cómo reproducir el TFM desde cero | 176 | Alguien de fuera lo consigue siguiendo el README |
| MIAX-181 | Revisión final de fugas de datos en todo el repositorio | 138, 157 | Auditoría escrita, punto por punto |
| MIAX-182 | Auditoría final de reproducibilidad | 180, 181 | Dos ejecuciones independientes coinciden |

---

## Épica 13 — Memoria y defensa

**Release objetivo:** `v1.0.0` · **Área:** todos

| ID | Tarea | Dep. | Hecho cuando |
|---|---|---|---|
| MIAX-183 | Índice de la memoria y reparto de capítulos | 015 | Cada capítulo tiene dueño |
| MIAX-184 | Capítulo: introducción y contexto | 183 | Borrador revisado por los tres |
| MIAX-185 | Capítulo: estado del arte | 183 | Con referencias cerradas |
| MIAX-186 | Capítulo: datos y universo | 031, 046 | Incluye el sesgo de supervivencia |
| MIAX-187 | Capítulo: metodología | 060, 076, 090 | Incluye las decisiones de diseño y su porqué |
| MIAX-188 | Capítulo: resultados estadísticos | 114, 141 | Con test de significatividad |
| MIAX-189 | Capítulo: resultados económicos | 156 | Con sensibilidad a coste y horizonte |
| MIAX-190 | Capítulo: estabilidad y análisis de régimen | 097 | La inestabilidad se presenta como resultado |
| MIAX-191 | Capítulo: limitaciones y amenazas a la validez | 181 | Honesto y explícito |
| MIAX-192 | Capítulo: conclusiones | 188–191 | Responde la pregunta del §3 del README |
| MIAX-193 | Bibliografía completa y formateada | 185 | Sin referencias huérfanas |
| MIAX-194 | Figuras y tablas finales unificadas de estilo | 158, 099 | Mismo estilo en todas |
| MIAX-195 | Revisión cruzada entre los tres integrantes | 192, 193, 194 | Cada uno ha leído el capítulo de los otros |
| MIAX-196 | Presentación de defensa y ensayo con preguntas previsibles | 195 | Ensayo cronometrado hecho |

---

## Cómo añadir un ticket

Si aparece trabajo nuevo, **no reutilices un número**. Se añade al final de su épica con el siguiente número libre y se anota su dependencia. El orden numérico es el orden histórico en que se decidió hacer algo, no una prioridad rígida.
