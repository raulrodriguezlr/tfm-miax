# Fixtures de Binance

Respuestas **crudas** de la API publica de Binance Spot (`https://api.binance.com`), grabadas una vez para que los tests de ingesta (MIAX-032) corran sin red. El JSON es el que devuelve Binance: filas de 12 campos con los numeros como texto, sin convertir.

**Fecha de grabacion:** 2026-10-13 (se actualiza al regenerar).

| Fichero | Endpoint | Contenido |
|---|---|---|
| `klines_BTCUSDT_1m.json` | `/api/v3/klines` | Lista de filas: BTCUSDT 1m, dia completo 2024-03-01 (00:00Z a 23:59Z, 1.440 velas). |
| `klines_LUNAUSDT_1m.json` | `/api/v3/klines` | Lista de filas: LUNAUSDT 1m, las 100 velas anteriores y las 100 posteriores al mayor hueco de la ventana 2022-05-10 a 2022-06-05 (el hueco de mayo de 2022). |
| `klines_1d.json` | `/api/v3/klines` | Dict `{simbolo: [filas]}` de velas 1d para BTCUSDT, ETHUSDT, DOGEUSDT, UNIUSDT, LINKUSDT, USDCUSDT y EOSUSDT: su primera vela diaria (`startTime=0`) mas las del 2023-10-01 al 2023-12-31. |
| `exchange_info.json` | `/api/v3/exchangeInfo` | Respuesta cruda recortada a unos 15 simbolos (los 7 anteriores, LUNAUSDT, SOLUSDT, ADAUSDT, BNBUSDT, ETHBTC y algunos pares en BREAK); cada entrada se conserva entera y el resto de claves de nivel superior no se tocan. |

## Como se localiza el hueco de LUNAUSDT

El script baja el 1m de LUNAUSDT del 2022-05-10 al 2022-06-05, busca el mayor salto de `open_time` entre velas consecutivas y guarda solo las 100 velas a cada lado. Si el salto es menor de 1 hora o a algun lado hay menos de 100 velas, el script falla en vez de grabar algo distinto. Asi no se hardcodean fechas frágiles del hueco.

## Regenerar

Usa la red y sobrescribe los cuatro ficheros (descarga todo antes de escribir, asi un fallo no deja un conjunto mezclado):

```bash
python scripts/dev/record_fixtures.py
```

Los tests no regeneran nada: leen estos ficheros tal cual. Al regenerar, `exchange_info.json` puede cambiar (estados de los pares); conviene revisar el diff.
