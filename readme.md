# README

## Introducción

Este proyecto implementa un flujo completo de **lectura**, **procesado**, **modelado** y **evaluación** de datos NBBO (National Best Bid and Offer) para predecir el **mid-price** de un instrumento financiero (“SYMBOL = A”) un minuto en el futuro. El objetivo final es aplicar esta predicción en estrategias de **market making**, permitiendo colocar órdenes de forma dinámica y optimizar la captura de spread.

---

## ¿Qué es Market Making?

El **market making** es una estrategia de trading en la que un participante (el "market maker") provee **liquidez** al mercado colocando simultáneamente órdenes de compra (bids) y órdenes de venta (asks) sobre un instrumento financiero. Al ofrecer siempre precios en ambos lados del libro de órdenes, el market maker busca capturar la diferencia (el *spread*) entre el precio al que compra y el precio al que vende.

* **Spread**: es la diferencia entre el mejor precio de venta (`BO`, Best Offer) y el mejor precio de compra (`BB`, Best Bid).
* **Mid-price**: punto medio entre `BB` y `BO`, calculado como `(BB + BO) / 2`.
* **BBSize / BOSize**: volumen disponible en ese nivel de best bid u offer.

El market maker gana pequeñas fracciones cada vez que sus órdenes cruzan en el mercado, acumulando beneficios por la repetición de estas operaciones.

---

## Estructura del Modelo

1. **Carga de datos**: Se lee un archivo SAS (`.sas7bdat`) con registros tick-a-tick de NBBO.
2. **Construcción del índice temporal**:

   * Convierte `DATE` (fecha) a tipo `datetime64`.
   * Transforma `TIME` (segundos desde medianoche) a un `timedelta`.
   * Combina fecha y hora para fijar un índice `datetime` sobre el DataFrame.
3. **Filtrado de instrumento**: Solo se procesan los ticks cuyo campo `SYMBOL` coincide con el activo de interés (por ejemplo, “A”).
4. **Resampleo**: Agrupa los ticks cada 60 segundos y calcula estadísticas dentro de cada ventana:

   * `mid`: último mid-price (`(BB + BO)/2`).
   * `BBSize`, `BOSize`: volúmenes totales (`sum`).
   * `BB`, `BO`: precios medios (`mean`).
5. **Feature engineering**:

   * **spread** = `BO - BB`.
   * **imbalance** = `(BBSize - BOSize)/(BBSize + BOSize)`.
   * **mid\_future**: target, mid-price un minuto adelante (`mid.shift(-1)`).
6. **Split train/test**: 80% de las ventanas para entrenamiento, 20% para prueba, manteniendo orden temporal.
7. **Modelado**:

   * Pipeline: `StandardScaler` + `RandomForestRegressor(n_estimators=100)`.
   * Entrenamiento y predicción.
8. **Evaluación**:

   * **MSE**, **RMSE**, **MAE**, **MAPE**, **R²**.
   * Predicción en tiempo real para la última ventana.

---

## Resultados Principales

* **MSE**: Error cuadrático medio.
* **RMSE**: \~0.21 (error en unidades de precio).
* **MAE**: \~0.13 (error absoluto medio).
* **MAPE**: \~0.22% (error porcentual medio).
* **R²**: \~0.71 (explica el 71% de la varianza).

Estos resultados indican que el modelo predice con alta fidelidad el mid-price a un minuto, con desviaciones típicas muy pequeñas respecto al precio real.

---

## Uso en Market Making

1. **Generación de señales**: Cada minuto, a partir de las features del intervalo actual, el modelo predice `mid_next`. Comparando con `mid_actual`:

   * Si `mid_next > mid_actual * (1 + ε)`: señal **BUY**.
   * Si `mid_next < mid_actual * (1 - ε)`: señal **SELL**.
   * En otro caso, **HOLD**.

2. **Colocación de órdenes**:

   * Se despliegan **órdenes limitadas** de compra en `(mid_next - spread/2)` y de venta en `(mid_next + spread/2)`.
   * El **size** (`BBSize`/`BOSize`) indica cuánto volumen se puede colocar sin mover demasiado el precio; se suelen usar fracciones (10–30%) para minimizar impacto.

### Ejemplo de uso práctico

* El modelo te dice: “este minuto el mid está en 50.00; al siguiente minuto lo veo en 50.20”.
* Con esa señal, envías **órdenes limitadas** de compra/venda alrededor de `mid_pred`, capturando el spread cuando tus órdenes sean ejecutadas.
* El **size** te guía para dimensionar tu orden: no metes todo tu capital, sino un porcentaje del volumen disponible.
* No se trata de “comprar todo y vender todo cada minuto”, sino de diseñar reglas de entrada/salida, límites de órdenes, gestión de riesgo y frecuencia de trading según tus costos y tolerancia al riesgo.

3. **Backtest y métricas**:

   * Simula tu estrategia aplicando las señales sobre datos históricos (`df_eval`).
   * Mide P\&L, tasa de aciertos, ratio de Sharpe y drawdown.

---

## Cómo ejecutar

1. Instalar dependencias:
```
pip install -r requirements.txt
```

2. Ejecutar el notebook para entrenar y obtener predicciones
