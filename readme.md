
## Introducción

Este proyecto implementa un flujo completo de procesamiento, modelado y evaluación de datos NBBO (National Best Bid and Offer) para **predecir la dirección futura del precio de un instrumento financiero (Sube/Baja)**. El objetivo es determinar si el **precio de cierre** de un activo (`SYMBOL = A`) aumentará o disminuirá significativamente en los próximos **10 minutos**. Esta predicción se puede aplicar en estrategias de trading cuantitativas, como el **market making**, para optimizar la colocación de órdenes y la captura de `spread`.

---

## ¿Qué es Market Making?

El market making es una estrategia de trading donde un participante (el "market maker") provee liquidez al mercado. Lo hace colocando simultáneamente órdenes de compra (`bids`) y órdenes de venta (`asks`) para un instrumento financiero. Al ofrecer precios en ambos lados del libro de órdenes, el market maker busca capturar la diferencia (el `spread`) entre el precio al que compra y el precio al que vende.

* **`Spread`**: Es la diferencia entre el mejor precio de venta (`Ask`, antes `BO`) y el mejor precio de compra (`Bid`, antes `BB`).
* **`Mid-price`**: Punto medio entre `Bid` y `Ask`, calculado como `(Bid + Ask) / 2`.
* **`BidQty` / `AskQty`**: Volumen disponible en el nivel del mejor precio de compra o venta, respectivamente (antes `BBSize`/`BOSize`).

El market maker obtiene pequeñas ganancias cada vez que sus órdenes son ejecutadas, acumulando beneficios a través de la repetición de estas operaciones.

---

## Estructura del Modelo

### 1. Carga y Re-muestreo de Datos

* **Carga de datos**: Se lee un archivo Parquet (`all_data.parquet`) que contiene registros `tick-a-tick` de NBBO.
* **Filtrado de instrumento**: Solo se procesan los ticks cuyo campo `SYMBOL` coincide con el activo de interés (`"A"`).
* **Renombrado de columnas**: Se ajustan los nombres de las columnas a `Bid`, `Ask`, `BidQty`, `AskQty`.
* **Construcción del índice temporal**:
    * `TIME` (segundos desde medianoche) se convierte a un `timedelta`.
    * Se combina `DATE` y el `timedelta` para crear un índice `datetime` sobre el DataFrame.
* **Re-muestreo**: Los ticks se agrupan cada **60 segundos** (`1min`), y se calculan las siguientes estadísticas dentro de cada ventana:
    * `Bid`, `Ask`: `first`, `max`, `min`, `last` (primer, máximo, mínimo y último precio).
    * `BidQty`, `AskQty`: `sum` (volumen total).
    * `NUMEX`: `mean`.

### 2. Ingeniería de Características

Se calculan nuevas variables (features) a partir de los datos re-muestreados para mejorar la capacidad predictiva del modelo:

* **`open`, `high`, `low`, `close`**: Precios de apertura, máximo, mínimo y cierre de la vela de 1 minuto, calculados a partir de los `Bid`/`Ask` correspondientes.
* **`spread`**: `Ask_last - Bid_last`.
* **`imbalance`**: `(BidQty_sum - AskQty_sum) / (BidQty_sum + AskQty_sum)`. Mide la presión de compra/venta.
* **`volume`**: `BidQty_sum + AskQty_sum`. Volumen total transado en la ventana.
* **Medias Móviles (`rolling means`)**: Para `imbalance` y `volume` en ventanas de **5, 10 y 30 minutos** (ej. `imbalance_ma_5`).
* **Volatilidad (`rolling std`)**: Desviación estándar del precio de cierre (`close`) en ventanas de **5, 10 y 30 minutos** (ej. `volatility_5`).

### 3. Creación del Target BINARIO

El objetivo del modelo es clasificar si el precio de cierre subirá o bajará significativamente en el futuro:

* **Horizonte de Predicción**: Se define `prediction_horizon = 10` minutos. Esto significa que el `target` se basará en el `close` 10 minutos más tarde.
* **`close_future`**: El precio de cierre del activo 10 minutos adelante (`close.shift(-10)`).
* **`target_price_change`**: La diferencia entre el `close_future` y el `close` actual.
* **Umbral (`threshold`)**: Se establece un `threshold` (por ejemplo, `0.01`) para definir un movimiento significativo:
    * Si `target_price_change > threshold`: `target_direction = 1` (Sube).
    * Si `target_price_change < -threshold`: `target_direction = 0` (Baja).
    * Si `abs(target_price_change) <= threshold`: Se marcan como `NaN` para ser eliminados, ya que no representan un movimiento "claro" hacia arriba o hacia abajo.

### 4. División de Datos y Modelado

* **Limpieza Final**: Se eliminan las filas con valores `NaN` (incluyendo las ventanas con `target_direction` indefinido y los `NaN` generados por las medias móviles).
* **`Split` Train/Test**: Los datos se dividen en un conjunto de entrenamiento (80%) y un conjunto de prueba (20%), **manteniendo el orden temporal** (`shuffle=False`).
* **Modelo**: Se utiliza **XGBoost (Extreme Gradient Boosting)**, un potente algoritmo de clasificación basado en árboles de decisión.
* **`GridSearchCV` para Series Temporales**:
    * Se emplea `TimeSeriesSplit` para una validación cruzada adecuada para datos secuenciales, evitando la fuga de información futura.
    * Se optimizan hiperparámetros clave del modelo XGBoost (`max_depth`, `learning_rate`, `n_estimators`, `colsample_bytree`) usando `GridSearchCV` para encontrar la mejor combinación.
    * Se calcula `scale_pos_weight` para manejar el desbalance de clases si lo hubiera (más "sube" que "baja" o viceversa).
    * La métrica de optimización es **ROC AUC (`roc_auc_score`)**, que es robusta para problemas de clasificación binaria, especialmente con desbalance de clases.
* **Entrenamiento y Predicción**: El mejor modelo encontrado por `GridSearchCV` se entrena y luego se utiliza para predecir la dirección en el conjunto de prueba.

---

### 5. Evaluación

La evaluación del modelo se realiza sobre el conjunto de prueba (`X_test`, `y_test`), utilizando métricas para clasificación binaria:

* **`Accuracy`**: Proporción de predicciones correctas.
* **`ROC AUC Score`**: Mide la capacidad del modelo para distinguir entre las clases positivas y negativas. Un valor más cercano a 1 indica mejor rendimiento.
* **`Classification Report`**: Proporciona `precision`, `recall` y `f1-score` para ambas clases (`Baja (0)` y `Sube (1)`).

---

### 6. Guardado del Modelo

El modelo XGBoost optimizado se guarda utilizando `joblib` para su posterior uso en inferencia o despliegue.

---

## Resultados Principales

Los resultados se basan en la evaluación del modelo XGBoost optimizado en el conjunto de prueba:

* **`Accuracy`**: [0.3772]
* **`ROC AUC Score`**: [0.7466]

El **Reporte de Clasificación** detallará la `precisión`, `recall` y `f1-score` para cada clase, lo que ayuda a entender el rendimiento del modelo para predecir movimientos de precio al alza (Sube) o a la baja (Baja).

Estos resultados indican la capacidad del modelo para clasificar correctamente la dirección futura del precio con un buen balance entre falsos positivos y falsos negativos, lo cual es crucial para la toma de decisiones en trading.

---

## Uso en Market Making (Clasificación Binaria)

A diferencia de predecir un `mid-price` exacto, este modelo genera una señal direccional:

* **Generación de señales**: Cada minuto (o al final de cada ventana de 1 minuto), el modelo predice la `target_direction` (Sube o Baja) para los próximos 10 minutos.
    * Si el modelo predice `1` (**Sube**): Señal `BUY` o `LONG`.
    * Si el modelo predice `0` (**Baja**): Señal `SELL` o `SHORT`.
* **Colocación de órdenes**:
    * **Señal `BUY`**: Un market maker podría sesgar su libro de órdenes colocando más volumen del lado de la `compra` o ajustando sus precios para estar más agresivo en la `compra`, esperando que el precio suba.
    * **Señal `SELL`**: Un market maker podría sesgar su libro de órdenes colocando más volumen del lado de la `venta` o ajustando sus precios para estar más agresivo en la `venta`, esperando que el precio baje.
    * El **`size`** (`BidQty`/`AskQty`) sigue siendo relevante para dimensionar las órdenes sin impactar significativamente el mercado.
