import pandas as pd
import numpy as np
import xgboost as xgb  # <--- Importamos XGBoost
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. Carga de Datos Mejorados ---

# Cargamos el nuevo dataframe con todas las características
# Este archivo ahora contiene los features, el target y las columnas 'mid' para evaluación
df = pd.read_parquet("model/train_data_ml_enhanced.parquet")

# --- 2. Definición de Features y Target ---

# El target es el cambio de precio que ya calculamos
y = df['target_price_change']

# Los features son todas las demás columnas, excepto las relacionadas con el target y el precio
X = df.drop(columns=['target_price_change', 'mid', 'mid_future'])

print(f"Entrenando con {len(X.columns)} features.")

# División temporal, crucial como siempre
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# --- 3. Entrenamiento del Modelo XGBoost ---

# Definimos el modelo XGBoost con algunos parámetros iniciales razonables
# A diferencia de RandomForest, XGBoost no necesita un Pipeline con StandardScaler
xgb_regressor = xgb.XGBRegressor(
    objective='reg:squarederror',  # Objetivo de la regresión
    n_estimators=1000,             # Número de árboles (iteraciones de boosting)
    learning_rate=0.05,            # Tasa de aprendizaje, un valor bajo es más robusto
    max_depth=5,                   # Profundidad máxima de los árboles para evitar sobreajuste
    subsample=0.8,                 # Proporción de muestras de entrenamiento a usar por árbol
    colsample_bytree=0.8,          # Proporción de features a usar por árbol
    random_state=42,               # Para reproducibilidad
    n_jobs=-1,                     # Usa todos los cores de la CPU
    early_stopping_rounds=50       # ¡MUY ÚTIL! Detiene el entrenamiento si el modelo no mejora
)

print("\nEntrenando modelo XGBoost...")

# Para usar early_stopping, necesitamos un conjunto de evaluación
# Usaremos una porción del final del set de entrenamiento para esto
X_train_part, X_val, y_train_part, y_val = train_test_split(X_train, y_train, test_size=0.1, shuffle=False)

xgb_regressor.fit(
    X_train_part, y_train_part,
    eval_set=[(X_val, y_val)],
    verbose=False  # Ponlo en True si quieres ver el progreso del entrenamiento
)

print("Entrenamiento completado.")

# El modelo predice el CAMBIO de precio (delta)
predicted_delta = xgb_regressor.predict(X_test)

# --- 4. Evaluación del Modelo ---

# Para la evaluación, necesitamos los precios reales del conjunto de prueba
mid_actual_test = df.loc[X_test.index, "mid"]
mid_true_future = df.loc[X_test.index, "mid_future"]

# Reconstruimos la predicción del precio final
y_pred_xgb = mid_actual_test + predicted_delta

# 1. Evaluación del modelo XGBoost
mse_xgb = mean_squared_error(mid_true_future, y_pred_xgb)
rmse_xgb = np.sqrt(mse_xgb)
r2_xgb = r2_score(mid_true_future, y_pred_xgb)

print("\n--- Resultados del Modelo XGBoost ---")
print(f"RMSE: {rmse_xgb}")
print(f"R²: {r2_xgb}")

# 2. Evaluación del modelo de Baseline (Persistencia)
y_pred_baseline = mid_actual_test  # La predicción es simplemente el precio actual
mse_baseline = mean_squared_error(mid_true_future, y_pred_baseline)
rmse_baseline = np.sqrt(mse_baseline)
r2_baseline = r2_score(mid_true_future, y_pred_baseline)

print("\n--- Resultados del Modelo Baseline (Ingenuo) ---")
print(f"RMSE Baseline: {rmse_baseline}")
print(f"R² Baseline: {r2_baseline}")

# --- 5. Análisis Detallado ---

# Comparamos el rendimiento relativo
improvement_rmse = (rmse_baseline - rmse_xgb) / rmse_baseline
print(f"\nMejora sobre el baseline (RMSE): {improvement_rmse:.4%}")

print("\n--- Comparación de Predicciones (últimas 30) ---")
df_eval = pd.DataFrame({
    "mid_actual": mid_actual_test,
    "mid_true": mid_true_future,
    "mid_pred_XGB": y_pred_xgb,
    "mid_pred_baseline": y_pred_baseline
}, index=X_test.index)

print(df_eval.tail(30))