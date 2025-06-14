# train_optimized.py

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, roc_auc_score
import time # Para medir el tiempo de ejecución
import joblib

# --- 1. Carga de Datos ---
df = pd.read_parquet("model/clean/train_data_ml_binary.parquet")

# --- 2. Definición de Features y Target ---
y = df['target_direction']
X = df.drop(columns=['target_direction'])
print(f"Entrenando con {len(X.columns)} features para clasificación binaria.")

# Dividimos en un conjunto de entrenamiento y uno de prueba final (holdout)
# El GridSearchCV solo se ejecutará sobre el conjunto de entrenamiento.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# --- 3. Configuración de GridSearchCV para Series Temporales ---

# a) Definir el validador cruzado para series temporales
# Usaremos 5 "splits" o divisiones.
tscv = TimeSeriesSplit(n_splits=5)

# b) Definir el estimador base
# Calculamos el `scale_pos_weight` sobre el conjunto de entrenamiento completo
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb_estimator = xgb.XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)

# c) Definir la parrilla de hiperparámetros a probar
# ¡ADVERTENCIA! Una parrilla grande puede tardar MUCHO tiempo.
# Esta es una parrilla de tamaño razonable para empezar.
param_grid = {
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.05],
    'n_estimators': [500, 1000],
    'colsample_bytree': [0.7, 0.8]
}

# d) Crear el objeto GridSearchCV
# - estimator: Nuestro modelo base
# - param_grid: El diccionario de parámetros a probar
# - cv: Nuestro validador de series temporales
# - scoring: La métrica que queremos maximizar. 'roc_auc' es la mejor para este problema.
# - verbose: Muestra el progreso del entrenamiento.
grid_search = GridSearchCV(
    estimator=xgb_estimator,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=tscv,
    verbose=2,
    n_jobs=-1 # Usa todos los cores, pero puede ser intensivo. Pon 1 si da problemas.
)

# --- 4. Ejecutar la Búsqueda y Entrenamiento ---
print("\nIniciando la búsqueda de los mejores hiperparámetros con GridSearchCV...")
start_time = time.time()

grid_search.fit(X_train, y_train) # ¡OJO! Se entrena solo con el conjunto de entrenamiento

end_time = time.time()
print(f"Búsqueda completada en {((end_time - start_time) / 60):.2f} minutos.")

# Mostramos los mejores parámetros encontrados
print("\nMejores parámetros encontrados:")
print(grid_search.best_params_)
print(f"\nMejor score ROC AUC durante la validación cruzada: {grid_search.best_score_:.4f}")

# El mejor modelo ya está re-entrenado con los mejores parámetros sobre todo X_train
best_model = grid_search.best_estimator_

# --- 5. Evaluación del Modelo OPTIMIZADO en el conjunto de prueba ---
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])

print("\n--- Resultados del Modelo XGBoost OPTIMIZADO en el conjunto de prueba final ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")

print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=['Baja (0)', 'Sube (1)']))


model_filename = 'model/models/xgb_model.joblib'
print(f"\nGuardando el mejor modelo en '{model_filename}'...")
joblib.dump(best_model, model_filename)
print("Modelo guardado exitosamente.")