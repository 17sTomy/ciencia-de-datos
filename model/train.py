# train.py - SCRIPT ACTUALIZADO PARA CLASIFICACIÓN

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_sample_weight

# --- 1. Carga de Datos ---
df = pd.read_parquet("model/train_data_ml_classification_1min.parquet")

# --- 2. Definición de Features y Target ---
y = df['target_direction']
X = df.drop(columns=['target_direction'])
print(f"Entrenando con {len(X.columns)} features.")

# División temporal
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# --- 3. Entrenamiento del Modelo XGBoost Classifier ---

# ¡CAMBIO DE MODELO! Usamos XGBClassifier
xgb_classifier = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=3,
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50,
    eval_metric='mlogloss'
)
print("\nEntrenando modelo XGBoost para CLASIFICACIÓN con PONDERACIÓN DE CLASES...")

# ¡NUEVO! Calculamos los pesos para las muestras de entrenamiento
# Esto le dará más importancia a las clases minoritarias


X_train_part, X_val, y_train_part, y_val = train_test_split(X_train, y_train, test_size=0.1, shuffle=False)

sample_weights = compute_sample_weight(
    class_weight='balanced',
    y=y_train_part
)

# Usamos 'mlogloss' como métrica de evaluación para clasificación
xgb_classifier.fit(
    X_train_part, y_train_part,
    sample_weight=sample_weights,
    eval_set=[(X_val, y_val)],
    verbose=False
)

print("Entrenamiento completado.")
y_pred = xgb_classifier.predict(X_test)

# --- 4. Evaluación del Modelo de Clasificación ---

# a) Baseline: Predecir siempre la clase más frecuente
baseline_prediction = y_train.mode()[0]
baseline_accuracy = accuracy_score(y_test, np.full_like(y_test, fill_value=baseline_prediction))
print("\n--- Resultados del Modelo Baseline (Clase más frecuente) ---")
print(f"Clase más frecuente: {baseline_prediction} (0:Baja, 1:Estable, 2:Sube)")
print(f"Accuracy Baseline: {baseline_accuracy:.4f}")

# b) Modelo XGBoost
accuracy_xgb = accuracy_score(y_test, y_pred)
print("\n--- Resultados del Modelo XGBoost ---")
print(f"Accuracy: {accuracy_xgb:.4f}")

# c) Reporte detallado y Matriz de Confusión
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=['Baja (0)', 'Estable (1)', 'Sube (2)']))

print("Matriz de Confusión:")
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['Baja', 'Estable', 'Sube'], cmap='Blues')
plt.show()

# --- 5. Análisis Detallado de Predicciones ---
df_eval = pd.DataFrame({
    'direction_true': y_test,
    'direction_pred': y_pred
}, index=X_test.index)
print("\n--- Comparación de Predicciones (últimas 30) ---")
print(df_eval.tail(30))