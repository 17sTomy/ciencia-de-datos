# clean.py - SCRIPT COMPLETO Y ACTUALIZADO

import pandas as pd
import numpy as np

# --- 1. Carga y Limpieza Inicial ---
df = pd.read_parquet("model/train_data.parquet")
df.rename(columns={
    'BB': 'Bid', "BO": "Ask", "BBSize": "BidQty", "BOSize": "AskQty",
    "BBASize": "BidTotalQty", "BOASize": "AskTotalQty"
}, inplace=True)
df = df[df["SYMBOL"] == "A"].copy()
df.dropna(subset=['Bid', 'Ask', 'BidQty', 'AskQty'], inplace=True)

# --- 2. Creación del Índice de Tiempo y Features Base ---
df["time_secs"] = df["TIME"].astype(float)
df["datetime"] = df["DATE"] + pd.to_timedelta(df["time_secs"], unit="s")
df = df.set_index("datetime").sort_index()
df['mid'] = (df['Bid'] + df['Ask']) / 2
df['spread'] = df['Ask'] - df['Bid']

# --- 3. Ingeniería de Características Avanzada ---
print("Iniciando ingeniería de características avanzada...")
df['imbalance'] = (df['BidQty'] - df['AskQty']) / (df['BidQty'] + df['AskQty'])
df['bid_ask_qty_ratio'] = df['BidQty'] / df['AskQty']
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df['bid_ask_qty_ratio'].fillna(method='ffill', inplace=True)
epsilon = 1e-6
df['bid_depth_pressure'] = df['BidQty'] / (df['BidTotalQty'] + epsilon)
df['ask_depth_pressure'] = df['AskQty'] / (df['AskTotalQty'] + epsilon)
windows = [5, 10, 30]
for window in windows:
    df[f'imbalance_ma_{window}'] = df['imbalance'].rolling(window=window).mean()
    df[f'spread_ma_{window}'] = df['spread'].rolling(window=window).mean()
    df[f'volatility_{window}'] = df['mid'].rolling(window=window).std()
df['log_return_1'] = np.log(df['mid'] / df['mid'].shift(1))
df['imbalance_diff_1'] = df['imbalance'].diff(1)
print("Ingeniería de características completada.")

# --- 4. Creación de la Variable Objetivo (TARGET) ---

# Nombramos esta sección de forma diferente porque es la parte clave que cambiamos
print("Creando target para CLASIFICACIÓN...")
prediction_horizon = 10
df['mid_future'] = df.groupby(df.index.date)['mid'].shift(-prediction_horizon)
df['target_price_change'] = df['mid_future'] - df['mid']

# ¡NUEVO! Convertimos el cambio de precio en clases
# Este `threshold` es un parámetro muy importante con el que puedes experimentar.
# Define qué tan grande debe ser un movimiento para no considerarlo "estable".
threshold = 0.005 

def assign_direction(change):
    """
    Asigna una clase numérica a un cambio de precio.
    - 2: Sube (movimiento positivo significativo)
    - 1: Estable (movimiento pequeño o nulo)
    - 0: Baja (movimiento negativo significativo)
    Usamos 0, 1, 2 porque es lo que XGBoost espera.
    """
    if change > threshold:
        return 2  # Sube
    elif change < -threshold:
        return 0  # Baja
    else:
        return 1  # Estable

# Aplicamos la función para crear nuestro target final
df['target_direction'] = df['target_price_change'].apply(assign_direction)


# --- 5. Limpieza Final y Guardado ---
df_ml = df.dropna()
feature_columns = [
    'imbalance', 'bid_ask_qty_ratio', 'bid_depth_pressure', 'ask_depth_pressure', 'spread',
    'imbalance_ma_5', 'imbalance_ma_10', 'imbalance_ma_30', 'spread_ma_5',
    'spread_ma_10', 'spread_ma_30', 'volatility_5', 'volatility_10',
    'volatility_30', 'log_return_1', 'imbalance_diff_1', 'NUMEX'
]
# Ahora guardamos nuestro nuevo target de clasificación
target_column = 'target_direction'
final_columns = [col for col in feature_columns if col in df_ml.columns] + [target_column]
df_final = df_ml[final_columns]
df_final.to_parquet("model/train_data_ml_classification.parquet")

print("\nProceso finalizado. Archivo para clasificación guardado.")