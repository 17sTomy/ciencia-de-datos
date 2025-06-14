# clean_resampled_1min.py

import pandas as pd
import numpy as np

print("Iniciando script de limpieza y re-muestreo a 1 minuto...")

# --- 1. Carga y Limpieza Inicial ---
df = pd.read_parquet("model/train_data.parquet")
df.rename(columns={'BB': 'Bid', "BO": "Ask", "BBSize": "BidQty", "BOSize": "AskQty"}, inplace=True)
df = df[df["SYMBOL"] == "A"].copy()
df.dropna(subset=['Bid', 'Ask', 'BidQty', 'AskQty'], inplace=True)

# --- 2. Creación del Índice de Tiempo ---
df["time_secs"] = df["TIME"].astype(float)
df["datetime"] = df["DATE"] + pd.to_timedelta(df["time_secs"], unit="s")
df = df.set_index("datetime").sort_index()

# --- 3. RE-MUESTREO (RESAMPLING) A 1 MINUTO ---
print("Agregando datos tick a frecuencia de 1 minuto...")

# Reglas de agregación para crear velas OHLC (Open, High, Low, Close) y volúmenes
agg_rules = {
    'Bid': ['first', 'max', 'min', 'last'],
    'Ask': ['first', 'max', 'min', 'last'],
    'BidQty': 'sum',
    'AskQty': 'sum',
    'NUMEX': 'mean'
}

# Aplicamos el resample con el código '1T' para 1 minuto
df_resampled = df.resample('1T').agg(agg_rules)
# Aplanamos los nombres de las columnas (ej. de ('Bid', 'first') a 'Bid_first')
df_resampled.columns = ['_'.join(col).strip() for col in df_resampled.columns.values]
df_resampled.dropna(inplace=True) # Eliminamos minutos en los que no hubo ni un tick

# --- 4. Ingeniería de Características sobre Datos de 1 Minuto ---
print("Calculando features para datos de 1 minuto...")

# a) Precios OHLC para el punto medio (mid-price)
df_resampled['open'] = (df_resampled['Bid_first'] + df_resampled['Ask_first']) / 2
df_resampled['high'] = (df_resampled['Bid_max'] + df_resampled['Ask_max']) / 2
df_resampled['low'] = (df_resampled['Bid_min'] + df_resampled['Ask_min']) / 2
df_resampled['close'] = (df_resampled['Bid_last'] + df_resampled['Ask_last']) / 2

# b) Spread, Imbalance y otros features basados en los datos del minuto
df_resampled['spread'] = df_resampled['Ask_last'] - df_resampled['Bid_last']
df_resampled['imbalance'] = (df_resampled['BidQty_sum'] - df_resampled['AskQty_sum']) / (df_resampled['BidQty_sum'] + df_resampled['AskQty_sum'])
df_resampled['volume'] = df_resampled['BidQty_sum'] + df_resampled['AskQty_sum']

# c) Features de ventana móvil (ahora las ventanas son minutos, no ticks)
windows = [5, 10, 30] # Ventanas de 5, 10, 30 minutos
for window in windows:
    df_resampled[f'imbalance_ma_{window}'] = df_resampled['imbalance'].rolling(window=window).mean()
    df_resampled[f'volume_ma_{window}'] = df_resampled['volume'].rolling(window=window).mean()
    df_resampled[f'volatility_{window}'] = df_resampled['close'].rolling(window=window).std()

# --- 5. Creación del Target de Clasificación ---
print("Creando target para clasificación...")

prediction_horizon = 10 # Horizonte de predicción: 10 minutos en el futuro
df_resampled['close_future'] = df_resampled['close'].shift(-prediction_horizon)
df_resampled['target_price_change'] = df_resampled['close_future'] - df_resampled['close']

# El threshold probablemente deba ser mayor para movimientos de minutos. ¡Experimenta con este valor!
threshold = 0.01

def assign_direction(change):
    if change > threshold: return 2  # Sube
    elif change < -threshold: return 0  # Baja
    else: return 1  # Estable

df_resampled['target_direction'] = df_resampled['target_price_change'].apply(assign_direction)

# --- 6. Limpieza Final y Guardado ---
df_final = df_resampled.dropna()

# Seleccionamos los features que creamos para nuestros datos de minutos
feature_columns = [
    'open', 'high', 'low', 'close', 'volume', 'spread', 'imbalance',
    'imbalance_ma_5', 'imbalance_ma_10', 'imbalance_ma_30',
    'volume_ma_5', 'volume_ma_10', 'volume_ma_30',
    'volatility_5', 'volatility_10', 'volatility_30',
    'NUMEX_mean'
]
target_column = 'target_direction'

final_columns = [col for col in feature_columns if col in df_final.columns] + [target_column]
df_final = df_final[final_columns]

# Guardamos en un nuevo archivo para no sobreescribir el anterior
df_final.to_parquet("model/train_data_ml_classification_1min.parquet")

print(f"\nProceso finalizado. Archivo de 1 minuto guardado con {df_final.shape[0]} filas.")
print("Columnas finales:", final_columns)