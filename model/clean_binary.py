import pandas as pd
import numpy as np

print("Iniciando script para CLASIFICACIÓN BINARIA (Sube/Baja)...")

# --- 1. Carga y Re-muestreo (Igual que antes) ---
df = pd.read_parquet("model/filter/all_data.parquet") # Usando el archivo unificado
df = df[df["SYMBOL"] == "A"].copy()
df.dropna(subset=['BB', 'BO', 'BBSize', 'BOSize'], inplace=True)
df.rename(columns={'BB': 'Bid', "BO": "Ask", "BBSize": "BidQty", "BOSize": "AskQty"}, inplace=True)
df["time_secs"] = df["TIME"].astype(float)
df["datetime"] = df["DATE"] + pd.to_timedelta(df["time_secs"], unit="s")
df = df.set_index("datetime").sort_index()

agg_rules = {'Bid': ['first', 'max', 'min', 'last'], 'Ask': ['first', 'max', 'min', 'last'], 'BidQty': 'sum', 'AskQty': 'sum', 'NUMEX': 'mean'}
df_resampled = df.resample('1min').agg(agg_rules)
df_resampled.columns = ['_'.join(col).strip() for col in df_resampled.columns.values]
df_resampled.dropna(inplace=True)

# --- 2. Ingeniería de Características (Igual que antes) ---
df_resampled['open'] = (df_resampled['Bid_first'] + df_resampled['Ask_first']) / 2
df_resampled['high'] = (df_resampled['Bid_max'] + df_resampled['Ask_max']) / 2
df_resampled['low'] = (df_resampled['Bid_min'] + df_resampled['Ask_min']) / 2
df_resampled['close'] = (df_resampled['Bid_last'] + df_resampled['Ask_last']) / 2
df_resampled['spread'] = df_resampled['Ask_last'] - df_resampled['Bid_last']
df_resampled['imbalance'] = (df_resampled['BidQty_sum'] - df_resampled['AskQty_sum']) / (df_resampled['BidQty_sum'] + df_resampled['AskQty_sum'])
df_resampled['volume'] = df_resampled['BidQty_sum'] + df_resampled['AskQty_sum']

windows = [5, 10, 30]
for window in windows:
    df_resampled[f'imbalance_ma_{window}'] = df_resampled['imbalance'].rolling(window=window).mean()
    df_resampled[f'volume_ma_{window}'] = df_resampled['volume'].rolling(window=window).mean()
    df_resampled[f'volatility_{window}'] = df_resampled['close'].rolling(window=window).std()

# --- 3. Creación del Target BINARIO ---
prediction_horizon = 10
df_resampled['close_future'] = df_resampled['close'].shift(-prediction_horizon)
df_resampled['target_price_change'] = df_resampled['close_future'] - df_resampled['close']

threshold = 0.01 # Umbral para definir un movimiento significativo

def assign_binary_direction(change):
    if change > threshold:
        return 1  # Sube
    elif change < -threshold:
        return 0  # Baja
    else:
        return np.nan # Marcamos los estables como NaN para eliminarlos

df_resampled['target_direction'] = df_resampled['target_price_change'].apply(assign_binary_direction)

# --- 4. Limpieza Final y Guardado ---
df_final = df_resampled.dropna() # Esto elimina las filas marcadas como NaN (estables)

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

df_final.to_parquet("model/clean/train_data_ml_binary.parquet")
print(f"\nProceso BINARIO finalizado. Archivo guardado con {df_final.shape[0]} filas.")