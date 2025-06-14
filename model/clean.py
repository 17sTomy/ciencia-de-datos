import pandas as pd
import numpy as np

# --- 1. Carga y Limpieza Inicial ---

# Cargamos el dataframe completo para aprovechar todas las columnas
df = pd.read_parquet("model/train_data.parquet")

# Renombramos columnas para mayor claridad
df.rename(columns={
    'BB': 'Bid', "BO": "Ask", 
    "BBSize": "BidQty", "BOSize": "AskQty",
    "BBASize": "BidTotalQty", "BOASize": "AskTotalQty" # Asumiendo estos nombres del schema
}, inplace=True)

# Filtramos por un único símbolo para empezar
df = df[df["SYMBOL"] == "A"].copy() # Usamos .copy() para evitar SettingWithCopyWarning

# Eliminamos filas donde la información de cotización básica es nula
df.dropna(subset=['Bid', 'Ask', 'BidQty', 'AskQty'], inplace=True)

# --- 2. Creación del Índice de Tiempo y Features Base ---

# Crear un índice de tiempo es fundamental para las operaciones de ventana móvil
df["time_secs"] = df["TIME"].astype(float)
df["datetime"] = df["DATE"] + pd.to_timedelta(df["time_secs"], unit="s")
df = df.set_index("datetime").sort_index() # Ordenar el índice es crucial

# Features base que ya tenías
df['mid'] = (df['Bid'] + df['Ask']) / 2
df['spread'] = df['Ask'] - df['Bid']

# --- 3. Ingeniería de Características Avanzada ---

print("Iniciando ingeniería de características avanzada...")

# a) Features de Imbalance y Ratios
df['imbalance'] = (df['BidQty'] - df['AskQty']) / (df['BidQty'] + df['AskQty'])
df['bid_ask_qty_ratio'] = df['BidQty'] / df['AskQty']
# Reemplazamos infinitos que puedan surgir si AskQty es 0
df.replace([np.inf, -np.inf], np.nan, inplace=True) 
df['bid_ask_qty_ratio'].fillna(method='ffill', inplace=True) # Llenamos NaNs con el valor anterior

# b) Features de Profundidad del Mercado (usando columnas adicionales)
# Asumimos que BidTotalQty y AskTotalQty no son cero para evitar divisiones.
# Una pequeña constante (epsilon) evita errores de división por cero.
epsilon = 1e-6 
df['bid_depth_pressure'] = df['BidQty'] / (df['BidTotalQty'] + epsilon)
df['ask_depth_pressure'] = df['AskQty'] / (df['AskTotalQty'] + epsilon)

# c) Features de Ventana Móvil (para capturar tendencias y suavizar ruido)
windows = [5, 10, 30]
for window in windows:
    df[f'imbalance_ma_{window}'] = df['imbalance'].rolling(window=window).mean()
    df[f'spread_ma_{window}'] = df['spread'].rolling(window=window).mean()
    df[f'volatility_{window}'] = df['mid'].rolling(window=window).std() # Volatilidad

# d) Features de Momento/Aceleración
df['log_return_1'] = np.log(df['mid'] / df['mid'].shift(1)) # Retorno logarítmico tick a tick
df['imbalance_diff_1'] = df['imbalance'].diff(1) # Cambio en el imbalance

print("Ingeniería de características completada.")

# --- 4. Creación de la Variable Objetivo (Target) ---

# Horizonte de predicción en ticks. Experimenta cambiando este valor (e.g., a 1, 5, 20)
prediction_horizon = 10 

# El objetivo es predecir el cambio en el precio 'N' ticks en el futuro.
# Usamos `groupby` por día para no mezclar datos entre días.
df['mid_future'] = df.groupby(df.index.date)['mid'].shift(-prediction_horizon)

# Nuestro target 'y' será el cambio de precio
df['target_price_change'] = df['mid_future'] - df['mid']

# --- 5. Limpieza Final y Guardado ---

# Eliminar todas las filas que tengan algún NaN generado durante la creación de features
# Esto es importante porque las ventanas móviles y los shifts crean NaNs al principio
df_ml = df.dropna()

# Definimos las columnas que serán features para el modelo
feature_columns = [
    'imbalance', 'bid_ask_qty_ratio', 'bid_depth_pressure', 'ask_depth_pressure', 'spread',
    'imbalance_ma_5', 'imbalance_ma_10', 'imbalance_ma_30',
    'spread_ma_5', 'spread_ma_10', 'spread_ma_30',
    'volatility_5', 'volatility_10', 'volatility_30',
    'log_return_1', 'imbalance_diff_1', 'NUMEX'
]

# Columnas que necesitamos para el target y la evaluación
utility_and_target_columns = ['mid', 'mid_future', 'target_price_change']

# Nos aseguramos de que todas las columnas seleccionadas existan en el DataFrame
final_columns = [col for col in feature_columns if col in df_ml.columns] + utility_and_target_columns
df_final = df_ml[final_columns]

# Guardamos el dataframe final listo para el entrenamiento
df_final.to_parquet("model/train_data_ml_enhanced.parquet")

print("\nProceso finalizado.")
print(f"DataFrame con features mejoradas guardado en 'model/train_data_ml_enhanced.parquet'")
print(f"Dimensiones del DataFrame final: {df_final.shape}")
print("Columnas finales:")
print(df_final.columns.tolist())