# backtest.py

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# --- 1. Configuración del Backtest ---
print("Iniciando simulación de backtesting...")

INITIAL_CAPITAL = 10000.0  # Capital inicial de la simulación
TRADE_QUANTITY = 100       # Número de acciones a vender en cada operación
COMMISSION_RATE = 0.001    # Comisión por operación (0.1%), se aplica al entrar y al salir
PREDICTION_HORIZON = 10    # Nuestro horizonte en minutos

# --- 2. Carga de Datos y Modelo ---

# Cargamos los datos que contienen los features Y los precios de mercado (close)
# Usamos el mismo set de prueba que en el entrenamiento para una comparación justa
try:
    full_df = pd.read_parquet("model/clean/train_data_ml_binary.parquet")
    model = joblib.load('model/models/xgb_model.joblib')
except FileNotFoundError as e:
    print(f"Error: No se encontró el archivo necesario. Asegúrate de que 'train_data_ml_binary.parquet' y 'xgb_model.joblib' existan.")
    print(e)
    exit()

# Separamos los datos de prueba (los mismos que usamos para evaluar el modelo)
test_set_size = int(len(full_df) * 0.2)
test_df = full_df.iloc[-test_set_size:]

X_test = test_df.drop(columns=['target_direction'])
y_pred = model.predict(X_test)
test_df['prediction'] = y_pred

# --- 3. Bucle de Simulación ---

capital = INITIAL_CAPITAL
position = 0  # 0: sin posición, -1: en posición de venta (short)
entry_price = 0
entry_time_index = None
trade_log = []
equity_curve = [INITIAL_CAPITAL] # Lista para registrar la evolución del capital

print(f"Simulando sobre {len(test_df)} minutos de datos de prueba...")

for i in range(len(test_df)):
    current_row = test_df.iloc[i]
    current_time = test_df.index[i]
    current_price = current_row['close']

    # --- Lógica de Salida ---
    # Si estamos en una posición, verificamos si es hora de salir
    if position == -1:
        time_in_position = i - entry_time_index
        if time_in_position >= PREDICTION_HORIZON:
            # Cerramos la posición
            exit_price = current_price
            
            # Calculamos P&L para una posición corta
            profit_loss = (entry_price - exit_price) * TRADE_QUANTITY
            
            # Aplicamos comisión de salida
            exit_commission = exit_price * TRADE_QUANTITY * COMMISSION_RATE
            net_pnl = profit_loss - exit_commission
            capital += net_pnl
            
            trade_log.append({
                "entry_time": entry_time,
                "exit_time": current_time,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": net_pnl
            })
            
            # Reseteamos el estado de la posición
            position = 0
            entry_price = 0
            entry_time_index = None

    # --- Lógica de Entrada ---
    # Si no estamos en una posición, verificamos si hay una señal para entrar
    if position == 0:
        if current_row['prediction'] == 0: # Señal de "Baja"
            # Abrimos posición de venta (short)
            entry_price = current_price
            
            # Aplicamos comisión de entrada
            entry_commission = entry_price * TRADE_QUANTITY * COMMISSION_RATE
            capital -= entry_commission
            
            position = -1
            entry_time = current_time
            entry_time_index = i

    # Registramos el capital al final de cada minuto para la curva de equidad
    equity_curve.append(capital)


# --- 4. Análisis de Resultados ---
print("\n--- Resultados del Backtest ---")

total_pnl = capital - INITIAL_CAPITAL
total_return_pct = (total_pnl / INITIAL_CAPITAL) * 100
n_trades = len(trade_log)

if n_trades > 0:
    wins = [t for t in trade_log if t['pnl'] > 0]
    losses = [t for t in trade_log if t['pnl'] <= 0]
    win_rate = (len(wins) / n_trades) * 100 if n_trades > 0 else 0
    avg_win = sum(t['pnl'] for t in wins) / len(wins) if len(wins) > 0 else 0
    avg_loss = sum(t['pnl'] for t in losses) / len(losses) if len(losses) > 0 else 0

    print(f"Capital Final: ${capital:,.2f}")
    print(f"Ganancia/Pérdida Total: ${total_pnl:,.2f}")
    print(f"Retorno Total: {total_return_pct:.2f}%")
    print(f"Número Total de Operaciones: {n_trades}")
    print(f"Tasa de Acierto (Win Rate): {win_rate:.2f}%")
    print(f"Ganancia Promedio por Operación Ganadora: ${avg_win:,.2f}")
    print(f"Pérdida Promedio por Operación Perdedora: ${avg_loss:,.2f}")
else:
    print("No se ejecutó ninguna operación según la estrategia.")

# --- 5. Visualización ---
if n_trades > 0:
    plt.figure(figsize=(12, 6))
    # Usamos `equity_curve[:-1]` para que coincida con el tamaño del índice de tiempo del test_df
    pd.Series(equity_curve[:-1], index=test_df.index).plot()
    plt.title('Curva de Equidad de la Estrategia "Francotirador"')
    plt.xlabel('Fecha y Hora')
    plt.ylabel('Capital')
    plt.grid(True)
    plt.tight_layout()
    plt.show()