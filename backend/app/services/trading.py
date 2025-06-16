import pandas as pd

def simulate_trading(
        df: pd.DataFrame, 
        model, 
        initial_capital=10000.0, 
        trade_quantity=100, 
        commission_rate=0.001, 
        prediction_horizon=10
    ) -> pd.DataFrame:
    
    features = df.drop(columns=["target_direction"])
    df["prediction"] = model.predict(features)

    capital = initial_capital
    position = 0
    entry_price = 0
    entry_index = None
    operations = 0
    correct = 0

    results = []

    for i, (_, row) in enumerate(df.iterrows()):
        if position == -1 and i - entry_index >= prediction_horizon:
            exit_price = row["close"]
            profit_loss = (entry_price - exit_price) * trade_quantity
            exit_commission = exit_price * trade_quantity * commission_rate
            capital += profit_loss - exit_commission
            operations += 1
            position = 0
            entry_price = 0
            entry_index = None

        if position == 0 and row["prediction"] == 0:
            entry_price = row["close"]
            entry_commission = entry_price * trade_quantity * commission_rate
            capital -= entry_commission
            position = -1
            entry_index = i

        if row["prediction"] == row["target_direction"]:
            correct += 1

        accuracy = correct / (i + 1)
        row_data = row.to_dict()
        row_data.update({
            "accuracy": accuracy,
            "capital": capital,
            "operations": operations,
        })
        results.append(row_data)

    return results
