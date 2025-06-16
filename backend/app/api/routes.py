import asyncio
import joblib
from backend.app.schemas.schemas import PricePredictionResponse
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import pandas as pd

router = APIRouter()

@router.get("/ping")
def ping():
    return {"message": "pong"}

@router.websocket("/ws/prices")
async def stream_prices(websocket: WebSocket):
    """Stream simulated price data every minute over a WebSocket."""
    await websocket.accept()
    try:
        df = pd.read_parquet("model/clean/train_data_ml_binary.parquet")
        model = joblib.load("model/models/xgb_model.joblib")
    except FileNotFoundError:
        await websocket.close(code=1003)
        return

    features = df.drop(columns=["target_direction"])
    df["prediction"] = model.predict(features)

    INITIAL_CAPITAL = 10000.0
    TRADE_QUANTITY = 100
    COMMISSION_RATE = 0.001
    PREDICTION_HORIZON = 10

    capital = INITIAL_CAPITAL
    position = 0
    entry_price = 0
    entry_index = None
    operations = 0
    correct = 0

    for i, (_, row) in enumerate(df.iterrows()):
        try:
            if position == -1 and i - entry_index >= PREDICTION_HORIZON:
                exit_price = row["close"]
                profit_loss = (entry_price - exit_price) * TRADE_QUANTITY
                exit_commission = exit_price * TRADE_QUANTITY * COMMISSION_RATE
                capital += profit_loss - exit_commission
                operations += 1
                position = 0
                entry_price = 0
                entry_index = None

            if position == 0 and row["prediction"] == 0:
                entry_price = row["close"]
                entry_commission = entry_price * TRADE_QUANTITY * COMMISSION_RATE
                capital -= entry_commission
                position = -1
                entry_index = i

            if row["prediction"] == row["target_direction"]:
                correct += 1

            accuracy = correct / (i + 1)

            bid = float(row.get("close", 0) - row.get("spread", 0) / 2)
            ask = float(row.get("close", 0) + row.get("spread", 0) / 2)

            payload = PricePredictionResponse(
                bid=bid,
                ask=ask,
                will_go_up=int(row["prediction"]),
                earnings=capital - INITIAL_CAPITAL,
                operations=operations,
                accuracy=accuracy,
            )
            await websocket.send_json(payload.dict())
            await asyncio.sleep(60)
        except WebSocketDisconnect:
            break
        except Exception:
            break