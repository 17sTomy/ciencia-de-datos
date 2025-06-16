import asyncio
import joblib
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import pandas as pd
from app.services.trading import simulate_trading
from app.schemas.schemas import PricePredictionResponse

router = APIRouter()

@router.get("/ping")
def ping():
    return {"message": "pong"}

@router.websocket("/ws/prices")
async def stream_prices(websocket: WebSocket):
    print("Connected")
    await websocket.accept()
    try:
        df = pd.read_parquet('model\clean\train_data_ml_binary.parquet')
        model = joblib.load('model\models\xgb_model.joblib')
    except FileNotFoundError as e:
        await websocket.close(code=1003)
        return

    try:
        results = simulate_trading(df, model)
        for row in results:
            bid = float(row.get("close", 0) - row.get("spread", 0) / 2)
            ask = float(row.get("close", 0) + row.get("spread", 0) / 2)

            payload = PricePredictionResponse(
                bid=bid,
                ask=ask,
                will_go_up=int(row["prediction"]),
                earnings=row["capital"],
                operations=row["operations"],
                accuracy=row["accuracy"],
                timestamp=row["timestamp"],
            )
            print(payload.dict())
            await websocket.send_json(payload.dict())
            await asyncio.sleep(3)
    except WebSocketDisconnect:
        print("WebSocket disconnected.")
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})