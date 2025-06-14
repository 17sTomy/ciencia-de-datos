from backend.app.schemas.schemas import PricePredictionResponse
from fastapi import APIRouter

router = APIRouter()

@router.get("/ping")
def ping():
    return {"message": "pong"}

@router.get("/predict", response_model=PricePredictionResponse)
def predict_price():
    bid = 100.5
    ask = 101.0
    will_go_up = 1
    earnings: float

    return PricePredictionResponse(
        bid=bid, 
        ask=ask, 
        will_go_up=will_go_up, 
        earnings=earnings
    )