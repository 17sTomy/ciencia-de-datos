from pydantic import BaseModel

class PricePredictionResponse(BaseModel):
    bid: float
    ask: float
    will_go_up: int  # 1 = sube, 0 = baja
    earnings: float
    operations: int
    accuracy: float
    timestamp: str
