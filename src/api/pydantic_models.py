from pydantic import BaseModel
import pandas as pd

class CustomerData(BaseModel):
    total_amount: float
    avg_amount: float
    transaction_count: int
    std_amount: float
    hour_median: float
    day_median: float
    month_median: float
    year_median: float

    def to_df(self):
        return pd.DataFrame([self.dict()])

class PredictionResponse(BaseModel):
    risk_probability: float
