from fastapi import FastAPI
from src.api.pydantic_models import CustomerData, PredictionResponse
import mlflow.pyfunc

app = FastAPI()

model = mlflow.pyfunc.load_model(model_uri="models:/credit_risk_model/Production")

@app.get("/")
def read_root():
    return {"message": "Credit Risk Model API is running"}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerData):
    input_df = data.to_df()
    prediction = model.predict(input_df)
    return PredictionResponse(risk_probability=prediction[0])
