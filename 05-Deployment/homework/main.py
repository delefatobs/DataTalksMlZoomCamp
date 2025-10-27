import pickle
from typing import Literal
from pydantic import BaseModel, Field
from fastapi import FastAPI
import uvicorn
import os


# -----------------------------
# Load pipeline
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "pipeline_v2.bin")

with open(MODEL_PATH, "rb") as f_in:
    pipeline = pickle.load(f_in)


# -----------------------------
# Input schema
# -----------------------------
class Client(BaseModel):
    lead_source: Literal["paid_ads", "organic_search", "referral", "other"]
    number_of_courses_viewed: int = Field(..., ge=0)
    annual_income: float = Field(..., ge=0.0)


# -----------------------------
# Response schema
# -----------------------------
class PredictResponse(BaseModel):
    subscription_probability: float
    subscribed: bool


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="subscription-prediction")


def predict_single(client):
    X = [client]
    result = pipeline.predict_proba(X)[0, 1]
    return float(result)


@app.get("/")
def root():
    return {"message": "Model API is running"}


@app.post("/predict", response_model=PredictResponse)
def predict(client: Client) -> PredictResponse:
    prob = predict_single(client.model_dump())

    return PredictResponse(
        subscription_probability=prob,
        subscribed=prob >= 0.5
    )


# -----------------------------
# Run with uvicorn (if run as script)
# -----------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


