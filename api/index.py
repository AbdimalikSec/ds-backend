from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import joblib
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "student_model.joblib")
model = joblib.load(MODEL_PATH)

class Input(BaseModel):
    weekly_self_study_hours: float = Field(..., ge=0, le=40)
    attendance_percentage: float = Field(..., ge=0, le=100)
    class_participation: float = Field(..., ge=0, le=10)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: Input):
    X = [[
        data.weekly_self_study_hours,
        data.attendance_percentage,
        data.class_participation
    ]]
    pred = float(model.predict(X)[0])

    # clamp to 0–100
    pred = max(0.0, min(100.0, pred))

    return {"predicted_total_score": pred}
