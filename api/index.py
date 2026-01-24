from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

app = FastAPI()

# Load model ONCE when server starts
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "model",
    "student_model.joblib"
)

model = joblib.load(MODEL_PATH)

class Input(BaseModel):
    weekly_self_study_hours: float
    attendance_percentage: float
    class_participation: float

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
    prediction = model.predict(X)[0]
    return {"predicted_total_score": float(prediction)}
