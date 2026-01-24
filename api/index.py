from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Input(BaseModel):
    weekly_self_study_hours: float
    attendance_percentage: float
    class_participation: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: Input):
    predicted_score = (
        data.weekly_self_study_hours * 2
        + data.attendance_percentage * 0.5
        + data.class_participation * 5
    )
    return {"predicted_total_score": predicted_score}
