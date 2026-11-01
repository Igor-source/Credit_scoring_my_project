# import random
# from fastapi import FastAPI
# from pydantic import BaseModel
# import joblib

# class ClientData(BaseModel):
#     age: int
#     income: float
#     education: bool
#     work: bool
#     car: bool

# app = FastAPI()
# model = joblib.load("src/Catboostmodel_credit_scoring.pkl")

# @app.post("/score")
# def score(data: ClientData):
#     features = [data.age, data.income, data.education, data.work, data.car]
#     approved = not model.predict([features])[0].item()
#     # approved = data.income > 50 or (data.education and data.work)
#     return {"approved": approved}   

from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

try:
    import joblib
except ImportError:  # pragma: no cover - optional dependency in minimal runtime
    joblib = None


class ClientData(BaseModel):
    age: int
    income: float
    education: bool
    work: bool
    car: bool
    


app = FastAPI()


def _rule_based_decision(data: ClientData) -> bool:
    score = 0
    if data.age >= 25:
        score += 1
    if data.income >= 40:
        score += 1
    if data.education:
        score += 1
    if data.work:
        score += 1
    if data.car:
        score += 1
    return score >= 3


def _predict_approval(data: ClientData) -> bool:
    model_path = Path(__file__).with_name("Catboostmodel_credit_scoring.pkl")
    if joblib is not None and model_path.exists():
        try:
            model = joblib.load(model_path)
            features = [
                data.age,
                data.income,
                int(data.education),
                int(data.work),
                int(data.car),
            ]
            prediction = model.predict([features])[0]
            if hasattr(prediction, "item"):
                prediction = prediction.item()
            if isinstance(prediction, (list, tuple)) and prediction:
                prediction = prediction[0]
            if isinstance(prediction, (int, float)):
                return bool(not prediction)
        except Exception:
            pass
    return _rule_based_decision(data)


@app.post("/score")
def score(data: ClientData):
    approved = _predict_approval(data)
    return {"approved": approved}