import random
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

class ClientData(BaseModel):
    age: int
    monthly_income: float  
    employment_type: str
    months_at_job: int
    education: str
    loan_amount: float
    loan_term_months: int
    dependents: int
    region: str
    requested_product: str

app = FastAPI()
model = joblib.load("logistic_regression_credit_scoring.pkl")

@app.post("/score")
def score(data: ClientData):
      # Формируем DataFrame с названиями признаков
    features = pd.DataFrame([{
        "age": data.age,
        "monthly_income": data.monthly_income,
        "employment_type": data.employment_type,
        "months_at_job": data.months_at_job,
        "education": data.education,
        "loan_amount": data.loan_amount,
        "loan_term_months": data.loan_term_months,
        "dependents": data.dependents,
        "region": data.region,
        "requested_product": data.requested_product
    }])

    # Получаем предсказанный класс
    prediction = int(model.predict(features)[0])

    # Получаем вероятность дефолта (класс 1)
    default_probability = float(
        model.predict_proba(features)[0, 1]
    )

    # Если класс 0 — одобряем, если 1 — отказываем
    approved = prediction == 0

    return {
        "approved": approved,
        "prediction": prediction,
        "default_probability": round(default_probability, 4)
    }
   

# from pathlib import Path

# from fastapi import FastAPI
# from pydantic import BaseModel

# try:
#     import joblib
# except ImportError:  # pragma: no cover - optional dependency in minimal runtime
#     joblib = None


# class ClientData(BaseModel):
#     age: int
#     monthly_income: float  
#     employment_type: str
#     months_at_job: int
#     education: str
#     loan_amount: float
#     loan_term_months: int
#     dependents: int
#     region: str
#     requested_product: str



# app = FastAPI()


# def _rule_based_decision(data: ClientData) -> bool:
#     score = 0
#     if data.age >= 25:
#         score += 1
#     if data.income >= 40:
#         score += 1
#     if data.education:
#         score += 1
#     if data.work:
#         score += 1
#     if data.car:
#         score += 1
#     return score >= 3


# def _predict_approval(data: ClientData) -> bool:
#     model_path = Path(__file__).with_name("Catboostmodel_credit_scoring.pkl")
#     if joblib is not None and model_path.exists():
#         try:
#             model = joblib.load(model_path)
#             features = [
#                 data.age,
#                 data.income,
#                 int(data.education),
#                 int(data.work),
#                 int(data.car),
#             ]
#             prediction = model.predict([features])[0]
#             if hasattr(prediction, "item"):
#                 prediction = prediction.item()
#             if isinstance(prediction, (list, tuple)) and prediction:
#                 prediction = prediction[0]
#             if isinstance(prediction, (int, float)):
#                 return bool(not prediction)
#         except Exception:
#             pass
#     return _rule_based_decision(data)


# @app.post("/score")
# def score(data: ClientData):
#     approved = _predict_approval(data)
#     return {"approved": approved}