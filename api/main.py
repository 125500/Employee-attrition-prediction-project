"""FastAPI app.  Run:  uvicorn api.main:app --reload"""
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src import config

app = FastAPI(title="Employee Attrition Prediction API")

FRONTEND = config.ROOT / "index.html"

try:
    pipeline = joblib.load(config.MODEL_PATH)
except FileNotFoundError:
    pipeline = None


class Employee(BaseModel):
    Age: int
    BusinessTravel: str
    DailyRate: int
    Department: str
    DistanceFromHome: int
    Education: int
    EducationField: str
    EnvironmentSatisfaction: int
    Gender: str
    HourlyRate: int
    JobInvolvement: int
    JobLevel: int
    JobRole: str
    JobSatisfaction: int
    MaritalStatus: str
    MonthlyIncome: int
    MonthlyRate: int
    NumCompaniesWorked: int
    OverTime: str
    PercentSalaryHike: int
    PerformanceRating: int
    RelationshipSatisfaction: int
    StockOptionLevel: int
    TotalWorkingYears: int
    TrainingTimesLastYear: int
    WorkLifeBalance: int
    YearsAtCompany: int
    YearsInCurrentRole: int
    YearsSinceLastPromotion: int
    YearsWithCurrManager: int


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(FRONTEND)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": pipeline is not None}


@app.post("/predict")
def predict(emp: Employee):
    if pipeline is None:
        raise HTTPException(503, "Model not found. Run `python -m src.train` first.")
    df = pd.DataFrame([emp.model_dump()])
    prob = float(pipeline.predict_proba(df)[0, 1])
    level = "High" if prob >= 0.6 else "Medium" if prob >= 0.4 else "Low"
    return {"attrition_probability": round(prob, 3), "risk_level": level}
