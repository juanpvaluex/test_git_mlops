from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import pandas as pd
import boto3
import json
import os
import joblib

app = FastAPI()

# model = mlflow.pyfunc.load_model("runs:/feb259a73e444a83be10fc87cb70bc5e/model")
model = joblib.load("pipeline_model/model.pkl")

# Cliente S3
s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AKIAS252VX4ZKWBC5Z57"),
    aws_secret_access_key=os.getenv("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"),
    region_name="us-east-2"
)

BUCKET_NAME = "jbo-0508-mlops"
FILE_NAME = "predicciones.json"

# Esquema de entrada
class InputData(BaseModel):
    StudentID: int
    Age: int
    Gender: int
    Ethnicity: int
    ParentalEducation: int
    StudyTimeWeekly: float
    Absences: int
    Tutoring: int
    ParentalSupport: int
    Extracurricular: int
    Sports: int
    Music: int
    Volunteering: int
    GPA: float

@app.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([data.dict()])

    pred = model.predict(df)
    result = {
        "prediction": float(pred[0])
    }

    with open(FILE_NAME, "w") as f:
        json.dump(result, f)

    s3.upload_file(FILE_NAME, BUCKET_NAME, FILE_NAME)

    return {
        "prediction": float(pred[0]),
        "s3_path": f"s3://{BUCKET_NAME}/{FILE_NAME}"
    }

