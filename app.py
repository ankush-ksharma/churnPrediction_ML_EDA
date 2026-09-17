from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import uvicorn

app = FastAPI(title="Churn Prediction API")

try:
    model = joblib.load("model/churn_model.pkl")
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.post("/predict")
def predict_chrun(customer: CustomerData):
    try:
        data = pd.DataFrame([customer.dict()])

        # Apply the same feature engineering used to train the pipeline
        service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        data['Service_Count'] = data[service_cols].apply(lambda x: (x == 'Yes').sum(), axis=1)

        data['AutoPayment'] = data['PaymentMethod'].isin(
            ['Bank transfer (automatic)', 'Credit card (automatic)']
        ).astype(int)

        data['FamilyCustomer'] = ((data['Partner'] == 'Yes') | (data['Dependents'] == 'Yes')).astype(int)

        data['CustomerType'] = pd.cut(
            data['tenure'],
            bins=[-1, 12, 36, float('inf')],
            labels=['New', 'Regular', 'Loyal']
        )

        support_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
        data['Support_Count'] = data[support_cols].apply(lambda x: (x == 'Yes').sum(), axis=1)

        # Predict
        prediction_val = model.predict(data)[0]
        prediction_prob = model.predict_proba(data)[0][1]

        return {
            "prediction": "Yes" if prediction_val == 1 else "No",
            "churn_probability": round(float(prediction_prob), 4)
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)