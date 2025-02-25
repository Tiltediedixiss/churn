from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd

# Load the model and encoders at startup
try:
    with open("customer_churn_model.pkl", "rb") as model_file:
        model_data = pickle.load(model_file)
        model = model_data["model"]
        features = model_data["features_name"]

    with open("encoders.pkl", "rb") as enc_file:
        encoders = pickle.load(enc_file)
except Exception as e:
    raise RuntimeError(f"Error loading model or encoders: {e}")

# Initialize FastAPI app
app = FastAPI(title="Customer Churn Prediction API")

# Define the input data model using Pydantic
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

@app.post("/predict/")
def predict_churn(data: CustomerData):
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([data.dict()])

        # Apply label encoding where needed
        for column, encoder in encoders.items():
            if column in input_df.columns:
                input_df[column] = encoder.transform(input_df[column])

        # Reorder features to match training order
        input_df = input_df[features]

        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        return {
            "prediction": "Churn" if prediction == 1 else "No Churn",
            "probability": round(probability, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")
    