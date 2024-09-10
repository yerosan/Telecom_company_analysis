import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd

# Create a FastAPI app instance
app = FastAPI()

# Load the trained model
model = joblib.load("model.pkl")

# Define a request body model
class UserData(BaseModel):
    engagement_score: float
    experience_score: float

class MultipleUserData(BaseModel):
    data: List[UserData]

# Define the prediction route
@app.post("/predict/")
def predict(data: MultipleUserData):
    # Convert the incoming data into a pandas DataFrame
    df = pd.DataFrame([user.dict() for user in data.data])

    # Make the prediction
    predictions = model.predict(df)
    
    return {"satisfaction_score": predictions.tolist()}

# Root route
@app.get("/")
def root():
    return {"message": "Welcome to the Satisfaction Score Prediction API"}
