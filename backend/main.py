from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import os

app = FastAPI()

# Load your trained model
MODEL_PATH = "model.pkl"
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)
else:
    model = None

# Define the expected input data format
class UserData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    # Add more features as needed

@app.get("/")
def read_root():
    return {"message": "Welcome to the Reverse Logistics API"}

@app.post("/predict/")
def predict(data: UserData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not found. Please upload the trained model.")

    try:
        # Convert input data to numpy array
        input_data = np.array([[data.feature1, data.feature2, data.feature3]])  # Modify as per your model's input structure
        
        # Make a prediction
        prediction = model.predict(input_data)
        
        # Return prediction result
        return {"prediction": str(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
