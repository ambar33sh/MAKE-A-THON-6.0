from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import os

app = FastAPI()

# Allow CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_methods=["*"],   # Allows all methods (POST, GET, etc.)
    allow_headers=["*"],   # Allows all headers
)

# Load trained model and label encoders
MODEL_PATH = "model.pkl"
ENCODERS_PATH = "labelencoder.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(ENCODERS_PATH):
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)
    with open(ENCODERS_PATH, "rb") as encoders_file:
        label_encoders = pickle.load(encoders_file)
else:
    model = None
    label_encoders = None

# Define expected input format (No sentiment column)
class UserData(BaseModel):
    category: str
    price_range: str
    brand: str
    price: float
    rating: int

@app.get("/")
def read_root():
    return {"message": "🚀 Product Return Prediction API is Running!"}

@app.post("/predict/")
def predict(data: UserData):
    if model is None or label_encoders is None:
        raise HTTPException(status_code=500, detail="Model or Label Encoders not found. Please upload the trained model and encoders.")

    try:
        # Encode categorical inputs
        try:
            product_category_encoded = label_encoders["Product Category"].transform([data.category])[0]
            price_range_encoded = label_encoders["Price Range"].transform([data.price_range])[0]
            brand_type_encoded = label_encoders["Brand Type"].transform([data.brand])[0]
        except KeyError as e:
            raise HTTPException(status_code=400, detail=f"Invalid input value: {str(e)}")

        # Prepare input for model (No sentiment column)
        product_data = np.array([[product_category_encoded, data.price, price_range_encoded, data.rating, brand_type_encoded]])

        # Make prediction
        prediction = model.predict(product_data)[0]

        # Generate response message
        result = "✅ Product might be returned." if prediction == 1 else "❌ Product is likely to be kept."

        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
