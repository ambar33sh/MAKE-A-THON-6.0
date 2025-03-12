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
    allow_origins=["*"],  
    allow_methods=["*"],   
    allow_headers=["*"],  
)

# Define file paths using absolute paths for better compatibility
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
ENCODERS_PATH = os.path.join(os.path.dirname(__file__), "label_encoders.pkl")

print(f"🔍 Checking for model at: {MODEL_PATH}")
print(f"🔍 Checking for encoders at: {ENCODERS_PATH}")

if os.path.exists(MODEL_PATH) and os.path.exists(ENCODERS_PATH):
    print("✅ Model and label encoders found. Loading...")
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)
    with open(ENCODERS_PATH, "rb") as encoders_file:
        label_encoders = pickle.load(encoders_file)
    print(f"📝 Available label encoders: {label_encoders.keys()}")  # Debugging step
else:
    print("❌ ERROR: Model or Label Encoders not found. Check file paths.")
    model = None
    label_encoders = None

class UserData(BaseModel):
    category: str
    price_range: str
    brand: str
    price: float
    rating: int
    sentiment: float  # ✅ Added missing feature

@app.get("/")
def read_root():
    return {"message": "🚀 Product Return Prediction API is Running!"}

@app.post("/predict/")
def predict(data: UserData):
    if model is None or label_encoders is None:
        return {"error": "Model or encoders are missing. Check deployment and file paths."}

    try:
        # Ensure keys exist before transformation
        if "Product Category" not in label_encoders or "Price Range" not in label_encoders or "Brand Type" not in label_encoders:
            return {"error": "Label encoders do not contain expected keys. Check training process."}

        product_category_encoded = label_encoders["Product Category"].transform([data.category])[0]
        price_range_encoded = label_encoders["Price Range"].transform([data.price_range])[0]
        brand_type_encoded = label_encoders["Brand Type"].transform([data.brand])[0]

        # ✅ Added sentiment feature to match frontend
        product_data = np.array([[product_category_encoded, data.price, price_range_encoded, data.rating, brand_type_encoded, data.sentiment]])
        
        prediction = model.predict(product_data)[0]

        result = "✅ Based on past behavior, this product might be returned." if prediction == 1 else "❌ This product is aligned with previous purchases and will likely be kept."

        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
