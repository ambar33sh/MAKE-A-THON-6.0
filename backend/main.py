from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle

app = FastAPI()

# Load the trained model
try:
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

class ProductDetails(BaseModel):
    category: str
    price: float
    rating: float
    other_features: dict  # Include any other features your model requires

@app.post("/predict/")
async def predict(details: ProductDetails):
    try:
        # Prepare the input data for prediction
        data = [details.category, details.price, details.rating]  # Modify this as per your model's input requirements

        # Make prediction
        prediction = model.predict([data])

        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")
