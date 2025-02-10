from fastapi import FastAPI, HTTPException
import pickle
import numpy as np
from pydantic import BaseModel
import os

app = FastAPI()

# Initialize model variable
model = None

# Try to load the ML model
try:
    model_path = "model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"The model file '{model_path}' was not found. Please ensure the model file exists in the project directory."
        )
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {str(e)}")
    # We'll handle the missing model in the predict endpoint

# Define the input schema
class InputData(BaseModel):
    features: list

@app.get("/health")
def health_check():
    return {
        "status": "up",
        "model_loaded": model is not None
    }

@app.post("/predict/")
def predict(data: InputData):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="The ML model is not loaded. Please ensure model.pkl exists and is valid."
        )
    
    try:
        prediction = model.predict(np.array([data.features]))
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}"
        )

# Run the server using: uvicorn main:app --reload
