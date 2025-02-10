from fastapi import FastAPI, HTTPException
import pickle
import numpy as np
from pydantic import BaseModel
import os
import httpx
from datetime import datetime, date
import math

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

class InputData(BaseModel):
    date: str = None

async def fetch_weather_data(date_str: str):
    url = f"https://d.meteostat.net/app/proxy/stations/hourly?station=42182&tz=Asia/Kolkata&start={date_str}&end={date_str}"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch weather data")
        return response.json()

async def check_if_holiday(date_str: str):
    # This is a placeholder - implement actual holiday checking logic
    # For now, returning False as default
    return False

def prepare_batch_inputs(weather_data, is_holiday):
    # Extract hourly data
    hours_data = weather_data['data']
    
    # Prepare features in correct order for each hour
    features = []
    for hour_data in hours_data:
        feature_row = [
            hour_data['temp'],      # Temperature
            hour_data['dwpt'],      # Dew Point
            hour_data['rhum'],      # Humidity
            hour_data['prcp'],      # Precipitation
            hour_data['wdir'],      # Wind Direction
            hour_data['wspd'],      # Wind Speed
            hour_data['pres'],      # Pressure
            hour_data['coco'],      # Cloud Cover
            1 if is_holiday else 0  # Is Holiday
        ]
        features.append(feature_row)
    
    # Split into batches of 6
    num_batches = math.ceil(len(features) / 6)
    batches = []
    
    for i in range(num_batches):
        start_idx = i * 6
        end_idx = min(start_idx + 6, len(features))
        batch = features[start_idx:end_idx]
        
        # Add batch number to each row
        for row in batch:
            row.append(i)  # Add batch number
            
        batches.append(batch)
    
    return batches

@app.get("/health")
def health_check():
    return {
        "status": "up",
        "model_loaded": model is not None
    }

@app.post("/predict/")
async def predict(data: InputData):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="The ML model is not loaded. Please ensure model.pkl exists and is valid."
        )
    
    try:
        # Use provided date or default to today
        predict_date = data.date if data.date else date.today().strftime("%Y-%m-%d")
        
        # Fetch weather data
        weather_data = await fetch_weather_data(predict_date)
        
        # Check if holiday
        is_holiday = await check_if_holiday(predict_date)
        
        # Prepare batches
        batches = prepare_batch_inputs(weather_data, is_holiday)
        
        # Make predictions for each batch
        all_predictions = []
        for batch in batches:
            prediction = model.predict(np.array(batch))
            all_predictions.extend(prediction.tolist())
        
        return {
            "date": predict_date,
            "predictions": all_predictions,
            "num_batches": len(batches),
            "total_hours": len(all_predictions)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}"
        )

# Run the server using: uvicorn main:app --reload
