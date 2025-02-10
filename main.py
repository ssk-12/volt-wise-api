from fastapi import FastAPI, HTTPException
import pickle
import numpy as np
from pydantic import BaseModel
import os
import httpx
from datetime import datetime, date, timedelta
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
    # Extract hourly data and ensure 24 hours
    hours_data = weather_data['data']
    
    # Create a 24-hour template with hourly intervals
    hours_24 = []
    base_time = datetime.strptime(hours_data[0]['time'], '%Y-%m-%d %H:%M:%S')
    
    for hour in range(24):
        hour_time = (base_time + timedelta(hours=hour)).strftime('%Y-%m-%d %H:%M:%S')
        
        # Find matching data or use interpolated/default values
        matching_data = next(
            (x for x in hours_data if x['time'].startswith(hour_time[:13])),
            None
        )
        
        if matching_data:
            hour_data = matching_data
        else:
            # If no data for this hour, use the closest available data
            hour_data = hours_data[-1] if hours_data else {
                'temp': 0, 'dwpt': 0, 'rhum': 0, 'prcp': 0,
                'wdir': 0, 'wspd': 0, 'pres': 0, 'coco': 0
            }
        
        hours_24.append(hour_data)
    
    # Prepare features in correct order for each hour
    features = []
    for hour_data in hours_24:
        feature_row = [
            float(hour_data['temp'] or 0),      # Temperature
            float(hour_data['dwpt'] or 0),      # Dew Point
            float(hour_data['rhum'] or 0),      # Humidity
            float(hour_data['prcp'] or 0),      # Precipitation
            float(hour_data['wdir'] or 0),      # Wind Direction
            float(hour_data['wspd'] or 0),      # Wind Speed
            float(hour_data['pres'] or 0),      # Pressure
            float(hour_data['coco'] or 0),      # Cloud Cover
            1.0 if is_holiday else 0.0,         # Is Holiday
            0.0                                 # Placeholder for batch number
        ]
        features.append(feature_row)
    
    # Split into batches of 7 entries each (24 hours = 4 batches with padding)
    batches = []
    for i in range(0, len(features), 7):
        batch = features[i:i+7]
        
        # If the batch is not complete (less than 7 entries), pad it
        while len(batch) < 7:
            # Create a copy of the last entry for padding
            padding_row = list(batch[-1] if batch else features[-1])
            batch.append(padding_row)
        
        # Update batch number for each row
        batch_num = i // 7
        for row in batch:
            row[9] = float(batch_num)  # Set batch number in the pre-allocated slot
        
        batches.append(np.array(batch, dtype=np.float32))
    
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
        hours = []
        
        for batch_idx, batch in enumerate(batches):
            batch_array = batch.reshape(1, 7, 10)  # Reshape to match expected input
            prediction = model.predict(batch_array)
            
            # Calculate the actual hours for this batch
            start_hour = batch_idx * 7
            for i, pred in enumerate(prediction[0]):
                hour = start_hour + i
                if hour < 24:  # Only include predictions for the first 24 hours
                    all_predictions.append(pred)
                    hours.append(f"{hour:02d}:00")
        
        # Ensure we have exactly 24 predictions
        all_predictions = all_predictions[:24]
        hours = hours[:24]
        
        return {
            "date": predict_date,
            "predictions": [
                {"hour": hour, "prediction": float(pred)} 
                for hour, pred in zip(hours, all_predictions)
            ],
            "num_batches": len(batches),
            "total_hours": 24
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}"
        )

# Run the server using: uvicorn main:app --reload
