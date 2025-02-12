from fastapi import FastAPI, HTTPException
import pickle
import numpy as np
from pydantic import BaseModel
import os
import httpx
from datetime import datetime
import json

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

async def fetch_weather_data(station, date):
    url = f"https://d.meteostat.net/app/proxy/stations/hourly?station={station}&tz=Asia/Kolkata&start={date}&end={date}"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()

def process_weather_data(data):
    processed_data = []
    for idx, record in enumerate(data):
        # Extract and order features according to model requirements
        features = [
            record['temp'],      # Temperature
            record['dwpt'],      # Dew Point
            record['rhum'],      # Humidity
            record['prcp'],      # Precipitation
            record['wdir'],      # Wind Direction
            record['wspd'],      # Wind Speed
            record['pres'],      # Pressure
            record['coco'],      # Cloud Cover
            0,                   # Is_Holiday (set to 0)
            idx                  # Index
        ]
        processed_data.append({
            'time': record['time'],
            'features': features
        })
    return processed_data

@app.get("/health")
def health_check():
    return {
        "status": "up",
        "model_loaded": model is not None
    }

@app.get("/predict/{station}/{date}")
async def predict_weather(station: str, date: str):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="The ML model is not loaded. Please ensure model.pkl exists and is valid."
        )
    
    try:
        # Fetch weather data
        weather_data = await fetch_weather_data(station, date)
        processed_records = process_weather_data(weather_data['data'])
        
        predictions = []
        # Process in groups of 7
        for i in range(0, len(processed_records), 7):
            batch = processed_records[i:i+7]
            if len(batch) == 7:  # Only process complete batches
                features = np.array([record['features'] for record in batch])
                prediction = model.predict(features)
                
                # Create prediction results
                for j, pred in enumerate(prediction):
                    predictions.append({
                        "timestamp": batch[j]['time'],
                        "prediction": float(pred),
                        "actual_temp": batch[j]['features'][0],
                        "actual_humidity": batch[j]['features'][2]
                    })
        
        return {
            "station": station,
            "date": date,
            "predictions": predictions
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}"
        )

# Run the server using: uvicorn main:app --reload
