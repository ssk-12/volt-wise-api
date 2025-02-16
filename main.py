import pickle
import requests
import numpy as np
from fastapi import FastAPI, Query
from datetime import datetime
import uvicorn

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "up"}

WEATHER_API_URL = "https://d.meteostat.net/app/proxy/stations/hourly?station=42182&tz=Asia/Kolkata&start={date}&end={date}"

@app.get("/predict")
def predict(date: str = Query(default=datetime.today().strftime('%Y-%m-%d'))):
    # Fetch weather data
    response = requests.get(WEATHER_API_URL.format(date=date))
    if response.status_code != 200:
        return {"error": "Failed to fetch weather data"}
    
    weather_data = response.json().get("data", [])
    if not weather_data:
        return {"error": "No weather data available"}
    
    predictions = []
    for entry in weather_data:
        hour = datetime.strptime(entry["time"], "%Y-%m-%d %H:%M:%S").hour
        
        features = [
            entry.get("temp", 25),  # TEMP
            entry.get("dwpt", 15),  # DWPT
            entry.get("rhum", 75),  # RHUM
            entry.get("prcp", 0),   # PRCP
            entry.get("wdir", 180), # WDIR
            entry.get("wspd", 5),   # WSPD
            entry.get("pres", 1013),# PRES
            entry.get("coco", 1),   # COCO
            hour,  # HOUR
            datetime.strptime(date, '%Y-%m-%d').weekday(),  # DAY_OF_WEEK
            datetime.strptime(date, '%Y-%m-%d').month   # MONTH
        ]
        
        # Reshape and predict
        input_array = np.array(features).reshape(1, -1)
        prediction = model.predict(input_array)
        predictions.append({"hour": hour, "prediction": prediction.tolist()})
    
    return {"predictions": predictions}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)