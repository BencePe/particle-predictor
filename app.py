import requests
import sqlite3
import time
from datetime import datetime
import json

# API keys and URLs (replace with your own keys if necessary)
OPENWEATHER_API_KEY = "83d50870977893b61ac48149455cf65a"
OPENAQ_API_KEY = "568eaee503e6178c9b3a6951a9e5941245040a6ffdf1eb334ab725cd1608d042"
WEATHER_URL = f"http://api.openweathermap.org/data/2.5/weather?q=Debrecen,hu&appid={OPENWEATHER_API_KEY}&units=metric"
# Location endpoint URL that returns sensor metadata, e.g., for location id 2178
LOCATION_URL = "https://api.openaq.org/v3/locations/2178"

# Database connection and table creation
conn = sqlite3.connect('data.db')
c = conn.cursor()

# Drop the existing table if it exists
c.execute('DROP TABLE IF EXISTS data')

# Create the table with the correct schema (storing JSON strings)
c.execute('''
    CREATE TABLE data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        weather_data TEXT,
        air_quality_data TEXT
    )
''')
conn.commit()

def fetch_weather_data():
    try:
        weather_response = requests.get(WEATHER_URL)
        weather_data = weather_response.json()
    except Exception as e:
        weather_data = {"error": str(e)}
    return weather_data

def fetch_air_quality_data():
    headers = {"X-API-Key": OPENAQ_API_KEY}
    try:
        response = requests.get(LOCATION_URL, headers=headers)
        air_quality_data = response.json()
    except Exception as e:
        air_quality_data = {"error": str(e)}
    return air_quality_data

def parse_weather_data(data):
    try:
        return {
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "description": data["weather"][0]["description"]
        }
    except KeyError as e:
        return {"error": f"Invalid weather data format: {e}"}

def parse_air_quality_data(data):
    try:
        # Initialize with default None values
        pm_data = {"pm10": None, "pm25": None}
        # Get the sensors list from the first result
        sensors = data["results"][0].get("sensors", [])
        for sensor in sensors:
            param = sensor.get("parameter", {}).get("name", "").lower()
            if param in pm_data:
                pm_data[param] = {
                    "sensor_id": sensor.get("id"),
                    "name": sensor.get("name"),
                    "units": sensor.get("parameter", {}).get("units"),
                    "displayName": sensor.get("parameter", {}).get("displayName")
                }
        return pm_data
    except Exception as e:
        return {"error": str(e)}

def save_data(weather, air_quality):
    now = datetime.now().isoformat()
    c.execute('INSERT INTO data (timestamp, weather_data, air_quality_data) VALUES (?, ?, ?)',
              (now, json.dumps(weather), json.dumps(air_quality)))
    conn.commit()

if __name__ == "__main__":
    while True:
        weather_raw = fetch_weather_data()
        aq_raw = fetch_air_quality_data()
        
        weather_parsed = parse_weather_data(weather_raw)
        aq_parsed = parse_air_quality_data(aq_raw)
        
        save_data(weather_parsed, aq_parsed)
        
        print(f"Data saved at: {datetime.now()}")
        print("-----------------------------------------------------------------------")
        print("")
        time.sleep(3000)  # Wait for 300 seconds (5 minutes)
