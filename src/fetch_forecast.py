from datetime import datetime
import pandas as pd
import requests
from requests_cache import logger
from config import DEBRECEN_LAT, DEBRECEN_LON, OW_API_KEY


def fetch_forecast_data():
    """
    Fetch weather forecast data for prediction.
    
    Returns:
        DataFrame: Pandas DataFrame with forecast data
    """
    if not OW_API_KEY:
        logger.warning("OpenWeather API key not configured")
        return None
        
    try:
        # OpenWeather API endpoint
        url = "https://api.openweathermap.org/data/2.5/forecast"
        params = {
            "lat": DEBRECEN_LAT,
            "lon": DEBRECEN_LON,
            "appid": OW_API_KEY,
            "units": "metric"
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            # Process forecast data
            forecast_records = []
            for item in data["list"]:
                dt = datetime.fromtimestamp(item["dt"])
                forecast_records.append({
                    "datetime": dt,
                    "temperature": item["main"]["temp"],
                    "humidity": item["main"]["humidity"],
                    "pressure": item["main"]["pressure"],
                    "wind_speed": item["wind"]["speed"],
                    "wind_dir": item["wind"]["deg"]
                })
                
            logger.info(f"Successfully fetched forecast data: {len(forecast_records)} records")
            return pd.DataFrame(forecast_records)
        else:
            logger.error(f"Error fetching forecast: {response.status_code}")
            return None
            
    except Exception as e:
        logger.error(f"Exception in fetch_forecast_data: {e}")
        return None