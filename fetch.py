import requests
from datetime import datetime

OPENWEATHER_API_KEY = "83d50870977893b61ac48149455cf65a"
OPENAQ_API_KEY = "568eaee503e6178c9b3a6951a9e5941245040a6ffdf1eb334ab725cd1608d042"
WEATHER_URL = f"http://api.openweathermap.org/data/2.5/weather?q=Debrecen,hu&appid={OPENWEATHER_API_KEY}&units=metric"

def set_date():
    #TODO: date_to and date_from needs to be defined in relation to datetime 
    return date_to , date_from
   
def fetch_weather_data():
    try:
        weather_response = requests.get(WEATHER_URL)
        weather_data = weather_response.json()
    except Exception as e:
        weather_data = {"error": str(e)}
    return weather_data

def fetch_air_quality_data():
    DATE_TO = set_date()[0]
    DATE_FROM = set_date()[1]
    HOURLY = f"https://api.openaq.org/v3/sensors/4272895/hours?datetime_to={DATE_TO}&datetime_from={DATE_FROM}&limit=100&page=1"
    headers = {"X-API-Key": OPENAQ_API_KEY}
    try:
        response = requests.get(HOURLY, headers=headers)
        air_quality_data = response.json()
    except Exception as e:
        air_quality_data = {"error": str(e)}
    return air_quality_data

def main():
    set_date()
    fetch_weather_data()
    fetch_air_quality_data()
    
if __name__ == "__main__":
    main()