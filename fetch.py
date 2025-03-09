import requests
# API keys and URLs (replace with your own keys if necessary)
OPENWEATHER_API_KEY = "83d50870977893b61ac48149455cf65a"
OPENAQ_API_KEY = "568eaee503e6178c9b3a6951a9e5941245040a6ffdf1eb334ab725cd1608d042"
WEATHER_URL = f"http://api.openweathermap.org/data/2.5/weather?q=Debrecen,hu&appid={OPENWEATHER_API_KEY}&units=metric"
# Location endpoint URL that returns sensor metadata, e.g., for location id 2178
LOCATION_URL = "https://api.openaq.org/v3/locations/2178"
def fetch_weather_data():
    try:
        weather_response = requests.get(WEATHER_URL)
        weather_data = weather_response.json()
    except Exception as e:
        weather_data = {"error": str(e)}
        print("Fetch weather")
    return weather_data

def fetch_air_quality_data():
    headers = {"X-API-Key": OPENAQ_API_KEY}
    try:
        response = requests.get(LOCATION_URL, headers=headers)
        air_quality_data = response.json()
    except Exception as e:
        air_quality_data = {"error": str(e)}
        print("Fetch AQ data")
    return air_quality_data

def main():
    fetch_weather_data()
    fetch_air_quality_data()

if __name__ == "__main__":
    main()