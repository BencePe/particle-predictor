import fetch
def parse_weather_data():
    weather = fetch.fetch_weather_data()
    try:
        return {
            "temperature": weather["main"]["temp"],
            "humidity": weather["main"]["humidity"],
            "description": weather["weather"][0]["description"]
        }
    except KeyError as e:
        return {"error": f"Invalid weather data format: {e}"}

def parse_air_quality_data():
    try:
        air_quality = fetch.fetch_air_quality_data()
        pm_data = {"pm10": None, "pm25": None}
        sensors = air_quality["results"][0].get("sensors", [])
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
    

def main():
    parse_weather_data()
    parse_air_quality_data()

if __name__ == "__main__":
    main()