import fetch
from dateutil import parser as dateutil_parser
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
        pm_data = fetch.fetch_air_quality_data()
        local_dt_str = pm_data["results"][0]["period"]["datetimeTo"]["local"]
        local_dt = dateutil_parser.isoparse(local_dt_str).replace(tzinfo=None)
        value = {
            "ds": local_dt.isoformat(),
            "y": pm_data["results"][0]["value"]
        }
        return value
    except Exception as e:
        return {"error": str(e)}

def main():
    parse_weather_data()
    parse_air_quality_data()
    
if __name__ == "__main__":
    main()