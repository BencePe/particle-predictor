import time
import db
import fetch
import parser
from datetime import datetime

if __name__ == "__main__":
    db.connect()
    while True:
        weather_raw = fetch.fetch_weather_data
        aq_raw = fetch.fetch_air_quality_data
        
        weather_parsed = parser.parse_weather_data()
        aq_parsed = parser.parse_air_quality_data()

        db.save_data()
        
        print(f"Data saved at: {datetime.now()}")
        print("-----------------------------------------------------------------------")
        print(weather_parsed)
        print("-----------------------------------------------------------------------")
        print(aq_parsed)
        print("-----------------------------------------------------------------------")
        time.sleep(300)  # Wait for 300 seconds (5 minutes)
