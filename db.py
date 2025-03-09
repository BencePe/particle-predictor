import parser
import sqlite3
from datetime import datetime

def connect():
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute('DROP TABLE IF EXISTS data')
    c.execute('''
        CREATE TABLE data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            weather_temperature REAL,
            weather_humidity REAL,
            weather_description TEXT,
            sensor_ds TEXT,
            sensor_y REAL
        )
    ''')
    conn.commit()
    conn.close()

#TODO: TimeScaleDB should be added instead of the mock DB

def save_data():
    weather = parser.parse_weather_data()
    sensor = parser.parse_air_quality_data()
    weather_temp = weather.get("temperature")
    weather_humidity = weather.get("humidity")
    weather_desc = weather.get("description")
    sensor_ds = sensor.get("ds")
    sensor_y = sensor.get("y")
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    now = datetime.now().isoformat()
    c.execute('''
        INSERT INTO data (timestamp, weather_temperature, weather_humidity, weather_description, sensor_ds, sensor_y)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (now, weather_temp, weather_humidity, weather_desc, sensor_ds, sensor_y))
    conn.commit()
    conn.close()

def main():
    connect()
    save_data()

if __name__ == "__main__":
    main()
