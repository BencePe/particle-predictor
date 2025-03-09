import parser
import sqlite3
from datetime import datetime
import json
def connect():
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

def save_data():
    weather = parser.parse_weather_data()
    air_quality = parser.parse_air_quality_data()
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    now = datetime.now().isoformat()
    print(weather, air_quality)
    c.execute('INSERT INTO data (timestamp, weather_data, air_quality_data) VALUES (?, ?, ?)',
              (now, json.dumps(weather), json.dumps(air_quality)))
    conn.commit()

def main():
    connect()
    save_data()
if __name__ == "__main__":
    main()