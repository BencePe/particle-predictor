#TODO:
## - Fetch data from TomTom, OpenAQ and OpenWeather for 2024
## - Fetch data from local sensors and combine them with present data
## - Limit to 1000 requests/hour

import requests
import time
import os
import json
from dotenv import load_dotenv

def fetchCurrentData():
    load_dotenv()
    OW_API_KEY = os.getenv("OW_API_KEY")
    TT_API_KEY = os.getenv("TT_API_KEY")

def fetchPredictionData():
    ...
    #TODO:
    ## - Fetch the weather forecast data for the prediction period (max. 5 days)

if __name__ == "__main__":
    ...