import os
from typing import List, Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OpenMeteoConfig(BaseModel):
    base_url: str = Field("https://api.open-meteo.com/v1/", description="Base URL for Open-Meteo APIs")
    air_quality_endpoint: str = Field("air-quality", description="Endpoint for air quality data")
    archive_endpoint: str = Field("archive", description="Endpoint for historical weather data")
    forecast_endpoint: str = Field("forecast", description="Endpoint for weather forecast data")
    historical_air_quality_url: str = Field("https://air-quality-api.open-meteo.com/v1/air-quality", description="Historical air quality url")
    historical_weather_url: str = Field("https://archive-api.open-meteo.com/v1/archive", description="Historical weather url")
    forecast_weather_url: str = Field("https://api.open-meteo.com/v1/forecast", description="Future weather url")

class OpenWeatherAPIConfig(BaseModel):
    api_key: str = Field(..., description="OpenWeatherMap API key")
    base_url: str = Field("https://api.openweathermap.org/data/2.5/weather", description="Base URL for OpenWeatherMap API")
    units: str = Field("metric", description="Units for weather data")

class ThingSpeakAPIConfig(BaseModel):
    channel_id: str = Field(..., description="ThingSpeak channel ID")
    read_api_key: str = Field(..., description="ThingSpeak read API key")
    base_url: str = Field("https://api.thingspeak.com", description="Base URL for ThingSpeak API")
    feeds_endpoint: str = Field("feeds.json", description="Endpoint for channel feeds")

class DataPathsConfig(BaseModel):
    data_dir: str = Field(os.path.join(os.getcwd(), "data"), description="Directory for storing data")
    model_dir: str = Field(os.path.join(os.getcwd(), "models"), description="Directory for storing models")

class DatabaseConfig(BaseModel):
    host: str = Field(..., description="Database host")
    port: int = Field(5432, description="Database port")
    database: str = Field(..., description="Database name")
    user: str = Field(..., description="Database user")
    password: str = Field(..., description="Database password")

class ModelParametersConfig(BaseModel):
    max_iter: int = Field(100, description="Maximum number of iterations for the GBT model")
    max_depth: int = Field(5, description="Maximum depth of each tree in the GBT model")
    step_size: float = Field(0.1, description="Step size for the GBT model")
    subsampling_rate: float = Field(1.0, description="Subsampling rate for the GBT model")
    seed: int = Field(42, description="Random seed for model training")
    num_trees: int = Field(100, description="Number of trees for the Random Forest model")
    max_depth_rf: int = Field(5, description="Maximum depth of each tree in the Random Forest model")

class CoordinatesConfig(BaseModel):
    latitude: float = Field(47.53, description="Debrecen latitude")
    longitude: float = Field(21.63, description="Debrecen longitude")
    elevation: float = Field(120.0, description="Debrecen elevation")
    start_date: str = Field("2023-01-01", description="Start date for historical data")
    end_date: str = Field("2024-01-01", description="End date for historical data")

class Config(BaseSettings):
    open_meteo: OpenMeteoConfig = Field(default_factory=OpenMeteoConfig)
    open_weather_api: OpenWeatherAPIConfig
    thing_speak_api: ThingSpeakAPIConfig
    data_paths: DataPathsConfig = Field(default_factory=DataPathsConfig)
    database: DatabaseConfig
    model_parameters: ModelParametersConfig = Field(default_factory=ModelParametersConfig)
    coordinates: CoordinatesConfig = Field(default_factory=CoordinatesConfig)

    model_config = SettingsConfigDict(env_nested_delimiter='__', frozen=True)

config = Config()