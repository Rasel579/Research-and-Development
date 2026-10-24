import requests

WEATHER_URI = "https://goweather.xyz/v2/weather/"

class WeatherApi:
    def __init__(self):
      self.uri = WEATHER_URI
    def get_weather(self, city: str):
        response = requests.get(self.uri + city)
        return response.json()