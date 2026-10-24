import requests

WEATHER_URI = "https://goweather.xyz/v2/weather/"

class WeatherApi:
    def __init__(self):
      self.uri = WEATHER_URI
    def get_weather(self, city: str):
        try:
            resp = requests.get(f"{self.uri}{city}", timeout=10)
            data = resp.json()
            temp = data["temperature"]
            wind = data["wind"]

            forecast_lines = []
            for day_data in data["forecast"]:
                day_name = day_data["day"]
                day_temp = day_data["temperature"]
                day_wind = day_data["wind"]
                forecast_lines.append(f"• {day_name}: {day_temp}, ветер {day_wind}")

            forecast_text = "\n".join(forecast_lines)
            return (
                f"📍 Погода в {city}:\n"
                f"Сейчас: температура {temp}, ветер {wind}\n\n"
                f"Прогноз:\n{forecast_text}"
            )
        except Exception as e:
            return f"❌ Ошибка: {e}"

if __name__ == "__main__":
    weather = WeatherApi()
    result = weather.get_weather(city="Сургут")
    print(result)