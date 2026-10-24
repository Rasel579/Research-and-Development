import os
import requests
import json
from WeatherApi import WeatherApi

OLLAMA_URL = os.environ['LLM_URI'] + '/v1/chat/completions'
TOOLS =  [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City and state"}
                },
                "required": ["location"]
            }
        }
    }
]

MODEL = "~/bonsai/Bonsai-demo/models/gguf/27B/Bonsai-27B-Q1_0.gguf"

class LLamaCpp:
    def __init__(self):
        self.uri = OLLAMA_URL
        self.tools = TOOLS
        self.model = MODEL
        self.weather_api = WeatherApi()

    def response(self, user_id, prompt):
        print(prompt)
        msg = {"role": "user", "content": prompt}
        response = requests.post(
            self.uri,
            json={
                "model": self.model,
                "messages": [msg],
                "stream" : False,
                "tools": self.tools
            }
        )
        result = response.json()
        print(result)
        choice = result["choices"][0]
        message = choice["message"]
        tools_calls = message.get("tool_calls")
        if tools_calls:
            tool_call = tools_calls[0]
            args = json.loads(tool_call["function"]["arguments"])
            weather_result = self.weather_api.get_weather(args["location"])
            response = requests.post(
            self.uri,
            json={
                "model": self.model,
                "messages": [msg, result["choices"][0]["message"],
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(weather_result)
                }]
            }
        )
        print(response.json())
        return response.json()["choices"][0]["message"]["content"]

