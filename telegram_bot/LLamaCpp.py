import os
import requests
import json

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
                    "location": {"type": "string", "description": "City and state"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
    }
]

class LLamaCpp:
    def __init__(self):
        self.uri = OLLAMA_URL
        self.tools = TOOLS

    def get_weather(city: str, unit: str = "celsius"):
        return {"city": city, "temperature": 22, "unit": unit, "condition": "sunny"}

    def response(self, user_id, prompt):
        msg = {"role": "user", "content": prompt}
        response = requests.post(
            self.uri,
            json={
                "model": "model.gguf",
                "messages": [msg],
                "tools": self.tools
            }
        )
        result = response.json()
        tools_calls = result["choices"]["messages"][0]["content"].get("tools_calls")
        if tools_calls:
            tool_call = tools_calls[0]
            args = json.loads(tool_call["function"]["arguments"])
            weather_result = {"temperature": 72, "condition": "sunny"}
            response = requests.post(
            self.uri,
            json={
                "model": "model.gguf",
                "messages": [msg, result["choices"][0]["message"],
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(weather_result)
                }]
            }
        )
        return response.json()["choices"][0]["message"]["content"]

