import os
import requests
import json
from WeatherApi import WeatherApi
import ShellTool

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
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Прочитать содержимое файла",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "путь к файлу"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Записать содержимое в файл",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Выполнить shell-команду и вернуть вывод",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"}
                },
                "required": ["command"]
            }
        }
    }
]

MODEL = "~/bonsai/Bonsai-demo/models/gguf/27B/Bonsai-27B-Q1_0.gguf"
SYSTEM_PROMPT = """Ты — агент-разработчик. Ты работаешь в локальной директории проекта.
                   Используй инструменты чтобы читать, писать файлы и выполнять команды.
                   Думай пошагово. Если задача неясна — уточни."""

class LLamaCpp:
    def __init__(self):
        self.uri = OLLAMA_URL
        self.tools = TOOLS
        self.model = MODEL
        self.weather_api = WeatherApi()
        self.TOOL_IMPL = { "read_file": ShellTool.read_file,
                           "write_file": ShellTool.write_file,
                           "run_command": ShellTool.run_command,
                           "get_weather": self.weather_api.get_weather,
                         }

    def response(self, user_id, prompt, vk, max_iters: int = 20):
        print(prompt)
        msg = [{"role": "system", "content": SYSTEM_PROMPT},
               {"role": "user", "content": prompt}]
        for i in range(max_iters):
            response = requests.post(
                self.uri,
                json={
                    "model": self.model,
                    "messages": msg,
                    "stream" : False,
                    "tools": self.tools
                }
            )
            result = response.json()
            print(result)
            choice = result["choices"][0]
            message = choice["message"]
            if not message.get("tool_calls"):
                return message
            msg.append(message)
            tools_calls = message.get("tool_calls")
            for tool in tools_calls:
                fname = tool["function"]["name"]
                args = json.loads(tool["function"]["arguments"])
                print(f"  🔧 {fname}({args})")
                try:
                        vk.messages.send(
                            user_id=user_id,
                            message=f"  🔧 {fname}({args})",
                            random_id=0
                        )
                except Exception as e:
                    print(f"Ошибка: {e}")
                try:
                    result = self.TOOL_IMPL[fname](**args)
                except Exception as e:
                    result = f"Ошибка: {e}"

                msg.append({
                    "role": "tool",
                    "tool_call_id": tool["id"],
                    "content": str(result)[:8000],
                })
        print("Лимит итераций, сорян 😅")
        return result["choices"][0]["message"]

