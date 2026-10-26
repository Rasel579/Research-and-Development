import os
import requests
import json
from WeatherApi import WeatherApi
import ShellTool
from GithubTool import GithubTool

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
    },
    {
        "type": "function",
        "function": {
            "name": "manage_repo_and_push",
            "description": "Определяет работу с Github, или создает новый репозиторий или добавляет новую ветку с push-request",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_name": {"type": "string"},
                    "repo_path": {"type": "string"},
                    "branch_name": {"type": "string"},
                    "commit_message": {"type": "string"}
                },
                "required": ["repo_name", "repo_path", "branch_name", "commit_message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "switch_branch_and_update_repo",
            "description": "переключает в ветку git (по-умолчанию main) и подтягивает изменения с удаленного репозитория",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_path": {"type": "string"},
                    "branch": {"type": "string"},
                },
                "required": ["repo_path"]
            }
        }
    }
]

MODEL = "~/bonsai/Bonsai-demo/models/gguf/27B/Bonsai-27B-Q1_0.gguf"
SYSTEM_PROMPT = """Ты — агент-разработчик. Ты работаешь в локальной директории проекта.
                   Используй инструменты чтобы читать, писать файлы и выполнять команды.
                   Думай пошагово. Если задача неясна — уточни.
                   Правило №1: НИКОГДА не создавай проект с нуля, если папка с таким именем 
                   уже существует. Сначала всегда используй инструмент ls,
                   чтобы проверить наличие директории. Если папка есть — просто переходи 
                   в неё (cd) и переключайся в main ветку и подтягивай изменения из github. 
                   Правило №2: главной веткой всегда должна быть - main, 
                   если не так то сделай главной main и отправь в репозиторий.
                   Правило №3: Когда найдешь директорию обнови главную ветку 
                   изменения с GitHub и создай новую ветку от нее и работай в ней.
                   Правила №4: Если папки с проектом не нашлось, 
                   то уже Создавай новую папку и создавай новый репозиторий в github
                   Правила №5: Для работы с Git и Github используй инструмент `manage_repo_and_push`"""

class LLamaCpp:
    def __init__(self, vk):
        self.TOOL_IMPL = None
        self.github_api = None
        self.uri = OLLAMA_URL
        self.tools = TOOLS
        self.model = MODEL
        self.weather_api = WeatherApi()
        self.vk = vk

    def response(self, user_id, prompt, vk, max_iters: int = 28):
        self.github_api = GithubTool(vk, user_id)
        self.TOOL_IMPL = { "read_file": ShellTool.read_file,
                           "write_file": ShellTool.write_file,
                           "run_command": ShellTool.run_command,
                           "get_weather": self.weather_api.get_weather,
                           "manage_repo_and_push": self.github_api.manage_repo_and_push,
                           "update_repo": self.github_api.update_repo,
                         }
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
            print("\n*****************\n")
            print(result)
            print("\n*****************\n")
            choice = result["choices"][0]
            message = choice["message"]
            if not message.get("tool_calls"):
                return message["content"]
            try:
                vk.messages.send(
                    user_id=user_id,
                    message=message["reasoning_content"],
                    random_id=0
                )
            except Exception as e:
                print(f"Ошибка: {e}")
            msg.append(message)
            tools_calls = message.get("tool_calls")
            for tool in tools_calls:
                fname = tool["function"]["name"]
                args = json.loads(tool["function"]["arguments"])
                print(f"🔧 {fname}({args})")
                try:
                  vk.messages.send(
                    user_id=user_id,
                    message=f"🔧 {fname}({args})",
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
                    "content": str(result)[:12000],
                })
        print("Лимит итераций, сорян 😅")
        try:
            vk.messages.send(
                user_id=user_id,
                message="Лимит итераций, сорян 😅",
                random_id=0
            )
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Ошибка: {e}"


