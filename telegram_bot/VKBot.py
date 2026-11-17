import requests
import vk_api
import os
from LLamaCpp import LLamaCpp
from bs4 import BeautifulSoup
from ddgs import DDGS
from vk_api.longpoll import VkLongPoll, VkEventType

from telegram_bot import ShellTool
from telegram_bot.GithubTool import GithubTool
from telegram_bot.WeatherApi import WeatherApi

TOKEN = os.environ['VK_BOT_API']
OLLAMA_MODEL = ""
OLLAMA_URL = ""
LLM_INSTRUCTIONS = "" """os.environ['LLM_INSTRUCTIONS']"""

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
conversations = {}


def search_web(query, num_results=3):
    try:
        results = []
        with DDGS() as ddg:
            search_results = list(ddg.text(query, max_results=num_results))
            for r in search_results:
                results.append(
                    f"Заголовок: {r.get('title', 'Нет заголовка')}\nСсылка: {r.get('href', '')}\nОписание: {r.get('body', '')}\n")

        if not results:
            print(f"Пустой поисковик : {results}")
            return "Ничего не найдено 😿"
        print(f"Поисковик с результатами : {results}")
        return "\n---\n".join(results)
    except Exception as e:
        print(f"Пустой сломался : {e}")
        return f"Поиск сломался: {e}"

def fetch_url_content(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (VKBot; friendly)'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'lxml')
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        text = soup.get_text(separator='\n', strip=True)
        return text[:2000] + "..." if len(text) > 2000 else text
    except Exception as e:
        print(f"Не удалось открыть сайт : {e}")
        return f"Не удалось открыть сайт: {e}"

def needs_search(user_text):
    search_triggers = [
        "найди", "поищи", "гугли", "что такое", "кто такой",
        "новости", "погода", "курс", "цена", "сколько стоит",
        "последние новости", "актуально", "свежая информация",
        "расскажи про", "что известно о"
    ]
    user_text_lower = user_text.lower()
    return any(trigger in user_text_lower for trigger in search_triggers)

def get_ollama_response(user_id, prompt):
    if user_id not in conversations:
        conversations[user_id] = []
    messages = [{"role": "system", "content": LLM_INSTRUCTIONS}] + conversations[user_id]

    if needs_search(prompt):
        search_results = search_web(prompt, num_results=10)
        enhanced_prompt = (
            f"Пользователь спросил: '{prompt}'\n\n"
            f"Вот что я нашел в интернете:\n{search_results}\n\n"
            f"Ответь на вопрос, используя эту информацию. Если информации недостаточно, так и скажи."
        )
        messages.append({"role": "user", "content": enhanced_prompt})
    else:
        messages.append({"role": "user", "content": prompt})

    try:
        response = requests.post(OLLAMA_URL, json={"model": OLLAMA_MODEL, "messages": messages, "stream": False, "think": False},timeout=600)
        response.raise_for_status()
        bot_answer = response.json().get("message", {}).get("content", "Ой, Марсик съел ответ 😿")

        conversations[user_id].append({"role": "user", "content": prompt})
        conversations[user_id].append({"role": "assistant", "content": bot_answer})

        if len(conversations[user_id]) > 20:
            conversations[user_id] = conversations[user_id][-20:]
        return bot_answer
    except Exception as e:
        return f'Что-то сломалось: {e}'

def answer_llama( llm: LLamaCpp, msg, user_id, vk):
    weather_api = WeatherApi()
    github_api = GithubTool(vk, user_id)
    tool_impl = { "read_file": ShellTool.read_file,
                  "write_file": ShellTool.write_file,
                  "run_command": ShellTool.run_command,
                  "get_weather": weather_api.get_weather,
                  "manage_repo_and_push": github_api.manage_repo_and_push,
                  "switch_branch_and_update_repo": github_api.update_repo,
                }
    return llm.response(user_id, msg, vk, TOOLS, tool_impl, SYSTEM_PROMPT)

def main():
    vk_session = vk_api.VkApi(token=TOKEN)
    vk = vk_session.get_api()
    longpoll = VkLongPoll(vk_session)

    print("Бот запущен и слушает сообщения...")

    for event in longpoll.listen():
        if event.type == VkEventType.MESSAGE_NEW and event.to_me and event.text:
            user_message = event.text.lower()
            if user_message.lower() in ['/очисти', '/clear']:
                conversations[event.user_id] = []
                vk.messages.send(user_id=event.user_id, message="Память очищена! 🧼", random_id=0)
                continue

            try:
                llama_cpp = LLamaCpp(event.user_id, vk)
                bot_answer = answer_llama(llama_cpp, user_message, event.user_id, vk)
                vk.messages.send(
                    user_id=event.user_id,
                    message=bot_answer,
                    random_id=0
                )
            except Exception as e:
                print(f"Ошибка отправки сообщения: {e}")

if __name__ == "__main__":
    main()