import requests
import vk_api
import os
from bs4 import BeautifulSoup
import ddgs
from vk_api.longpoll import VkLongPoll, VkEventType

TOKEN = os.environ['VK_BOT_API']
GROUP_ID = 123456789
OLLAMA_MODEL = os.environ['LLM_MODEL']
OLLAMA_URL = os.environ['LLM_URI'] + '/api/chat'
LLM_INSTRUCTIONS = os.environ['LLM_INSTRUCTIONS']

conversations = {}


def search_web(query, num_results=3):
    try:
        results = []
        with ddgs:
            search_results = list(ddgs.DDGS.text(query, max_results=num_results))
            for r in search_results:
                results.append(
                    f"Заголовок: {r.get('title', 'Нет заголовка')}\nСсылка: {r.get('href', '')}\nОписание: {r.get('body', '')}\n")

        if not results:
            return "Ничего не найдено 😿"

        return "\n---\n".join(results)
    except Exception as e:
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
        # Сначала ищем
        search_results = search_web(prompt, num_results=2)
        enhanced_prompt = (
            f"Пользователь спросил: '{prompt}'\n\n"
            f"Вот что я нашел в интернете:\n{search_results}\n\n"
            f"Ответь на вопрос, используя эту информацию. Если информации недостаточно, так и скажи."
        )
        messages.append({"role": "user", "content": enhanced_prompt})
    else:
        messages.append({"role": "user", "content": prompt})

    try:
        response = requests.post(OLLAMA_URL, json={"model": OLLAMA_MODEL, "messages": messages, "stream": False},
                                 timeout=120)
        response.raise_for_status()
        bot_answer = response.json().get("message", {}).get("content", "Ой, Марсик съел ответ 😿")

        conversations[user_id].append({"role": "user", "content": prompt})
        conversations[user_id].append({"role": "assistant", "content": bot_answer})

        if len(conversations[user_id]) > 20:
            conversations[user_id] = conversations[user_id][-20:]
        return bot_answer
    except Exception as e:
        return f'Что-то сломалось: {e}'

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

            bot_answer = get_ollama_response(event.user_id, user_message)
            vk.messages.send(
                user_id=event.user_id,
                message=bot_answer,
                random_id=0
            )

if __name__ == "__main__":
    main()