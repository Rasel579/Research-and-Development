import requests
import vk_api
import os
from vk_api.longpoll import VkLongPoll, VkEventType

TOKEN = os.environ['VK_BOT_API']
GROUP_ID = 123456789
OLLAMA_MODEL = os.environ['LLM_MODEL']
OLLAMA_URL = os.environ['LLM_URI'] + '/api/generate'
LLM_INSTRUCTIONS = os.environ['LLM_INSTRUCTIONS']

def get_ollama_response(prompt):
    data = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "system": LLM_INSTRUCTIONS,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_URL, json=data)
        return response.json().get('response', 'Ой, ничего не вышло 😿')
    except Exception as e:
        return f'Что-то сломалось: {e}'

def main():
    vk_session = vk_api.VkApi(token=TOKEN)
    vk = vk_session.get_api()
    longpoll = VkLongPoll(vk_session)

    print("Бот запущен и слушает сообщения...")

    for event in longpoll.listen():
        if event.type == VkEventType.MESSAGE_NEW and event.to_me and event.text:
            peer_id = event.peer_id
            user_message = event.text.lower()
            bot_answer = get_ollama_response(user_message)
            vk.messages.send(
                user_id=event.user_id,
                message=bot_answer,
                random_id=0
            )

if __name__ == "__main__":
    main()