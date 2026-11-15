import os
import time
import requests
import vk_api
from bs4 import BeautifulSoup
from vk_api.longpoll import VkEventType, VkLongPoll
from vk_api.vk_api import VkApiMethod
from vk_api.upload import VkUpload
from urllib.parse import quote
from LLamaCpp import LLamaCpp

TOOLS =  [
    {
        "type": "function",
        "function": {
            "name": "post_meme",
            "description": "Запостить мем в группу",
            "parameters": {
                "type": "object",
                "properties": {
                    "meme": {"type": "dict", "description": "Словарь с мемами полученный из тулзы get_meme"},
                    "message": {"type": "string", "description": "Подпись к посту"}
                },
                "required": ["meme"]
            }
        }
    },
{
        "type": "function",
        "function": {
            "name": "get_meme",
            "description": "Получает мем из источника",
            "parameters": {
                "type": "dict",
                "properties": {
                    "type": {"type": "string", "description": "Категория мема"}
                },
                "required": ["type"]
            }
        }
    }
]

MEME_URI = os.environ['MEME_URI']
TOKEN = os.environ['VK_BOT_API']
MEME_GROUP = os.environ['GROUP_ID']

SYSTEM_PROMPT = """ # РОЛЬ
Ты — Макс, мем-сомелье и главный развлекатор VK-сообщества. 
Твоя задача: поддерживать ламповую атмосферу в чате/группе, реагировать на сообщения 
пользователей уместными шутками, реакциями и, главное, отправлять мемы, когда это нужно.

# ХАРАКТЕР
- Ты жизнерадостный, с отличным чувством юмора, немного саркастичный, но добрый.
- Общаешься на "ты", используешь современный сленг, эмодзи (в меру).
- Ты в теме мемов, но не перегибаешь — не кидаешь устаревшие "превед медвед" из 2007-го.
- Чувствуешь контекст: понимаешь, когда люди грустят, радуются, троллят друг друга.

# ТВОИ ВОЗМОЖНОСТИ
У тебя есть тулза `post_meme`, которая отправляет 
мем на стену группы ВК и get_meme для поиска мемов из источника

# КОГДА ИСПОЛЬЗОВАТЬ ТУЛЗУ
✅ Человек явно просит мем / шутку / поднять настроение
✅ В чате затяжная пауза, нужна разрядка
✅ Обсуждают что-то, под что идеально ложится известный мем
✅ Кто-то жалуется на понедельник, дедлайн, баги — самое время поддержать
✅ Админ дал команду "кинь мемас"

# КОГДА НЕ ИСПОЛЬЗОВАТЬ ТУЛЗУ ❌
🚫 Человек задает серьезный вопрос (технический, рабочий, личный)
🚫 В группе обсуждают что-то грустное или важное
🚫 Ты уже кидал мем меньше 10 минут назад (не спамь!)
🚫 Нет подходящей картинки под ситуацию — лучше пошутить текстом

# ПРАВИЛА
1. Перед отправкой мема кратко прокомментируй его в чате — объясни, почему именно сейчас.
2. Текст мема должен быть коротким, ёмким, без душноты.
3. Никогда не используй мемы на чувствительные темы (политика, трагедии, болезни).
4. Если не уверен, уместен ли мем — лучше ответь текстом.
5. Не повторяйся: если вчера кидал мем про программистов, сегодня найди другую тему.

# ПРИМЕРЫ ПОВЕДЕНИЯ

Пользователь: "Блин, опять понедельник..."
Ты: "Понедельник — это когда пятница была давно, а следующая еще не скоро 😩 
Держи, братан, классика:"
[вызывает send_meme_to_vk("Понедельничек...", "memes/monday.jpg")]

Пользователь: "Как в Python сделать асинхронный запрос?"
Ты: "О, серьезный вопрос! Смотри, юзаешь aiohttp..." 
[тулзу НЕ вызывает — тут не до мемов]

Пользователь: "А можешь чего-нибудь веселого кинуть?"
Ты: "О, это я люблю! 😎 Лови:"
[вызывает meme = get_meme("Программирование") и вызывает
post_meme(meme,"Когда код скомпилился с первого раза"]"""
class VKAgent:
    def __init__(self, vk_session: VkApiMethod):
        self.vk_session = vk_session
        self.vk = vk_session.get_api()
        self.tools = TOOLS
        self.upload = VkUpload(vk_session)

    def get_meme(self, type: str, limit: int = 3, min_rating: int = 0 ):
        url = f"{MEME_URI}/tag/{quote(type)}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": MEME_URI
        }
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"❌ Ошибка при запросе: {e}")
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        memes = []

        for story in soup.select("div.story")[:limit * 3]:
            if len(memes) >= limit:
                break
            rating_el = story.select_one("div.story__rating-count")
            rating = 0
            if rating_el:
                try:
                    rating = int("".join(filter(lambda x: x.isdigit() or x == '-', rating_el.text)))
                except ValueError:
                    pass

            if rating < min_rating:
                continue

            img_el = story.select_one("div.story-image img")
            if not img_el:
                continue

            img_url = img_el.get("data-large-image") or img_el.get("data-src") or img_el.get("src", "")
            if img_url.startswith("//"):
                img_url = "https:" + img_url

            title_el = story.select_one("a.story__title-link")
            title = title_el.text.strip() if title_el else "Безымянный мем"
            post_url = title_el["href"] if title_el and title_el.has_attr("href") else ""
            if post_url and not post_url.startswith("http"):
                post_url = MEME_URI + post_url

            memes.append({
                "title": title,
                "image_url": img_url,
                "post_url": post_url,
                "rating": rating
            })
        return memes

    def post_meme(self, meme: dict, message: str = ""):
        print(f"📥 Скачиваем мем: {meme['title']}")

        img_response = requests.get(meme["image_url"], timeout=10)
        temp_filename = f"temp_meme_{int(time.time())}.jpg"

        with open(temp_filename, "wb") as f:
            f.write(img_response.content)

        try:
            photo = self.upload.photo_wall(photos=temp_filename, group_id=MEME_GROUP)
            photo_id = photo[0]["id"]
            owner_id = photo[0]["owner_id"]
            attachment = f"photo{owner_id}_{photo_id}"

            self.vk.wall.post(
              owner_id=-MEME_GROUP,
              message=message,
              attachments=attachment,
              from_group=1
            )
            print(f"✅ Мем успешно опубликован! 🎉")
            return True
        except vk_api.exceptions.ApiError as e:
           print(f"❌ Ошибка: {e}")
           return False
        finally:
          if os.path.exists(temp_filename):
               os.remove(temp_filename)

def answer_llama( llm: LLamaCpp, msg, user_id, vk):
    agent = VKAgent(vk)
    tool_impl = { "post_meme": agent.post_meme,
                  "get_meme": agent.get_meme,
                }
    return llm.response(user_id, msg,TOOLS, tool_impl, SYSTEM_PROMPT, vk)

def main():
    vk_session = vk_api.VkApi(token=TOKEN)
    vk = vk_session.get_api()
    longpoll = VkLongPoll(vk_session)
    llama_cpp = LLamaCpp(vk)
    print("Бот запущен и слушает сообщения...")

    for event in longpoll.listen():
        if event.type == VkEventType.MESSAGE_NEW and event.to_me and event.text:
            user_message = event.text.lower()
            try:
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

