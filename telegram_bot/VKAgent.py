import os
import time
import requests
import vk_api
from bs4 import BeautifulSoup
from vk_api.longpoll import VkEventType, VkLongPoll
from vk_api.vk_api import VkApiMethod
from vk_api.upload import VkUpload
from LLamaCpp import LLamaCpp

TOOLS =  [
    {
        "type": "function",
        "function": {
            "name": "post_meme",
            "description": "Публикует мем в группу. Скачивает картинку по image_url, загружает её на стену сообщества и создаёт пост с подписью. Используется после тулзы get_meme, которая возвращает готовый объект мема.",
            "parameters": {
                "type": "object",
                "properties": {
                    "meme": {
                        "type": "object",
                        "description": "Объект мема, полученный из тулзы get_meme",
                        "properties": {
                            "image_url": {
                                "type": "string",
                                "description": "Прямая ссылка на изображение мема (jpg/png/gif)"
                            },
                            "title": {
                                "type": "string",
                                "description": "Заголовок мема, используется как подпись по умолчанию"
                            },
                            "post_url": {
                                "type": "string",
                                "description": "Ссылка на исходный пост (для указания источника)"
                            },
                            "rating": {
                                "type": "integer",
                                "description": "Рейтинг мема на источнике"
                            }
                        },
                        "required": ["image_url", "title"]
                    },
                    "message": {
                        "type": "string",
                        "description": "Кастомная подпись к посту. Если не указана — используется title мема. Можно добавить эмодзи, хэштеги или свой текст."
                    },
                    "silent": {
                        "type": "boolean",
                        "description": "Если true — пост публикуется без уведомления подписчиков (mute). По умолчанию false."
                    }
                },
                "required": ["meme"]
            }
        }
    },
{
    "type": "function",
    "function": {
        "name": "get_meme",
        "description": "Ищет и возвращает мем по заданной категории (тегу). Возвращает объект мема, который можно напрямую передать в тулзу post_meme для публикации в группу. Использует парсинг.",
        "parameters": {
            "type": "object",
            "properties": {
                "meme_type": {
                    "type": "string",
                    "description": "Тег или категория мема.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Количество мемов, которые нужно вернуть. По умолчанию 1, максимум 10.",
                    "default": 1,
                    "minimum": 1,
                    "maximum": 10
                },
            },
            "required": ["meme_type"]
        }
    }
}
]

MEME_URI = os.environ['MEME_URI']
TOKEN = os.environ['VK_BOT_API']
MEME_GROUP = os.environ['GROUP_ID']
SERVICE_TOKEN = os.environ['SERVICE_TOKEN']

SYSTEM_PROMPT = """ # РОЛЬ
Ты — Макс, мем-сомелье и главный развлекатор сообщества. 
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
[вызывает post_meme("Понедельничек...", "memes/monday.jpg")]

Пользователь: "Как в Python сделать асинхронный запрос?"
Ты: "О, серьезный вопрос! Смотри, юзаешь aiohttp..." 
[тулзу НЕ вызывает — тут не до мемов]

Пользователь: "А можешь чего-нибудь веселого кинуть?"
Ты: "О, это я люблю! 😎 Лови:"
[вызывает meme = get_meme("Программирование") и вызывает
post_meme(meme,"Когда код скомпилился с первого раза"]"""
class VKAgent:
    def __init__(self, vk_session: VkApiMethod, vk_group: VkApiMethod):
        self.vk_session = vk_session
        self.vk = vk_session
        self.tools = TOOLS
        self.upload = VkUpload(vk_group)

    def get_meme(self, meme_type: str, limit: int = 3, min_rating: int = 0 ):
        url = f"{MEME_URI}/tag/{meme_type}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": MEME_URI,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
        }
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"❌ Ошибка при запросе: {e}")
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        memes = []
        stories = soup.select("article.story")
        for story in stories:
            if len(memes) >= limit:
                break
            rating_str = story.get("data-rating", "0")
            try:
                rating = int(rating_str)
            except ValueError:
                rating = 0

            if rating < min_rating:
                continue

            img_el = story.select_one("div.story-image img")
            img_url = None
            if img_el:
                img_url = img_el.get("data-large-image") or img_el.get("data-src") or img_el.get("src", "")
            else:
                carousel_img = story.select_one("div.carousel__item img")
                if carousel_img:
                    img_url = carousel_img.get("data-large-image") or carousel_img.get("data-src") or carousel_img.get(
                        "src", "")
            if not img_url:
                continue

            img_url = img_url.strip()
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

    def post_meme(self, meme: dict, message: str = "", silent: bool = False):
        print(f"📥 Скачиваем мем: {meme['title']}")

        img_response = requests.get(meme["image_url"], timeout=10)
        temp_filename = f"temp_meme_{int(time.time())}.jpg"

        with open(temp_filename, "wb") as f:
            f.write(img_response.content)

        try:
            photo = self.upload.photo_wall(photos=temp_filename, group_id=int(MEME_GROUP))
            photo_id = photo[0]["id"]
            owner_id = photo[0]["owner_id"]
            attachment = f"photo{owner_id}_{photo_id}"

            result = self.vk.wall.post(
              owner_id=-int(MEME_GROUP),
              message=message,
              attachments=attachment,
              from_group=1,
              mute=silent
            )
            print(f"✅ Мем успешно опубликован! 🎉")
            return {
                "success": True,
                "post_id": result["post_id"],
                "post_url": f"https://vk.com/wall-{MEME_GROUP}_{result['post_id']}"
            }
        except vk_api.exceptions.ApiError as e:
           print(f"❌ Ошибка: {e}")
           return  {
            "success": False,
            "post_id": "❌ Ошибка: " + e.error.message,
           }
        finally:
          if os.path.exists(temp_filename):
               os.remove(temp_filename)

def answer_llama( llm: LLamaCpp, msg, user_id, vk, vk_service):
    agent = VKAgent(vk,vk_service)
    tool_impl = { "post_meme": agent.post_meme,
                  "get_meme": agent.get_meme,
                }
    return llm.response(user_id, msg, vk, TOOLS, tool_impl, SYSTEM_PROMPT)

def main():
    vk_session = vk_api.VkApi(token=TOKEN)
    vk = vk_session.get_api()
    longpoll = VkLongPoll(vk_session)
    vk_session_service = vk_api.VkApi(token=SERVICE_TOKEN)
    vk_service = vk_session_service.get_api()
    llama_cpp = LLamaCpp(vk)
    print("Бот запущен и слушает сообщения...")

    for event in longpoll.listen():
        if event.type == VkEventType.MESSAGE_NEW and event.to_me and event.text:
            user_message = event.text.lower()
            try:
                bot_answer = answer_llama(llama_cpp, user_message, event.user_id, vk,vk_service)
                vk.messages.send(
                    user_id=event.user_id,
                    message=bot_answer,
                    random_id=0
                )
            except Exception as e:
                print(f"Ошибка отправки сообщения: {e}")



def test():
    vk_session = vk_api.VkApi(token=TOKEN)
    vk = vk_session.get_api()
    vk_session_service = vk_api.VkApi(token=SERVICE_TOKEN)
    vk_service = vk_session_service.get_api()
    agent = VKAgent(vk, vk_service)
    memes = {
        "title": 'title',
        "image_url": 'https://cs16.pikabu.ru/s/2026/11/15/17/sjs4esre_lg.png',
        "post_url": 'https://pikabu.ru/story/deystvitelno_14529468',
        "rating": 1
    }

    response = agent.post_meme(memes, "example")
    if "error" in response:
        print("Ошибка:", response["error"])
    else:
        print(f"Успех!")

if __name__ == "__main__":
    main()

