import os
import time
from urllib.parse import quote, urlsplit, urlunsplit, urljoin

import requests
import vk_api
from bs4 import BeautifulSoup
from vk_api.longpoll import VkEventType, VkLongPoll
from vk_api.vk_api import VkApiMethod
from vk_api.upload import VkUpload
from LLamaCpp import LLamaCpp

BLOCK_TAGS = {"текст", "18+", "nsfw", "реклама", "блоги компаний",
              "мат", "негатив", "просьба", "ответ на пост"}

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
        "description": "Ищет и возвращает мем по заданной категории (тегу). Возвращает объект мема, который можно проверить соотвествие описания картинки заданной тематики через тулзу `img_ocr`. После проверки можно и постить через тулзу `post_meme` Использует парсинг.",
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
    },
},
{
    "type": "function",
    "function": {
        "name": "img_ocr",
        "description": "Описывает найденный мем по картинке",
        "parameters": {
            "type": "object",
            "properties": {
                "img_uri": {
                    "type": "string",
                    "description": "Путь к файлу",
                },
            },
            "required": ["img_uri"]
        }
    },
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
также есть тулза `get_meme`, которая получает мем по тегу,
и есть тулза `img_ocr` для описания мема из полученного `get_meme`

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
1. Перед отправкой мема проверь полученный мем.
2. Перед отправкой мема кратко прокомментируй его в чате — объясни, почему именно сейчас.
3. Текст мема должен быть коротким, ёмким, без душноты.
4. Никогда не используй мемы на чувствительные темы (политика, трагедии, болезни).
5. Если не уверен, уместен ли мем — лучше ответь текстом.
6. Не повторяйся: если вчера кидал мем про программистов, сегодня найди другую тему.

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

def _clean_url(u: str) -> str:
    u = (u or "").strip()
    if u.startswith("//"):
        u = "https:" + u
    p = urlsplit(u)
    return urlunsplit((p.scheme, p.netloc, p.path, "", ""))

class VKAgent:
    def __init__(self, user_id, vk_session: VkApiMethod, vk_group: VkApiMethod):
        self.user_id = user_id
        self.vk_session = vk_session
        self.vk = vk_session
        self.tools = TOOLS
        self.upload = VkUpload(vk_group)

    def get_meme(self, meme_type: str, limit: int = 3, min_rating: int = 0 ):
        url = f"{MEME_URI}/tag/{quote(meme_type)}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": MEME_URI,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "ru-RU,ru;q=0.9",
        }
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"❌ Ошибка при запросе: {e}")
            self.vk_session.messages.send(
                    user_id=self.user_id,
                    message=f"❌ Ошибка при запросе: {e}",
                    random_id=0
                )
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        memes = []
        stats = []
        stories = soup.select("article.story[data-story-id]")

        def rej(reason):
            stats[reason] = stats.get(reason, 0) + 1
        for story in stories:
            if len(memes) >= limit:
                break
            if (story.select_one(".story__placeholder, .story__promo, .story__advert-info")
                    or story.get("data-is-fixed-story") == "true"):
                rej("ad")
                continue
            raw = story.get("data-rating")
            if raw is None:
                rej("no_rating")
                continue
            try:
                rating = int(raw)
            except ValueError:
                rating = 0

            if rating < min_rating:
                continue
            tags = [a.get("data-tag", "").strip()
                    for a in story.select("a.tags__tag")]
            tagset = {t.lower().strip("[]") for t in tags if t}
            if tagset & BLOCK_TAGS:
                rej("blocked_tags")
                continue

            imgs = []
            for img in story.select("img.story-image__image, .carousel__item img"):
                u = _clean_url(img.get("data-large-image")
                               or img.get("data-src") or img.get("src"))
                if u and u not in imgs:
                    imgs.append(u)
            if not imgs:
                rej("no_image")
                continue
            title = story.select_one("a.story__title-link")

            memes.append({
                "title": title.get_text(strip=True) if title else "",
                "image_url": imgs[0],
                "post_url": _clean_url(urljoin(MEME_URI, title["href"])) if title and title.has_attr("href") else "",
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
    agent = VKAgent(user_id, vk,vk_service)
    tool_impl = { "post_meme": agent.post_meme,
                  "get_meme": agent.get_meme,
                  "img_ocr": llm.img_ocr
                }
    return llm.response(user_id, msg, vk, TOOLS, tool_impl, SYSTEM_PROMPT)

def main():
    vk_session = vk_api.VkApi(token=TOKEN)
    vk = vk_session.get_api()
    longpoll = VkLongPoll(vk_session)
    vk_session_service = vk_api.VkApi(token=SERVICE_TOKEN)
    vk_service = vk_session_service.get_api()
    print("Бот запущен и слушает сообщения...")

    for event in longpoll.listen():
        if event.type == VkEventType.MESSAGE_NEW and event.to_me and event.text:
            user_message = event.text.lower()
            try:
                llama_cpp = LLamaCpp(event.user_id, vk)
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
    agent = VKAgent(1,vk, vk_service)
    memes = {
        "title": 'title',
        "image_url": 'https://cs16.pikabu.ru/s/2026/11/15/17/sjs4esre_lg.png',
        "post_url": 'https://pikabu.ru/story/deystvitelno_14529468',
        "rating": 1
    }

    response = agent.get_meme("баг", 1)
    if "error" in response:
        print("Ошибка:", response["error"])
    else:
        print(f"Успех! {response}")

if __name__ == "__main__":
    main()

