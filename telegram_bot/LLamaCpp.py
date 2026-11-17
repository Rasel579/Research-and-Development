import base64
import os
import time

import requests
import json

OLLAMA_URL = os.environ['LLM_URI'] + '/v1/chat/completions'
MODEL = "~/bonsai/Bonsai-demo/models/gguf/27B/Bonsai-27B-Q1_0.gguf"

class LLamaCpp:
    def __init__(self, user_id, vk):
        self.TOOL_IMPL = None
        self.github_api = None
        self.uri = OLLAMA_URL
        self.tools = None
        self.model = MODEL
        self.vk = vk
        self.user_id = user_id

    def response(self, user_id, prompt, vk, tools, tool_impl, system_prompt, max_iters: int = 28):
        self.tools = tools
        self.TOOL_IMPL = tool_impl
        print(prompt)
        msg = [{"role": "system", "content": system_prompt},
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

    def img_ocr(self, img_uri):
        img_response = requests.get(img_uri, timeout=10)
        temp_filename = f"temp_meme_{int(time.time())}.jpg"

        with open(temp_filename, "wb") as f:
            f.write(img_response.content)
        with open(temp_filename, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode("utf-8")
        mime = "image/jpeg"

        msg = [ {"role": "user", "content": [
                {"type": "text", "text": "Опиши этот мем в одном предложении на русском. Что на нём происходит? Какая шутка?"},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_base64}"}}
            ]}]
        try:
          response = requests.post(
                  self.uri,
                 json={
                "model": self.model,
                "messages": msg,
                "max_tokens": 150,
                "temperature": 0.3
            },
            timeout=60)
          response.raise_for_status()
        except Exception as e:
            print(f"Ошибка отправки сообщения: {e}")
            self.vk.messages.send(
                user_id=self.user_id,
                message=f"Ошибка отправки сообщения: {e}",
                random_id=0
            )
        description = response.json()["choices"][0]["message"]["content"].strip()
        print(f"🤖 Vision сказал: {description}")
        self.vk.messages.send(
            user_id=self.user_id,
            message=description,
            random_id=0
        )
        import os
        os.remove(temp_filename)
        return description




