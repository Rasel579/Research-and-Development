import os
import requests
import json

OLLAMA_URL = os.environ['LLM_URI'] + '/v1/chat/completions'
MODEL = "~/bonsai/Bonsai-demo/models/gguf/27B/Bonsai-27B-Q1_0.gguf"

class LLamaCpp:
    def __init__(self, vk):
        self.TOOL_IMPL = None
        self.github_api = None
        self.uri = OLLAMA_URL
        self.tools = None
        self.model = MODEL
        self.vk = vk

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


