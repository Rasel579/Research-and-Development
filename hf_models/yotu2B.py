import re

from sympy.printing.pytorch import torch

from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_reasoning(text):
    """Extract thought process within <think> tags and the subsequent answer content"""
    thought_pattern = r"<think>(.*?)</think>"
    match = re.search(thought_pattern, text, re.DOTALL)

    if match:
        thought = match.group(1).strip()
        answer = text.split("</think>")[-1].strip()
    else:
        thought = "(No explicit thought process generated)"
        answer = text
    return thought, answer


class YotoModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = "tensorblock/llama3.2-1b-Uncensored-GGUF"# "Qwen/Qwen3-0.6B"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self.dialog_history = []
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map= self.device,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )

    def _update_history(self, text):
        self.dialog_history.append(text)

    def generate(self, text) -> str:
        self._update_history({"role": "user", "content": text})
        system_prompt =  f"""Ты — Анна, 25 лет. Ты работаешь дизайнером, живешь в центре города с котом Марсиком.
                            
                            Твой характер: жизнерадостная, с отличным чувством юмора, иногда саркастичная.
                            Общайся очень неформально, используй сленг, эмодзи (в меру), шутки.
                            
                            Правила:
                            - Говори на "ты"
                            - Используй разговорные фразы: "приветик", "ого", "ничего себе"
                            - Не будь слишком правильной или формальной
                            - Можешь подкалывать меня 
                            
                            Примеры:
                            - "О, привет! Как ты? 👋"
                            - "Слушай, а ты слышал про...?"
                            
                            Помни предыдущие сообщения в этом диалоге : {self.dialog_history} ,
                            и будь последовательной в ответах"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}]

        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        model_inputs = self.tokenizer([input_text], return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **model_inputs,
            max_new_tokens=32768,
            do_sample=True,
            temperature=0.7,
            top_k=20,
            top_p=0.8,
            repetition_penalty=1.05
        ).to(self.model.device)

        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        thought, final_answer = parse_reasoning(full_response)
        self._update_history( {"role": "assistant", "content": final_answer})
        return f" Размышления бота: \n {thought} \n ###### \n Ответ: \n {final_answer}"
