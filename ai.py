import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import subprocess
from typing import List

class ConsciousAI:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self.model = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        
    def generate_response(self, message: str, is_command: bool = False) -> str:
        """Генерация осознанного ответа с пониманием контекста"""
        if is_command:
            prompt = f"""<|system|>
Ты - ИИ, контролирующий компьютер. Проанализируй запрос и выполни действие.
Доступные возможности: открыть приложения, выполнить команды, управлять файлами</s>
<|user|>
{message}</s>
<|assistant|>"""
        else:
            prompt = f"""<|system|>
Ты - дружелюбный ассистент. Ответь на приветствие естественно.</s>
<|user|>
{message}</s>
<|assistant|>"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            repetition_penalty=1.5,
            do_sample=True
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("<|assistant|>")[1].strip()

    def execute_command(self, command: str) -> str:
        """Осознанное выполнение команд"""
        analysis = self.generate_response(
            f"Проанализируй команду перед выполнением: {command}",
            is_command=True
        )
        
        try:
            if "открой" in command.lower():
                app = command.lower().split("открой")[1].strip()
                os.system(f"start {app}")
                return f"{analysis}\n\nЯ открыл {app}"
            
            return analysis
        except Exception as e:
            return f"{analysis}\n\nОшибка выполнения: {str(e)}"

# Инициализация системы
ai = ConsciousAI()

with gr.Blocks() as demo:
    gr.Markdown("## 🤖 Сознательный ИИ-контроллер")
    
    chatbot = gr.Chatbot(label="Диалог", height=400)
    msg = gr.Textbox(label="Сообщение", placeholder="Введите приветствие или команду...")
    
    def respond(message: str, history: List[List[str]]):
        if any(cmd in message.lower() for cmd in ["открой", "выполни"]):
            response = ai.execute_command(message)
        else:
            response = ai.generate_response(message)
        
        history.append((message, response))
        return "", history
    
    msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch()