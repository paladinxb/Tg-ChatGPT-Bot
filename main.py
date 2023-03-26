import telebot
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
os.environ["HUGGINGFACE_TOKEN"] = "Your Token"
model_name = "Whatever model name"
# Загрузка модели и токенизатора ChatGPT
openai_api_key = "Your openai API key"
model = AutoModelForCausalLM.from_pretrained("model name", 
                                              from_tf=False, 
                                              gradient_checkpointing=False, 
                                              api_key=openai_api_key)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)


# Инициализация бота Telegram
bot = telebot.TeleBot("token of tg bot")
# Обработчик входящих сообщений от пользователей
@bot.message_handler(func=lambda message: True)

def handle_message(message):
    # Генерация ответа с помощью ChatGPT
    input_ids = tokenizer.encode(message.text, return_tensors="pt")
    output_ids = model.generate(input_ids)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Отправка ответа пользователю через Telegram бота
    bot.send_message(message.chat.id, output_text)

# Запуск бота
bot.polling()
