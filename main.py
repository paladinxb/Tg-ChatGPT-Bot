import telebot
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
os.environ["HUGGINGFACE_TOKEN"] = "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"
model_name = "EleutherAI/gpt-j-6B"
# Загрузка модели и токенизатора ChatGPT
openai_api_key = "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"
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", 
                                              from_tf=False, 
                                              gradient_checkpointing=False, 
                                              api_key=openai_api_key)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)


# Инициализация бота Telegram
bot = telebot.TeleBot("5891690608:AAE6Mw5-M_HLkhXwSa1Cz_MYsy-OPXaFdos")

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