from transformers import GPTLanguageModel
from bpe import  BasicTokenizer
import torch
import os
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackContext
import asyncio


def get_vocab_size(tokenizer_param: BasicTokenizer):
    return len(tokenizer_param.vocab) + len(tokenizer_param.special_tokens)


class Bot:
    def __init__(self, token: str, model_path: str, tokenizer_pass: str):
        self.device = None
        self.bot = None
        self.model = None
        self.tokenizer = None
        self.token = token
        self.model_path = model_path
        self.tokenizer_pass = tokenizer_pass
        self.load_model()
        self.setup_bot()



    def load_model(self):
        tokenizer = BasicTokenizer()
        tokenizer.load(model_file=self.tokenizer_pass)
        self.tokenizer = tokenizer

        block_size = 256
        n_embedding = 512
        n_head = 8
        n_layer = 4
        dropout = 0.2
        batch_size = 64
        vocab_size = get_vocab_size(tokenizer)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = GPTLanguageModel(
            vocab_size=vocab_size,
            block_size=block_size,
            n_embeddings=n_embedding,
            n_head=n_head,
            device=self.device,
            n_layers=n_layer,
            dropout=dropout
        )
        model.to(self.device)
        model = torch.compile(model)

        checkpoint = torch.load(self.model_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        self.model = model

    def setup_bot(self):
       self.bot = Application.builder().token(self.token).build()
       self.bot.add_handler(CommandHandler("start", self.start_command))
       self.bot.add_handler(CommandHandler("help", self.help_command))
       self.bot.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

    async def start_command(self, update: Update, context: CallbackContext):
        await update.message.reply_text(
            "🚀 Добро пожаловать в AI-бота!\n"
            "Отправьте мне текст, и я сгенерирую продолжение.\n"
            "Команды:\n"
            "/start - начать работу\n"
            "/help - помощь"        )

    async  def help_command(self, update: Update, context: CallbackContext):
        await update.message.reply_text(
            "📖 Помощь по использованию бота:\n"
            "• Просто напишите сообщение - я отвечу\n"
            "• Бот использует языковую модель для генерации текста"
        )

    async def handle_message(self, update: Update, context: CallbackContext):
        user_text = update.message.text
        if len(user_text.split()) < 3:
            await update.message.reply_text("Напишите более развернутое сообщение для генерации.")
            return
        status_message = await update.message.reply_text("💭 Думаю...")
        response = await self.generate_text(user_text)
        await status_message.edit_text(f"🤖 {response}")

    async def generate_text(self, user_text: str):
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, self._generate_sync, user_text
            )
            return response

        except Exception as e:
            return "⚠️ Произошла ошибка при генерации. Попробуйте еще раз."

    def _generate_sync(self, user_text: str):
        input_tokens = self.tokenizer.encode(user_text)
        input_tokens = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            output = self.model.generate(input_tokens, 100)
        a = output[0]
        response = self.tokenizer.decode(a.tolist())
        if response.startswith(user_text):
            response = response[len(user_text):].strip()
        return response

    def run_bot(self):
        print("🤖 Запускаю Telegram бота...")
        self.bot.run_polling()


if __name__ == '__main__':
   BOT_API = os.getenv('TELEGRAM_TOKEN')
   MODEL_PATH = f'../output/pretrain/v3/checkpoint100.pth'
   TOKENIZER_PATH = '../output/tokenizer/tokenzier_v1.model'
   bot = Bot(BOT_API, MODEL_PATH, TOKENIZER_PATH)
   bot.run_bot()

