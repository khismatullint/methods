
import os
import logging
import asyncio
from contextlib import suppress
from typing import Dict, Optional

import httpx
from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.utils import executor
from dotenv import load_dotenv

# ────────────────────────────
# Инициализация окружения
# ────────────────────────────
load_dotenv()

# ────────────────────────────
# Настройка логирования
# ────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ────────────────────────────
# Конфигурация приложения
# ────────────────────────────
class Config:
    """Класс для работы с конфигурацией приложения"""

    def __init__(self) -> None:
        self.API_TOKEN = self._get_env_var("TELEGRAM_API_TOKEN")
        self.YANDEX_GPT_API_KEY = self._get_env_var("YANDEX_GPT_API_KEY")
        self.YANDEX_FOLDER_ID = self._get_env_var("YANDEX_FOLDER_ID")
        self.HYPERBOLIC_API_KEY = self._get_env_var("HYPERBOLIC_API_KEY")

    @staticmethod
    def _get_env_var(name: str) -> str:
        """Получение переменной окружения с валидацией"""
        value = os.getenv(name)
        if not value:
            logger.error("❌ Отсутствует обязательная переменная окружения: %s", name)
            exit(1)
        return value


# ────────────────────────────
# FSM‑состояния
# ────────────────────────────
class Form(StatesGroup):
    profession = State()
    experience = State()
    goals = State()
    skills = State()
    preferences = State()


# ────────────────────────────
# Базовый HTTP‑клиент
# ────────────────────────────
class APIClient:
    """Базовый класс для API‑клиентов"""

    def __init__(self, timeout: int = 30) -> None:
        # глобальный тайм‑аут для всех запросов
        self.client = httpx.AsyncClient(timeout=timeout)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.client.aclose()


# ────────────────────────────
# YandexGPT
# ────────────────────────────
class YandexGPTClient(APIClient):
    BASE_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

    def __init__(self, api_key: str, folder_id: str, timeout: int = 30) -> None:
        super().__init__(timeout)
        self.headers = {
            "Authorization": f"Api-Key {api_key}",
            "Content-Type": "application/json",
        }
        self.model_uri = f"gpt://{folder_id}/yandexgpt-lite"

    async def generate_roadmap(self, user_data: Dict[str, str]) -> Optional[str]:
        payload = {
            "modelUri": self.model_uri,
            "completionOptions": {
                "stream": False,
                "temperature": 0.7,
                "maxTokens": 1500,
            },
            "messages": [
                {
                    "role": "system",
                    "text": "Ты карьерный консультант. Составь детальный персонализированный роадмап.",
                },
                {"role": "user", "text": self._build_prompt(user_data)},
            ],
        }

        try:
            response = await self.client.post(
                self.BASE_URL, headers=self.headers, json=payload
            )
            response.raise_for_status()
            return response.json()["result"]["alternatives"][0]["message"]["text"]
        except Exception as exc:  # noqa: BLE001
            logger.error("YandexGPT Error: %s", exc, exc_info=True)
            return None

    @staticmethod
    def _build_prompt(data: Dict[str, str]) -> str:
        """Формирование промпта с безопасным доступом к ключам"""
        return (
            f"Составь детальный карьерный роадмап для профессии {data.get('profession', '-')}. "
            f"Учитывая что у пользователя текущий опыт: {data.get('experience', '-')}, "
            f"карьерные цели: {data.get('goals', '-')}, текущие навыки: {data.get('skills', '-')}, "
            f"и предпочтения: {data.get('preferences', '-')}. Включи:\n"
            "1. Поэтапный план развития\n2. Рекомендуемые обучающие ресурсы\n"
            "3. Ключевые навыки для развития\n4. Рекомендации по нетворкингу\n"
            "5. Потенциальные карьерные треки"
        )


# ────────────────────────────
# Hyperbolic (Llama‑3.3‑70B‑Instruct)
# ────────────────────────────
class HyperbolicClient(APIClient):
    BASE_URL = "https://api.hyperbolic.xyz/v1/chat/completions"

    def __init__(self, api_key: str, timeout: int = 30) -> None:
        super().__init__(timeout)
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    async def generate_analysis(self, prompt: str) -> Optional[str]:
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "model": "meta-llama/Llama-3.3-70B-Instruct",
            "max_tokens": 512,
            "temperature": 0.1,
            "top_p": 0.9,
        }
        try:
            response = await self.client.post(
                self.BASE_URL, headers=self.headers, json=payload
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            logger.error("Hyperbolic API Error: %s", exc, exc_info=True)
            return None


# ────────────────────────────
# Основной бот
# ────────────────────────────
class RoadmapGeneratorBot:
    """Бот, объединяющий обе модели"""

    # порядок вопросов (строки, а не State‑объекты)
    _STATE_ORDER = [
        "profession",
        "experience",
        "goals",
        "skills",
        "preferences",
    ]

    _QUESTIONS = {
        "profession": "📌 Назовите профессию или должность, для которой хотите получить роадмап:",
        "experience": "🎯 Какой у вас текущий уровень опыта?\n(начинающий/средний/профессионал)",
        "goals": "🚀 Какие главные карьерные цели вы хотите достичь в ближайшие 3 года?",
        "skills": "💡 Перечислите ваши текущие ключевые навыки (через запятую):",
        "preferences": "🌟 Есть ли особые предпочтения или ограничения?\n(удаленная работа, индустрия и т.д.)",
    }

    def __init__(self, config: Config) -> None:
        self.bot = Bot(token=config.API_TOKEN)
        self.dp = Dispatcher(self.bot, storage=MemoryStorage())
        self.yandex_client = YandexGPTClient(
            config.YANDEX_GPT_API_KEY, config.YANDEX_FOLDER_ID
        )
        self.hyperbolic_client = HyperbolicClient(config.HYPERBOLIC_API_KEY)

        # регистрация хэндлеров
        self.dp.register_message_handler(self.start_command, commands=["start"], state="*")
        self.dp.register_message_handler(
            self.process_answer,
            state=[
                Form.profession,
                Form.experience,
                Form.goals,
                Form.skills,
                Form.preferences,
            ],
        )

    # ── Handlers ────────────────────────────── #
    async def start_command(self, message: types.Message):
        await message.answer(
            "🌟 Добро пожаловать в CareerRoadmapBot!\n"
            "Я помогу составить персонализированный карьерный план развития.\n"
            "Ответьте на несколько вопросов для начала:"
        )
        await self._ask_question(message, Form.profession)

    async def _ask_question(self, message: types.Message, state: State):
        await state.set()
        await message.answer(self._QUESTIONS[state.state.split(":")[1]])

    async def process_answer(self, message: types.Message, state: FSMContext):
        """Обрабатывает ответ пользователя и двигает форму дальше"""
        current_state_full = await state.get_state()  # напр. 'Form:profession'
        if not current_state_full:
            return

        state_name = current_state_full.split(":")[1]
        async with state.proxy() as data:
            data[state_name] = message.text.strip()

        # определить следующее состояние
        try:
            idx = self._STATE_ORDER.index(state_name)
            next_state_name = self._STATE_ORDER[idx + 1] if idx + 1 < len(self._STATE_ORDER) else None
        except ValueError:
            next_state_name = None

        if next_state_name:
            await self._ask_question(message, getattr(Form, next_state_name))
        else:
            await self._generate_and_send_roadmap(message, state)

    # ── Генерация роадмапа ───────────────────── #
    async def _generate_and_send_roadmap(self, message: types.Message, state: FSMContext):
        await message.answer("🔍 Анализирую данные... Это займет 1‑2 минуты")

        try:
            user_data = await state.get_data()
            base_prompt = YandexGPTClient._build_prompt(user_data)

            yandex_resp, hyperbolic_resp = await asyncio.gather(
                self.yandex_client.generate_roadmap(user_data),
                self.hyperbolic_client.generate_analysis(
                    f"Дай дополнительные рекомендации для этого запроса: {base_prompt}"
                ),
            )

            await message.answer(self._format_responses(yandex_resp, hyperbolic_resp), parse_mode="Markdown")
        except Exception as exc:  # noqa: BLE001
            logger.error("Ошибка генерации: %s", exc, exc_info=True)
            await message.answer("⚠️ Произошла ошибка при генерации роадмапа")
        finally:
            with suppress(Exception):
                await state.finish()
                await message.answer(
                    "✅ Готово! Можете начать заново с /start"
                )

    @staticmethod
    def _format_responses(yandex: Optional[str], hyperbolic: Optional[str]) -> str:
        def safe(text: Optional[str], default: str) -> str:
            return text.strip() if text else default

        return (
            "🚀 **Основной роадмап от YandexGPT:**\n\n" + safe(yandex, "Не удалось получить основной роадмап") + "\n\n" +
            "🔍 **Дополнительные рекомендации от Llama‑3:**\n\n" + safe(hyperbolic, "Не удалось получить дополнительные рекомендации")
        )

    # ── Запуск ───────────────────────────────── #
    def run(self):
        logger.info("Запуск бота…")
        executor.start_polling(self.dp, skip_updates=True)


# ────────────────────────────
# Entry point
# ────────────────────────────
if __name__ == "__main__":
    cfg = Config()
    RoadmapGeneratorBot(cfg).run()


