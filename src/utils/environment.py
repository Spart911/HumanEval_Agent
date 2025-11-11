"""Утилиты для работы с окружением и конфигурацией."""
import os
import logging
import torch
from typing import Dict, Any, Optional
from huggingface_hub import login

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_dotenv_variables() -> Dict[str, str]:
    """Загрузка переменных окружения из .env файла."""
    env_vars = {}
    try:
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
    except FileNotFoundError:
        logger.warning("Файл .env не найден")
    except Exception as e:
        logger.error(f"Ошибка при загрузке файла .env: {e}")

    return env_vars


def login_to_huggingface(api_key: Optional[str] = None) -> bool:
    """Вход в HuggingFace Hub с использованием API ключа.

    Args:
        api_key: API ключ HuggingFace. Если None, пытается получить из окружения.

    Returns:
        True, если вход успешен, иначе False.
    """
    if api_key is None:
        env_vars = load_dotenv_variables()
        api_key = env_vars.get('KEY_HUGGINGFACE')

    if api_key:
        try:
            login(token=api_key)
            logger.info("Успешный вход в HuggingFace Hub")
            return True
        except Exception as e:
            logger.error(f"Не удалось войти в HuggingFace Hub: {e}")
            return False
    else:
        logger.warning("API ключ HuggingFace не найден")
        return False


def get_device() -> torch.device:
    """Получить подходящее устройство для работы с моделью.

    Returns:
        torch.device: CUDA/CPU/MPS.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Используется устройство CUDA")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Используется устройство MPS (MacOS Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        logger.info("Используется устройство CPU")

    return device


def get_model_paths() -> Dict[str, str]:
    """Получить пути к моделям из переменных окружения.

    Returns:
        Словарь с ключами MAIN_MODEL_PATH и BASE_MODEL_PATH.
    """
    env_vars = load_dotenv_variables()
    model_paths = {
        'main_model_path': env_vars.get('MAIN_MODEL_PATH'),
        'base_model_path': env_vars.get('BASE_MODEL_PATH')
    }

    if not model_paths['main_model_path']:
        logger.warning("MAIN_MODEL_PATH не найден в переменных окружения")
    if not model_paths['base_model_path']:
        logger.warning("BASE_MODEL_PATH не найден в переменных окружения")

    return model_paths
