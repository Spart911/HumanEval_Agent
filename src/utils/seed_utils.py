"""Утилиты для установки случайных seed'ов для повторяемости результатов."""
import os
import random
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def set_random_seed(seed: int) -> None:
    """
    Установка глобального случайного seed'а для Python, NumPy и PyTorch.

    Args:
        seed: Значение seed'а для установки.
    """
    # Установка seed'а для Python random
    random.seed(seed)

    # Установка seed'а для Python hash randomization
    os.environ['PYTHONHASHSEED'] = str(seed)

    try:
        # Установка seed'а для NumPy
        import numpy as np
        np.random.seed(seed)
        logger.info(f"Установлен NumPy random seed: {seed}")
    except ImportError:
        logger.warning("NumPy не установлен, пропускаю установку NumPy seed")

    try:
        # Установка seed'а для PyTorch
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Дополнительная фиксация для генераторов CUDA
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.info(f"Установлен PyTorch random seed: {seed}")
    except ImportError:
        logger.warning("PyTorch не установлен, пропускаю установку PyTorch seed")

    try:
        # Установка seed'а для Transformers
        import transformers
        transformers.set_seed(seed)
        # Дополнительная фиксация для генерации
        if hasattr(transformers, 'torch'):
            transformers.torch.manual_seed(seed)
        logger.info(f"Установлен Transformers seed: {seed}")
    except ImportError:
        logger.warning("Transformers не установлен, пропускаю установку Transformers seed")
    except Exception as e:
        logger.warning(f"Ошибка при установке Transformers seed: {e}")

    # Проверка установки seed
    test_random = random.randint(0, 1000)
    logger.info(f"Установлен глобальный random seed: {seed} (test: {test_random})")


def make_generation_deterministic(generation_config: dict) -> dict:
    """
    Модификация конфигурации генерации для детерминированной генерации.

    Args:
        generation_config: Исходная конфигурация генерации.

    Returns:
        Модифицированная конфигурация с детерминированными параметрами.
    """
    deterministic_config = generation_config.copy()

    # Для детерминированной генерации отключаем сэмплирование
    deterministic_config['do_sample'] = False
    deterministic_config['temperature'] = 0.0
    deterministic_config['top_k'] = 1
    deterministic_config['top_p'] = 1.0

    # Убираем случайные параметры, которые могут влиять на генерацию
    if 'num_beams' in deterministic_config:
        deterministic_config['num_beams'] = 1

    logger.info("Конфигурация генерации изменена на детерминированную")
    return deterministic_config
