"""Утилиты для установки случайных seed'ов для повторяемости результатов."""
import os
import random
import logging
import sys
from typing import Optional

logger = logging.getLogger(__name__)


def set_random_seed(seed: int) -> None:
    """
    Установка seed'ов для Python/NumPy (и PyTorch, если он уже импортирован).

    Args:
        seed: Значение seed'а для установки.
    """
    # Установка seed'а для Python random
    random.seed(seed)

    # Установка seed'а для Python hash randomization (важно: для полного эффекта лучше задавать как env ДО старта процесса)
    os.environ['PYTHONHASHSEED'] = str(seed)

    try:
        # Установка seed'а для NumPy
        import numpy as np
        np.random.seed(seed)
        logger.info(f"Установлен NumPy random seed: {seed}")
    except ImportError:
        logger.warning("NumPy не установлен, пропускаю установку NumPy seed")

    # ВАЖНО: не импортируем torch принудительно (чтобы не нарушать раннюю настройку env для детерминированности).
    # Если torch уже импортирован где-то выше по стеку, можем дополнительно засеять его генераторы.
    if "torch" in sys.modules:
        try:
            import torch  # noqa: F401
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            logger.info(f"Установлен PyTorch random seed: {seed}")
        except Exception as e:
            logger.warning(f"Не удалось установить PyTorch seed (torch уже импортирован): {e}")

    try:
        # Установка seed'а для Transformers
        import transformers
        transformers.set_seed(seed)
        logger.info(f"Установлен Transformers seed: {seed}")
    except ImportError:
        logger.warning("Transformers не установлен, пропускаю установку Transformers seed")
    except Exception as e:
        logger.warning(f"Ошибка при установке Transformers seed: {e}")

    logger.info(f"Seed установлен: {seed}")


def configure_determinism_env() -> None:
    """
    Настраивает переменные окружения для максимальной детерминированности.

    Критично: эту функцию нужно вызывать ДО первого импорта/инициализации torch/CUDA,
    иначе часть настроек (например, CUBLAS_WORKSPACE_CONFIG) не подействует.
    """
    # Отключаем многопоточность
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

    # Отключаем TF32 для точности
    os.environ.setdefault('NVIDIA_TF32_OVERRIDE', '0')

    # Требование PyTorch для детерминированности CuBLAS (CUDA >= 10.2)
    # См. сообщение: "set CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8 before running"
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

    logger.info("Переменные окружения для детерминированности настроены (CUBLAS_WORKSPACE_CONFIG и т.п.)")


def enable_torch_determinism(strict: bool = True) -> None:
    """
    Включает детерминированные алгоритмы PyTorch.

    Args:
        strict: Если True — PyTorch будет падать на недетерминированных операциях.
                Если False — будет лишь предупреждать (если поддерживается версией).
    """
    try:
        import torch
    except Exception as e:
        logger.warning(f"PyTorch недоступен, пропускаю enable_torch_determinism(): {e}")
        return

    # CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # TF32
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False

    # Глобальные детерминированные алгоритмы
    try:
        # PyTorch >= 1.8: есть warn_only
        torch.use_deterministic_algorithms(True, warn_only=not strict)  # type: ignore[arg-type]
    except TypeError:
        # Старые версии PyTorch без warn_only
        torch.use_deterministic_algorithms(True)

    logger.info(f"PyTorch детерминированный режим включен (strict={strict})")


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
