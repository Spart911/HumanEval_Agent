"""Утилиты для загрузки и управления моделями."""
import os
import torch
import logging
from typing import Union, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, QuantoConfig
from peft import PeftModel
from src.utils.environment import get_device, login_to_huggingface

logger = logging.getLogger(__name__)


def setup_quantization(config: Dict[str, Any] = None) -> QuantoConfig:
    """Настройка конфигурации квантизации для загрузки модели."""
    if config is None:
        config = {}
    weights_type = config.get("weights", "int8")
    quantization_config = QuantoConfig(weights=weights_type)
    logger.info(f"Настройка квантизации {weights_type}")
    return quantization_config


def _expand_local_path(path: str) -> str:
    """Разворачивает ~ и проверяет локальное существование пути."""
    expanded = os.path.expanduser(path)
    if os.path.exists(expanded):
        return expanded
    return path  # если это repo_id (например, "Qwen/Qwen2.5-Coder-3B")


def load_model(
    model_path: str,
    base_model_path: str,
    device: Union[str, torch.device] = "auto",
    quantization_config: QuantoConfig = None,
    use_lora: bool = False,
    trust_remote_code: bool = True,
    torch_dtype: torch.dtype = torch.float16
) -> tuple:
    """
    Загрузка языковой модели с опциональной квантизацией и LoRA.
    """
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


    login_to_huggingface()

    model_path = _expand_local_path(model_path)
    if base_model_path:
        base_model_path = _expand_local_path(base_model_path)

    if device == "auto":
        device = get_device()
    elif isinstance(device, str):
        device = torch.device(device)

    if not os.path.exists(model_path) and "/" not in model_path:
        logger.error(f"Модель {model_path} не найдена локально и не указано имя репозитория.")
        raise ValueError(f"Model path {model_path} does not exist")

    # ---------------------- Выбор класса модели ----------------------
    try:
        from transformers import Qwen2ForCausalLM
        ModelClass = Qwen2ForCausalLM
        logger.info("Используем класс Qwen2ForCausalLM")
    except Exception:
        ModelClass = AutoModelForCausalLM
        logger.info("Используем универсальный класс AutoModelForCausalLM")

    try:
        # ---------------------- Загружаем токенизатор ----------------------
        logger.info(f"Загружаем токенизатор из {base_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=trust_remote_code,
            local_files_only=True
        )

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            logger.info("Установлен pad_token_id равным eos_token_id")

        # ---------------------- Аргументы загрузки модели ----------------------
        load_kwargs = {
            "trust_remote_code": trust_remote_code,
            "torch_dtype": torch_dtype,
            "local_files_only": True,
        }
        if quantization_config:
            load_kwargs["quantization_config"] = quantization_config

        if device.type == "cuda":
            load_kwargs["device_map"] = "cuda"
        else:
            load_kwargs["device_map"] = None

        # ---------------------- Загрузка модели ----------------------
        logger.info(f"Загружаем базовую модель из {AutoModelForCausalLM}")
        base_model = AutoModelForCausalLM.from_pretrained(base_model_path, **load_kwargs)

        logger.info(f"Загружаем адаптер LoRA из {model_path}")
        model = PeftModel.from_pretrained(base_model, model_path, local_files_only=True)
        model = model.to(device)

        model.eval()
        logger.info(f"✅ Модель успешно загружена на устройство {device}")

        return tokenizer, model, device

    except Exception as e:
        logger.error(f"❌ Не удалось загрузить модель: {e}")
        raise


def load_model_with_lora(
    lora_path: str,
    base_model_path: str,
    device: Union[str, torch.device] = "auto",
    torch_dtype: torch.dtype = torch.float16
) -> tuple:
    """Упрощённая загрузка модели с адаптером LoRA."""
    return load_model(
        model_path=lora_path,
        base_model_path=base_model_path,
        device=device,
        use_lora=True,
        torch_dtype=torch_dtype
    )


def load_base_model_only(
    base_model_path: str,
    device: Union[str, torch.device] = "auto",
    torch_dtype: torch.dtype = torch.float16,
    trust_remote_code: bool = True
) -> tuple:
    """
    Загрузка только базовой модели без каких-либо адаптеров или дообученных версий.

    Args:
        base_model_path: Путь к базовой модели
        device: Устройство для загрузки модели
        torch_dtype: Тип данных для модели
        trust_remote_code: Доверять удаленному коду

    Returns:
        Кортеж (tokenizer, model, device)
    """
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    login_to_huggingface()

    base_model_path = _expand_local_path(base_model_path)

    if device == "auto":
        device = get_device()
    elif isinstance(device, str):
        device = torch.device(device)

    if not os.path.exists(base_model_path) and "/" not in base_model_path:
        logger.error(f"Базовая модель {base_model_path} не найдена локально и не указано имя репозитория.")
        raise ValueError(f"Base model path {base_model_path} does not exist")

    try:
        # ---------------------- Загружаем токенизатор ----------------------
        logger.info(f"Загружаем токенизатор из {base_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=trust_remote_code,
            local_files_only=True
        )

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            logger.info("Установлен pad_token_id равным eos_token_id")

        # ---------------------- Аргументы загрузки модели ----------------------
        load_kwargs = {
            "trust_remote_code": trust_remote_code,
            "torch_dtype": torch_dtype,
            "local_files_only": True,
        }

        if device.type == "cuda":
            load_kwargs["device_map"] = "cuda"
        else:
            load_kwargs["device_map"] = None

        # ---------------------- Загрузка базовой модели ----------------------
        logger.info(f"Загружаем базовую модель из {base_model_path}")
        model = AutoModelForCausalLM.from_pretrained(base_model_path, **load_kwargs)

        model = model.to(device)
        model.eval()
        logger.info(f"✅ Базовая модель успешно загружена на устройство {device}")

        return tokenizer, model, device

    except Exception as e:
        logger.error(f"❌ Не удалось загрузить базовую модель: {e}")
        raise


def load_merged_model(
    lora_path: str,
    base_model_path: str,
    save_path: str = None,
    device: Union[str, torch.device] = "auto",
    torch_dtype: torch.dtype = torch.float16
) -> tuple:
    """Создание и опциональное сохранение объединённой модели LoRA + базовой."""
    tokenizer, lora_model, device = load_model_with_lora(lora_path, base_model_path, device, torch_dtype)

    logger.info("Объединяем адаптер LoRA с базовой моделью")
    merged_model = lora_model.merge_and_unload()

    if save_path:
        save_path = _expand_local_path(save_path)
        logger.info(f"💾 Сохраняем объединённую модель в {save_path}")
        os.makedirs(save_path, exist_ok=True)
        merged_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

    return tokenizer, merged_model, device
