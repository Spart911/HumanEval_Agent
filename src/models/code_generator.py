"""Утилиты генерации кода для моделей LLM."""
import torch
import logging
import textwrap
from typing import Optional, Dict, Any
from transformers import AutoTokenizer


from ..utils.code_utils import check_code_in_subprocess, strip_code_fences

logger = logging.getLogger(__name__)


def generate_code_with_model(
        prompt: str,
        tokenizer: AutoTokenizer,
        model,
        device: torch.device,
        generation_config: Optional[Dict[str, Any]] = None,
        iterations: int = 3
) -> str:
    """
    Генерация кода с использованием модели и итеративного исправления ошибок.

    Args:
        prompt: Промпт задачи.
        tokenizer: Токенизатор для модели.
        model: Языковая модель.
        device: Устройство для выполнения генерации.
        generation_config: Опциональная конфигурация для параметров генерации.
        iterations: Количество попыток генерации/исправления кода.

    Returns:
        Лучший сгенерированный код после итеративного улучшения.
    """
    # Конфигурация генерации по умолчанию
    default_config = {
        "max_new_tokens": 400,
        "do_sample": True,
        "temperature": 0.6,
        "top_k": 40,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id
    }

    # Обновление конфигурацией от пользователя
    if generation_config:
        default_config.update(generation_config)

    base_prompt = prompt
    current_prompt = prompt
    last_successful = ""
    all_steps = []

    for step in range(iterations):
        logger.info(f"🧩 Итерация {step + 1}/{iterations}: уточнение кода...")

        # Токенизация промпта
        inputs = tokenizer(current_prompt, return_tensors="pt", padding=True, truncation=True)
        # Переносим тензоры на устройство
        inputs = {k: v.to(device) for k, v in inputs.items()}

        try:
            # Генерация кода
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                    **default_config
                )

            # Декодирование сгенерированного текста
            gen_suffix = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

            # Очищаем ограждения markdown и выравниваем
            gen_suffix = strip_code_fences(gen_suffix)
            gen_suffix = textwrap.dedent(gen_suffix).strip()
            all_steps.append(gen_suffix)

            # Валидация сгенерированного кода в subprocess
            success, error_msg = check_code_in_subprocess(gen_suffix, timeout=6)

            if success:
                logger.info(f"✅ Валидация кода пройдена на итерации {step + 1}")
                last_successful = gen_suffix
                return last_successful.strip()
            else:
                logger.error(f"❌ Валидация кода не пройдена: {error_msg}")

                # Подготовка промпта для следующей итерации с информацией об ошибке
                current_prompt = (
                    f"{base_prompt}\\n\\n"
                    f"Previous solution:\\n{gen_suffix}\\n\\n"
                    f"The last attempt failed during automatic checking with the following error (exact text):\\n"
                    f"```\\n{error_msg}\\n```\\n\\n"
                    "Analyze the code carefully and return a corrected, fully functional, syntactically correct Python function.\\n"
                    "Keep the same function name and parameters. Fix all indentation and syntax errors and any runtime error "
                    "reported above. Do not include any explanations, comments or additional text — return only the updated code.\\n"
                )
                last_error = error_msg
                # Переходим к следующей итерации
        except Exception as e:
            logger.error(f"Ошибка во время генерации: {e}")
            continue

    logger.warning("⚠️ Итеративное уточнение не дало корректного кода.")

    if last_successful:
        logger.info("Возвращаем последнюю успешную скомпилированную версию.")
        return last_successful.strip()
    else:
        logger.info("Возвращаем последнюю сгенерированную версию (без успешной валидации).")
        return all_steps[-1].strip() if all_steps else ""


def generate_single_turn_code(
        prompt: str,
        tokenizer: AutoTokenizer,
        model,
        device: torch.device,
        generation_config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Генерация кода за один проход без валидации.

    Args:
        prompt: Промпт задачи.
        tokenizer: Токенизатор для модели.
        model: Языковая модель.
        device: Устройство для выполнения генерации.
        generation_config: Опциональная конфигурация для параметров генерации.

    Returns:
        Сгенерированный код без валидации.
    """
    # Конфигурация генерации по умолчанию
    default_config = {
        "max_new_tokens": 400,
        "do_sample": True,
        "temperature": 0.6,
        "top_k": 40,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id
    }

    # Обновление конфигурацией от пользователя
    if generation_config:
        default_config.update(generation_config)

    # Токенизация промпта
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    try:
        # Генерация кода
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                **default_config
            )

        # Декодирование и очистка
        gen_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        gen_text = strip_code_fences(gen_text)
        gen_text = textwrap.dedent(gen_text).strip()

        return gen_text
    except Exception as e:
        logger.error(f"Ошибка во время генерации кода: {e}")
        return ""