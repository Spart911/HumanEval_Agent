"""Утилиты генерации кода для моделей LLM."""
import torch
import logging
import textwrap
from typing import Optional, Dict, Any
from transformers import AutoTokenizer


from ..utils.code_utils import check_code_in_subprocess, strip_code_fences

logger = logging.getLogger(__name__)


def _clean_and_prepare_gen_kwargs(tokenizer: AutoTokenizer, user_cfg: Optional[Dict[str, Any]], input_len: int) -> Dict[str, Any]:
    """Собираем kwargs для model.generate, фильтруем None и выставляем безопасный max_length.

    Notes:
    - На некоторых версиях transformers игнорируется max_new_tokens, если не указан max_length.
      Поэтому выставляем max_length = input_len + max_new_tokens для совместимости.
    - Исключаем значения None, чтобы не затирать корректные pad/eos token id.
    - Для детерминированной генерации (do_sample=False) удаляем параметры sampling.
    """
    base_cfg: Dict[str, Any] = {
        "max_new_tokens": 400,
        "do_sample": True,
        "temperature": 0.6,
        "top_k": 40,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if user_cfg:
        # убираем None значения
        cleaned = {k: v for k, v in user_cfg.items() if v is not None}
        base_cfg.update(cleaned)

    # Для детерминированной генерации (greedy decoding) удаляем параметры sampling
    if base_cfg.get("do_sample") is False:
        # Удаляем параметры, которые используются только при sampling
        base_cfg.pop("temperature", None)
        base_cfg.pop("top_k", None)
        base_cfg.pop("top_p", None)
        logger.info("Детерминированный режим: используется greedy decoding (do_sample=False)")

    # Удаляем seed из параметров генерации - он не используется model.generate()
    # seed устанавливается отдельно через set_seed() перед генерацией
    base_cfg.pop("seed", None)

    # убедимся, что pad/eos заданы
    if base_cfg.get("pad_token_id") is None:
        base_cfg["pad_token_id"] = tokenizer.eos_token_id
    if base_cfg.get("eos_token_id") is None:
        base_cfg["eos_token_id"] = tokenizer.eos_token_id

    # совместимость с версиями, ожидающими max_length
    max_new = int(base_cfg.get("max_new_tokens", 256))
    base_cfg["max_length"] = int(input_len + max_new)


    return base_cfg


def _generate_single_step(
    prompt: str,
    tokenizer: AutoTokenizer,
    model,
    device: torch.device,
    generation_config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Внутренняя функция для генерации кода за один шаг.
    Используется как в generate_code_with_model, так и в generate_single_turn_code
    для обеспечения идентичности результатов.
    
    Args:
        prompt: Промпт задачи.
        tokenizer: Токенизатор для модели.
        model: Языковая модель.
        device: Устройство для выполнения генерации.
        generation_config: Опциональная конфигурация для параметров генерации.
    
    Returns:
        Сгенерированный код после обработки.
    """
    # Токенизация промпта
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Вычисляем реальную длину промпта без padding для детерминированности
    # Используем attention_mask для определения реальной длины
    if "attention_mask" in inputs:
        actual_input_len = inputs["attention_mask"].sum().item()
    else:
        actual_input_len = inputs["input_ids"].shape[1]

    # Сбор совместимых kwargs для generate
    gen_kwargs = _clean_and_prepare_gen_kwargs(tokenizer, generation_config, actual_input_len)

    # Генерация кода
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            **gen_kwargs
        )

    # Декодирование сгенерированного текста
    gen_suffix = tokenizer.decode(outputs[0][actual_input_len:], skip_special_tokens=True).strip()

    # Очищаем ограждения markdown и выравниваем
    gen_suffix = strip_code_fences(gen_suffix)
    gen_suffix = textwrap.dedent(gen_suffix).strip()

    # Преобразуем литералы "\n" и "\t" в реальные переносы/табуляции
    gen_suffix = gen_suffix.replace("\\n", "\n").replace("\\t", "\t")

    return gen_suffix


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
    # Параметры генерации будут собраны динамически ниже в gen_kwargs

    base_prompt = prompt
    current_prompt = prompt
    last_successful = ""
    all_steps = []

    for step in range(iterations):
        logger.info(f"🧩 Итерация {step + 1}/{iterations}: уточнение кода...")

        try:
            # Используем общую функцию генерации для обеспечения идентичности
            gen_suffix = _generate_single_step(
                prompt=current_prompt,
                tokenizer=tokenizer,
                model=model,
                device=device,
                generation_config=generation_config
            )
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
                    f"{base_prompt}\n\n"
                    f"Previous solution:\n{gen_suffix}\n\n"
                    f"The last attempt failed during automatic checking with the following error (exact text):\n"
                    f"```\n{error_msg}\n```\n\n"
                    "Analyze the code carefully and return a corrected, fully functional, syntactically correct Python function.\n"
                    "Keep the same function name and parameters. Fix all indentation and syntax errors and any runtime error "
                    "reported above. Do not include any explanations, comments or additional text — return only the updated code.\n"
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
    Использует ту же логику, что и первая итерация generate_code_with_model.

    Args:
        prompt: Промпт задачи.
        tokenizer: Токенизатор для модели.
        model: Языковая модель.
        device: Устройство для выполнения генерации.
        generation_config: Опциональная конфигурация для параметров генерации.

    Returns:
        Сгенерированный код без валидации.
    """
    try:
        # Используем общую функцию генерации для обеспечения идентичности с первой итерацией
        # generate_code_with_model
        return _generate_single_step(
            prompt=prompt,
            tokenizer=tokenizer,
            model=model,
            device=device,
            generation_config=generation_config
        )
    except Exception as e:
        logger.error(f"Ошибка во время генерации кода: {e}")
        return ""