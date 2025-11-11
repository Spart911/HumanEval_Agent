"""Утилиты загрузки датасетов для бенчмаркинга."""
import logging
from typing import Dict, List, Any, Union
from datasets import Dataset, load_dataset, DownloadConfig

logger = logging.getLogger(__name__)



def load_humaneval_dataset(split: str = "test") -> Dataset:
    """
    Загрузка датасета HumanEval для бенчмаркинга генерации кода.
    """
    try:
        # Добавляем конфигурацию загрузки с timeout
        download_config = DownloadConfig()
        dataset = load_dataset("openai_humaneval", split=split, download_config=download_config)
        logger.info(f"Загружен датасет HumanEval с {len(dataset)} задачами")
        return dataset
    except Exception as e:
        logger.error(f"Не удалось загрузить датасет HumanEval: {e}")
        raise

def load_mbpp_dataset(split: str = "test") -> Dataset:
    """
    Загрузка датасета MBPP (Mostly Basic Python Programming).

    Args:
        split: Часть датасета для загрузки.

    Returns:
        Датасет, содержащий задачи MBPP.
    """
    try:
        dataset = load_dataset("mbpp", split=split)
        logger.info(f"Загружен датасет MBPP с {len(dataset)} задачами")
        return dataset
    except Exception as e:
        logger.error(f"Не удалось загрузить датасет MBPP: {e}")
        raise


def load_custom_dataset(path: str, split: str = None) -> Dataset:
    """
    Загрузка пользовательского датасета из указанного пути.

    Args:
        path: Путь к датасету.
        split: Опциональная часть датасета для загрузки.

    Returns:
        Датасет, загруженный из указанного пути.
    """
    try:
        if split:
            dataset = load_dataset(path, split=split)
        else:
            dataset = load_dataset(path)

        logger.info(f"Загружен пользовательский датасет из {path} с {len(dataset)} примерами")
        return dataset
    except Exception as e:
        logger.error(f"Не удалось загрузить пользовательский датасет из {path}: {e}")
        raise


def prepare_humaneval_examples(dataset: Dataset) -> List[Dict[str, Any]]:
    """
    Подготовка примеров HumanEval в стандартизированном формате.

    Args:
        dataset: Датасет HumanEval.

    Returns:
        Список стандартизированных словарей с примерами.
    """
    examples = []
    for item in dataset:
        examples.append({
            "task_id": item["task_id"],
            "prompt": item["prompt"],
            "canonical_solution": item["canonical_solution"],
            "test_code": item["test"],
            "entry_point": item["entry_point"]
        })
    return examples


def prepare_mbpp_examples(dataset: Dataset) -> List[Dict[str, Any]]:
    """
    Prepare MBPP examples in a standardized format.

    Args:
        dataset: The MBPP dataset.

    Returns:
        List of standardized example dictionaries.
    """
    examples = []
    for item in dataset:
        examples.append({
            "task_id": item.get("task_id", f"mbpp_{len(examples)}"),
            "prompt": item["text"],
            "canonical_solution": item["code"],
            "test_case_list": item["test_list"],
            "entry_point": None  # MBPP doesn't explicitly provide entry points
        })
    return examples


def preprocess_dataset(examples: List[Dict[str, Any]], dataset_type: str = "humaneval") -> List[Dict[str, Any]]:
    """
    Preprocess dataset examples for benchmarking.

    Args:
        examples: List of raw examples.
        dataset_type: Type of dataset ("humaneval", "mbpp", etc.)

    Returns:
        Preprocessed examples ready for benchmarking.
    """
    processed = []

    if dataset_type == "humaneval":
        for example in examples:
            processed.append({
                "task_id": example["task_id"],
                "prompt": example["prompt"],
                "test_code": example["test_code"],
                "entry_point": example["entry_point"]
            })
    elif dataset_type == "mbpp":
        for example in examples:
            processed.append({
                "task_id": example["task_id"],
                "prompt": example["prompt"],
                "test_code": example["test_case_list"],
                "entry_point": example["entry_point"]
            })
    else:
        # Generic preprocessing for unknown dataset types
        for example in examples:
            processed.append({
                "task_id": example.get("task_id", f"generic_{len(processed)}"),
                "prompt": example["prompt"],
                "test_code": example.get("test_code", ""),
                "entry_point": example.get("entry_point")
            })

    return processed