#!/usr/bin/env python3
"""
Тест повторяемости генерации кода.
Запускает генерацию несколько раз и проверяет, что результаты идентичны.
"""
import sys
import os
from pathlib import Path

# Добавляем директорию src в Python путь для импорта модулей
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

import hashlib
import logging
from src.config import parse_cli_args, load_config
from src.benchmarks import BenchmarkManager
from src.utils import login_to_huggingface, set_random_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_generation_hash(generated_code: str) -> str:
    """Получить хэш сгенерированного кода для сравнения."""
    return hashlib.md5(generated_code.encode('utf-8')).hexdigest()


def test_reproducibility(seed: int, limit: int = 5, runs: int = 3):
    """Тестирование повторяемости генерации."""
    print(f"🧪 Тестирование повторяемости с seed={seed}, limit={limit}, runs={runs}")
    print("=" * 60)

    results = []

    for run in range(runs):
        print(f"\n🏃‍♂️ Запуск {run + 1}/{runs}")

        # Устанавливаем seed для каждого запуска
        set_random_seed(seed)

        # Создаем менеджер бенчмарка
        benchmark_manager = BenchmarkManager(
            model_path="pdg_trained_github",
            base_model_path="~/Jupyter/Qwen2.5-Coder-3B",
            use_lora=True,
            device="cuda",
            seed=seed,
            deterministic_generation=True
        )

        # Загружаем модель
        if not benchmark_manager.load_model():
            print("❌ Не удалось загрузить модель")
            return False

        # Запускаем бенчмарк
        result = benchmark_manager.run_humaneval(
            limit=limit,
            iterations=1,  # Одна итерация для простоты
            generation_config={"max_new_tokens": 200, "temperature": 0.0, "do_sample": False},
            verbose=False,
            use_agent_chain=False  # Без итеративного исправления
        )

        if result:
            # Сохраняем результаты для сравнения
            run_results = {
                "pass_rate": result["pass_rate"],
                "passed_examples": result["passed_examples"],
                "total_examples": result["total_examples"],
                "config": result["config"]
            }
            results.append(run_results)
            print(f"✅ Запуск {run + 1}: {result['pass_rate']:.2f}% ({result['passed_examples']}/{result['total_examples']})")
        else:
            print(f"❌ Запуск {run + 1} провалился")
            return False

    # Проверяем повторяемость
    print("
🔍 Проверка повторяемости:"    all_same = True

    for i in range(1, len(results)):
        if results[i]["pass_rate"] != results[0]["pass_rate"]:
            print(f"❌ Разные результаты: запуск 1: {results[0]['pass_rate']}%, запуск {i+1}: {results[i]['pass_rate']}%")
            all_same = False
        else:
            print(f"✅ Запуск {i+1}: совпадает с запуском 1")

    if all_same:
        print("
🎉 ПОЛНАЯ ПОВТОРЯЕМОСТЬ ДОСТИГНУТА!"        return True
    else:
        print("
⚠️  ПОВТОРЯЕМОСТЬ НЕ ДОСТИГНУТА"        return False


if __name__ == "__main__":
    # Вход в HuggingFace
    login_to_huggingface()

    # Тестируем с разными seed'ами
    test_cases = [
        {"seed": 42, "limit": 3, "runs": 3},
        {"seed": 123, "limit": 3, "runs": 3},
    ]

    all_passed = True
    for test_case in test_cases:
        success = test_reproducibility(**test_case)
        if not success:
            all_passed = False

    if all_passed:
        print("\n🏆 ВСЕ ТЕСТЫ ПРОШЛИ! Повторяемость достигнута.")
    else:
        print("\n💥 НЕКОТОРЫЕ ТЕСТЫ ПРОВАЛИЛИСЬ! Повторяемость не достигнута.")



