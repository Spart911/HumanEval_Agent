#!/usr/bin/env python3
"""
Демонстрационный скрипт для запуска бенчмарка на русском языке.
"""
import sys
import os
from pathlib import Path

# Добавляем директорию src в Python путь для импорта модулей
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from benchmarks import BenchmarkManager


def main():
    print("🚀 Запуск демо-бенчмарка с русифицированными сообщениями")
    print("=" * 50)

    # Создаем менеджер бенчмарка
    manager = BenchmarkManager(
        model_path="pdg_trained_github",
        base_model_path="~/Jupyter/Qwen2.5-Coder-3B",
        use_lora=True,
        device="cuda"
    )

    print(f"Путь к модели: {manager.model_path}")
    print("Использование LoRA: Да")
    print("Автоматический выбор устройства: Включено")

    print("\n⚠️  Примечание:")
    print("   Это демонстрация без фактической загрузки модели.")
    print("   Для полного запуска бенчмарка используйте:")
    print("   python main.py --limit 5 --verbose")

    print("\nДля получения справки:")
    print("python main.py --help")


if __name__ == "__main__":
    main()