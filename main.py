#!/usr/bin/env python3
"""
Основной входной файл для запуска бенчмарков генерации кода LLM.
"""
import sys
import os
import logging
from pathlib import Path

# Добавляем директорию src в Python путь для импорта модулей
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from src.config import parse_cli_args, load_config
from src.benchmarks import BenchmarkManager
from src.utils import login_to_huggingface

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Основная функция для запуска бенчмарка."""
    # Вход в HuggingFace
    login_to_huggingface()

    # Парсинг аргументов командной строки
    cli_args = parse_cli_args()

    # Загрузка конфигурации
    config = load_config(cli_args.config, cli_args)

    # Проверка конфигурации
    if not config.model.model_path:
        logger.error("Путь к модели не указан. Используйте --model-path или установите MAIN_MODEL_PATH в .env")
        return 1

    # Настройка логирования на основе флага verbose
    if not config.benchmark.verbose:
        logging.getLogger().setLevel(logging.WARNING)


    # Создание менеджера бенчмарка
    benchmark_manager = BenchmarkManager(
        model_path=config.model.model_path,
        base_model_path=config.model.base_model_path,
        use_lora=config.model.use_lora,
        device=config.model.device
    )

    # Загрузка модели
    if not benchmark_manager.load_model():
        logger.error("Не удалось загрузить модель. Продолжение невозможно.")
        return 1

    try:
        # Run appropriate benchmark
        if config.benchmark.dataset == "humaneval":
            result = benchmark_manager.run_humaneval(
                limit=config.benchmark.limit,
                iterations=config.benchmark.iterations,
                generation_config=config.generation.__dict__,
                verbose=config.benchmark.verbose
            )

            if result:
                logger.info(f"Бенчмарк HumanEval завершен. Pass@1: {result['pass_rate']:.2f}%")

                # Сохранение результатов
                if config.benchmark.output_file:
                    benchmark_manager.save_results(config.benchmark.output_file)
                    logger.info(f"Результаты сохранены в {config.benchmark.output_file}")

                # Вывод сводки
                if config.benchmark.verbose:
                    benchmark_manager.print_summary()
                return 0
            else:
                logger.error("Бенчмарк не удался.")
                return 1
        else:
            logger.error(f"Неподдерживаемый датасет: {config.benchmark.dataset}")
            return 1

    except KeyboardInterrupt:
        logger.info("Бенчмарк прерван пользователем")
        return 130
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
