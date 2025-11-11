"""Утилиты для обработки и валидации кода."""
import re
import textwrap
import types
import os
import sys
import subprocess
import tempfile
import pathlib
import py_compile
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


def strip_code_fences(code: str) -> str:
    """Удаление ограждений кода (```python ... ```) и пробелов.

    Args:
        code: Строка кода, возможно обернутая в markdown-ограждения.

    Returns:
        Строка кода без ограждений.
    """
    if not code:
        return code

    # Удаляем ограждения кода типа ```python ... ``` или ```py ...
    code = re.sub(r"^\s*```(?:py(?:thon)?)?\s*\n", "", code, flags=re.IGNORECASE)
    code = re.sub(r"\n\s*```\s*$", "", code, flags=re.IGNORECASE)
    return code


def check_code_in_subprocess(code: str, timeout: int = 5) -> Tuple[bool, str]:
    """
    Проверка кода в отдельном процессе Python:
    1) py_compile для проверки синтаксиса
    2) Импорт модуля через importlib в подпроцессе для ловли ошибок времени выполнения при импорте

    Args:
        code: Строка кода для валидации.
        timeout: Тайм-аут в секундах для выполнения подпроцесса.

    Returns:
        Кортеж (success, error_message). Пустой error_message в случае успеха.
    """
    try:
        if not code or not code.strip():
            return False, "EmptyCodeError: код пуст"

        # Очищаем потенциальные блоки markdown и выравниваем отступы
        code_clean = strip_code_fences(code)
        code_clean = textwrap.dedent(code_clean).strip()

        if not code_clean:
            return False, "EmptyCodeError: код пуст после удаления ограждений/выравнивания"

        # Создаем временный файл с кодом
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
            path = pathlib.Path(f.name)
            f.write(code_clean)
            f.flush()

        # 1) Проверка синтаксиса
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as e:
            return False, f"SyntaxError: {str(e)}"

        # 2) Пытаемся импортировать модуль в отдельном процессе
        runner = textwrap.dedent(f"""
        import importlib.util, sys, traceback
        path = r\"{str(path)}\"
        try:
            spec = importlib.util.spec_from_file_location('generated_check_module', path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception:
            traceback.print_exc()
            raise
        """)

        # Используем sys.executable для избежания зависимости от 'python' в PATH
        python_exe = sys.executable or "python"
        env = os.environ.copy()
        # Отключаем параллелизм tokenizers в подпроцессе (убирает предупреждение о форке)
        env["TOKENIZERS_PARALLELISM"] = "false"
        # Гарантируем вывод в UTF-8
        env["PYTHONUNBUFFERED"] = "1"

        proc = subprocess.run(
            [python_exe, "-c", runner],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env
        )

        if proc.returncode != 0:
            # Возвращаем stderr или stdout
            err = (proc.stderr.strip() or proc.stdout.strip())
            return False, err

        return True, ""

    except subprocess.TimeoutExpired:
        return False, "TimeoutExpired: проверка кода заняла слишком много времени"
    except Exception as ex:
        return False, f"CheckerError: {type(ex).__name__}: {ex}"
    finally:
        # Пытаемся удалить временный файл (если остался)
        try:
            if 'path' in locals() and path.exists():
                path.unlink()
        except Exception:
            pass


def extract_functions_only(text: str) -> str:
    """Извлечение кода функций из текста, удаляя основные блоки и доктесты.

    Args:
        text: Текст, содержащий Python код.

    Returns:
        Строка только с определениями функций.
    """
    text = text.replace("\\n", "\n")

    # Убираем блок if __name__ == '__main__'
    main_pattern = r"(if\s+__name__\s*==\s*['\"]__main__['\"]\s*:.*)"
    text = re.sub(main_pattern, '', text, flags=re.DOTALL)

    # Убираем импорты и вызовы doctest
    doctest_pattern = r"import\s+doctest\s*\n\s*doctest\.testmod\(\)"
    text = re.sub(doctest_pattern, '', text, flags=re.DOTALL)

    # Дополнительно удаляем markdown-ограждения и выравниваем
    text = strip_code_fences(text)

    return text.strip()


def is_syntax_valid(code: str) -> bool:
    """Проверка синтаксической корректности кода.

    Args:
        code: Строка кода для валидации.

    Returns:
        True если синтаксически корректен, иначе False.
    """
    try:
        if not code.strip():
            return False
        compile(code, "<string>", "exec")
        return True
    except SyntaxError:
        return False


def run_humaneval_test(generated_code: str, test_code: str, entry_point: str) -> bool:
    """Запуск теста HumanEval для сгенерированного кода.

    Args:
        generated_code: Сгенерированный код для тестирования.
        test_code: Код теста для выполнения.
        entry_point: Имя тестируемой функции.

    Returns:
        True если тест пройден, иначе False.
    """
    if not is_syntax_valid(generated_code):
        logger.error(f"❌ СИНТАКСИЧЕСКАЯ ОШИБКА в коде для {entry_point}")
        return False

    module = types.ModuleType("generated_module")
    try:
        exec(generated_code, module.__dict__)
        exec(test_code, module.__dict__)
        logger.info(f"✅ Тесты HumanEval для {entry_point} пройдены успешно.")
        return True
    except Exception as e:
        logger.error(f"❌ ОШИБКА ВЫПОЛНЕНИЯ ТЕСТА для {entry_point}: {type(e).__name__}: {e}")
        return False