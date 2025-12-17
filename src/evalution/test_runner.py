"""Test runners for evaluating generated code."""
import logging
import types
from typing import List, Dict, Any, Union

from ..utils.code_utils import is_syntax_valid, run_humaneval_test

logger = logging.getLogger(__name__)


def run_humaneval_test(generated_code: str, test_code: str, entry_point: str) -> bool:
    """
    Run HumanEval test for generated code.

    Args:
        generated_code: Generated code to test.
        test_code: Test code to execute.
        entry_point: Name of the function being tested.

    Returns:
        True if test passes, False otherwise.
    """
    if not is_syntax_valid(generated_code):
        logger.error(f"❌ SYNTAX ERROR in code for {entry_point}")
        return False

    module = types.ModuleType("generated_module")
    try:
        exec(generated_code, module.__dict__)
        exec(test_code, module.__dict__)
        logger.info(f"✅ HumanEval tests for {entry_point} passed successfully.")
        return True
    except Exception as e:
        logger.error(f"❌ TEST EXECUTION ERROR for {entry_point}: {type(e).__name__}: {e}")
        return False


def run_test_suite(test_code: str, generated_code: str, timeout: float = 5.0) -> Dict[str, Any]:
    """
    Run a test suite against generated code.

    Args:
        test_code: The test code to execute.
        generated_code: The generated function code.
        timeout: Timeout for test execution in seconds.

    Returns:
        Dictionary with test results.
    """
    results = {
        "passed": False,
        "error": None,
        "error_type": None,
        "execution_time": None
    }

    if not is_syntax_valid(generated_code):
        results["error"] = "SYNTAX ERROR"
        results["error_type"] = "SyntaxError"
        return results

    import time
    import importlib.util
    import sys

    try:
        start_time = time.time()

        # Create temporary module and execute code
        test_module = types.ModuleType("test_module")
        exec(generated_code, test_module.__dict__)
        exec(test_code, test_module.__dict__)

        execution_time = time.time() - start_time
        results["passed"] = True
        results["execution_time"] = execution_time

    except Exception as e:
        results["error"] = str(e)
        results["error_type"] = type(e).__name__

    return results


def run_mbpp_test(generated_code: str, test_cases: List[str]) -> bool:
    """
    Run MBPP test for generated code.

    Args:
        generated_code: Generated code to test.
        test_cases: List of test case strings to execute.

    Returns:
        True if all test cases pass, False otherwise.
    """
    if not is_syntax_valid(generated_code):
        logger.error("❌ SYNTAX ERROR in generated code")
        return False

    module = types.ModuleType("generated_module")

    try:
        # Execute the generated code
        exec(generated_code, module.__dict__)

        # Run each test case
        for i, test_case in enumerate(test_cases):
            try:
                exec(test_case, module.__dict__)
            except Exception as e:
                logger.error(f"❌ Test case {i + 1} failed: {e}")
                return False

        logger.info("✅ All MBPP test cases passed")
        return True
    except Exception as e:
        logger.error(f"❌ MBPP test execution failed: {e}")
        return False


def evaluate_single_example(generated_code: str, test_code: str, entry_point: str = None,
                            test_type: str = "humaneval") -> Dict[str, Any]:
    """
    Evaluate a single generated code example.

    Args:
        generated_code: Generated code to evaluate.
        test_code: Tests to run against the code.
        entry_point: Function name being tested.
        test_type: Type of test to run ("humaneval", "mbpp", etc.)

    Returns:
        Dictionary with evaluation results.
    """
    if test_type == "humaneval":
        passed = run_humaneval_test(generated_code, test_code, entry_point)
        return {"passed": passed, "test_type": test_type}
    elif test_type == "mbpp":
        passed = run_mbpp_test(generated_code, test_code)
        return {"passed": passed, "test_type": test_type}
    else:
        # Default test runner
        results = run_test_suite(test_code, generated_code)
        results["test_type"] = test_type
        return results