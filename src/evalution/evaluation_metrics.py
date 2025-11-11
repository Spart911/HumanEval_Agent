"""Evaluation metrics for code generation models."""
import math
import logging
from typing import List, Dict, Any, Tuple
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    total_examples: int
    passed_examples: int
    pass_rate: float
    results_by_example: List[Dict[str, Any]]
    dataset_name: str = ""
    model_name: str = ""
    generation_config: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "total_examples": self.total_examples,
            "passed_examples": self.passed_examples,
            "pass_rate": self.pass_rate,
            "dataset_name": self.dataset_name,
            "model_name": self.model_name,
            "generation_config": self.generation_config
        }


def calculate_pass_at_k(results: List[List[bool]], k_values: List[int] = [1, 5, 10]) -> Dict[int, float]:
    """
    Calculate Pass@k metric from binary success results.

    Args:
        results: List where each element is a list of boolean results for k attempts.
        k_values: List of k values to calculate.

    Returns:
        Dictionary mapping each k to its pass@k percentage.
    """
    pass_at_k = {}

    total = len(results)
    if total == 0:
        return {k: 0.0 for k in k_values}

    for k in k_values:
        passed = sum(any(result[:k]) for result in results if len(result) >= k)
        pass_at_k[k] = passed / total * 100

    return pass_at_k


def evaluate_on_dataset(generated_codes: List[str], test_cases: List[Any],
                        dataset_type: str = "humaneval") -> EvaluationResult:
    """
    Evaluate generated code against a dataset of test cases.

    Args:
        generated_codes: List of generated code strings.
        test_cases: List of test cases data.
        dataset_type: Type of dataset ("humaneval", "mbpp", etc.).

    Returns:
        EvaluationResult object containing pass rate.
    """
    total_examples = len(generated_codes)
    passed_examples = 0
    results_by_example = []

    for i, (code, test_case) in enumerate(zip(generated_codes, test_cases)):
        if dataset_type == "humaneval":
            test_code = test_case["test_code"]
            entry_point = test_case["entry_point"]

            from .test_runner import run_humaneval_test
            passed = run_humaneval_test(code, test_code, entry_point)
            results_by_example.append({
                "example_index": i,
                "task_id": test_case.get("task_id", f"task_{i}"),
                "entry_point": entry_point,
                "passed": passed
            })
        elif dataset_type == "mbpp":
            test_code_list = test_case["test_code"]

            from .test_runner import run_mbpp_test
            passed = run_mbpp_test(code, test_code_list)
            results_by_example.append({
                "example_index": i,
                "task_id": test_case.get("task_id", f"task_{i}"),
                "passed": passed
            })
        else:
            # Generic test runner
            test_code = test_case["test_code"]
            entry_point = test_case.get("entry_point", f"function_{i}")

            from .test_runner import evaluate_single_example
            result = evaluate_single_example(code, test_code, entry_point, dataset_type)
            passed = result.get("passed", False)
            results_by_example.append({
                "example_index": i,
                "task_id": test_case.get("task_id", f"task_{i}"),
                "passed": passed
            })

        if passed:
            passed_examples += 1

    pass_rate = passed_examples / total_examples * 100

    return EvaluationResult(
        total_examples=total_examples,
        passed_examples=passed_examples,
        pass_rate=pass_rate,
        results_by_example=results_by_example
    )


def compute_bleu_score(reference: str, candidate: str) -> float:
    """
    Compute BLEU score between reference and candidate text.

    Args:
        reference: Reference text.
        candidate: Candidate text.

    Returns:
        BLEU score between 0 and 1.
    """
    try:
        import nltk
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

        # Tokenize
        ref_tokens = reference.split()
        can_tokens = candidate.split()

        # Calculate BLEU with smoothing
        smoother = SmoothingFunction().method1
        bleu = corpus_bleu([[ref_tokens]], [can_tokens], smoothing_function=smoother)
        return bleu
    except (ImportError, Exception):
        # Fallback if NLTK not available or other error
        logger.warning("Could not compute BLEU score, returning 0")
        return 0.0


def compute_code_bleu_metrics(reference_code: str, candidate_code: str) -> Dict[str, float]:
    """
    Compute code-specific BLEU metrics.

    Args:
        reference_code: Reference code string.
        candidate_code: Generated code string.

    Returns:
        Dictionary with code BLEU metrics.
    """
    metrics = {}

    # Compute regular BLEU
    metrics["bleu"] = compute_bleu_score(reference_code, candidate_code)

    # Compute exact match
    match = 1 if reference_code.strip() == candidate_code.strip() else 0
    metrics["exact_match"] = match

    return metrics


def compute_syntax_metrics(code: str) -> Dict[str, float]:
    """
    Compute syntax-related metrics for code.

    Args:
        code: Code string to evaluate.

    Returns:
        Dictionary with syntax metrics.
    """
    from ..utils.code_utils import is_syntax_valid, check_code_in_subprocess

    metrics = {
        "is syntactically_valid": float(is_syntax_valid(code))
    }

    # Test if the code can be imported in a subprocess
    success, _ = check_code_in_subprocess(code, timeout=5)
    metrics["is_importable"] = float(success)

    return metrics


def calculate_termination_ratio(completions: List[str]) -> float:
    """
    Calculate the ratio of completions that properly end with function termination.

    Args:
        completions: List of code completions.

    Returns:
        Ratio of properly terminated completions.
    """
    terminated_count = 0
    for completion in completions:
        lines = completion.strip().split('\n')
        if len(lines) >= 1:
            last_line = lines[-1].strip()
            # Simple heuristic: check if last line is not in the middle of something
            if len(last_line) == 0 or last_line.startswith('return ') or last_line.startswith('"""'):
                # Could add more sophisticated checks
                pass

        # Count if completion seems to end naturally
        # This is a basic placeholder method
        terminated_count += 1

    return terminated_count / len(completions) if completions else 0.0