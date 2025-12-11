"""HumanEval benchmark runner for code generation models."""
import logging
import time
import torch
from typing import Dict, Any, Optional, List
from datasets import Dataset

from src.data import load_humaneval_dataset, prepare_humaneval_examples
from src.models import generate_code_with_model, generate_single_turn_code
from src.evalution import evaluate_on_dataset, EvaluationResult
from src.utils import extract_functions_only


logger = logging.getLogger(__name__)


def run_humaneval_example(
        task_id: str,
        tokenizer,
        model,
        device: torch.device,
        generation_config: Optional[Dict[str, Any]] = None,
        iterations: int = 3,
        use_agent_chain: bool = True
) -> Dict[str, Any]:
    """
    Run benchmark on a single HumanEval example.

    Args:
        task_id: Task ID to run.
        tokenizer: Tokenizer for generation.
        model: Model for generation.
        device: Device to run on.
        generation_config: Configuration for generation parameters.
        iterations: Number of self-correction iterations.

    Returns:
        Dictionary with results for this example.
    """
    dataset = load_humaneval_dataset()
    examples = prepare_humaneval_examples(dataset)

    # Sort examples by task_id for deterministic ordering
    examples = sorted(examples, key=lambda x: x["task_id"])

    # Find the requested example
    example = None
    for ex in examples:
        if ex["task_id"] == task_id:
            example = ex
            break

    if example is None:
        logger.error(f"Task ID {task_id} not found in dataset")
        return {"error": f"Task ID {task_id} not found"}

    print(f"\n--- [Example] ID: {task_id} ---")
    print(f"Function: {example['entry_point']}")

    try:
            start_time = time.time()

            # Generate code
            if use_agent_chain:
                generated_code_part = generate_code_with_model(
                    prompt="<prompt>" + example["prompt"] + "<prompt><setting>code generation without docstring</setting>",
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    generation_config=generation_config,
                    iterations=iterations
                )
            else:
                generated_code_part = generate_single_turn_code(
                    prompt="<prompt>" + example["prompt"] + "<prompt><setting>code generation without docstring</setting>",
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    generation_config=generation_config
                )

            # Combine with prompt
            full_generated_text = example["prompt"] + generated_code_part
            generated_function = extract_functions_only(full_generated_text)

            if not generated_function.strip():
                logger.error("❌ Failed to extract executable code from model response")
                return {
                    "task_id": task_id,
                    "entry_point": example["entry_point"],
                    "error": "Could not extract executable code",
                    "passed": False
                }

            full_code_to_test = generated_function

            # Evaluate the code
            result = evaluate_on_dataset([full_code_to_test],
                                         [{"test_code": example["test_code"],
                                           "entry_point": example["entry_point"],
                                           "task_id": task_id}])

            gen_time = time.time() - start_time

            print(f"Generated code for {example['entry_point']}:")
            preview = generated_code_part
            print(preview)
            print(f"Result: {'PASS' if result.pass_rate > 0 else 'FAIL'}")
            print("-" * 50)

            return {
            "task_id": task_id,
            "entry_point": example["entry_point"],
            "generated_code": generated_code_part,
            "generated_function": generated_function,
            "passed": result.pass_rate > 0,
            "generation_time": gen_time
        }

    except Exception as e:
        logger.error(f"❌ Error during generation for {example['entry_point']}: {e}")
        return {
            "task_id": task_id,
            "entry_point": example["entry_point"],
            "error": str(e),
            "passed": False
        }


def run_full_humaneval_benchmark(
        tokenizer,
        model,
        device: torch.device,
        generation_config: Optional[Dict[str, Any]] = None,
        iterations: int = 3,
        limit: Optional[int] = None,
        output_file: Optional[str] = None,
        verbose: bool = True,
        use_agent_chain: bool = True
) -> EvaluationResult:
    """
    Run full HumanEval benchmark on provided model.

    Args:
        tokenizer: Tokenizer for generation.
        model: Model for generation.
        device: Device to run on.
        generation_config: Configuration for generation parameters.
        iterations: Number of self-correction iterations.
        limit: Optional limit on number of examples to run.
        output_file: Optional file path to save detailed results.
        verbose: Whether to print detailed progress.

    Returns:
        EvaluationResult with overall benchmark metrics.
    """
    # Load dataset
    human_eval = load_humaneval_dataset()
    examples = prepare_humaneval_examples(human_eval)

    # Sort examples by task_id for deterministic ordering
    examples = sorted(examples, key=lambda x: x["task_id"])

    # Apply limit if specified
    if limit:
        examples = examples[:limit]

    total_problems = len(examples)

    if verbose:
        print("=" * 60)
        print(f"🚀 Starting HumanEval benchmark ({total_problems} problems)")
        print(f"Model: {getattr(model.config, '_name_or_path', str(model.__class__))} on {device}")
        print("=" * 60)

    # Prepare containers for results
    generated_codes = []
    test_cases = []
    detailed_results = []

    # Run benchmark
    start_time = time.time()

    for i, example in enumerate(examples):
        task_id = example["task_id"]
        prompt_for_model = example["prompt"]
        # test_code = example["test"]
        entry_point = example["entry_point"]

        if verbose:
            print(f"\n--- [Problem {i + 1}/{total_problems}] ID: {task_id} ---")
            print(f"Function: {entry_point}")

        try:
            # Generate code
            if use_agent_chain:
                generated_code_part = generate_code_with_model(
                    prompt="<setting>code generation without docstring</setting><setting>don't use List, use list</setting><prompt>" + prompt_for_model + "<prompt>",
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    generation_config=generation_config,
                    iterations=iterations
                )
            else:
                generated_code_part = generate_single_turn_code(
                    prompt="<setting>code generation without docstring</setting><setting>don't use List, use list</setting><prompt>" + prompt_for_model + "<prompt>",
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    generation_config=generation_config
                )

            # Combine with prompt
            full_generated_text = prompt_for_model + generated_code_part
            generated_function = extract_functions_only(full_generated_text)

            if not generated_function.strip():
                logger.error("❌ Failed to extract executable code from model response")
                # Still add to results but mark as failed
                generated_codes.append("")
                test_cases.append(example)
                detailed_results.append({
                    "task_id": task_id,
                    "entry_point": entry_point,
                    "error": "Could not extract executable code",
                    "passed": False
                })
                continue

            full_code_to_test = generated_function
            generated_codes.append(full_code_to_test)
            test_cases.append(example)

            if verbose:
                print(f"Generated code for {example['entry_point']}:")
                preview = generated_code_part
                print(preview)
                print("-" * 50)

        except Exception as e:
            logger.error(f"❌ Error during generation for {entry_point}: {e}")
            generated_codes.append("")
            test_cases.append(example)
            detailed_results.append({
                "task_id": task_id,
                "entry_point": entry_point,
                "error": str(e),
                "passed": False
            })
            continue

    # Evaluate all generated code
    result = evaluate_on_dataset(generated_codes, test_cases, "humaneval")
    total_time = time.time() - start_time

    if verbose:
        print("\n" + "=" * 60)
        print("✨ HUMAN-EVAL BENCHMARK RESULTS ✨")
        print(f"🎯 Total problems: {total_problems}")
        print(f"🟢 Passed tests (Pass@1): {result.passed_examples}")
        print(f"📊 Pass@1: {result.pass_rate:.2f}%")
        print(f"⏱️ Total time: {total_time:.2f} seconds")
        print(f"⚡ Average per problem: {total_time / total_problems:.2f} seconds")
        print("=" * 60)

    # Save detailed results if requested
    if output_file:
        import json
        with open(output_file, "w") as f:
            json.dump({
                "summary": result.to_dict(),
                "detailed_results": detailed_results,
                "benchmark_config": {
                    "iterations": iterations,
                    "generation_config": generation_config,
                    "total_time": total_time,
                    "avg_time_per_problem": total_time / total_problems
                }
            }, f, indent=2)
        logger.info(f"Results saved to {output_file}")

    return result