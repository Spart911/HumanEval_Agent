"""Benchmark manager for coordinating and running multiple benchmarks."""
import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime

from .humaneval_benchmark import run_full_humaneval_benchmark
from src.models import load_model, load_model_with_lora, load_base_model_only

logger = logging.getLogger(__name__)


class BenchmarkManager:
    def __init__(
        self,
        model_path: str,
        base_model_path: Optional[str] = None,
        use_lora: bool = False,
        device: str = "auto",
        use_base_model_only: bool = False,
    ):
        self.model_path = model_path
        self.base_model_path = base_model_path
        self.use_lora = use_lora
        self.device_name = device
        self.use_base_model_only = use_base_model_only
        self.model = None
        self.tokenizer = None
        self.device = None
        self.metadata: Dict[str, Any] = {"initialized": True, "model_loaded": False}
        self.benchmark_results: Dict[str, Any] = {}

    def load_model(self) -> bool:
        """Load model (optionally with LoRA or base model only)."""
        try:
            logger.info("Loading model...")

            if self.use_base_model_only:
                if not self.base_model_path:
                    logger.error("Base model path is required when using --use-base-model-only")
                    return False
                logger.info("Loading base model only (without fine-tuned adapters)...")
                self.tokenizer, self.model, self.device = load_base_model_only(
                    self.base_model_path,
                    device=self.device_name,
                )
            else:
                self.tokenizer, self.model, self.device = load_model_with_lora(
                    self.model_path,
                    self.base_model_path,
                    device=self.device_name,
                )

            logger.info("Model loaded successfully.")
            self.metadata["model_loaded"] = True
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.metadata["model_loaded"] = False
            return False

    def run_humaneval(
        self,
        limit: Optional[int] = None,
        iterations: int = 3,
        generation_config: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
        use_agent_chain: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Run HumanEval benchmark.

        Args:
            limit: Optional limit on number of examples.
            iterations: Number of self-correction iterations.
            generation_config: Optional generation configuration.
            verbose: Whether to print detailed output.

        Returns:
            Dictionary with benchmark results, or None if model not loaded.
        """
        if self.model is None:
            logger.error("Model not loaded. Call load_model() first.")
            return None

        try:
            result = run_full_humaneval_benchmark(
                self.tokenizer,
                self.model,
                self.device,
                generation_config=generation_config,
                iterations=iterations,
                limit=limit,
                verbose=verbose,
                use_agent_chain=use_agent_chain,
            )

            self.benchmark_results["humaneval"] = {
                "pass_rate": result.pass_rate,
                "passed_examples": result.passed_examples,
                "total_examples": result.total_examples,
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "iterations": iterations,
                    "generation_config": generation_config,
                    "limit": limit,
                },
            }

            return self.benchmark_results["humaneval"]

        except Exception as e:
            logger.error(f"Error running HumanEval benchmark: {e}")
            return None

    def save_results(self, filepath: str) -> bool:
        """Save benchmark results to file."""
        try:
            results_to_save = {
                "metadata": self.metadata,
                "benchmark_results": self.benchmark_results,
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(results_to_save, f, indent=2, ensure_ascii=False)

            logger.info(f"Results saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return False

    def load_results(self, filepath: str) -> bool:
        """Load benchmark results from file."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.metadata = data.get("metadata", {})
            self.benchmark_results = data.get("benchmark_results", {})

            logger.info(f"Results loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load results: {e}")
            return False

    def print_summary(self):
        """Print summary of benchmark results."""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        print(f"Model: {self.model_path}")
        print(f"Device: {self.device}")
        print(f"LoRA: {self.use_lora}")
        print()

        for benchmark_name, results in self.benchmark_results.items():
            print(f"{benchmark_name.upper()}:")
            print(f"  Pass Rate: {results['pass_rate']:.2f}%")
            print(f"  Passed: {results['passed_examples']}/{results['total_examples']}")
            print(f"  Timestamp: {results['timestamp']}")
            print()

        print("=" * 60)
