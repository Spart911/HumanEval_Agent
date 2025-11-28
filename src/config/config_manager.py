"""Configuration management for benchmarking."""
import os
import logging
import argparse
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import yaml

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model loading."""
    model_path: str = ""
    base_model_path: Optional[str] = None
    use_lora: bool = False
    device: str = "auto"
    torch_dtype: str = "float16"
    quantization: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationConfig:
    """Configuration for code generation."""
    max_new_tokens: int = 400
    temperature: float = 0.6
    top_k: int = 40
    top_p: float = 0.95
    do_sample: bool = True
    num_beams: int = 1
    num_return_sequences: int = 1
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""
    dataset: str = "humaneval"
    limit: Optional[int] = None
    iterations: int = 3
    output_file: Optional[str] = None
    save_detailed: bool = False
    verbose: bool = True
    use_agent_chain: bool = True


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)

    def __post_init__(self):
        # Set default values based on environment variables
        if not self.model.model_path:
            env_vars = self._load_env_variables()
            self.model.model_path = env_vars.get('MAIN_MODEL_PATH', '')
            self.model.base_model_path = env_vars.get('BASE_MODEL_PATH')

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'model': self.model.__dict__,
            'generation': self.generation.__dict__,
            'benchmark': self.benchmark.__dict__
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create configuration from dictionary."""
        config = cls()

        if 'model' in config_dict:
            for key, value in config_dict['model'].items():
                if hasattr(config.model, key):
                    setattr(config.model, key, value)

        if 'generation' in config_dict:
            for key, value in config_dict['generation'].items():
                if hasattr(config.generation, key):
                    setattr(config.generation, key, value)

        if 'benchmark' in config_dict:
            for key, value in config_dict['benchmark'].items():
                if hasattr(config.benchmark, key):
                    setattr(config.benchmark, key, value)

        return config

    def _load_env_variables(self) -> Dict[str, str]:
        """Load environment variables from .env file."""
        env_vars = {}
        try:
            with open('.env', 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        env_vars[key.strip()] = value.strip()
        except FileNotFoundError:
            logger.warning(".env file not found")
        except Exception as e:
            logger.error(f"Error loading .env file: {e}")

        return env_vars


def load_config(config_path: Optional[str] = None, cli_args: Optional[argparse.Namespace] = None) -> Config:
    """
    Load configuration from file, CLI arguments, and environment variables.

    Args:
        config_path: Path to YAML config file.
        cli_args: Parsed CLI arguments.

    Returns:
        Complete Config object.
    """
    # Start with default configuration
    config = Config()

    # Load from YAML file if provided
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)

            if yaml_config:
                config = Config.from_dict(yaml_config)
                logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")

    # Override with CLI arguments if provided
    if cli_args:
        # Model configuration
        if hasattr(cli_args, 'model_path') and cli_args.model_path:
            config.model.model_path = cli_args.model_path
        if hasattr(cli_args, 'base_model_path') and cli_args.base_model_path:
            config.model.base_model_path = cli_args.base_model_path
        if hasattr(cli_args, 'use_lora'):
            config.model.use_lora = cli_args.use_lora
        if hasattr(cli_args, 'device') and cli_args.device:
            config.model.device = cli_args.device

        # Generation configuration
        if hasattr(cli_args, 'max_new_tokens'):
            config.generation.max_new_tokens = cli_args.max_new_tokens
        if hasattr(cli_args, 'temperature'):
            config.generation.temperature = cli_args.temperature

        # Benchmark configuration
        if hasattr(cli_args, 'dataset') and cli_args.dataset:
            config.benchmark.dataset = cli_args.dataset
        if hasattr(cli_args, 'limit'):
            config.benchmark.limit = cli_args.limit
        if hasattr(cli_args, 'iterations'):
            config.benchmark.iterations = cli_args.iterations
        if hasattr(cli_args, 'output_file') and cli_args.output_file:
            config.benchmark.output_file = cli_args.output_file
        if hasattr(cli_args, 'verbose'):
            config.benchmark.verbose = cli_args.verbose
        if hasattr(cli_args, 'no_use_agent_chain') and cli_args.no_use_agent_chain:
            config.benchmark.use_agent_chain = False

    return config


def parse_cli_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run LLM benchmarks")

    # Model configuration
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--model-path', type=str, help='Path to model')
    model_group.add_argument('--base-model-path', type=str, help='Path to base model (for LoRA)')
    model_group.add_argument('--use-lora', action='store_true', help='Load model as LoRA')
    model_group.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto', help='Device to use')

    # Generation configuration
    gen_group = parser.add_argument_group('Generation Configuration')
    gen_group.add_argument('--max-new-tokens', type=int, help='Maximum new tokens to generate')
    gen_group.add_argument('--temperature', type=float, help='Generation temperature')

    # Benchmark configuration
    bench_group = parser.add_argument_group('Benchmark Configuration')
    bench_group.add_argument('--dataset', choices=['humaneval', 'mbpp'], default='humaneval',
                             help='Dataset to benchmark')
    bench_group.add_argument('--limit', type=int, help='Limit number of examples to run')
    bench_group.add_argument('--iterations', type=int, default=3, help='Number of self-correction iterations')
    bench_group.add_argument('--output-file', type=str, help='Output file for results')
    bench_group.add_argument('--config', type=str, help='Configuration file path')
    bench_group.add_argument('--verbose', action='store_true', default=True, help='Verbose output')
    bench_group.add_argument('--no-use-agent-chain', action='store_true', help='Disable agent chain for iterative code correction')

    # Load arguments CLI arguments
    args = parser.parse_args()

    return args